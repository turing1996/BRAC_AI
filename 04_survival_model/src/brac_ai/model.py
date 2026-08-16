from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .pretrained_vit.model import ViT
from .pretrained_vit.utils import load_pretrained_weights


class TokenCrossAttention(nn.Module):
    """H&E CLS query attending to morphology tokens."""

    def __init__(self, dim: int = 768, heads: int = 12, dropout: float = 0.1, gamma_init: float = 0.05) -> None:
        super().__init__()
        self.he_norm = nn.LayerNorm(dim, eps=1e-6)
        self.morphology_norm = nn.LayerNorm(dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(dim, eps=1e-6)
        self.gamma = nn.Parameter(torch.tensor([float(gamma_init)]))

    def forward(self, he_tokens: torch.Tensor, morphology_tokens: torch.Tensor) -> torch.Tensor:
        q = self.he_norm(he_tokens[:, :1])
        kv = self.morphology_norm(morphology_tokens)
        # Match the previous Morphology-MLP implementation exactly: request
        # averaged attention weights even though they are not consumed downstream.
        # This avoids switching to a different optimized attention kernel when
        # reproducing earlier runs.
        attended, _ = self.attention(
            q,
            kv,
            kv,
            need_weights=True,
            average_attn_weights=True,
        )
        return self.out_norm(he_tokens[:, :1] + self.gamma * self.dropout(attended))


class CoxHead(nn.Module):
    def __init__(self, dim: int = 768, hidden: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MorphologyMLPEncoder(nn.Module):
    """Patch-wise MLP encoder for a 2-channel 384x384 morphology / UMAP map."""

    def __init__(
        self,
        image_size: int = 384,
        patch_size: int = 16,
        in_channels: int = 2,
        dim: int = 768,
        hidden: int = 1024,
        depth: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by morphology_patch_size")
        if depth < 0:
            raise ValueError("morphology_depth must be non-negative")
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        grid = self.image_size // self.patch_size
        self.num_tokens = grid * grid
        patch_dim = self.in_channels * self.patch_size * self.patch_size
        self.patch_mlp = nn.Sequential(
            nn.LayerNorm(patch_dim, eps=1e-6),
            nn.Linear(patch_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        self.blocks = nn.ModuleList([ResidualMLPBlock(dim, hidden, dropout) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, morphology_map: torch.Tensor) -> torch.Tensor:
        if morphology_map.ndim != 4:
            raise ValueError(f"Expected BxCxHxW morphology map; got {tuple(morphology_map.shape)}")
        _, channels, height, width = morphology_map.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} morphology channels; got {channels}")
        if (height, width) != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected {self.image_size}x{self.image_size} morphology map; got {height}x{width}"
            )
        patches = F.unfold(
            morphology_map,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)
        tokens = self.patch_mlp(patches) + self.position_embedding
        for block in self.blocks:
            tokens = block(tokens)
        return self.output_norm(tokens)


class MorphologyMLPCox(nn.Module):
    def __init__(self, cfg: dict, initialize_pretrained: bool = True) -> None:
        super().__init__()
        self.ViTbranch1 = ViT("B_16_imagenet1k", pretrained=False)
        weights = cfg.get("pretrained_weights") if initialize_pretrained else None
        if weights:
            path = Path(weights)
            if not path.is_file():
                raise FileNotFoundError(f"Pretrained ViT weights not found: {path}")
            load_pretrained_weights(self.ViTbranch1, weights_path=str(path))

        self.morphology_encoder = MorphologyMLPEncoder(
            image_size=int(cfg.get("image_size", 384)),
            patch_size=int(cfg.get("morphology_patch_size", 16)),
            in_channels=int(cfg.get("morphology_channels", 2)),
            hidden=int(cfg.get("morphology_hidden_dim", 1024)),
            depth=int(cfg.get("morphology_depth", 2)),
            dropout=float(cfg.get("morphology_dropout", 0.15)),
        )
        self.cross_attention = TokenCrossAttention(
            heads=int(cfg.get("cross_attention_heads", 12)),
            dropout=float(cfg.get("cross_attention_dropout", 0.10)),
            gamma_init=float(cfg.get("cross_attention_gamma_init", 0.05)),
        )
        self.os_head = CoxHead(
            hidden=int(cfg.get("head_hidden_dim", 64)),
            dropout=float(cfg.get("head_dropout", 0.30)),
        )
        self.dfs_head = CoxHead(
            hidden=int(cfg.get("head_hidden_dim", 64)),
            dropout=float(cfg.get("head_dropout", 0.30)),
        )

    def configure_backbone_trainability(self, last_n_blocks: int) -> None:
        for parameter in self.ViTbranch1.parameters():
            parameter.requires_grad_(False)
        if last_n_blocks < 0:
            for parameter in self.ViTbranch1.parameters():
                parameter.requires_grad_(True)
        elif last_n_blocks > 0:
            for block in self.ViTbranch1.transformer.blocks[-last_n_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)

    def enforce_frozen_backbone_eval(self) -> None:
        self.ViTbranch1.eval()
        for block in self.ViTbranch1.transformer.blocks:
            if any(parameter.requires_grad for parameter in block.parameters()):
                block.train()

    def forward(self, image: torch.Tensor, morphology_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        he_tokens = self.ViTbranch1.return_high_level_fm(image)
        morphology_tokens = self.morphology_encoder(morphology_map)
        fused = self.cross_attention(he_tokens, morphology_tokens)[:, 0]
        return self.os_head(fused), self.dfs_head(fused)


def build_model(model_config: dict, initialize_pretrained: bool = True) -> MorphologyMLPCox:
    if int(model_config.get("image_size", 384)) != 384:
        raise ValueError("The bundled ViT-B/16 backbone requires image_size=384")
    return MorphologyMLPCox(model_config, initialize_pretrained=initialize_pretrained)
