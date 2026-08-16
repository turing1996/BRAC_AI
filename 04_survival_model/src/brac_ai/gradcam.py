from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from .checkpoint import load_checkpoint
from .config import load_config
from .data import discover_ids, make_dataset
from .model import build_model


class ViTGradCAMHook:
    """Capture activations and output gradients from a ViT token layer."""

    def __init__(self, layer: nn.Module) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module: nn.Module, _inputs: tuple[object, ...], output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("Grad-CAM target layer must return a Tensor")
        self.activations = output
        self.gradients = None
        output.register_hook(self._gradient_hook)

    def _gradient_hook(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient

    def close(self) -> None:
        self._handle.remove()


def _resolve_target_layer(model: nn.Module, block_index: int) -> nn.Module:
    blocks = model.ViTbranch1.transformer.blocks
    n_blocks = len(blocks)
    resolved = block_index if block_index >= 0 else n_blocks + block_index
    if resolved < 0 or resolved >= n_blocks:
        raise ValueError(f"target-block {block_index} is out of range for {n_blocks} H&E ViT blocks")
    return blocks[resolved].norm1


def _token_gradcam(
    model: nn.Module,
    hook: ViTGradCAMHook,
    image: torch.Tensor,
    morphology_map: torch.Tensor,
    endpoint: str,
) -> tuple[np.ndarray, float]:
    model.zero_grad(set_to_none=True)
    os_risk, dfs_risk = model(image, morphology_map)
    target = os_risk.sum() if endpoint == "os" else dfs_risk.sum()
    risk_value = float(target.detach().cpu().item())
    target.backward()

    if hook.activations is None or hook.gradients is None:
        raise RuntimeError("Failed to capture Grad-CAM activations/gradients")

    activations = hook.activations.detach()
    gradients = hook.gradients.detach()
    if activations.ndim != 3 or gradients.shape != activations.shape:
        raise RuntimeError(
            f"Expected BxSxD target activations/gradients; got {tuple(activations.shape)} and {tuple(gradients.shape)}"
        )
    if activations.shape[0] != 1:
        raise ValueError("Grad-CAM requires batch size 1")

    # Remove the CLS token. The remaining 576 tokens correspond to a 24x24
    # H&E patch grid for the 384x384 ViT-B/16 input.
    patch_activations = activations[0, 1:, :]
    patch_gradients = gradients[0, 1:, :]
    n_tokens = int(patch_activations.shape[0])
    grid = int(round(n_tokens ** 0.5))
    if grid * grid != n_tokens:
        raise RuntimeError(f"H&E patch token count {n_tokens} cannot be reshaped to a square grid")

    # Standard Grad-CAM channel weighting: global-average-pool gradients over
    # spatial tokens, then compute the weighted activation sum and ReLU.
    channel_weights = patch_gradients.mean(dim=0)
    cam = (patch_activations * channel_weights.unsqueeze(0)).sum(dim=-1)
    cam = torch.relu(cam).reshape(1, 1, grid, grid)
    cam = F.interpolate(
        cam,
        size=(int(image.shape[-2]), int(image.shape[-1])),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    cam_min = cam.min()
    cam_max = cam.max()
    if float((cam_max - cam_min).abs().cpu()) > 1e-12:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)
    return cam.cpu().numpy().astype(np.float32), risk_value


def _save_visualizations(
    image_path: Path,
    cam_384: np.ndarray,
    heatmap_path: Path,
    overlay_path: Path,
    alpha: float,
    cmap_name: str,
) -> None:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        original = np.asarray(rgb, dtype=np.float32) / 255.0

    cam_img = Image.fromarray(np.uint8(np.clip(cam_384, 0.0, 1.0) * 255), mode="L")
    cam_original = np.asarray(
        cam_img.resize((width, height), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    colored = cmap(np.clip(cam_original, 0.0, 1.0))[..., :3].astype(np.float32)
    overlay = np.clip((1.0 - alpha) * original + alpha * colored, 0.0, 1.0)

    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(heatmap_path, cam_original, cmap=cmap_name, vmin=0.0, vmax=1.0)
    Image.fromarray(np.uint8(overlay * 255.0), mode="RGB").save(overlay_path)


def _select_ids(all_ids: list[str], requested: list[str] | None, max_samples: int | None) -> list[str]:
    if requested:
        available = set(all_ids)
        missing = [sample_id for sample_id in requested if sample_id not in available]
        if missing:
            raise ValueError(f"Requested sample IDs not found in cohort: {missing[:5]}")
        selected = list(requested)
    else:
        selected = list(all_ids)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        selected = selected[:max_samples]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate H&E ViT Grad-CAM maps for the minimal Morphology-MLP Cox model."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--cohort",
        choices=("tcga_train", "tcga_validation", "external"),
        default="external",
    )
    parser.add_argument("--endpoint", choices=("os", "dfs", "both"), default="both")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help="Visualize only this exact sample ID. Repeat the option for multiple samples.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--target-block",
        type=int,
        default=-1,
        help="H&E ViT block whose norm1 token representation is used for Grad-CAM; default=-1 (last block).",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap opacity in the overlay [0,1].")
    parser.add_argument("--cmap", default="jet", help="Matplotlib colormap for the heatmap/overlay.")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0,1]")

    cfg = load_config(args.config)
    requested_device = str(cfg.get("training", {}).get("device", "auto"))
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)

    model = build_model(cfg["model"], initialize_pretrained=False).to(device)
    payload = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    root = Path(cfg["data"][args.cohort])
    all_ids = discover_ids(root)
    sample_ids = _select_ids(all_ids, args.sample_id, args.max_samples)
    dataset = make_dataset(root, cfg, sample_ids)

    output_root = Path(
        args.output
        or Path(cfg["training"]["output_dir"]) / f"gradcam_{args.cohort}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    endpoints = ("os", "dfs") if args.endpoint == "both" else (args.endpoint,)
    target_layer = _resolve_target_layer(model, args.target_block)
    hook = ViTGradCAMHook(target_layer)
    rows: list[dict[str, object]] = []

    iterator = range(len(dataset))
    if not args.no_progress:
        iterator = tqdm(iterator, desc=f"Grad-CAM {args.cohort}", unit="sample")

    try:
        with torch.enable_grad():
            for index in iterator:
                item = dataset[index]
                sample_id = str(item["sample_id"])
                image = item["image"].unsqueeze(0).to(device, non_blocking=True)
                morphology_map = item["morphology_map"].unsqueeze(0).to(device, non_blocking=True)
                image_path = dataset._image_path(sample_id)

                row: dict[str, object] = {
                    "sample_id": sample_id,
                    "patient_id": str(item["patient_id"]),
                    "checkpoint_epoch": int(payload.get("epoch", -1)) if isinstance(payload, dict) else -1,
                    "target_block": int(args.target_block),
                }
                for endpoint in endpoints:
                    cam, risk = _token_gradcam(model, hook, image, morphology_map, endpoint)
                    endpoint_dir = output_root / endpoint
                    heatmap_path = endpoint_dir / "heatmap" / f"{sample_id}.png"
                    overlay_path = endpoint_dir / "overlay" / f"{sample_id}.png"
                    _save_visualizations(
                        image_path=image_path,
                        cam_384=cam,
                        heatmap_path=heatmap_path,
                        overlay_path=overlay_path,
                        alpha=float(args.alpha),
                        cmap_name=str(args.cmap),
                    )
                    row[f"{endpoint}_risk"] = risk
                    row[f"{endpoint}_heatmap"] = str(heatmap_path)
                    row[f"{endpoint}_overlay"] = str(overlay_path)
                rows.append(row)
    finally:
        hook.close()

    fieldnames: list[str] = ["sample_id", "patient_id", "checkpoint_epoch", "target_block"]
    for endpoint in endpoints:
        fieldnames.extend([f"{endpoint}_risk", f"{endpoint}_heatmap", f"{endpoint}_overlay"])
    with (output_root / "gradcam_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Grad-CAM complete: {len(rows)} samples -> {output_root}")


if __name__ == "__main__":
    main()
