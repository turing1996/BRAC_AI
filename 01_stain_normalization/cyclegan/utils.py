"""Shared checkpoint, reproducibility, and preview helpers."""

import random
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid


class ReplayBuffer:
    def __init__(self, max_size: int = 50) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.data: list[torch.Tensor] = []

    def push_and_pop(self, batch: torch.Tensor) -> torch.Tensor:
        returned = []
        for item in batch.detach():
            item = item.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(item)
                returned.append(item)
            elif random.random() > 0.5:
                index = random.randrange(self.max_size)
                returned.append(self.data[index].clone())
                self.data[index] = item
            else:
                returned.append(item)
        return torch.cat(returned)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_weights(module: torch.nn.Module) -> None:
    name = module.__class__.__name__
    if "Conv" in name and hasattr(module, "weight"):
        torch.nn.init.normal_(module.weight.data, 0.0, 0.02)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def load_weights(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)


def save_weights(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def save_preview(tensors: list[torch.Tensor], path: Path) -> None:
    grid = make_grid(
        torch.cat([tensor[:1].detach().cpu() for tensor in tensors]),
        nrow=2,
        normalize=True,
        value_range=(-1, 1),
    )
    array = (
        grid.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, quality=95)
