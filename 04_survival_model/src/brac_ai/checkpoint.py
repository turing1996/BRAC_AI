from __future__ import annotations
from pathlib import Path
import torch


def save_checkpoint(path, model, epoch, config, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "epoch": int(epoch),
        "config": config,
        "metrics": metrics,
    }, path)


def load_checkpoint(model, path, device="cpu"):
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    state = payload.get("state_dict", payload)
    if state and all(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return payload
