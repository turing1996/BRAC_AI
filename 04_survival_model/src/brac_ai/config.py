from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    root = path.parent
    for key in ("tcga_train", "tcga_validation", "external"):
        value = config["data"][key]
        if not Path(value).is_absolute():
            config["data"][key] = str((root / value).resolve())
    value = config["model"].get("pretrained_weights")
    if value and not Path(value).is_absolute():
        config["model"]["pretrained_weights"] = str((root / value).resolve())
    value = config["training"]["output_dir"]
    if not Path(value).is_absolute():
        config["training"]["output_dir"] = str((root / value).resolve())
    return config
