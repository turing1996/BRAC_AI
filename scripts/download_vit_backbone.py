#!/usr/bin/env python
"""Download the ImageNet-pretrained ViT-B/16 checkpoint expected by BRAC-AI."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URL = "https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16_imagenet1k.pth"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "04_survival_model" / "pretrained" / "B_16_imagenet1k.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        print(f"Already exists: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {args.output}")
    urllib.request.urlretrieve(URL, args.output)
    print("Done")


if __name__ == "__main__":
    main()
