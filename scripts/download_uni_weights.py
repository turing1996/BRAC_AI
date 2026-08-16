#!/usr/bin/env python
"""Download the gated original UNI checkpoint after the user has obtained access."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "weights" / "uni"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id="MahmoodLab/UNI",
        filename="pytorch_model.bin",
        local_dir=str(args.output_dir),
    )
    print(f"UNI checkpoint available at: {path}")


if __name__ == "__main__":
    main()
