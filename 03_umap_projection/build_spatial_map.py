#!/usr/bin/env python
"""Reconstruct a two-channel spatial morphology map from tile-level UMAP coordinates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def read_coordinates(path: Path) -> np.ndarray:
    """Read row-aligned x,y tile coordinates from the UNI feature-index CSV."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Coordinate CSV contains no rows: {path}")
    required = {"feature_index", "x", "y"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Coordinate CSV must contain {sorted(required)}")

    indexed: list[tuple[int, int, int]] = []
    for row in rows:
        if row["x"] == "" or row["y"] == "":
            continue
        indexed.append((int(row["feature_index"]), int(row["x"]), int(row["y"])))
    if not indexed:
        raise ValueError(f"No valid x,y coordinates were found in {path}")
    indexed.sort(key=lambda item: item[0])
    return np.asarray(indexed, dtype=np.int64)


def load_umap(path: Path, key: str = "umap") -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive.files:
                raise KeyError(f"UMAP key {key!r} not found in {path}; available={archive.files}")
            arr = np.asarray(archive[key])
    else:
        raise ValueError("--umap must be .npy or .npz")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected UMAP array [N,2], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("UMAP array contains NaN/Inf")
    return arr


def resize_channel(channel: np.ndarray, size: int) -> np.ndarray:
    """Resize one float channel using bilinear interpolation without quantization."""
    image = Image.fromarray(np.asarray(channel, dtype=np.float32), mode="F")
    resized = image.resize((size, size), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def build_map(umap: np.ndarray, coordinates: np.ndarray, size: int, stride: int | None) -> np.ndarray:
    feature_indices = coordinates[:, 0]
    if feature_indices.min() < 0 or feature_indices.max() >= len(umap):
        raise ValueError("feature_index values exceed the UMAP row range")

    x = coordinates[:, 1]
    y = coordinates[:, 2]
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    if stride is None:
        dx = np.diff(unique_x)
        dy = np.diff(unique_y)
        candidates = np.concatenate([dx[dx > 0], dy[dy > 0]])
        stride = int(np.gcd.reduce(candidates.astype(np.int64))) if candidates.size else 1
    if stride <= 0:
        raise ValueError("stride must be positive")

    x0, y0 = int(x.min()), int(y.min())
    gx = ((x - x0) // stride).astype(np.int64)
    gy = ((y - y0) // stride).astype(np.int64)
    width = int(gx.max()) + 1
    height = int(gy.max()) + 1
    grid = np.zeros((height, width, 2), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.int32)

    for idx, xx, yy in zip(feature_indices, gx, gy):
        grid[yy, xx] += umap[idx]
        counts[yy, xx] += 1
    occupied = counts > 0
    if np.any(counts > 1):
        grid[occupied] /= counts[occupied][:, None]

    if size != height or size != width:
        grid = np.stack(
            [resize_channel(grid[..., channel], size) for channel in range(2)],
            axis=-1,
        )
    return np.asarray(grid, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--umap", type=Path, required=True, help="Tile UMAP .npz/.npy file")
    parser.add_argument("--coordinates", type=Path, required=True, help="Row-aligned UNI CSV index")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy morphology map")
    parser.add_argument("--umap-key", default="umap")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Tile stride in level-0 pixels; inferred from coordinates when omitted.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.size <= 0:
        parser.error("--size must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; pass --overwrite to replace it")

    umap = load_umap(args.umap, args.umap_key)
    coordinates = read_coordinates(args.coordinates)
    morphology = build_map(umap, coordinates, args.size, args.stride)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.save(handle, morphology)
    print(f"Saved {morphology.shape} float32 morphology map -> {args.output}")


if __name__ == "__main__":
    main()
