#!/usr/bin/env python
"""Tile SVS/TIFF whole-slide images into fixed-size image patches."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image
from tqdm import tqdm


WSI_SUFFIXES = {".svs", ".tif", ".tiff"}


class SlideReader(Protocol):
    @property
    def level_count(self) -> int: ...

    def level_dimensions(self, level: int) -> tuple[int, int]: ...

    def level_downsample(self, level: int) -> float: ...

    def read_region(
        self, x_level0: int, y_level0: int, level: int, width: int, height: int
    ) -> Image.Image: ...

    def read_patch(self, x: int, y: int, level: int, size: int) -> Image.Image: ...

    def close(self) -> None: ...


def import_openslide():
    """Import OpenSlide, honoring OPENSLIDE_PATH on Windows."""
    dll_dir = os.environ.get("OPENSLIDE_PATH")
    try:
        if dll_dir and hasattr(os, "add_dll_directory"):
            with os.add_dll_directory(dll_dir):
                import openslide
        else:
            import openslide
        return openslide, None
    except (ImportError, OSError) as exc:
        return None, exc


class OpenSlideReader:
    def __init__(self, path: Path, openslide_module) -> None:
        self._slide = openslide_module.OpenSlide(str(path))

    @property
    def level_count(self) -> int:
        return self._slide.level_count

    def level_dimensions(self, level: int) -> tuple[int, int]:
        return tuple(self._slide.level_dimensions[level])

    def level_downsample(self, level: int) -> float:
        return float(self._slide.level_downsamples[level])

    def read_region(
        self, x_level0: int, y_level0: int, level: int, width: int, height: int
    ) -> Image.Image:
        return self._slide.read_region(
            (x_level0, y_level0), level, (width, height)
        ).convert("RGB")

    def read_patch(self, x: int, y: int, level: int, size: int) -> Image.Image:
        downsample = self.level_downsample(level)
        x_level0 = int(round(x * downsample))
        y_level0 = int(round(y * downsample))
        return self.read_region(x_level0, y_level0, level, size, size)

    def close(self) -> None:
        self._slide.close()


class PillowTiffReader:
    """Fallback for TIFF files that are not readable by OpenSlide."""

    def __init__(self, path: Path) -> None:
        Image.MAX_IMAGE_PIXELS = None
        self._image = Image.open(path)
        self._dimensions: list[tuple[int, int]] = []
        for frame in range(getattr(self._image, "n_frames", 1)):
            self._image.seek(frame)
            self._dimensions.append(self._image.size)
        self._image.seek(0)

    @property
    def level_count(self) -> int:
        return len(self._dimensions)

    def level_dimensions(self, level: int) -> tuple[int, int]:
        return self._dimensions[level]

    def level_downsample(self, level: int) -> float:
        width0, height0 = self._dimensions[0]
        width, height = self._dimensions[level]
        return max(width0 / width, height0 / height)

    def read_region(
        self, x_level0: int, y_level0: int, level: int, width: int, height: int
    ) -> Image.Image:
        downsample = self.level_downsample(level)
        x = int(round(x_level0 / downsample))
        y = int(round(y_level0 / downsample))
        self._image.seek(level)
        return self._image.crop((x, y, x + width, y + height)).convert("RGB")

    def read_patch(self, x: int, y: int, level: int, size: int) -> Image.Image:
        downsample = self.level_downsample(level)
        return self.read_region(
            int(round(x * downsample)), int(round(y * downsample)), level, size, size
        )

    def close(self) -> None:
        self._image.close()


def open_slide(path: Path) -> SlideReader:
    openslide, import_error = import_openslide()
    if openslide is not None:
        try:
            return OpenSlideReader(path, openslide)
        except openslide.OpenSlideUnsupportedFormatError:
            if path.suffix.lower() == ".svs":
                raise

    if path.suffix.lower() in {".tif", ".tiff"}:
        return PillowTiffReader(path)

    detail = f" ({import_error})" if import_error else ""
    raise RuntimeError(
        "Reading SVS requires OpenSlide. Install openslide-python and the OpenSlide "
        f"runtime (on Windows, openslide-bin is convenient){detail}"
    )


def tissue_fraction(image: Image.Image, background_threshold: int, dark_threshold: int) -> float:
    """Estimate non-background tissue while excluding transparent/black scanner borders."""
    rgb = np.asarray(image, dtype=np.uint8)
    non_white = np.any(rgb < background_threshold, axis=2)
    non_dark = np.mean(rgb, axis=2) > dark_threshold
    return float(np.mean(non_white & non_dark))


def find_wsi_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in WSI_SUFFIXES:
            raise ValueError(f"Unsupported WSI suffix: {input_path.suffix}")
        return [input_path]
    return sorted(
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in WSI_SUFFIXES
    )


def slide_output_dir(slide_path: Path, input_path: Path, output_root: Path) -> Path:
    if input_path.is_file():
        return output_root / slide_path.stem
    relative = slide_path.relative_to(input_path)
    return output_root / relative.parent / slide_path.stem


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "patch_path", "slide_path", "level", "x", "y", "level_x", "level_y",
        "patch_size", "tissue_fraction",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def tile_one_slide(slide_path: Path, input_path: Path, output_root: Path, args) -> tuple[int, int]:
    reader = open_slide(slide_path)
    output_dir = slide_output_dir(slide_path, input_path, output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not 0 <= args.level < reader.level_count:
            raise ValueError(
                f"level {args.level} is invalid for {slide_path.name}; "
                f"available levels: 0..{reader.level_count - 1}"
            )

        width, height = reader.level_dimensions(args.level)
        downsample = reader.level_downsample(args.level)
        xs = range(0, width - args.patch_size + 1, args.stride)
        ys = range(0, height - args.patch_size + 1, args.stride)
        total_candidates = len(xs) * len(ys)
        kept = 0
        rows: list[dict[str, object]] = []

        positions = ((x, y) for y in ys for x in xs)
        progress = tqdm(
            positions,
            total=total_candidates,
            desc=slide_path.name,
            unit="patch",
            leave=False,
        )
        for x, y in progress:
            patch = reader.read_patch(x, y, args.level, args.patch_size)
            fraction = tissue_fraction(patch, args.background_threshold, args.dark_threshold)
            if fraction < args.min_tissue_fraction:
                continue

            x_level0 = int(round(x * downsample))
            y_level0 = int(round(y * downsample))
            filename = f"{slide_path.stem}_{y_level0}_{x_level0}.{args.image_format}"
            patch_path = output_dir / filename
            if args.overwrite or not patch_path.exists():
                save_kwargs = {"quality": args.jpeg_quality} if args.image_format == "jpg" else {}
                patch.save(patch_path, **save_kwargs)

            rows.append(
                {
                    "patch_path": filename,
                    "slide_path": str(slide_path),
                    "level": args.level,
                    "x": x_level0,
                    "y": y_level0,
                    "level_x": x,
                    "level_y": y,
                    "patch_size": args.patch_size,
                    "tissue_fraction": f"{fraction:.6f}",
                }
            )
            kept += 1
            progress.set_postfix(kept=kept, refresh=False)

        write_manifest(output_dir / "patches.csv", rows)
        return kept, total_candidates
    finally:
        reader.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tile .svs/.tif/.tiff WSI files into fixed-size tissue patches."
    )
    parser.add_argument("--input", type=Path, required=True, help="A WSI file or a directory searched recursively.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "patches",
        help="Patch output root (default: uni-infer/patches).",
    )
    parser.add_argument("--patch-size", type=int, default=256, help="Square patch size in level pixels.")
    parser.add_argument("--stride", type=int, default=None, help="Grid stride; defaults to patch size.")
    parser.add_argument("--level", type=int, default=0, help="WSI pyramid level to tile (default: 0).")
    parser.add_argument(
        "--min-tissue-fraction", type=float, default=0.10,
        help="Minimum estimated tissue fraction in [0,1]; use 0 to retain every full patch.",
    )
    parser.add_argument(
        "--background-threshold", type=int, default=220,
        help="RGB values at or above this threshold are treated as white background.",
    )
    parser.add_argument(
        "--dark-threshold", type=int, default=15,
        help="Pixels with mean RGB at or below this threshold are treated as scanner borders.",
    )
    parser.add_argument("--image-format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing patch images.")
    return parser


def validate_args(parser: argparse.ArgumentParser, args) -> None:
    if not args.input.exists():
        parser.error(f"input does not exist: {args.input}")
    if args.patch_size <= 0:
        parser.error("--patch-size must be positive")
    if args.stride is None:
        args.stride = args.patch_size
    if args.stride <= 0:
        parser.error("--stride must be positive")
    if not 0 <= args.min_tissue_fraction <= 1:
        parser.error("--min-tissue-fraction must be between 0 and 1")
    if not 0 <= args.background_threshold <= 255 or not 0 <= args.dark_threshold <= 255:
        parser.error("pixel thresholds must be between 0 and 255")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    input_path = args.input.resolve()
    output_root = args.output_dir.resolve()
    wsi_files = find_wsi_files(input_path)
    if not wsi_files:
        parser.error(f"no .svs/.tif/.tiff files found under {input_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(wsi_files)} WSI(s); output: {output_root}")
    failed: list[tuple[Path, Exception]] = []
    total_kept = 0
    started = time.time()
    for index, slide_path in enumerate(wsi_files, start=1):
        try:
            kept, candidates = tile_one_slide(slide_path, input_path, output_root, args)
            total_kept += kept
            print(f"[{index}/{len(wsi_files)}] {slide_path.name}: kept {kept}/{candidates}")
        except Exception as exc:  # Continue other slides, but return a failing exit code.
            failed.append((slide_path, exc))
            print(f"[{index}/{len(wsi_files)}] ERROR {slide_path}: {exc}", file=sys.stderr)

    print(f"Finished: {total_kept} patches in {time.time() - started:.1f}s")
    if failed:
        print(f"Failed slides: {len(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
