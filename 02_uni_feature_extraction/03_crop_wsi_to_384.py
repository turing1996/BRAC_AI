#!/usr/bin/env python
"""Crop each WSI to the valid-patch bounding box and resize it to 384x384."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
import time
from pathlib import Path

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "wsi_384"


def load_wsi_module():
    """Reuse the tested OpenSlide/Pillow readers from script 01."""
    script_path = SCRIPT_DIR / "01_tile_wsi.py"
    spec = importlib.util.spec_from_file_location("uni_infer_tile_wsi", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import WSI reader from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_manifests(patches_dir: Path, manifest_name: str) -> list[Path]:
    direct = patches_dir / manifest_name
    if direct.is_file():
        return [direct]
    return sorted(path for path in patches_dir.rglob(manifest_name) if path.is_file())


def read_manifest(manifest_path: Path) -> tuple[Path, int, int, list[dict[str, str]]]:
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("manifest contains no valid patches")

    required = {"slide_path", "level", "x", "y", "patch_size"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"manifest is missing columns: {', '.join(sorted(missing))}")

    slide_values = {row["slide_path"].strip() for row in rows}
    level_values = {int(row["level"]) for row in rows}
    patch_sizes = {int(row["patch_size"]) for row in rows}
    if len(slide_values) != 1:
        raise ValueError(f"expected one slide_path, found {len(slide_values)}")
    if len(level_values) != 1 or len(patch_sizes) != 1:
        raise ValueError("all rows must use the same level and patch_size")

    slide_path = Path(next(iter(slide_values)))
    if not slide_path.is_absolute():
        slide_path = (manifest_path.parent / slide_path).resolve()
    return slide_path, next(iter(level_values)), next(iter(patch_sizes)), rows


def relocate_slide(recorded_path: Path, wsi_root: Path | None) -> Path:
    if recorded_path.is_file():
        return recorded_path
    if wsi_root is None:
        raise FileNotFoundError(f"recorded WSI does not exist: {recorded_path}")
    matches = sorted(path for path in wsi_root.rglob(recorded_path.name) if path.is_file())
    if not matches:
        raise FileNotFoundError(
            f"recorded WSI does not exist and {recorded_path.name!r} was not found under {wsi_root}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple files named {recorded_path.name!r} were found under {wsi_root}; "
            "restore the original slide_path or use an unambiguous WSI root"
        )
    return matches[0]


def valid_bbox_level0(
    rows: list[dict[str, str]],
    patch_size: int,
    patch_level_downsample: float,
    slide_width: int,
    slide_height: int,
    margin: int,
) -> tuple[int, int, int, int]:
    patch_extent = max(1, int(math.ceil(patch_size * patch_level_downsample)))
    xs = [int(row["x"]) for row in rows]
    ys = [int(row["y"]) for row in rows]
    x0 = max(0, min(xs) - margin)
    y0 = max(0, min(ys) - margin)
    x1 = min(slide_width, max(xs) + patch_extent + margin)
    y1 = min(slide_height, max(ys) + patch_extent + margin)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid bounding box: {(x0, y0, x1, y1)}")
    return x0, y0, x1, y1


def choose_read_level(reader, crop_width: int, crop_height: int, target_size: int) -> int:
    """Use the coarsest level that still supplies at least target-size detail."""
    desired_downsample = max(1.0, min(crop_width, crop_height) / target_size)
    eligible = [
        level
        for level in range(reader.level_count)
        if reader.level_downsample(level) <= desired_downsample
    ]
    return max(eligible, key=reader.level_downsample) if eligible else 0


def resize_image(image: Image.Image, size: int, mode: str, background: tuple[int, int, int]) -> Image.Image:
    if mode == "stretch":
        return image.resize((size, size), Image.Resampling.LANCZOS)

    copy = image.copy()
    copy.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), background)
    x = (size - copy.width) // 2
    y = (size - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def output_path_for_manifest(
    manifest_path: Path, patches_dir: Path, output_dir: Path, image_format: str
) -> Path:
    group_dir = manifest_path.parent
    if group_dir == patches_dir:
        relative = Path(patches_dir.name)
    else:
        relative = group_dir.relative_to(patches_dir)
    return output_dir / relative.parent / f"{relative.name}.{image_format}"


def crop_one(manifest_path: Path, patches_dir: Path, output_dir: Path, wsi_module, args):
    recorded_slide, patch_level, patch_size, rows = read_manifest(manifest_path)
    slide_path = relocate_slide(recorded_slide, args.wsi_root)
    reader = wsi_module.open_slide(slide_path)
    try:
        if not 0 <= patch_level < reader.level_count:
            raise ValueError(
                f"patch level {patch_level} is unavailable; WSI has {reader.level_count} level(s)"
            )
        slide_width, slide_height = reader.level_dimensions(0)
        bbox = valid_bbox_level0(
            rows,
            patch_size,
            reader.level_downsample(patch_level),
            slide_width,
            slide_height,
            args.margin,
        )
        x0, y0, x1, y1 = bbox
        crop_width, crop_height = x1 - x0, y1 - y0
        read_level = choose_read_level(reader, crop_width, crop_height, args.size)
        read_downsample = reader.level_downsample(read_level)
        read_width = max(1, int(math.ceil(crop_width / read_downsample)))
        read_height = max(1, int(math.ceil(crop_height / read_downsample)))
        region = reader.read_region(x0, y0, read_level, read_width, read_height)
        result = resize_image(region, args.size, args.resize_mode, tuple(args.pad_color))

        output_path = output_path_for_manifest(
            manifest_path, patches_dir, output_dir, args.image_format
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite or not output_path.exists():
            save_kwargs = {"quality": args.jpeg_quality} if args.image_format == "jpg" else {}
            result.save(output_path, **save_kwargs)

        return {
            "output_path": str(output_path),
            "manifest_path": str(manifest_path),
            "slide_path": str(slide_path),
            "valid_patch_count": len(rows),
            "bbox_x0": x0,
            "bbox_y0": y0,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_width": crop_width,
            "bbox_height": crop_height,
            "patch_level": patch_level,
            "read_level": read_level,
            "read_level_downsample": read_downsample,
            "read_width": read_width,
            "read_height": read_height,
            "output_width": args.size,
            "output_height": args.size,
            "resize_mode": args.resize_mode,
        }
    finally:
        reader.close()


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use script 01 patches.csv files to crop each WSI to its minimum valid-patch "
            "bounding rectangle, then resize to a fixed square image."
        )
    )
    parser.add_argument(
        "--patches-dir", type=Path, required=True,
        help="A script-01 patch directory, or a root containing patch directories.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output root (default: uni-infer/wsi_384).",
    )
    parser.add_argument(
        "--wsi-root", type=Path, default=None,
        help="Optional WSI search root if slide_path values in patches.csv are no longer valid.",
    )
    parser.add_argument("--manifest-name", default="patches.csv")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument(
        "--resize-mode", choices=("stretch", "pad"), default="stretch",
        help="stretch: resize directly; pad: preserve aspect ratio and add borders.",
    )
    parser.add_argument(
        "--pad-color", type=int, nargs=3, default=(255, 255, 255),
        metavar=("R", "G", "B"),
    )
    parser.add_argument(
        "--margin", type=int, default=0,
        help="Optional margin around the valid bounding box, in level-0 pixels.",
    )
    parser.add_argument("--image-format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args) -> None:
    if not args.patches_dir.is_dir():
        parser.error(f"patches directory does not exist: {args.patches_dir}")
    if args.wsi_root is not None and not args.wsi_root.is_dir():
        parser.error(f"WSI root does not exist: {args.wsi_root}")
    if args.size <= 0 or args.margin < 0:
        parser.error("--size must be positive and --margin cannot be negative")
    if any(value < 0 or value > 255 for value in args.pad_color):
        parser.error("--pad-color values must be between 0 and 255")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    patches_dir = args.patches_dir.resolve()
    output_dir = args.output_dir.resolve()
    args.wsi_root = args.wsi_root.resolve() if args.wsi_root else None
    manifests = find_manifests(patches_dir, args.manifest_name)
    if not manifests:
        parser.error(f"no {args.manifest_name!r} files found under {patches_dir}")

    wsi_module = load_wsi_module()
    print(f"Found {len(manifests)} manifest(s); output: {output_dir}")
    summaries: list[dict[str, object]] = []
    failures: list[tuple[Path, Exception]] = []
    started = time.time()
    for index, manifest_path in enumerate(manifests, start=1):
        try:
            summary = crop_one(manifest_path, patches_dir, output_dir, wsi_module, args)
            summaries.append(summary)
            print(
                f"[{index}/{len(manifests)}] {Path(summary['slide_path']).name}: "
                f"bbox=({summary['bbox_x0']}, {summary['bbox_y0']}, "
                f"{summary['bbox_x1']}, {summary['bbox_y1']}) -> {args.size}x{args.size}"
            )
        except Exception as exc:
            failures.append((manifest_path, exc))
            print(f"[{index}/{len(manifests)}] ERROR {manifest_path}: {exc}", file=sys.stderr)

    write_summary(output_dir / "crop_summary.csv", summaries)
    print(
        f"Finished in {time.time() - started:.1f}s: succeeded={len(summaries)}, "
        f"failed={len(failures)}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
