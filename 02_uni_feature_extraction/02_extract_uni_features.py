#!/usr/bin/env python
"""Extract 1024-dimensional UNI features from pathology image patches."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "weights" / "uni" / "pytorch_model.bin"
COORDINATE_PATTERN = re.compile(r"_(?P<y>-?\d+)_(?P<x>-?\d+)$")


class PatchDataset(Dataset):
    def __init__(self, paths: list[Path], input_root: Path, transform) -> None:
        self.paths = paths
        self.input_root = input_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        relative_path = path.relative_to(self.input_root).as_posix()
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
            return tensor, relative_path, ""
        except Exception as exc:
            return None, relative_path, f"{type(exc).__name__}: {exc}"


def safe_collate(batch):
    tensors, paths, errors = [], [], []
    for tensor, path, error in batch:
        if tensor is None:
            errors.append((path, error))
        else:
            tensors.append(tensor)
            paths.append(path)
    stacked = torch.stack(tensors) if tensors else None
    return stacked, paths, errors


def discover_groups(input_root: Path) -> dict[Path, list[Path]]:
    grouped: dict[Path, list[Path]] = defaultdict(list)
    for path in sorted(input_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            grouped[path.parent].append(path)
    return dict(sorted(grouped.items(), key=lambda item: str(item[0]).lower()))


def output_stem(group_dir: Path, input_root: Path, output_root: Path) -> Path:
    relative = group_dir.relative_to(input_root)
    if relative == Path("."):
        return output_root / input_root.name
    return output_root / relative


def artifact_path(stem: Path, suffix: str) -> Path:
    """Append a suffix without treating dots in a directory name as an extension."""
    return stem.parent / f"{stem.name}{suffix}"


def parse_coordinates(path: str) -> tuple[int | None, int | None]:
    match = COORDINATE_PATTERN.search(Path(path).stem)
    if match is None:
        return None, None
    return int(match.group("x")), int(match.group("y"))


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_numpy_save(array: np.ndarray, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def write_index_csv(path: Path, image_paths: list[str], coordinates: list[tuple[int | None, int | None]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_index", "patch_path", "filename", "x", "y"])
        for index, (image_path, (x, y)) in enumerate(zip(image_paths, coordinates)):
            writer.writerow([index, image_path, Path(image_path).name, "" if x is None else x, "" if y is None else y])
    temporary.replace(path)


def load_uni(checkpoint: Path, device: torch.device):
    """Instantiate the original UNI ViT-L/16 and load a local pretrained checkpoint.

    The architecture and image transform follow the official MahmoodLab UNI
    manual-loading example for the original 1024-dimensional UNI model.
    The encoder is returned in evaluation mode and is used only under
    ``torch.inference_mode`` below, so no UNI weights are fine-tuned here.
    """
    if not checkpoint.is_file():
        raise FileNotFoundError(f"UNI checkpoint not found: {checkpoint}")

    import timm
    from torchvision import transforms

    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return model, transform


def extract_group(
    group_dir: Path,
    image_files: list[Path],
    input_root: Path,
    model,
    transform,
    device: torch.device,
    args,
) -> tuple[torch.Tensor, list[str], list[tuple[str, str]]]:
    dataset = PatchDataset(image_files, input_root, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=safe_collate,
    )
    all_features: list[torch.Tensor] = []
    all_paths: list[str] = []
    all_errors: list[tuple[str, str]] = []
    amp_enabled = device.type == "cuda" and args.amp != "none"
    amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16

    progress = tqdm(loader, desc=group_dir.name, unit="batch", leave=False)
    with torch.inference_mode():
        for batch, paths, errors in progress:
            all_errors.extend(errors)
            if batch is None:
                continue
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                features = model(batch)
            if features.ndim != 2:
                raise RuntimeError(f"Expected a [N,D] model output, received {tuple(features.shape)}")
            features = features.detach().float().cpu()
            if not torch.isfinite(features).all():
                raise RuntimeError("UNI produced NaN or infinite feature values")
            all_features.append(features)
            all_paths.extend(paths)

    if not all_features:
        raise RuntimeError(f"No readable images in {group_dir}")
    return torch.cat(all_features, dim=0), all_paths, all_errors


def save_group(
    stem: Path,
    features: torch.Tensor,
    paths: list[str],
    errors: list[tuple[str, str]],
    group_dir: Path,
    checkpoint: Path,
    args,
) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    coordinates = [parse_coordinates(path) for path in paths]
    coordinate_tensor = torch.tensor(
        [[-1 if x is None else x, -1 if y is None else y] for x, y in coordinates],
        dtype=torch.int64,
    )
    if args.output_format in {"pt", "both"}:
        payload = {
            "features": features,
            "paths": paths,
            "filenames": [Path(path).name for path in paths],
            "coordinates": coordinate_tensor,
            "coordinate_order": "x,y; -1 denotes unavailable",
            "group_dir": str(group_dir),
            "encoder": "UNI ViT-L/16",
            "checkpoint": str(checkpoint),
            "feature_dim": features.shape[1],
        }
        atomic_torch_save(payload, artifact_path(stem, ".pt"))
    if args.output_format in {"npy", "both"}:
        atomic_numpy_save(features.numpy(), artifact_path(stem, ".npy"))

    write_index_csv(artifact_path(stem, ".csv"), paths, coordinates)
    if errors:
        error_path = stem.with_name(stem.name + "_errors.csv")
        temporary = error_path.with_suffix(error_path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["patch_path", "error"])
            writer.writerows(errors)
        temporary.replace(error_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract pretrained UNI features from patch directories.")
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="Patch directory. Every directory containing images is saved as one feature group.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR / "feature",
        help="Feature output root (default: 02_uni_feature_extraction/feature).",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers; 0 is safest on Windows.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:0.")
    parser.add_argument(
        "--amp", choices=("none", "fp16", "bf16"), default="fp16",
        help="CUDA mixed precision mode; ignored on CPU.",
    )
    parser.add_argument(
        "--output-format", choices=("pt", "npy", "both"), default="pt",
        help="Feature array format. CSV path/coordinate index is always written.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers cannot be negative")

    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    groups = discover_groups(input_root)
    if not groups:
        parser.error(f"no supported patch images found under {input_root}")

    device = resolve_device(args.device)
    print(f"Found {sum(map(len, groups.values()))} images in {len(groups)} group(s)")
    print(f"Loading UNI from {checkpoint} on {device} ...")
    model, transform = load_uni(checkpoint, device)
    print("UNI loaded; extracting features")

    processed = 0
    skipped = 0
    failed: list[tuple[Path, Exception]] = []
    started = time.time()
    for index, (group_dir, image_files) in enumerate(groups.items(), start=1):
        stem = output_stem(group_dir, input_root, output_root)
        expected_files = []
        if args.output_format in {"pt", "both"}:
            expected_files.append(artifact_path(stem, ".pt"))
        if args.output_format in {"npy", "both"}:
            expected_files.append(artifact_path(stem, ".npy"))
        if all(path.exists() for path in expected_files) and not args.overwrite:
            print(f"[{index}/{len(groups)}] skip existing: {', '.join(map(str, expected_files))}")
            skipped += 1
            continue
        try:
            features, paths, errors = extract_group(
                group_dir, image_files, input_root, model, transform, device, args
            )
            save_group(stem, features, paths, errors, group_dir, checkpoint, args)
            processed += 1
            print(
                f"[{index}/{len(groups)}] {group_dir}: {tuple(features.shape)}"
                + (f", unreadable={len(errors)}" if errors else "")
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            failed.append((group_dir, exc))
            print(
                f"[{index}/{len(groups)}] CUDA out of memory for {group_dir}; "
                "rerun with a smaller --batch-size",
                file=sys.stderr,
            )
        except Exception as exc:
            failed.append((group_dir, exc))
            print(f"[{index}/{len(groups)}] ERROR {group_dir}: {exc}", file=sys.stderr)

    print(
        f"Finished in {time.time() - started:.1f}s: processed={processed}, "
        f"skipped={skipped}, failed={len(failed)}, output={output_root}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
