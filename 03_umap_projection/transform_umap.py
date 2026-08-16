#!/usr/bin/env python
"""Transform pre-extracted features with a trained UMAP model and save .npz."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import joblib

from umap_io import (
    discover_feature_files,
    is_relative_to,
    load_feature_array,
    save_umap_npz,
    transform_in_batches,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained UMAP model, transform .npy/.npz features, and save "
            "compressed .npz files whose primary key is 'umap'."
        )
    )
    parser.add_argument("input", type=Path, help="Feature file or directory.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("tcga_train_umap.pkl"),
        help="Trained UMAP model (default: tcga_train_umap.pkl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "For one input file: output .npz or output directory. "
            "For a directory: output directory."
        ),
    )
    parser.add_argument(
        "--feature-key",
        help="Array key for .npz input; inferred for common keys or a single-array archive.",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Search input directories recursively."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Transform at most this many rows per model.transform call.",
    )
    parser.add_argument(
        "--copy-npz-metadata",
        action="store_true",
        help="Copy non-feature arrays from an input .npz into its output archive.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output .npz files."
    )
    return parser


def load_model(model_path: Path) -> Any:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model does not exist: {model_path}")
    try:
        model = joblib.load(model_path)
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        raise RuntimeError(
            f"Could not load {model_path}: missing Python module {missing!r}. "
            "Install 03_umap_projection/requirements.txt in the same environment used to run this script."
        ) from exc
    if not callable(getattr(model, "transform", None)):
        raise TypeError(f"Loaded object has no callable transform method: {type(model)!r}")
    return model


def output_mapping(
    input_path: Path, files: list[Path], output_arg: Path
) -> list[tuple[Path, Path]]:
    output_arg = output_arg.expanduser().resolve()
    if input_path.is_file():
        if output_arg.suffix.lower() == ".npz":
            return [(files[0], output_arg)]
        return [(files[0], output_arg / f"{files[0].stem}_umap.npz")]

    if output_arg.suffix.lower() == ".npz":
        raise ValueError("Directory input requires --output to be a directory path.")
    input_root = input_path.expanduser().resolve()
    mapping: list[tuple[Path, Path]] = []
    seen_outputs: set[Path] = set()
    for source in files:
        relative = source.relative_to(input_root)
        destination = output_arg / relative.parent / f"{source.stem}_umap.npz"
        if destination in seen_outputs:
            raise ValueError(
                f"Multiple inputs map to the same output path: {destination}. "
                "Rename same-stem .npy/.npz inputs or process them separately."
            )
        seen_outputs.add(destination)
        mapping.append((source, destination))
    return mapping


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    files = discover_feature_files(input_path, recursive=args.recursive)
    # Avoid feeding old result archives back in when output is inside a recursive input tree.
    if input_path.is_dir() and is_relative_to(output_path, input_path):
        files = [path for path in files if not is_relative_to(path, output_path)]
        if not files:
            raise FileNotFoundError("No input feature files remain after excluding --output.")
    mapping = output_mapping(input_path, files, output_path)
    same_path = [(source, destination) for source, destination in mapping if source == destination]
    if same_path:
        raise ValueError(
            f"Refusing to overwrite an input feature file: {same_path[0][0]}"
        )
    if not args.overwrite:
        existing = [destination for _, destination in mapping if destination.exists()]
        if existing:
            preview = ", ".join(str(path) for path in existing[:3])
            suffix = " ..." if len(existing) > 3 else ""
            raise FileExistsError(
                f"{len(existing)} output file(s) already exist: {preview}{suffix}. "
                "Pass --overwrite to replace them."
            )

    print(f"Loading model: {model_path}", flush=True)
    print(
        "Security note: only load .pkl/joblib models from a trusted source.", flush=True
    )
    model = load_model(model_path)
    print(f"Transforming {len(mapping)} feature file(s).", flush=True)

    total_rows = 0
    started = time.time()
    for index, (source, destination) in enumerate(mapping, start=1):
        features, selected_key = load_feature_array(
            source, feature_key=args.feature_key
        )
        print(
            f"[{index}/{len(mapping)}] {source.name}: {features.shape} -> {destination}",
            flush=True,
        )
        embedding = transform_in_batches(
            model=model, features=features, batch_size=args.batch_size
        )
        save_umap_npz(
            output_path=destination,
            embedding=embedding,
            source_path=source,
            feature_shape=features.shape,
            model_path=model_path,
            input_feature_key=selected_key,
            copy_npz_metadata=args.copy_npz_metadata,
        )
        total_rows += features.shape[0]

    elapsed = time.time() - started
    print(
        f"Done: {total_rows:,} rows in {elapsed:.1f} s. "
        "Load coordinates with np.load(path)['umap'].",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
