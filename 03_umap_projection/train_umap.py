#!/usr/bin/env python
"""Train a reusable UMAP model from pre-extracted feature arrays."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from umap_io import discover_feature_files, prepare_training_data


def optional_int(value: str) -> int | None:
    if value.strip().lower() in {"none", "null"}:
        return None
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train UMAP from one .npy/.npz feature file or a directory of files."
    )
    parser.add_argument("input", type=Path, help="Feature file or directory.")
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("umap_model_clean.pkl"),
        help="Destination joblib/pickle model (default: umap_model_clean.pkl).",
    )
    parser.add_argument(
        "--output-embedding",
        type=Path,
        help="Training embedding .npz (default: MODEL_training_umap.npz).",
    )
    parser.add_argument(
        "--skip-embedding-output",
        action="store_true",
        help="Do not save the training-set UMAP coordinates.",
    )
    parser.add_argument(
        "--feature-key",
        help="Array key for .npz input; inferred for common keys or a single-array archive.",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Search input directories recursively."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Uniformly sample at most this many rows before training.",
    )
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument(
        "--random-state",
        type=optional_int,
        default=None,
        metavar="INT|none",
        help=(
            "Random seed. Default 'none' matches the legacy training and permits "
            "parallel UMAP; an integer improves reproducibility."
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--compress",
        type=int,
        choices=range(0, 10),
        default=0,
        metavar="0-9",
        help="joblib compression level; 0 is fastest and best for very large models.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing model, embedding, and metadata outputs.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def import_umap() -> Any:
    try:
        import umap
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "umap-learn is not installed in this Python environment. "
            "Install 03_umap_projection/requirements.txt first."
        ) from exc
    return umap


def main() -> None:
    args = build_parser().parse_args()
    if args.n_neighbors < 2:
        raise SystemExit("--n-neighbors must be at least 2.")
    if args.n_components < 1:
        raise SystemExit("--n-components must be at least 1.")
    if not 0.0 <= args.min_dist:
        raise SystemExit("--min-dist must be non-negative.")

    model_path = args.output_model.expanduser().resolve()
    embedding_path = args.output_embedding
    if embedding_path is None:
        embedding_path = model_path.with_name(f"{model_path.stem}_training_umap.npz")
    embedding_path = embedding_path.expanduser().resolve()
    metadata_path = model_path.with_name(f"{model_path.name}.json")

    paths = discover_feature_files(args.input, recursive=args.recursive)
    input_files = {path.resolve() for path in paths}
    output_files = {model_path, metadata_path}
    if not args.skip_embedding_output:
        output_files.add(embedding_path)
    overlap = input_files & output_files
    if overlap:
        raise ValueError(
            "Refusing to overwrite an input feature file: "
            + ", ".join(str(path) for path in sorted(overlap))
        )
    if not args.overwrite:
        existing = sorted(path for path in output_files if path.exists())
        if existing:
            raise FileExistsError(
                "Output already exists: "
                + ", ".join(str(path) for path in existing)
                + ". Pass --overwrite to replace it."
            )
    print(f"Found {len(paths)} feature file(s).", flush=True)
    features, data_info = prepare_training_data(
        paths=paths,
        feature_key=args.feature_key,
        max_samples=args.max_samples,
        random_state=args.random_state,
    )
    print(
        f"Training data: {features.shape}, dtype={features.dtype}; "
        f"using {data_info.used_rows:,}/{data_info.total_rows:,} rows.",
        flush=True,
    )

    umap = import_umap()
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        metric=args.metric,
        min_dist=args.min_dist,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )
    started = time.time()
    embedding = np.asarray(reducer.fit_transform(features), dtype=np.float32)
    elapsed = time.time() - started
    print(f"UMAP fit completed in {elapsed / 60:.2f} min: {embedding.shape}.", flush=True)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving transform-capable model to: {model_path}", flush=True)
    joblib.dump(reducer, model_path, compress=args.compress)

    if not args.skip_embedding_output:
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "umap": embedding,
            "feature_shape": np.asarray(features.shape, dtype=np.int64),
            "total_input_rows": np.asarray(data_info.total_rows, dtype=np.int64),
            "source_files": np.asarray(
                [str(path.resolve()) for path in data_info.source_files]
            ),
            "model_file": np.asarray(str(model_path)),
        }
        if data_info.sampled_global_indices is not None:
            payload["sampled_global_indices"] = data_info.sampled_global_indices
        np.savez_compressed(embedding_path, **payload)
        print(f"Saved training embedding to: {embedding_path}", flush=True)

    metadata = {
        "model_file": str(model_path),
        "source_files": [str(path.resolve()) for path in data_info.source_files],
        "total_input_rows": data_info.total_rows,
        "training_rows": data_info.used_rows,
        "feature_dim": data_info.feature_dim,
        "parameters": {
            "n_neighbors": args.n_neighbors,
            "n_components": args.n_components,
            "metric": args.metric,
            "min_dist": args.min_dist,
            "random_state": args.random_state,
            "n_jobs": args.n_jobs,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "umap_learn": getattr(umap, "__version__", "unknown"),
        },
        "fit_seconds": elapsed,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved model metadata to: {metadata_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (FileExistsError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
