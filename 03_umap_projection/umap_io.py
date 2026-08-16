"""Shared I/O and validation helpers for the clean UMAP command-line tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SUPPORTED_SUFFIXES = {".npy", ".npz"}
DEFAULT_FEATURE_KEYS = (
    "features",
    "feature",
    "feats",
    "embeddings",
    "embedding",
    "x",
    "data",
    "arr_0",
)


@dataclass(frozen=True)
class FeatureFileInfo:
    path: Path
    rows: int
    columns: int
    dtype: str
    feature_key: str | None


@dataclass(frozen=True)
class TrainingDataInfo:
    source_files: tuple[Path, ...]
    total_rows: int
    used_rows: int
    feature_dim: int
    sampled_global_indices: np.ndarray | None


def discover_feature_files(input_path: Path, recursive: bool = False) -> list[Path]:
    """Return a stable, sorted list of supported feature files."""
    input_path = input_path.expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported feature file: {input_path}. Expected .npy or .npz."
            )
        return [input_path.resolve()]

    iterator: Iterable[Path]
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    files = sorted(
        (
            path.resolve()
            for path in iterator
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: str(path).casefold(),
    )
    if not files:
        scope = "recursively" if recursive else "directly"
        raise FileNotFoundError(
            f"No .npy/.npz feature files found {scope} under: {input_path}"
        )
    return files


def _choose_npz_key(keys: Sequence[str], requested_key: str | None) -> str:
    if requested_key is not None:
        if requested_key not in keys:
            raise KeyError(
                f"Feature key {requested_key!r} was not found. Available keys: {list(keys)}"
            )
        return requested_key

    for candidate in DEFAULT_FEATURE_KEYS:
        if candidate in keys:
            return candidate
    if len(keys) == 1:
        return keys[0]
    raise KeyError(
        "Could not infer the feature array in an .npz containing multiple arrays. "
        f"Available keys: {list(keys)}. Pass --feature-key explicitly."
    )


def _validate_array_shape(array: np.ndarray, path: Path) -> np.ndarray:
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(
            f"Feature array must have shape (n_samples, n_features); "
            f"got {array.shape} in {path}."
        )
    if array.shape[1] == 0:
        raise ValueError(f"Feature array has zero columns: {path}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Feature array must be numeric; got {array.dtype} in {path}.")
    return array


def load_feature_array(
    path: Path,
    feature_key: str | None = None,
    mmap_mode: str | None = "r",
) -> tuple[np.ndarray, str | None]:
    """Load one 2-D feature array without allowing pickled NumPy objects."""
    path = path.expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
        return _validate_array_shape(array, path), None
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = _choose_npz_key(archive.files, feature_key)
            array = np.asarray(archive[key])
        return _validate_array_shape(array, path), key
    raise ValueError(f"Unsupported feature file: {path}. Expected .npy or .npz.")


def inspect_feature_files(
    paths: Sequence[Path], feature_key: str | None = None
) -> list[FeatureFileInfo]:
    infos: list[FeatureFileInfo] = []
    expected_dim: int | None = None
    for path in paths:
        array, selected_key = load_feature_array(path, feature_key=feature_key)
        rows, columns = array.shape
        if expected_dim is None:
            expected_dim = columns
        elif columns != expected_dim:
            raise ValueError(
                "All feature files must have the same feature dimension. "
                f"Expected {expected_dim}, got {columns} in {path}."
            )
        infos.append(
            FeatureFileInfo(
                path=path,
                rows=rows,
                columns=columns,
                dtype=str(array.dtype),
                feature_key=selected_key,
            )
        )
    return infos


def assert_all_finite(array: np.ndarray, chunk_rows: int = 100_000) -> None:
    """Check a possibly memory-mapped array without creating one huge boolean array."""
    for start in range(0, array.shape[0], chunk_rows):
        stop = min(start + chunk_rows, array.shape[0])
        if not np.isfinite(array[start:stop]).all():
            bad_local = np.argwhere(~np.isfinite(array[start:stop]))[0]
            bad_row = start + int(bad_local[0])
            bad_col = int(bad_local[1])
            raise ValueError(
                f"Feature array contains NaN/Inf at row {bad_row}, column {bad_col}."
            )


def prepare_training_data(
    paths: Sequence[Path],
    feature_key: str | None,
    max_samples: int | None,
    random_state: int | None,
    copy_chunk_rows: int = 100_000,
) -> tuple[np.ndarray, TrainingDataInfo]:
    """Load all rows or a uniform sample across one or more feature files."""
    infos = inspect_feature_files(paths, feature_key=feature_key)
    total_rows = sum(info.rows for info in infos)
    if total_rows == 0:
        raise ValueError("The input contains no feature rows.")

    feature_dim = infos[0].columns
    if max_samples is not None and max_samples <= 0:
        raise ValueError("--max-samples must be a positive integer.")
    used_rows = min(max_samples, total_rows) if max_samples is not None else total_rows
    needs_sampling = used_rows < total_rows

    # Keep a single full float32 .npy memory-mapped instead of duplicating it in RAM.
    if len(paths) == 1 and not needs_sampling:
        array, _ = load_feature_array(paths[0], feature_key=feature_key)
        if array.dtype == np.float32:
            assert_all_finite(array)
            return array, TrainingDataInfo(
                source_files=tuple(paths),
                total_rows=total_rows,
                used_rows=used_rows,
                feature_dim=feature_dim,
                sampled_global_indices=None,
            )

    sampled_indices: np.ndarray | None = None
    if needs_sampling:
        rng = np.random.default_rng(random_state)
        sampled_indices = np.sort(
            rng.choice(total_rows, size=used_rows, replace=False).astype(np.int64)
        )

    output = np.empty((used_rows, feature_dim), dtype=np.float32)
    output_cursor = 0
    global_start = 0

    for info in infos:
        array, _ = load_feature_array(info.path, feature_key=feature_key)
        global_stop = global_start + info.rows
        if sampled_indices is None:
            for local_start in range(0, info.rows, copy_chunk_rows):
                local_stop = min(local_start + copy_chunk_rows, info.rows)
                block = np.asarray(array[local_start:local_stop], dtype=np.float32)
                block_rows = block.shape[0]
                output[output_cursor : output_cursor + block_rows] = block
                output_cursor += block_rows
        else:
            left = int(np.searchsorted(sampled_indices, global_start, side="left"))
            right = int(np.searchsorted(sampled_indices, global_stop, side="left"))
            local_indices = sampled_indices[left:right] - global_start
            if local_indices.size:
                block = np.asarray(array[local_indices], dtype=np.float32)
                output[output_cursor : output_cursor + block.shape[0]] = block
                output_cursor += block.shape[0]
        global_start = global_stop

    if output_cursor != used_rows:
        raise RuntimeError(
            f"Internal row-count mismatch: assembled {output_cursor}, expected {used_rows}."
        )
    assert_all_finite(output)
    return output, TrainingDataInfo(
        source_files=tuple(paths),
        total_rows=total_rows,
        used_rows=used_rows,
        feature_dim=feature_dim,
        sampled_global_indices=sampled_indices,
    )


def expected_model_feature_dim(model: Any) -> int | None:
    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None:
        return int(n_features)
    raw_data = getattr(model, "_raw_data", None)
    shape = getattr(raw_data, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[1])
    return None


def transform_in_batches(
    model: Any,
    features: np.ndarray,
    batch_size: int | None,
) -> np.ndarray:
    """Run model.transform, optionally limiting the number of query rows per call."""
    assert_all_finite(features)
    expected_dim = expected_model_feature_dim(model)
    if expected_dim is not None and features.shape[1] != expected_dim:
        raise ValueError(
            f"Model expects {expected_dim} input features, but input has "
            f"{features.shape[1]}."
        )

    if features.shape[0] == 0:
        n_components = int(getattr(model, "n_components", 2))
        return np.empty((0, n_components), dtype=np.float32)
    if batch_size is not None and batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")

    chunks: list[np.ndarray] = []
    step = features.shape[0] if batch_size is None else batch_size
    for start in range(0, features.shape[0], step):
        stop = min(start + step, features.shape[0])
        transformed = np.asarray(model.transform(features[start:stop]), dtype=np.float32)
        if transformed.ndim != 2 or transformed.shape[0] != stop - start:
            raise ValueError(
                "model.transform returned an invalid shape: "
                f"{transformed.shape} for {stop - start} input rows."
            )
        chunks.append(transformed)
    return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


def save_umap_npz(
    output_path: Path,
    embedding: np.ndarray,
    source_path: Path,
    feature_shape: tuple[int, int],
    model_path: Path,
    input_feature_key: str | None,
    copy_npz_metadata: bool = False,
) -> None:
    """Save UMAP coordinates plus compact provenance in one compressed .npz."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "umap": np.asarray(embedding, dtype=np.float32),
        "source_file": np.asarray(str(source_path.resolve())),
        "feature_shape": np.asarray(feature_shape, dtype=np.int64),
        "model_file": np.asarray(str(model_path.resolve())),
    }
    if input_feature_key is not None:
        payload["input_feature_key"] = np.asarray(input_feature_key)

    if copy_npz_metadata and source_path.suffix.lower() == ".npz":
        with np.load(source_path, allow_pickle=False) as archive:
            for key in archive.files:
                if key == input_feature_key:
                    continue
                output_key = key if key not in payload else f"input_{key}"
                payload[output_key] = np.asarray(archive[key])

    np.savez_compressed(output_path, **payload)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

