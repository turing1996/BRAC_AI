from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


CLEAN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLEAN_ROOT))

import train_umap  # noqa: E402
import transform_umap  # noqa: E402
from umap_io import (  # noqa: E402
    discover_feature_files,
    prepare_training_data,
    transform_in_batches,
)


class FakeUMAP:
    """Tiny UMAP-compatible test double; it is not used by production code."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.n_components = kwargs["n_components"]

    def fit_transform(self, features):
        features = np.asarray(features, dtype=np.float32)
        self.n_features_in_ = features.shape[1]
        self._mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        return self.transform(features)

    def transform(self, features):
        features = np.asarray(features, dtype=np.float32)
        return (features - self._mean)[:, : self.n_components]


def test_directory_loading_and_sampling(tmp_path):
    first = np.arange(24, dtype=np.float32).reshape(6, 4)
    second = np.arange(20, dtype=np.float32).reshape(5, 4)
    np.save(tmp_path / "a.npy", first)
    np.savez_compressed(tmp_path / "b.npz", features=second)

    paths = discover_feature_files(tmp_path)
    sampled, info = prepare_training_data(
        paths,
        feature_key=None,
        max_samples=7,
        random_state=42,
    )

    assert [path.name for path in paths] == ["a.npy", "b.npz"]
    assert sampled.shape == (7, 4)
    assert sampled.dtype == np.float32
    assert info.total_rows == 11
    assert info.used_rows == 7
    assert info.sampled_global_indices is not None
    assert np.isfinite(sampled).all()


def test_train_then_transform_cli_contract(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    train_features = rng.normal(size=(40, 8)).astype(np.float32)
    query_features = rng.normal(size=(13, 8)).astype(np.float32)
    train_path = tmp_path / "train.npy"
    query_path = tmp_path / "query.npz"
    model_path = tmp_path / "model.pkl"
    output_path = tmp_path / "query_umap.npz"
    np.save(train_path, train_features)
    np.savez_compressed(
        query_path,
        features=query_features,
        patch_id=np.asarray([f"patch_{index}" for index in range(13)]),
    )

    fake_umap_module = SimpleNamespace(UMAP=FakeUMAP, __version__="test-double")
    monkeypatch.setattr(train_umap, "import_umap", lambda: fake_umap_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_umap.py",
            str(train_path),
            "--output-model",
            str(model_path),
            "--random-state",
            "42",
        ],
    )
    train_umap.main()

    assert model_path.is_file()
    training_output = tmp_path / "model_training_umap.npz"
    assert training_output.is_file()
    with np.load(training_output, allow_pickle=False) as archive:
        assert archive["umap"].shape == (40, 2)
        assert archive["umap"].dtype == np.float32
    metadata = json.loads((tmp_path / "model.pkl.json").read_text(encoding="utf-8"))
    assert metadata["training_rows"] == 40
    assert metadata["feature_dim"] == 8

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "transform_umap.py",
            str(query_path),
            "--model",
            str(model_path),
            "--output",
            str(output_path),
            "--batch-size",
            "5",
            "--copy-npz-metadata",
        ],
    )
    transform_umap.main()

    with np.load(output_path, allow_pickle=False) as archive:
        assert set(archive.files) >= {
            "umap",
            "source_file",
            "feature_shape",
            "model_file",
            "input_feature_key",
            "patch_id",
        }
        assert archive["umap"].shape == (13, 2)
        assert archive["umap"].dtype == np.float32
        assert archive["feature_shape"].tolist() == [13, 8]
        assert archive["patch_id"].tolist() == [
            f"patch_{index}" for index in range(13)
        ]
        assert np.isfinite(archive["umap"]).all()


def test_feature_dimension_mismatch_is_rejected():
    model = FakeUMAP(n_components=2)
    model.n_features_in_ = 8
    model._mean = np.zeros(8, dtype=np.float32)

    with pytest.raises(ValueError, match="expects 8"):
        transform_in_batches(
            model,
            np.zeros((3, 7), dtype=np.float32),
            batch_size=2,
        )


def test_transform_refuses_to_overwrite_input_npz(tmp_path, monkeypatch):
    feature_path = tmp_path / "features.npz"
    np.savez_compressed(feature_path, features=np.zeros((3, 8), dtype=np.float32))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "transform_umap.py",
            str(feature_path),
            "--model",
            str(tmp_path / "unused.pkl"),
            "--output",
            str(feature_path),
            "--overwrite",
        ],
    )

    with pytest.raises(ValueError, match="overwrite an input"):
        transform_umap.main()
