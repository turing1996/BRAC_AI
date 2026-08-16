from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

SAMPLE_PATTERN = re.compile(r"^(?P<os>[01])_(?P<dfs>[01])_(?P<body>.+)$")
TCGA_PATTERN = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def discover_ids(root: str | Path) -> list[str]:
    he_dir = Path(root) / "HE"
    if not he_dir.is_dir():
        raise FileNotFoundError(f"Missing HE directory: {he_dir}")
    return sorted(
        p.stem for p in he_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")
    )


def parse_events(sample_id: str) -> tuple[float, float]:
    match = SAMPLE_PATTERN.match(sample_id)
    if not match:
        raise ValueError(f"Sample ID must start with '<OS event>_<DFS event>_': {sample_id}")
    return float(match.group("os")), float(match.group("dfs"))


def patient_id(sample_id: str) -> str:
    tcga = TCGA_PATTERN.search(sample_id)
    if tcga:
        return tcga.group(1).upper()
    match = SAMPLE_PATTERN.match(sample_id)
    body = match.group("body") if match else sample_id
    return re.sub(r"(?:_\d+){1,3}$", "", body)


def assert_patient_disjoint(left_ids: Sequence[str], right_ids: Sequence[str]) -> None:
    overlap = sorted({patient_id(x) for x in left_ids} & {patient_id(x) for x in right_ids})
    if overlap:
        raise ValueError(f"TCGA train/validation patient overlap detected; examples: {overlap[:5]}")


def sample_file_stems(sample_id: str) -> list[str]:
    """Support both event-prefixed and raw slide basenames for morphology files."""
    stems = [sample_id]
    match = SAMPLE_PATTERN.match(sample_id)
    if match and match.group("body") not in stems:
        stems.append(match.group("body"))
    return stems


def resolve_morphology_file(root: Path, subdir: str, sample_id: str) -> Path:
    searched: list[Path] = []
    for stem in sample_file_stems(sample_id):
        path = root / subdir / f"{stem}.npy"
        searched.append(path)
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing morphology map for {sample_id}; searched: {searched}")


class SurvivalDataset(Dataset):
    def __init__(self, root: str | Path, config: dict, sample_ids: Sequence[str] | None = None) -> None:
        self.root = Path(root)
        self.ids = list(sample_ids) if sample_ids is not None else discover_ids(self.root)
        self.image_size = int(config["model"]["image_size"])
        self.morphology_channels = int(config["model"].get("morphology_channels", 2))
        self.morphology_subdir = str(config["data"].get("morphology_subdir", "umap"))
        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self) -> int:
        return len(self.ids)

    def _image_path(self, sample_id: str) -> Path:
        files = [self.root / "HE" / f"{sample_id}{s}" for s in IMAGE_SUFFIXES]
        files = [p for p in files if p.is_file()]
        if len(files) != 1:
            raise FileNotFoundError(f"Expected one H&E image for {sample_id}; found {files}")
        return files[0]

    def _load_morphology(self, sample_id: str) -> torch.Tensor:
        path = resolve_morphology_file(self.root, self.morphology_subdir, sample_id)
        array = np.load(path, allow_pickle=False)
        if array.ndim != 3:
            raise ValueError(f"Expected a 3-D morphology map for {sample_id}; got {array.shape}")

        # Accept the historical H x W x 2 representation and a pre-transposed 2 x H x W representation.
        if array.shape[-1] == self.morphology_channels:
            array = np.moveaxis(array, -1, 0)
        elif array.shape[0] == self.morphology_channels:
            pass
        else:
            raise ValueError(
                f"Expected HxWx{self.morphology_channels} or {self.morphology_channels}xHxW morphology map "
                f"for {sample_id}; got {array.shape}"
            )
        if tuple(array.shape[1:]) != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected morphology spatial size {self.image_size}x{self.image_size} for {sample_id}; "
                f"got {tuple(array.shape[1:])}"
            )
        array = np.asarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"Morphology map contains NaN/Inf: {path}")
        return torch.from_numpy(array)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = self.ids[index]
        time_path = self.root / "sur_time" / f"{sample_id}.csv"
        if not time_path.is_file():
            raise FileNotFoundError(f"Missing survival time file: {time_path}")
        with Image.open(self._image_path(sample_id)) as image:
            image_tensor = self.image_transform(image.convert("RGB")).float()
        times = np.atleast_1d(np.loadtxt(time_path, delimiter=",", dtype=np.float32))
        if times.size < 2:
            raise ValueError(f"Expected OS,DFS times for {sample_id}; got {times}")
        os_event, dfs_event = parse_events(sample_id)
        return {
            "sample_id": sample_id,
            "patient_id": patient_id(sample_id),
            "image": image_tensor,
            "morphology_map": self._load_morphology(sample_id),
            "os_time": torch.tensor(float(times[0]), dtype=torch.float32),
            "dfs_time": torch.tensor(float(times[1]), dtype=torch.float32),
            "os_event": torch.tensor(os_event, dtype=torch.float32),
            "dfs_event": torch.tensor(dfs_event, dtype=torch.float32),
        }


def make_dataset(root: str | Path, config: dict, sample_ids: Sequence[str] | None = None) -> SurvivalDataset:
    return SurvivalDataset(root, config, sample_ids)
