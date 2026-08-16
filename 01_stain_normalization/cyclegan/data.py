"""Unpaired two-domain image dataset."""

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


class UnpairedImageDataset(Dataset):
    def __init__(
        self,
        root: Path,
        mode: str = "train",
        image_size: int = 400,
        unaligned: bool = True,
    ) -> None:
        domain_a = root / mode / "A"
        domain_b = root / mode / "B"
        if not domain_a.is_dir() or not domain_b.is_dir():
            raise FileNotFoundError(
                f"Expected domain folders {domain_a} and {domain_b}. "
                "See README.md for the dataset layout."
            )

        self.files_a = list_images(domain_a)
        self.files_b = list_images(domain_b)
        if not self.files_a or not self.files_b:
            raise ValueError("Both domain folders must contain at least one image.")

        self.unaligned = unaligned
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.12), Image.Resampling.BICUBIC),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,) * 3, (0.5,) * 3),
            ]
        )

    def __len__(self) -> int:
        return max(len(self.files_a), len(self.files_b))

    def __getitem__(self, index: int):
        path_a = self.files_a[index % len(self.files_a)]
        if self.unaligned:
            path_b = random.choice(self.files_b)
        else:
            path_b = self.files_b[index % len(self.files_b)]

        with Image.open(path_a) as image_a:
            tensor_a = self.transform(image_a.convert("RGB"))
        with Image.open(path_b) as image_b:
            tensor_b = self.transform(image_b.convert("RGB"))
        return {"A": tensor_a, "B": tensor_b}
