"""Apply the supplied CycleGAN generator to one image or a directory of images."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision.transforms import functional as transform_functional

from cyclegan.data import IMAGE_SUFFIXES
from cyclegan.models import Generator
from cyclegan.utils import load_weights, select_device


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Image file or image directory")
    parser.add_argument("--output", type=Path, required=True, help="Output file or directory")
    parser.add_argument("--direction", choices=["A2B", "B2A"], default="A2B")
    parser.add_argument("--weights", type=Path, help="Override the bundled generator checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--recursive", action="store_true", help="Search input subdirectories")
    return parser.parse_args()


def input_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    iterator = path.rglob("*") if recursive else path.iterdir()
    files = sorted(item for item in iterator if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise ValueError(f"No supported images found in {path}")
    return files


def output_path(source: Path, input_root: Path, destination: Path) -> Path:
    if input_root.is_file():
        if destination.suffix:
            return destination
        return destination / source.name
    return destination / source.relative_to(input_root)


def prepare(image: Image.Image, device: torch.device):
    tensor = transform_functional.to_tensor(image.convert("RGB"))
    tensor = transform_functional.normalize(tensor, (0.5,) * 3, (0.5,) * 3).unsqueeze(0)
    height, width = tensor.shape[-2:]
    pad_height = (-height) % 4
    pad_width = (-width) % 4
    if pad_height or pad_width:
        tensor = functional.pad(tensor, (0, pad_width, 0, pad_height), mode="reflect")
    return tensor.to(device), height, width


def save_tensor(tensor: torch.Tensor, path: Path) -> None:
    tensor = tensor.squeeze(0).detach().cpu().add(1).div(2).clamp(0, 1)
    image = transform_functional.to_pil_image(tensor)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=95, subsampling=0)
    else:
        image.save(path)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    default_name = "8_netG_A2B.pth" if args.direction == "A2B" else "8_netG_B2A.pth"
    weights = args.weights or PROJECT_ROOT / "weights" / default_name

    model = Generator().to(device)
    load_weights(model, weights, device)
    model.eval()
    files = input_files(args.input, args.recursive)
    print(f"Device: {device}; direction: {args.direction}; images: {len(files)}")

    with torch.inference_mode():
        for index, source in enumerate(files, start=1):
            with Image.open(source) as image:
                tensor, height, width = prepare(image, device)
            result = model(tensor)[..., :height, :width]
            destination = output_path(source, args.input, args.output)
            save_tensor(result, destination)
            print(f"[{index}/{len(files)}] {source} -> {destination}")


if __name__ == "__main__":
    main()
