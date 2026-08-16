"""Train the CBCGA <-> TCGA CycleGAN with command-line configuration."""

import argparse
import csv
import itertools
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cyclegan.data import UnpairedImageDataset
from cyclegan.models import Discriminator, Generator
from cyclegan.utils import (
    ReplayBuffer,
    initialize_weights,
    load_weights,
    save_preview,
    save_weights,
    seed_everything,
    select_device,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "examples" / "dataset")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "CBCGA2TCGA_400")
    parser.add_argument("--epochs", type=int, default=400, help="Total number of epochs")
    parser.add_argument("--start-epoch", type=int, default=None)
    parser.add_argument("--decay-start-epoch", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--preview-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--init-weights-dir",
        type=Path,
        help="Optional directory containing <epoch>_net{G_A2B,G_B2A,D_A,D_B}.pth",
    )
    parser.add_argument("--init-epoch", type=int, default=8)
    return parser.parse_args()


def set_learning_rate(optimizers, base_lr: float, epoch: int, total: int, decay_start: int) -> float:
    if not 0 <= decay_start < total:
        raise ValueError("decay-start-epoch must be in [0, epochs).")
    factor = 1.0 - max(0, epoch - decay_start) / (total - decay_start)
    current_lr = base_lr * factor
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = current_lr
    return current_lr


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = UnpairedImageDataset(args.data_root, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    generator_a2b = Generator().to(device)
    generator_b2a = Generator().to(device)
    discriminator_a = Discriminator().to(device)
    discriminator_b = Discriminator().to(device)
    networks = [generator_a2b, generator_b2a, discriminator_a, discriminator_b]
    for network in networks:
        network.apply(initialize_weights)

    if args.init_weights_dir:
        names = {
            generator_a2b: "netG_A2B",
            generator_b2a: "netG_B2A",
            discriminator_a: "netD_A",
            discriminator_b: "netD_B",
        }
        for network, name in names.items():
            load_weights(
                network,
                args.init_weights_dir / f"{args.init_epoch}_{name}.pth",
                device,
            )
        print("Loaded initialization weights; optimizer state is newly initialized.")

    start_epoch = args.start_epoch
    if start_epoch is None:
        start_epoch = args.init_epoch + 1 if args.init_weights_dir else 0

    criterion_gan = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    optimizer_g = torch.optim.Adam(
        itertools.chain(generator_a2b.parameters(), generator_b2a.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    optimizer_da = torch.optim.Adam(discriminator_a.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_db = torch.optim.Adam(discriminator_b.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizers = [optimizer_g, optimizer_da, optimizer_db]
    fake_a_buffer = ReplayBuffer()
    fake_b_buffer = ReplayBuffer()

    log_path = args.output_dir / "losses.csv"
    new_log = not log_path.exists() or start_epoch == 0
    log_file = log_path.open("w" if new_log else "a", newline="", encoding="utf-8")
    fields = ["epoch", "step", "lr", "loss_G", "loss_identity", "loss_GAN", "loss_cycle", "loss_D"]
    writer = csv.DictWriter(log_file, fieldnames=fields)
    if new_log:
        writer.writeheader()

    print(f"Device: {device}; training pairs per epoch: {len(dataset)}")
    try:
        for epoch in range(start_epoch, args.epochs):
            lr = set_learning_rate(optimizers, args.lr, epoch, args.epochs, args.decay_start_epoch)
            for step, batch in enumerate(loader, start=1):
                real_a = batch["A"].to(device, non_blocking=True)
                real_b = batch["B"].to(device, non_blocking=True)

                optimizer_g.zero_grad(set_to_none=True)
                same_b = generator_a2b(real_b)
                same_a = generator_b2a(real_a)
                loss_identity = (
                    criterion_identity(same_a, real_a) + criterion_identity(same_b, real_b)
                ) * 5.0
                fake_b = generator_a2b(real_a)
                fake_a = generator_b2a(real_b)
                prediction_fake_b = discriminator_b(fake_b)
                prediction_fake_a = discriminator_a(fake_a)
                loss_gan = criterion_gan(prediction_fake_b, torch.ones_like(prediction_fake_b))
                loss_gan += criterion_gan(prediction_fake_a, torch.ones_like(prediction_fake_a))
                recovered_a = generator_b2a(fake_b)
                recovered_b = generator_a2b(fake_a)
                loss_cycle = (
                    criterion_cycle(recovered_a, real_a) + criterion_cycle(recovered_b, real_b)
                ) * 10.0
                loss_g = loss_identity + loss_gan + loss_cycle
                loss_g.backward()
                optimizer_g.step()

                optimizer_da.zero_grad(set_to_none=True)
                prediction_real_a = discriminator_a(real_a)
                prediction_fake_a = discriminator_a(fake_a_buffer.push_and_pop(fake_a))
                loss_da = 0.5 * (
                    criterion_gan(prediction_real_a, torch.ones_like(prediction_real_a))
                    + criterion_gan(prediction_fake_a, torch.zeros_like(prediction_fake_a))
                )
                loss_da.backward()
                optimizer_da.step()

                optimizer_db.zero_grad(set_to_none=True)
                prediction_real_b = discriminator_b(real_b)
                prediction_fake_b = discriminator_b(fake_b_buffer.push_and_pop(fake_b))
                loss_db = 0.5 * (
                    criterion_gan(prediction_real_b, torch.ones_like(prediction_real_b))
                    + criterion_gan(prediction_fake_b, torch.zeros_like(prediction_fake_b))
                )
                loss_db.backward()
                optimizer_db.step()
                loss_d = loss_da + loss_db

                row = {
                    "epoch": epoch,
                    "step": step,
                    "lr": lr,
                    "loss_G": loss_g.item(),
                    "loss_identity": loss_identity.item(),
                    "loss_GAN": loss_gan.item(),
                    "loss_cycle": loss_cycle.item(),
                    "loss_D": loss_d.item(),
                }
                writer.writerow(row)
                log_file.flush()
                if step == 1 or step % args.log_every == 0 or step == len(loader):
                    print(
                        f"epoch {epoch:03d}/{args.epochs - 1:03d} step {step:04d}/{len(loader):04d} "
                        f"G={loss_g.item():.4f} D={loss_d.item():.4f}"
                    )

            if (epoch + 1) % args.preview_every == 0:
                save_preview(
                    [real_a, fake_b, real_b, fake_a],
                    args.output_dir / "previews" / f"epoch_{epoch:03d}.jpg",
                )
            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                save_weights(generator_a2b, args.output_dir / f"{epoch}_netG_A2B.pth")
                save_weights(generator_b2a, args.output_dir / f"{epoch}_netG_B2A.pth")
                save_weights(discriminator_a, args.output_dir / f"{epoch}_netD_A.pth")
                save_weights(discriminator_b, args.output_dir / f"{epoch}_netD_B.pth")
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
