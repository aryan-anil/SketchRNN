import argparse
import math
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from train.dataset import SketchDataset, load_quickdraw_ndjson
from model.model import DecoderLSTM, EncoderLSTM


def parse_args():
    parser = argparse.ArgumentParser(description="Train a SketchRNN VAE on QuickDraw ndjson data.")
    parser.add_argument("--data", default="data/full_simplified_bicycle.ndjson", help="Path to QuickDraw .ndjson file.")
    parser.add_argument("--out-dir", default="runs/bicycle", help="Directory for checkpoints.")
    parser.add_argument("--load-checkpoint", default=r"C:\Users\Aryan\Documents\SketchRNN\runs\bicycle\latest.pt", help="Path to checkpoint to resume training from.")
    parser.add_argument("--save-interval", type=int, default=5, help="Save checkpoint every N epochs.")
    parser.add_argument("--max-samples", type=int, default=50000, help="Maximum sketches to load.")
    parser.add_argument("--max-seq-len", type=int, default=250, help="Drop sketches longer than this.")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-z", type=int, default=128)
    parser.add_argument("--enc-hidden-size", type=int, default=256)
    parser.add_argument("--dec-hidden-size", type=int, default=512)
    parser.add_argument("--n-distributions", type=int, default=20)
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--kl-anneal-steps", type=int, default=2000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def reconstruction_loss(dist, q_log_probs, target, mask):
    target_xy = target[..., :2]
    target_pen = target[..., 2:5].argmax(dim=-1)

    sigma_x = dist.sigma_x.clamp_min(1e-5)
    sigma_y = dist.sigma_y.clamp_min(1e-5)
    rho = dist.rho_xy.clamp(-1 + 1e-5, 1 - 1e-5)
    x = target_xy[..., 0].unsqueeze(-1)
    y = target_xy[..., 1].unsqueeze(-1)

    norm_x = (x - dist.mu_x) / sigma_x
    norm_y = (y - dist.mu_y) / sigma_y
    z = norm_x.pow(2) + norm_y.pow(2) - 2 * rho * norm_x * norm_y
    one_minus_rho2 = (1 - rho.pow(2)).clamp_min(1e-5)

    log_pdf = (
        -math.log(2 * math.pi)
        - torch.log(sigma_x)
        - torch.log(sigma_y)
        - 0.5 * torch.log(one_minus_rho2)
        - z / (2 * one_minus_rho2)
    )
    log_mix = F.log_softmax(dist.pi_logits, dim=-1) + log_pdf
    xy_nll = -torch.logsumexp(log_mix, dim=-1)

    pen_nll = F.nll_loss(
        q_log_probs.reshape(-1, 3),
        target_pen.reshape(-1),
        reduction="none",
    ).reshape_as(mask)

    return masked_mean(xy_nll, mask), masked_mean(pen_nll, mask)


def kl_loss(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()


def run_epoch(encoder, decoder, loader, optimizer, device, args, step, train: bool):
    encoder.train(train)
    decoder.train(train)

    total_loss = 0.0
    total_xy = 0.0
    total_pen = 0.0
    total_kl = 0.0

    for data, mask in loader:
        data = data.to(device)
        mask = mask.to(device).transpose(0, 1)
        inputs = data[:, :-1].transpose(0, 1)
        target = data[:, 1:].transpose(0, 1)

        with torch.set_grad_enabled(train):
            z, mu, log_var = encoder(inputs)
            dist, q_log_probs, _ = decoder(inputs, z)
            xy_loss, pen_loss = reconstruction_loss(dist, q_log_probs, target, mask)
            kl = kl_loss(mu, log_var)
            anneal = min(1.0, step / max(1, args.kl_anneal_steps))
            loss = xy_loss + pen_loss + args.kl_weight * anneal * kl

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), args.grad_clip)
                optimizer.step()
                step += 1

        total_loss += loss.item()
        total_xy += xy_loss.item()
        total_pen += pen_loss.item()
        total_kl += kl.item()

    batches = max(1, len(loader))
    return {
        "loss": total_loss / batches,
        "xy": total_xy / batches,
        "pen": total_pen / batches,
        "kl": total_kl / batches,
        "step": step,
    }


def save_checkpoint(path, encoder, decoder, optimizer, args, scale, epoch, step, best_val, metrics):
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "scale": scale,
            "epoch": epoch,
            "step": step,
            "best_val": best_val,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path, encoder, decoder, optimizer, device):
    """Load checkpoint and return epoch, step, best_val to resume training."""
    checkpoint = torch.load(path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    best_val = checkpoint.get("best_val", float("inf"))
    scale = checkpoint.get("scale", 1.0)
    
    print(f"Loaded checkpoint from epoch {epoch}, step {step}, best_val={best_val:.4f}, scale={scale:.4f}")
    return epoch, step, best_val, scale


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sketches = load_quickdraw_ndjson(args.data, max_items=args.max_samples)
    full_dataset = SketchDataset(sketches, max_seq_len=args.max_seq_len)
    val_size = max(1, int(len(full_dataset) * args.val_fraction))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    encoder = EncoderLSTM(args.d_z, args.enc_hidden_size).to(device)
    decoder = DecoderLSTM(args.d_z, args.dec_hidden_size, args.n_distributions).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

    # Initialize training state
    start_epoch = 1
    step = 0
    best_val = float("inf")
    
    # Load checkpoint if specified
    if args.load_checkpoint is not None:
        checkpoint_path = Path(args.load_checkpoint)
        if checkpoint_path.exists():
            start_epoch, step, best_val, loaded_scale = load_checkpoint(
                checkpoint_path, encoder, decoder, optimizer, device
            )
            start_epoch += 1  # Resume from next epoch
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(f"Warning: Checkpoint {args.load_checkpoint} not found. Starting from scratch.")

    print(f"Loaded {len(full_dataset)} sketches. scale={full_dataset.scale:.4f} device={device}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(encoder, decoder, train_loader, optimizer, device, args, step, train=True)
        step = train_metrics["step"]
        val_metrics = run_epoch(encoder, decoder, val_loader, optimizer, device, args, step, train=False)

        print(
            f"epoch {epoch:03d} "
            f"train loss={train_metrics['loss']:.4f} xy={train_metrics['xy']:.4f} pen={train_metrics['pen']:.4f} kl={train_metrics['kl']:.4f} "
            f"val loss={val_metrics['loss']:.4f} xy={val_metrics['xy']:.4f} pen={val_metrics['pen']:.4f} kl={val_metrics['kl']:.4f}"
        )

        # Save latest checkpoint
        latest_path = out_dir / "latest.pt"
        save_checkpoint(latest_path, encoder, decoder, optimizer, args, full_dataset.scale, epoch, step, best_val, val_metrics)

        # Save best checkpoint
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(out_dir / "best.pt", encoder, decoder, optimizer, args, full_dataset.scale, epoch, step, best_val, val_metrics)
            print(f"  → New best validation loss: {best_val:.4f}")

        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            interval_path = out_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            save_checkpoint(interval_path, encoder, decoder, optimizer, args, full_dataset.scale, epoch, step, best_val, val_metrics)
            print(f"  → Saved checkpoint: {interval_path}")


if __name__ == "__main__":
    main()