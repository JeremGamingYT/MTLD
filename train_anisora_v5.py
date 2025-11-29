from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from anisora_v5_model import AnisoraV5

import torch.nn.functional as F


class SyntheticVideoDataset(Dataset):
    """A simple dataset of moving coloured squares.

    Each sample is a sequence of ``seq_len`` frames of size ``frame_size``
    × ``frame_size`` with a single square of random colour moving across
    the frame.  The square bounces off the edges and its size and speed
    are randomly chosen.  This toy dataset is useful for demonstrating
    temporal coherence and basic motion understanding without relying on
    external files.
    """

    def __init__(
        self,
        num_samples: int = 100,
        seq_len: int = 16,
        frame_size: int = 64,
        square_size_range: Tuple[int, int] = (8, 16),
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.frame_size = frame_size
        self.square_size_range = square_size_range

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Random square parameters
        size = np.random.randint(self.square_size_range[0], self.square_size_range[1] + 1)
        x_pos = np.random.randint(0, self.frame_size - size)
        y_pos = np.random.randint(0, self.frame_size - size)
        # Random velocity
        vx = np.random.choice([-1, 1]) * np.random.randint(1, 4)
        vy = np.random.choice([-1, 1]) * np.random.randint(1, 4)
        # Random colour
        colour = np.random.rand(3)
        frames = []
        for _ in range(self.seq_len):
            frame = np.zeros((3, self.frame_size, self.frame_size), dtype=np.float32)
            frame[:, y_pos : y_pos + size, x_pos : x_pos + size] = colour[:, None, None]
            frames.append(frame)
            # Update position and bounce off edges
            x_pos += vx
            y_pos += vy
            if x_pos < 0 or x_pos + size > self.frame_size:
                vx *= -1
                x_pos = max(0, min(x_pos, self.frame_size - size))
            if y_pos < 0 or y_pos + size > self.frame_size:
                vy *= -1
                y_pos = max(0, min(y_pos, self.frame_size - size))
        video = np.stack(frames, axis=0)  # (T, C, H, W)
        # Convert to tensor and scale to [-1, 1]
        video = torch.from_numpy(video) * 2.0 - 1.0
        return video


def train(model: AnisoraV5, dataloader: DataLoader, device: torch.device, epochs: int = 10, lr: float = 1e-3) -> None:
    """Train the VAE model on the provided dataloader.

    Args:
        model: The AnisoraV5 instance.
        dataloader: DataLoader providing video tensors of shape (B, T, C, H, W).
        device: Target device.
        epochs: Number of epochs to train for.
        lr: Learning rate.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_rec = 0.0
        total_kl = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, kl = model(batch)
            # Reconstruction loss: MSE between input and reconstructions
            rec_loss = F.mse_loss(recon, batch, reduction="none")
            rec_loss = rec_loss.mean(dim=[1, 2, 3, 4])  # per sample
            kl_loss = kl  # per sample
            # Beta‑VAE tradeoff; tune beta as needed
            beta = 1e-3
            loss = (rec_loss + beta * kl_loss).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
            total_rec += rec_loss.sum().item()
            total_kl += kl_loss.sum().item()
        n = len(dataloader.dataset)
        print(
            f"Epoch {epoch:02d} | Loss: {total_loss / n:.4f} | REC: {total_rec / n:.4f} | KL: {total_kl / n:.4f}"
        )


def save_video_grid(videos: torch.Tensor, path: str, n_rows: int = 2) -> None:
    """Save a grid of generated videos as an animated GIF.

    The videos tensor must have shape (N, T, C, H, W).  Frames from each
    sample are arranged in rows, and samples are stacked in columns.  The
    output file is saved as a GIF using Pillow so that external
    dependencies are not required.
    """
    from PIL import Image

    # Normalize to [0, 255] and clamp
    videos = videos.cpu().clamp(-1, 1)
    videos = (videos + 1) / 2  # [0, 1]
    N, T, C, H, W = videos.shape
    n_cols = (N + n_rows - 1) // n_rows
    frames = []
    for t in range(T):
        canvas = torch.zeros((3, n_rows * H, n_cols * W))
        for i in range(N):
            r = i % n_rows
            c = i // n_rows
            canvas[:, r * H : (r + 1) * H, c * W : (c + 1) * W] = videos[i, t]
        frame_array = (canvas.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frames.append(Image.fromarray(frame_array))
    # Save as GIF
    if frames:
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=250,
            loop=0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and sample Anisora V5")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of synthetic training samples")
    parser.add_argument("--seq-len", type=int, default=16, help="Length of video sequences")
    parser.add_argument("--frame-size", type=int, default=64, help="Height/width of frames")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    # Create dataset and dataloader
    dataset = SyntheticVideoDataset(
        num_samples=args.num_samples, seq_len=args.seq_len, frame_size=args.frame_size
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # Instantiate model
    model = AnisoraV5(
        in_channels=3,
        frame_size=args.frame_size,
        seq_len=args.seq_len,
        feat_dim=128,
        hidden_dim=256,
        latent_dim=64,
    ).to(device)
    # Train model
    train(model, dataloader, device, epochs=args.epochs)
    # Sample videos
    samples = model.sample(batch_size=4).to(torch.device("cpu"))
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "generated.gif")
    save_video_grid(samples, save_path)
    print(f"Generated samples saved to {save_path}")


if __name__ == "__main__":
    main()