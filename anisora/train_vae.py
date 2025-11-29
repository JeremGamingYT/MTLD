"""
train_vae
=========

This script provides a simple training loop for the ``Image2VideoVAE`` model
defined in ``image2video_vae.py``.  It is designed to work in a
notebook environment and requires only minimal dependencies: PyTorch
for the model and training, OpenCV for reading video files, and
Pillow for saving sample outputs.  The training loop is intentionally
lightweight but includes sufficient logging to monitor progress and
debug issues.  It can learn from a very small dataset of anime
videos (e.g. ten clips of ~20 seconds) and then synthesise new
sequences given a single conditioning image.

Usage (from a terminal or notebook cell)::

    # Train the baseline VAE (simpler but less detailed)
    python train_vae.py --dataset ./anime_clips --epochs 10 --batch-size 4 \
        --seq-len 16 --frame-size 64 --model-type vae --save-dir ./outputs

    # Train the U‑Net VAE (better detail preservation)
    python train_vae.py --dataset ./anime_clips --epochs 10 --batch-size 4 \
        --seq-len 16 --frame-size 64 --model-type unet-vae --save-dir ./outputs

The script will traverse the ``dataset`` directory, extract fixed
length sequences from each video, train the chosen model, and write
sample animations after each epoch.  Logs are printed to stdout
and can be captured in a notebook for inspection.  If GPU is
available (CUDA), it will be used automatically.

Notes
-----
* The dataset loader will decode every frame of each video; to
  accelerate loading, consider downsampling your videos (e.g. to
  64×64 pixels and a lower frame rate) ahead of time.
* The VAE uses a simple L1 reconstruction loss and KL divergence.
  Adjust the hyper‐parameters (e.g. learning rate, KL weight) as
  needed for your specific dataset.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Import both models.  They define the same API but use different architectures.
from image2video_vae import Image2VideoVAE
from image2video_unet_vae import Image2VideoUNetVAE


class VideoSequenceDataset(Dataset):
    """Dataset that extracts fixed‐length sequences from a list of video files.

    Each item returns a conditioning image (the first frame in the
    sequence) and the full target sequence.  Frames are resized to
    ``frame_size`` and normalised to the range ``[-1, 1]``.

    Parameters
    ----------
    video_paths : list of str or Path
        Paths to video files.
    seq_len : int
        Number of frames per sample sequence.
    frame_size : int
        Spatial resolution of frames (square).  Videos will be
        resized to ``(frame_size, frame_size)``.
    sample_step : int, optional
        Step size between frames when slicing sequences.  For
        example, ``sample_step=2`` will take every second frame.
    max_samples_per_video : int or None, optional
        Maximum number of sequences to extract from each video.  If
        ``None``, all possible sequences will be used.
    """

    def __init__(
        self,
        video_paths: List[Path],
        seq_len: int = 16,
        frame_size: int = 64,
        sample_step: int = 1,
        max_samples_per_video: int | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.frame_size = frame_size
        self.sample_step = sample_step
        self.video_paths = [Path(p) for p in video_paths]
        self.max_samples_per_video = max_samples_per_video
        # Precompute index mapping: list of tuples (video_index, start_frame)
        self.indices: List[Tuple[int, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        for vid_idx, path in enumerate(self.video_paths):
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            # Determine how many sequences can be extracted
            max_start = frame_count - (self.seq_len * self.sample_step)
            # Skip videos that are too short
            if max_start <= 0:
                continue
            # Determine number of samples: either all possible or truncated
            if self.max_samples_per_video is None:
                step = self.seq_len * self.sample_step
                starts = list(range(0, max_start + 1, step))
            else:
                # Uniformly sample start positions
                possible_starts = list(range(0, max_start + 1))
                if len(possible_starts) <= self.max_samples_per_video:
                    starts = possible_starts
                else:
                    rng = np.random.default_rng(12345 + vid_idx)
                    starts = rng.choice(possible_starts, size=self.max_samples_per_video, replace=False).tolist()
                    starts.sort()
            # Append to indices
            for s in starts:
                self.indices.append((vid_idx, s))

    def __len__(self) -> int:
        return len(self.indices)

    def _read_frames(self, path: Path, start: int) -> np.ndarray:
        """Read ``seq_len`` frames from ``path`` starting at ``start`` with ``sample_step``.

        Returns an array of shape (seq_len, H, W, 3) in BGR format.
        """
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video {path}")
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames: List[np.ndarray] = []
        for i in range(self.seq_len):
            ret, frame = cap.read()
            if not ret:
                break
            # Resize
            frame = cv2.resize(frame, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)
            frames.append(frame)
            # Skip frames
            if self.sample_step > 1:
                next_pos = start + (i + 1) * self.sample_step
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_pos)
        cap.release()
        if len(frames) < self.seq_len:
            # Pad with last frame
            while len(frames) < self.seq_len:
                frames.append(frames[-1].copy())
        return np.stack(frames, axis=0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vid_idx, start_frame = self.indices[idx]
        path = self.video_paths[vid_idx]
        frames_bgr = self._read_frames(path, start_frame)
        # Convert to RGB and normalise to [-1, 1]
        frames_rgb = frames_bgr[:, :, :, ::-1]  # BGR to RGB
        frames_rgb = frames_rgb.astype(np.float32) / 127.5 - 1.0
        frames_tensor = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2)  # (T, C, H, W)
        cond_img = frames_tensor[0]
        target_seq = frames_tensor
        return cond_img, target_seq


def train(model: Image2VideoVAE, dataloader: DataLoader, epochs: int, device: torch.device,
          lr: float = 3e-4, kl_weight: float = 1e-3, save_dir: Path | None = None) -> None:
    """Train the VAE model.

    Args:
        model: Instance of ``Image2VideoVAE``.
        dataloader: Dataloader over (cond_img, target_seq) pairs.
        epochs: Number of epochs to train.
        device: Device to run on (CPU or CUDA).
        lr: Learning rate for Adam optimizer.
        kl_weight: Weight applied to the KL divergence term.
        save_dir: Directory to save sample outputs; if ``None``, samples are not saved.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    global_step = 0
    for epoch in range(epochs):
        model.train()
        total_rec, total_kl = 0.0, 0.0
        for i, (cond_img, target_seq) in enumerate(dataloader):
            cond_img = cond_img.to(device)
            target_seq = target_seq.to(device)
            optimizer.zero_grad()
            recon_seq, mu, kl, _ = model(cond_img, target_seq)
            # Reconstruction loss (L1)
            rec_loss = F.l1_loss(recon_seq, target_seq, reduction='mean')
            kl_loss = kl.mean()
            loss = rec_loss + kl_weight * kl_loss
            loss.backward()
            optimizer.step()
            total_rec += rec_loss.item()
            total_kl += kl_loss.item()
            global_step += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                print(
                    f"Epoch [{epoch+1}/{epochs}] Step [{i+1}/{len(dataloader)}] "
                    f"Rec: {rec_loss.item():.4f} KL: {kl_loss.item():.4f}"
                )
        # Epoch summary
        mean_rec = total_rec / len(dataloader)
        mean_kl = total_kl / len(dataloader)
        print(f"Epoch {epoch+1} finished. Mean Rec: {mean_rec:.4f}, Mean KL: {mean_kl:.4f}")
        # Sample and save
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            model.eval()
            with torch.no_grad():
                # Sample first 4 conditioning images from dataset
                cond_samples, _ = next(iter(dataloader))
                cond_samples = cond_samples[:4].to(device)
                gen_seq = model.sample(cond_samples)
                # Save to GIF
                save_path = save_dir / f"sample_epoch{epoch+1}.gif"
                frames = (gen_seq.clamp(-1, 1) * 127.5 + 127.5).byte().cpu().numpy()
                frames = frames.transpose(0, 1, 3, 4, 2)  # (B, T, H, W, C)
                # Create a grid by tiling the generated sequences side by side
                b, t, h, w, c = frames.shape
                # Build list of images per frame index
                gif_frames: List[np.ndarray] = []
                for f in range(t):
                    # Tile horizontally
                    row = np.concatenate([frames[b_idx, f] for b_idx in range(b)], axis=1)
                    gif_frames.append(row)
                # Save using PIL
                pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in gif_frames]
                pil_frames[0].save(
                    save_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=100,
                    loop=0,
                )
                print(f"Saved sample animation to {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Image2VideoVAE on a small dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to directory containing training videos")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini batch size")
    parser.add_argument("--seq-len", type=int, default=16, help="Length of video sequences to train on")
    parser.add_argument("--frame-size", type=int, default=64, help="Spatial resolution of frames (64 or 128)")
    parser.add_argument("--sample-step", type=int, default=1, help="Step between frames when sampling sequences")
    parser.add_argument("--max-samples-per-video", type=int, default=None, help="Maximum number of sequences per video")
    parser.add_argument("--kl-weight", type=float, default=1e-3, help="Weight for KL divergence term")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--save-dir", type=str, default="samples", help="Directory to save sample outputs")
    parser.add_argument(
        "--model-type",
        type=str,
        default="vae",
        choices=["vae", "unet-vae"],
        help="Type of model architecture to use: 'vae' or 'unet-vae'",
    )
    parser.add_argument(
        "--base-dim",
        type=int,
        default=64,
        help="Base dimension for convolution layers (used in unet-vae).",
    )
    parser.add_argument(
        "--cond-dim",
        type=int,
        default=256,
        help="Dimension of conditioning image embedding.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden size for GRU and sequence encoder (unet-vae).",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Latent dimensionality (unet-vae).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = Path(args.dataset)
    if not video_dir.is_dir():
        raise ValueError(f"{video_dir} is not a directory")
    # Collect video paths
    video_paths = []
    for ext in ("*.mp4", "*.mov", "*.mkv", "*.webm", "*.avi"):
        video_paths += list(video_dir.glob(ext))
    if not video_paths:
        raise ValueError(f"No video files found in {video_dir}")
    dataset = VideoSequenceDataset(
        video_paths,
        seq_len=args.seq_len,
        frame_size=args.frame_size,
        sample_step=args.sample_step,
        max_samples_per_video=args.max_samples_per_video,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use single worker for notebook compatibility
        drop_last=True,
    )
    # Instantiate model according to args.model_type
    if args.model_type == "vae":
        model = Image2VideoVAE(
            frame_size=args.frame_size,
            seq_len=args.seq_len,
            in_channels=3,
            cond_dim=args.cond_dim,
            feat_dim=128,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
        )
    elif args.model_type == "unet-vae":
        model = Image2VideoUNetVAE(
            frame_size=args.frame_size,
            seq_len=args.seq_len,
            in_channels=3,
            base_dim=args.base_dim,
            cond_dim=args.cond_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train(
        model,
        dataloader,
        epochs=args.epochs,
        device=device,
        lr=args.lr,
        kl_weight=args.kl_weight,
        save_dir=Path(args.save_dir),
    )


if __name__ == "__main__":
    main()