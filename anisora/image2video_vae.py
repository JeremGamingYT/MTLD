"""
image2video_vae
================

This module defines a lightweight variational autoencoder (VAE) for
conditional image‑to‑video synthesis.  The model is designed to learn
from a small collection of short anime clips and then generate novel
video sequences given a single input image.  It is intentionally
compact and uses only standard PyTorch components.

Overview
--------

The VAE consists of three main parts:

* **ImageEncoder**: Encodes the input (conditioning) image into a
  feature vector.  This branch is shared across all training
  sequences.
* **SequenceEncoder**: Encodes the sequence of target frames (during
  training) using a 2D convolutional encoder followed by a GRU.  It
  produces a final hidden state summarising the motion in the clip,
  which is then mapped to a latent distribution.
* **Decoder**: Given a sampled latent vector and the conditioning
  image embedding, generates a sequence of latent states via a GRU and
  decodes each state into an RGB frame using a transposed convolution
  network.

During training, the model reconstructs the full sequence of target
frames and optimises a combination of reconstruction loss and
Kullback–Leibler divergence.  At inference time, only the
conditioning image is needed; a latent vector is sampled from a
standard normal distribution and fed to the decoder to produce an
original sequence.

This implementation is original and does not reuse code from external
repositories.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """Simple convolutional encoder for the conditioning image."""

    def __init__(self, in_channels: int = 3, out_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim
        # Compute flatten dimension dynamically
        dummy = torch.zeros(1, in_channels, 64, 64)
        conv_out = self.conv(dummy)
        flat_dim = conv_out.numel() // conv_out.size(0)
        self.fc = nn.Linear(flat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        h = self.fc(h)
        return h


class SequenceEncoder(nn.Module):
    """Encode a sequence of frames into a latent representation."""

    def __init__(self, in_channels: int = 3, feat_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        # Frame encoder
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # Flatten dimension
        dummy = torch.zeros(1, in_channels, 64, 64)
        conv_out = self.frame_encoder(dummy)
        flat_dim = conv_out.numel() // conv_out.size(0)
        self.fc = nn.Linear(flat_dim, feat_dim)
        # Temporal model
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of frames.

        Args:
            x: Tensor of shape (B, T, C, H, W).
        Returns:
            final hidden state of shape (B, hidden_dim).
        """
        B, T, C, H, W = x.shape
        feats = []
        for t in range(T):
            f = self.frame_encoder(x[:, t])
            f = f.flatten(1)
            feats.append(self.fc(f))
        feats = torch.stack(feats, dim=1)
        _, h_n = self.gru(feats)
        return h_n[-1]


class LatentHead(nn.Module):
    """Map sequence encoder output to latent parameters and sample."""

    def __init__(self, hidden_dim: int = 256, latent_dim: int = 64) -> None:
        super().__init__()
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return z, mu, kl


class Decoder(nn.Module):
    """Decode a latent vector and conditioning image into a sequence of frames."""

    def __init__(
        self,
        latent_dim: int = 64,
        cond_dim: int = 256,
        hidden_dim: int = 256,
        feat_dim: int = 128,
        out_channels: int = 3,
        seq_len: int = 16,
        frame_size: int = 64,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.frame_size = frame_size
        # Project latent and conditioning image to initial hidden state
        self.init_fc = nn.Linear(latent_dim + cond_dim, hidden_dim)
        # Project latent and conditioning image to GRU input
        self.in_fc = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        # Map hidden states to frame features
        self.out_fc = nn.Linear(hidden_dim, feat_dim)
        # Build a dynamic upsampling decoder. We start from a 1×1 feature map
        # and repeatedly upsample by factor 2 until reaching the target spatial
        # resolution. Between upsampling steps we apply a convolution to
        # gradually decrease the number of channels.
        import math
        num_upsamples = int(math.log2(frame_size))
        assert 2 ** num_upsamples == frame_size, (
            f"frame_size must be a power of 2, got {frame_size}"
        )
        # Define a sequence of output channels for each intermediate layer.
        # Start with feat_dim and progressively decrease. The final layer
        # produces out_channels.
        hidden_dims = []
        # We'll create (num_upsamples - 1) hidden layers before the final output
        # Choose a simple geometric progression if possible
        # Example for frame_size=64 (num_upsamples=6): hidden_dims length =5
        # Values: 256, 128, 64, 32, 16
        base = 256
        for _ in range(max(num_upsamples - 1, 0)):
            hidden_dims.append(base)
            base = max(base // 2, out_channels)
        # If the list is shorter than needed, fill with out_channels
        while len(hidden_dims) < max(num_upsamples - 1, 0):
            hidden_dims.append(out_channels)
        # Build convolution layers
        layers = []
        in_ch = feat_dim
        for i in range(num_upsamples):
            # Determine output channels: for all but last layer, pick from
            # hidden_dims; final layer outputs out_channels
            if i < num_upsamples - 1:
                out_ch = hidden_dims[i]
            else:
                out_ch = out_channels
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            in_ch = out_ch
        self.conv_layers = nn.ModuleList(layers)
        self.activation = nn.ReLU(inplace=True)
        self.final_activation = nn.Tanh()

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Decode latent and conditioning into frames.

        Args:
            z: Latent tensor of shape (B, latent_dim).
            cond: Conditioning image embedding of shape (B, cond_dim).
        Returns:
            Frames tensor of shape (B, T, C, H, W).
        """
        B = z.size(0)
        # Concatenate latent and conditioning image
        cat = torch.cat([z, cond], dim=1)
        # Initial hidden state
        h0 = torch.tanh(self.init_fc(cat)).unsqueeze(0)  # (1, B, hidden_dim)
        # Prepare repeated input
        inp = torch.tanh(self.in_fc(cat)).unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, hidden_dim)
        out, _ = self.gru(inp, h0)
        # Frame features
        feats = self.out_fc(out)  # (B, T, feat_dim)
        # Decode each feature to an image
        frames: list[torch.Tensor] = []
        for t in range(self.seq_len):
            f = feats[:, t]
            # Start from a (B, feat_dim, 1, 1) tensor
            f = f.view(B, -1, 1, 1)
            x = f
            for i, conv in enumerate(self.conv_layers):
                # Upsample if not the first iteration; upsampling is done
                # before the convolution except for the first layer
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                x = conv(x)
                if i < len(self.conv_layers) - 1:
                    x = self.activation(x)
                else:
                    x = self.final_activation(x)
            frames.append(x)
        return torch.stack(frames, dim=1)


class Image2VideoVAE(nn.Module):
    """Full VAE model for conditional image‑to‑video generation."""

    def __init__(
        self,
        frame_size: int = 64,
        seq_len: int = 16,
        in_channels: int = 3,
        cond_dim: int = 256,
        feat_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.image_encoder = ImageEncoder(in_channels=in_channels, out_dim=cond_dim)
        self.sequence_encoder = SequenceEncoder(in_channels=in_channels, feat_dim=feat_dim, hidden_dim=hidden_dim)
        self.latent_head = LatentHead(hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = Decoder(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            feat_dim=feat_dim,
            out_channels=in_channels,
            seq_len=seq_len,
            frame_size=frame_size,
        )

    def forward(self, cond_img: torch.Tensor, target_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass during training.

        Args:
            cond_img: Tensor of shape (B, C, H, W) representing the conditioning image.
            target_seq: Tensor of shape (B, T, C, H, W) of the target video sequence.
        Returns:
            recon_seq: Reconstructed video sequence (B, T, C, H, W).
            mu: Latent mean (B, latent_dim).
            kl: KL divergence per sample (B,).
            cond_emb: Conditioning image embedding (B, cond_dim).
        """
        B, T, C, H, W = target_seq.shape
        assert T == self.seq_len, f"Expected sequence length {self.seq_len}, got {T}"
        cond_emb = self.image_encoder(cond_img)
        enc_h = self.sequence_encoder(target_seq)
        z, mu, kl = self.latent_head(enc_h)
        recon_seq = self.decoder(z, cond_emb)
        return recon_seq, mu, kl, cond_emb

    @torch.no_grad()
    def sample(self, cond_img: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate a video sequence given a conditioning image.

        Args:
            cond_img: Tensor of shape (B, C, H, W).
            device: Optional device for sampling.
        Returns:
            Tensor of shape (B, T, C, H, W) representing the generated sequence.
        """
        device = device or cond_img.device
        cond_emb = self.image_encoder(cond_img)
        B = cond_img.size(0)
        latent_dim = self.latent_head.fc_mu.out_features
        z = torch.randn(B, latent_dim, device=device)
        frames = self.decoder(z, cond_emb)
        return frames


__all__ = [
    "Image2VideoVAE",
]