"""
image2video_unet_vae
====================

This module defines an upgraded conditional image‑to‑video VAE that
employs a U‑Net style decoder with skip connections.  The goal is to
provide a richer mapping from latent codes to video frames while
retaining high‑frequency details from the conditioning image.

The model is composed of the following parts:

* **CondEncoder**: Extracts a hierarchy of feature maps from the
  conditioning image.  Four scales (32×32, 16×16, 8×8, 4×4) are
  produced, and the coarsest feature is also pooled to obtain a
  global embedding for the latent pathway.
* **SequenceEncoder**: Encodes a sequence of video frames into a
  hidden vector using a convolutional encoder followed by a GRU.
* **LatentHead**: Maps the hidden vector to a Gaussian distribution
  and samples a latent code.  Computes the KL divergence term.
* **DecoderUNet**: Generates a sequence of frames from the latent
  codes and conditioning features.  A GRU produces a hidden state
  sequence from the latent code and conditioning embedding; each
  hidden state is projected to a small spatial map which is then
  upsampled through a series of convolutions with skip connections
  from the CondEncoder.
* **Image2VideoUNetVAE**: Top‑level module that combines all
  components.

The architecture remains simple enough to train on a small dataset
but is more expressive than a plain VAE, offering better detail
preservation and temporal dynamics.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CondEncoder(nn.Module):
    """Encode the conditioning image into multi‑scale features and a global embedding."""

    def __init__(self, in_channels: int = 3, base_dim: int = 64, cond_dim: int = 256) -> None:
        super().__init__()
        # Four stages: produce features at 32x32,16x16,8x8,4x4 for input 64x64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )  # -> 32x32, base_dim
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )  # -> 16x16, 2*base_dim
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )  # -> 8x8, 4*base_dim
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_dim * 4, base_dim * 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )  # -> 4x4, 8*base_dim
        # Global pooling for cond embedding
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_dim * 8, cond_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # x: (B, C, H, W), assume H=W=64
        f1 = self.conv1(x)  # 32x32
        f2 = self.conv2(f1)  # 16x16
        f3 = self.conv3(f2)  # 8x8
        f4 = self.conv4(f3)  # 4x4
        # Flatten and produce cond embedding
        pooled = self.global_pool(f4).flatten(1)
        cond_emb = self.fc(pooled)
        return cond_emb, [f4, f3, f2, f1]


class SequenceEncoder(nn.Module):
    """Encode a sequence of frames into a latent representation."""

    def __init__(self, in_channels: int = 3, base_dim: int = 64, hidden_dim: int = 512) -> None:
        super().__init__()
        # Frame encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )  # -> 32x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )  # -> 16x16
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )  # -> 8x8
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_dim * 4, base_dim * 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )  # -> 4x4
        # Linear to reduce to feature vector
        self.fc = nn.Linear(base_dim * 8 * 4 * 4, base_dim * 8)
        # GRU to aggregate over time
        self.gru = nn.GRU(input_size=base_dim * 8, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence x of shape (B, T, C, H, W)."""
        B, T, C, H, W = x.shape
        feats = []
        for t in range(T):
            f = x[:, t]
            h1 = self.conv1(f)
            h2 = self.conv2(h1)
            h3 = self.conv3(h2)
            h4 = self.conv4(h3)
            h4_flat = h4.flatten(1)
            feats.append(self.fc(h4_flat))
        feats = torch.stack(feats, dim=1)
        _, h_n = self.gru(feats)
        return h_n[-1]


class LatentHead(nn.Module):
    """Map hidden representation to latent distribution."""

    def __init__(self, hidden_dim: int = 512, latent_dim: int = 128) -> None:
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


class DecoderUNet(nn.Module):
    """U‑Net style decoder that generates frames using skip connections."""

    def __init__(
        self,
        latent_dim: int = 128,
        cond_dim: int = 256,
        hidden_dim: int = 512,
        base_dim: int = 64,
        out_channels: int = 3,
        seq_len: int = 16,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        # GRU to produce hidden states for each frame
        self.init_fc = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.in_fc = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        # Project GRU hidden state to coarse spatial map (4×4, 8*base_dim)
        self.fc_proj = nn.Linear(hidden_dim, base_dim * 8 * 4 * 4)
        # Convolution layers for upsampling with skip connections
        # Stage 1: 4→8
        self.up1 = nn.Sequential(
            nn.Conv2d(base_dim * 8 + base_dim * 4, base_dim * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Stage 2: 8→16
        self.up2 = nn.Sequential(
            nn.Conv2d(base_dim * 4 + base_dim * 2, base_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Stage 3: 16→32
        self.up3 = nn.Sequential(
            nn.Conv2d(base_dim * 2 + base_dim, base_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Stage 4: 32→64
        self.up4 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Final output conv
        self.out_conv = nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, cond_emb: torch.Tensor, cond_feats: List[torch.Tensor]) -> torch.Tensor:
        """Generate a sequence of frames.

        Args:
            z: Latent codes (B, latent_dim).
            cond_emb: Conditioning embedding (B, cond_dim).
            cond_feats: List of conditioning feature maps [f4, f3, f2, f1]
                with shapes [(B, 8*base, 4,4), (B,4*base,8,8), (B,2*base,16,16), (B,base,32,32)].
        Returns:
            frames: (B, T, C, H, W) generated sequence.
        """
        B = z.size(0)
        # Prepare GRU input: same for all time steps
        cat = torch.cat([z, cond_emb], dim=1)
        h0 = torch.tanh(self.init_fc(cat)).unsqueeze(0)
        inp = torch.tanh(self.in_fc(cat)).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.gru(inp, h0)
        frames: List[torch.Tensor] = []
        for t in range(self.seq_len):
            h_t = out[:, t]
            # Project to 4×4 spatial map
            x = self.fc_proj(h_t)  # (B, 8*base*4*4)
            x = x.view(B, -1, 4, 4)  # (B, 8*base, 4,4)
            # Retrieve cond features
            f4, f3, f2, f1 = cond_feats
            # Stage 1: upsample to 8x8 and merge with f3
            x = F.interpolate(x, scale_factor=2, mode="nearest")  # 4→8
            x = torch.cat([x, f3], dim=1)
            x = self.up1(x)  # -> 8
            # Stage 2: 8→16, merge with f2
            x = F.interpolate(x, scale_factor=2, mode="nearest")  # 8→16
            x = torch.cat([x, f2], dim=1)
            x = self.up2(x)  # -> 16
            # Stage 3: 16→32, merge with f1
            x = F.interpolate(x, scale_factor=2, mode="nearest")  # 16→32
            x = torch.cat([x, f1], dim=1)
            x = self.up3(x)  # -> 32
            # Stage 4: 32→64 (no skip here)
            x = F.interpolate(x, scale_factor=2, mode="nearest")  # 32→64
            x = self.up4(x)
            # Final conv to RGB
            x = self.out_conv(x)
            x = torch.tanh(x)
            frames.append(x)
        return torch.stack(frames, dim=1)


class Image2VideoUNetVAE(nn.Module):
    """High‑level VAE combining all components."""

    def __init__(
        self,
        frame_size: int = 64,
        seq_len: int = 16,
        in_channels: int = 3,
        base_dim: int = 64,
        cond_dim: int = 256,
        hidden_dim: int = 512,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.cond_encoder = CondEncoder(in_channels=in_channels, base_dim=base_dim, cond_dim=cond_dim)
        self.sequence_encoder = SequenceEncoder(in_channels=in_channels, base_dim=base_dim, hidden_dim=hidden_dim)
        self.latent_head = LatentHead(hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = DecoderUNet(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            base_dim=base_dim,
            out_channels=in_channels,
            seq_len=seq_len,
        )

    def forward(self, cond_img: torch.Tensor, target_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C, H, W = target_seq.shape
        assert T == self.seq_len, f"Expected seq_len={self.seq_len}, got {T}"
        cond_emb, cond_feats = self.cond_encoder(cond_img)
        seq_h = self.sequence_encoder(target_seq)
        z, mu, kl = self.latent_head(seq_h)
        recon_seq = self.decoder(z, cond_emb, cond_feats)
        return recon_seq, mu, kl, cond_emb

    @torch.no_grad()
    def sample(self, cond_img: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or cond_img.device
        cond_emb, cond_feats = self.cond_encoder(cond_img)
        B = cond_img.size(0)
        latent_dim = self.latent_head.fc_mu.out_features
        z = torch.randn(B, latent_dim, device=device)
        frames = self.decoder(z, cond_emb, cond_feats)
        return frames


__all__ = [
    "Image2VideoUNetVAE",
]