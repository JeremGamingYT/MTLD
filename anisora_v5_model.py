from __future__ import annotations

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoEncoder(nn.Module):
    """Encode individual video frames into feature vectors.

    This encoder operates on one frame at a time.  It consists of a
    sequence of 2D convolutional layers with ReLU activations followed by
    spatial pooling.  The final output is a flattened vector of size
    ``feat_dim``.

    Args:
        in_channels: Number of input channels (e.g. 3 for RGB).
        feat_dim: Dimensionality of the output feature vector.
    """

    def __init__(self, in_channels: int = 3, feat_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # Compute output size after convolutions assuming square input 64x64
        dummy = torch.zeros(1, in_channels, 64, 64)
        with torch.no_grad():
            dummy_out = self.conv(dummy)
        conv_out_dim = dummy_out.numel() // dummy_out.size(0)
        self.fc = nn.Linear(conv_out_dim, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, C, H, W).
        Returns:
            A tensor of shape (B, feat_dim).
        """
        h = self.conv(x)
        h = h.flatten(1)
        h = self.fc(h)
        return h


class TemporalReasoner(nn.Module):
    """Aggregate per‑frame features over time using a GRU.

    The reasoner returns both the sequence of hidden states and the final
    hidden state.  It can operate bidirectionally, but by default runs
    forward only to avoid leaking information about the future.

    Args:
        feat_dim: Dimensionality of input feature vectors.
        hidden_dim: Number of hidden units in the GRU.
        num_layers: Number of stacked GRU layers.
        bidirectional: Whether to use a bidirectional GRU.
    """

    def __init__(
        self,
        feat_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Tensor of shape (B, T, feat_dim).
        Returns:
            Tuple of (all_hidden_states, final_hidden_state).
            - all_hidden_states has shape (B, T, hidden_dim).
            - final_hidden_state has shape (B, hidden_dim).
        """
        output, h_n = self.gru(x)
        # h_n shape: (num_layers * num_directions, B, hidden_dim)
        # For single direction, take the last layer
        if self.bidirectional:
            # Concatenate the last hidden states from both directions
            final = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            final = h_n[-1]
        return output, final


class LatentSampler(nn.Module):
    """Map the final hidden state to a latent distribution and sample.

    Implements the standard VAE reparameterisation trick.  Returns the
    sampled latent vector along with the computed KL divergence against the
    unit Gaussian.

    Args:
        hidden_dim: Dimensionality of the input hidden state.
        latent_dim: Dimensionality of the latent vector.
    """

    def __init__(self, hidden_dim: int = 256, latent_dim: int = 64) -> None:
        super().__init__()
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            h: Tensor of shape (B, hidden_dim).
        Returns:
            Tuple of (z, kl_divergence).
        """
        mu = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # Compute KL divergence for regularisation
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return z, kl


class TemporalDecoder(nn.Module):
    """Generate a sequence of hidden states from a latent vector.

    The decoder uses a GRU to unfold the latent vector over time.  A
    conditioning projection injects the latent vector into the initial
    hidden state as well as at every time step via concatenation.

    Args:
        latent_dim: Dimensionality of the sampled latent vector.
        hidden_dim: Number of hidden units in the GRU.
        feat_dim: Dimensionality of the output feature per frame.
        num_layers: Number of stacked GRU layers.
        seq_len: Length of the sequence to generate.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        feat_dim: int = 128,
        num_layers: int = 1,
        seq_len: int = 16,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.init_proj = nn.Linear(latent_dim, hidden_dim)
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, feat_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Tensor of shape (B, latent_dim).
        Returns:
            Tensor of shape (B, seq_len, feat_dim).
        """
        B = z.size(0)
        # Initial hidden state
        h0 = torch.tanh(self.init_proj(z)).unsqueeze(0)  # (1, B, hidden_dim)
        # Prepare input sequence: replicate latent vector seq_len times
        conditioning = torch.tanh(self.input_proj(z)).unsqueeze(1)  # (B, 1, hidden_dim)
        decoder_inputs = conditioning.repeat(1, self.seq_len, 1)
        out, _ = self.gru(decoder_inputs, h0)
        feats = self.fc(out)
        return feats


class FrameDecoder(nn.Module):
    """Decode per‑frame feature vectors back into images.

    A small transposed convolutional network reconstructs frames from the
    feature vectors produced by the temporal decoder.  The spatial
    resolution can be controlled via the ``output_size`` parameter.

    Args:
        feat_dim: Dimensionality of the input feature vector.
        out_channels: Number of output channels (e.g. 3 for RGB).
        output_size: Spatial dimension of the output images (height = width).
    """

    def __init__(self, feat_dim: int = 128, out_channels: int = 3, output_size: int = 64) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim, 256 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.output_size = output_size

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            feats: Tensor of shape (B, T, feat_dim).
        Returns:
            Tensor of shape (B, T, C, H, W).
        """
        B, T, D = feats.shape
        x = self.fc(feats.view(-1, D))  # (B*T, 256*8*8)
        x = x.view(B * T, 256, 8, 8)
        x = self.deconv(x)  # (B*T, C, H, W)
        x = x.view(B, T, -1, self.output_size, self.output_size)
        return x


class AnisoraV5(nn.Module):
    """Full Anisora V5 model.

    This class combines all sub‑modules into a single end‑to‑end model.
    When called, it encodes an input video clip, samples a latent vector,
    decodes a new sequence and returns the reconstruction along with the
    KL divergence term.  During inference you can call the ``sample``
    method to generate completely new clips from random latent vectors.

    Args:
        in_channels: Number of channels in the input frames.
        frame_size: Spatial resolution of the frames (height = width).
        seq_len: Number of frames per clip.
        feat_dim: Dimensionality of intermediate feature vectors.
        hidden_dim: Hidden size for the temporal models.
        latent_dim: Dimensionality of the latent vector.
    """

    def __init__(
        self,
        in_channels: int = 3,
        frame_size: int = 64,
        seq_len: int = 16,
        feat_dim: int = 128,
        hidden_dim: int = 256,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.encoder = VideoEncoder(in_channels=in_channels, feat_dim=feat_dim)
        self.reasoner = TemporalReasoner(feat_dim=feat_dim, hidden_dim=hidden_dim)
        self.sampler = LatentSampler(hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = TemporalDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            feat_dim=feat_dim,
            seq_len=seq_len,
        )
        self.frame_decoder = FrameDecoder(feat_dim=feat_dim, out_channels=in_channels, output_size=frame_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.

        Args:
            x: Tensor of shape (B, T, C, H, W).
        Returns:
            Tuple of (reconstructed_x, kl_divergence), where
                reconstructed_x has shape (B, T, C, H, W) and
                kl_divergence has shape (B,) representing KL for each sample.
        """
        B, T, C, H, W = x.shape
        assert T == self.seq_len, f"Expected sequence length {self.seq_len}, got {T}"
        # Encode frames individually
        enc_feats = []
        for t in range(T):
            frame = x[:, t]
            enc_feats.append(self.encoder(frame))  # (B, feat_dim)
        feats = torch.stack(enc_feats, dim=1)  # (B, T, feat_dim)
        # Reason over time
        _, final_h = self.reasoner(feats)
        # Sample latent
        z, kl = self.sampler(final_h)
        # Decode sequence
        dec_feats = self.decoder(z)  # (B, T, feat_dim)
        # Decode frames
        recon_x = self.frame_decoder(dec_feats)  # (B, T, C, H, W)
        return recon_x, kl

    @torch.no_grad()
    def sample(self, batch_size: int = 1, seq_len: Optional[int] = None, device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate a batch of new video clips by sampling random latents.

        Args:
            batch_size: Number of samples to generate.
            seq_len: Length of the sequence; defaults to the model's seq_len.
            device: Device to generate on.
        Returns:
            A tensor of shape (batch_size, T, C, H, W) of generated videos.
        """
        device = device or next(self.parameters()).device
        seq_len = seq_len or self.seq_len
        # Sample from standard normal
        z = torch.randn(batch_size, self.sampler.fc_mean.out_features, device=device)
        # Decode latent into sequence features
        dec_feats = self.decoder(z)
        # Decode frames
        videos = self.frame_decoder(dec_feats)
        return videos


__all__ = [
    "VideoEncoder",
    "TemporalReasoner",
    "LatentSampler",
    "TemporalDecoder",
    "FrameDecoder",
    "AnisoraV5",
]