"""
anisora_v5_generate
===================

This module contains functions to generate anime‑style video sequences
from a single input image using the style parameters extracted by
``anisora_v5_style``.  It does not require any deep learning
frameworks and uses only NumPy and OpenCV.  The key idea is to apply
an anime filter to the input image (edge detection + colour
quantisation) and to create a temporal sequence of crops and warps
according to the motion statistics derived from a small training set.

The main functions are:

``anime_filter`` – Converts an input RGB image into an anime‑like
    representation by smoothing colours, enhancing edges and mapping
    colours to a palette.

``generate_video`` – Produces a sequence of frames by applying random
    motions sampled from the training distribution to the filtered image
    and rescales it back to the desired frame size.  The output can be
    saved or further processed.

Example usage::

    from anisora_v5_style import load_style
    from anisora_v5_generate import generate_video
    style = load_style("style.json")
    frames = generate_video("image.jpg", style, output_len=80)
    # ``frames`` is a NumPy array of shape (T, H, W, C)
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, List

import cv2
import numpy as np


def _quantise_colours(img: np.ndarray, palette: List[List[int]]) -> np.ndarray:
    """Map each pixel in ``img`` to the closest colour in the palette.

    Args:
        img: Input image as a 3D NumPy array (H, W, C) with values in [0, 255].
        palette: List of RGB colours (each a list of 3 ints).
    Returns:
        A colour‑quantised image of the same shape as ``img``.
    """
    pal = np.array(palette, dtype=np.float32)  # (P, 3)
    # Flatten image to (N, 3)
    flat = img.reshape(-1, 3).astype(np.float32)
    # Compute distances to palette colours
    dists = np.sqrt(((flat[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2))
    indices = np.argmin(dists, axis=1)
    quantised = pal[indices].astype(np.uint8)
    return quantised.reshape(img.shape)


def anime_filter(img: np.ndarray, palette: List[List[int]], edge_thickness: float) -> np.ndarray:
    """Apply an anime‑style filter to an image.

    The filter performs bilateral smoothing, edge enhancement and colour
    quantisation according to the palette.  Edge thickness controls the
    dilation applied to detected edges: thicker edges produce stronger
    outlines.

    Args:
        img: RGB input image as a NumPy array in [0, 255].
        palette: Colour palette as produced by style extraction.
        edge_thickness: Relative edge thickness (>1 thickens, <1 thins).
    Returns:
        An anime‑style image with the same shape as ``img``.
    """
    # Bilateral filter to smooth colours while preserving edges
    smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    # Edge detection with adaptive threshold
    gray = cv2.cvtColor(smooth, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Adjust edge thickness by dilation
    ksize = max(1, int(round(edge_thickness)))
    kernel = np.ones((ksize, ksize), np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=1)
    # Invert edges to create a mask
    edge_mask = 255 - thick_edges
    # Colour quantisation
    quantised = _quantise_colours(smooth, palette)
    # Combine quantised colours with edges (darken edges)
    edge_mask_3 = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2RGB)
    anime_img = cv2.bitwise_and(quantised, edge_mask_3)
    return anime_img


def _generate_motion_sequence(length: int, mean: Tuple[float, float], std: Tuple[float, float]) -> List[Tuple[int, int]]:
    """Sample a sequence of integer translation vectors from a Gaussian distribution."""
    dxs = np.random.normal(loc=mean[0], scale=max(std[0], 1e-3), size=length)
    dys = np.random.normal(loc=mean[1], scale=max(std[1], 1e-3), size=length)
    return [(int(round(dx)), int(round(dy))) for dx, dy in zip(dxs, dys)]


def generate_video(
    image_path: str,
    style: Dict[str, object],
    output_len: int = 80,
    output_size: Tuple[int, int] = (256, 256),
    motion_scale: float = 1.0,
) -> np.ndarray:
    """Generate an anime‑style video from a single image.

    The function reads the input image, applies the anime filter based on the
    provided style parameters, and then produces ``output_len`` frames by
    panning and shifting the image according to sampled motion vectors.

    Args:
        image_path: Path to the input image.
        style: Dictionary containing at least the keys ``palette``, ``edge_thickness``,
            ``motion_mean`` and ``motion_std`` as produced by ``extract_style_statistics``.
        output_len: Number of frames to generate.
        output_size: Desired (height, width) of the output frames.
        motion_scale: Multiplier applied to the sampled motion vectors; use values >1
            to exaggerate motion or <1 to smooth it.
    Returns:
        A NumPy array of shape (output_len, H, W, 3) containing the generated frames.
    """
    # Read and filter the image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    palette: List[List[int]] = style.get("palette", [[128, 128, 128]])
    edge_thickness: float = float(style.get("edge_thickness", 1.0))
    anime_img = anime_filter(img_rgb, palette, edge_thickness)
    # Resize anime image to be larger than output to allow cropping
    H_out, W_out = output_size
    # Make sure the working image is at least as big as desired output
    scale_factor = max(H_out / anime_img.shape[0], W_out / anime_img.shape[1]) * 1.5
    new_h = int(round(anime_img.shape[0] * scale_factor))
    new_w = int(round(anime_img.shape[1] * scale_factor))
    big_img = cv2.resize(anime_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Generate motion sequence
    motion_mean: Tuple[float, float] = tuple(style.get("motion_mean", (0.0, 0.0)))  # type: ignore
    motion_std: Tuple[float, float] = tuple(style.get("motion_std", (1.0, 1.0)))  # type: ignore
    motions = _generate_motion_sequence(output_len, motion_mean, motion_std)
    # Initialise starting crop position at centre
    cx = (new_w - W_out) // 2
    cy = (new_h - H_out) // 2
    frames = []
    for dx, dy in motions:
        cx += int(round(dx * motion_scale))
        cy += int(round(dy * motion_scale))
        # Clamp crop coordinates to valid range
        cx = max(0, min(cx, new_w - W_out))
        cy = max(0, min(cy, new_h - H_out))
        crop = big_img[cy : cy + H_out, cx : cx + W_out]
        frames.append(crop.astype(np.uint8))
    return np.stack(frames, axis=0)


__all__ = [
    "anime_filter",
    "generate_video",
]