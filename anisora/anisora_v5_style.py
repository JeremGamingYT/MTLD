"""
anisora_v5_style
================

This module defines functions to extract stylistic and motion features from
a small set of anime videos.  The extracted statistics are later used
by the generative pipeline to produce new video sequences from a single
input image.  The implementation is deliberately simple and does not
rely on deep learning frameworks.  Instead, it uses OpenCV and
scikit‑learn to analyse colour palettes, edge characteristics and
approximate motion patterns.

The key functions are:

``extract_style_statistics`` – Given a directory of video files, this
  function reads a sample of frames, computes a colour palette via
  k‑means clustering, estimates average edge thickness, and collects
  coarse motion vectors between consecutive frames.

``save_style`` / ``load_style`` – Utilities to persist the extracted
  style parameters to/from disk as JSON files.

Example usage::

    from anisora_v5_style import extract_style_statistics, save_style
    style = extract_style_statistics("./dataset", num_videos=10, frames_per_video=30)
    save_style(style, "style.json")

These parameters can then be passed to the generator to control the
appearance and motion of newly generated clips.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans


def _read_video_frames(path: str, num_frames: int) -> List[np.ndarray]:
    """Read a subset of frames from a video file.

    Args:
        path: Path to the video file.
        num_frames: Number of frames to sample evenly across the video.
    Returns:
        A list of RGB images as NumPy arrays.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def _compute_colour_palette(frames: List[np.ndarray], num_colours: int = 8) -> List[List[int]]:
    """Compute a representative colour palette from a list of frames.

    We flatten all frames, sample a subset of pixels, and apply k‑means
    clustering to find the dominant colours.  The resulting palette is
    returned as a list of RGB integer triplets.
    """
    # Sample up to 50000 pixels randomly to avoid memory issues
    pixels = np.concatenate([f.reshape(-1, 3) for f in frames], axis=0)
    if pixels.shape[0] > 50000:
        idx = np.random.choice(pixels.shape[0], 50000, replace=False)
        pixels = pixels[idx]
    # K‑means clustering
    kmeans = KMeans(n_clusters=num_colours, random_state=0).fit(pixels)
    centres = kmeans.cluster_centers_
    palette = [list(map(int, c)) for c in centres]
    return palette


def _estimate_edge_thickness(frame: np.ndarray) -> float:
    """Estimate average edge thickness in a frame using Canny and morphology.

    The function applies Canny edge detection, dilates the edges with a
    3×3 kernel and computes the ratio between dilated and original edge
    pixels.  A higher ratio indicates thicker lines.  The average over
    many frames yields a global estimate.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    original_count = np.count_nonzero(edges)
    dilated_count = np.count_nonzero(dilated)
    if original_count == 0:
        return 1.0
    return dilated_count / original_count


def _extract_motion_vectors(frames: List[np.ndarray]) -> List[Tuple[float, float]]:
    """Compute coarse motion vectors between consecutive frames.

    Rather than computing full optical flow, we convert each frame to
    grayscale, threshold to emphasise structure and compute the centre of
    mass.  Differences in centres between frames serve as approximate
    motion vectors.  Returns a list of (dx, dy) pairs.
    """
    vectors: List[Tuple[float, float]] = []
    prev_com = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Invert so that white regions correspond to objects
        thresh = 255 - thresh
        # Compute centre of mass
        ys, xs = np.nonzero(thresh)
        if len(xs) == 0:
            continue
        com = (np.mean(xs), np.mean(ys))
        if prev_com is not None:
            dx = com[0] - prev_com[0]
            dy = com[1] - prev_com[1]
            vectors.append((float(dx), float(dy)))
        prev_com = com
    return vectors


def extract_style_statistics(
    dataset_dir: str,
    num_videos: int = 10,
    frames_per_video: int = 30,
    num_palette_colours: int = 8,
) -> Dict[str, object]:
    """Extract style and motion statistics from a dataset of videos.

    Args:
        dataset_dir: Path to a directory containing video files.
        num_videos: Maximum number of videos to sample from the directory.
        frames_per_video: Number of frames to sample per video.
        num_palette_colours: Number of colours to include in the palette.
    Returns:
        A dictionary containing the colour palette, average edge thickness and
        motion statistics (mean and standard deviation of dx and dy).
    """
    dataset_path = Path(dataset_dir)
    video_paths = [p for p in dataset_path.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}]
    if not video_paths:
        raise FileNotFoundError(f"No video files found in {dataset_dir}")
    selected_paths = video_paths[:num_videos]
    all_frames: List[np.ndarray] = []
    all_edges: List[float] = []
    all_vectors: List[Tuple[float, float]] = []
    for vid in selected_paths:
        frames = _read_video_frames(str(vid), frames_per_video)
        all_frames.extend(frames)
        # Compute edge thickness for each frame
        for f in frames:
            all_edges.append(_estimate_edge_thickness(f))
        # Compute motion vectors for this video
        vecs = _extract_motion_vectors(frames)
        all_vectors.extend(vecs)
    # Colour palette
    palette = _compute_colour_palette(all_frames, num_palette_colours)
    # Edge thickness
    avg_edge_thickness = float(np.mean(all_edges)) if all_edges else 1.0
    # Motion statistics
    if all_vectors:
        dxs, dys = zip(*all_vectors)
        motion_mean = (float(np.mean(dxs)), float(np.mean(dys)))
        motion_std = (float(np.std(dxs)), float(np.std(dys)))
    else:
        motion_mean = (0.0, 0.0)
        motion_std = (1.0, 1.0)
    return {
        "palette": palette,
        "edge_thickness": avg_edge_thickness,
        "motion_mean": motion_mean,
        "motion_std": motion_std,
    }


def save_style(style: Dict[str, object], path: str) -> None:
    """Save style parameters to a JSON file."""
    with open(path, "w") as f:
        json.dump(style, f, indent=2)


def load_style(path: str) -> Dict[str, object]:
    """Load style parameters from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


__all__ = [
    "extract_style_statistics",
    "save_style",
    "load_style",
]