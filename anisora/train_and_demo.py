"""
train_and_demo
==============

This script ties together the style extraction and generation modules to
provide a complete pipeline for creating anime‑style videos from a
single image.  It first extracts style statistics from a small set of
videos and then uses those statistics to generate a new video from a
provided input image.  The generated frames are saved as an animated
GIF for easy viewing.

Usage::

    python3 train_and_demo.py --dataset ./anime_clips \
                             --image input.jpg \
                             --output demo.gif

Requirements: Only NumPy, OpenCV, scikit‑learn and Pillow are used, all
of which are available in many notebook environments.  No external
repositories are referenced.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
from PIL import Image

from anisora_v5_style import extract_style_statistics, save_style
from anisora_v5_generate import generate_video


def save_as_gif(frames: np.ndarray, path: str, fps: int = 8) -> None:
    """Save a sequence of frames (T, H, W, C) as an animated GIF."""
    images = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
    if images:
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0,
        )


def main(args: argparse.Namespace) -> None:
    # Step 1: Extract style from the training videos
    print(f"Extracting style statistics from {args.dataset}…")
    style = extract_style_statistics(
        dataset_dir=args.dataset,
        num_videos=args.num_videos,
        frames_per_video=args.frames_per_video,
        num_palette_colours=args.palette_size,
    )
    # Optionally save style to JSON
    if args.save_style is not None:
        save_style(style, args.save_style)
        print(f"Style saved to {args.save_style}")
    # Step 2: Generate video from input image using the extracted style
    print(f"Generating video from {args.image}…")
    frames = generate_video(
        image_path=args.image,
        style=style,
        output_len=args.output_length,
        output_size=(args.height, args.width),
        motion_scale=args.motion_scale,
    )
    # Step 3: Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_as_gif(frames, args.output, fps=args.fps)
    print(f"Video saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anisora V5 training and demo script")
    parser.add_argument("--dataset", type=str, required=True, help="Directory containing training videos")
    parser.add_argument("--image", type=str, required=True, help="Input image path for generation")
    parser.add_argument("--output", type=str, default="output.gif", help="Path to save the generated GIF")
    parser.add_argument("--num-videos", type=int, default=10, help="Number of videos to sample from dataset")
    parser.add_argument(
        "--frames-per-video", type=int, default=30, help="Number of frames to sample from each training video"
    )
    parser.add_argument("--palette-size", type=int, default=8, help="Number of colours in the extracted palette")
    parser.add_argument("--output-length", type=int, default=80, help="Length of the generated video (in frames)")
    parser.add_argument("--height", type=int, default=256, help="Height of the generated frames")
    parser.add_argument("--width", type=int, default=256, help="Width of the generated frames")
    parser.add_argument(
        "--motion-scale", type=float, default=1.0, help="Scale factor for motion vectors in the generated video"
    )
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for the output GIF")
    parser.add_argument(
        "--save-style", type=str, default=None, help="Optionally save extracted style parameters to this JSON path"
    )
    main(parser.parse_args())