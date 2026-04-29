#!/usr/bin/env python3
"""
Quickly visualize BEV features saved as .pt (torch.Tensor) or .npy (ndarray).

Use ``tools/generate_bev_features.py`` to create ``*_vtransform.pt`` and
``*_decoder_neck.pt`` from a NAVSIM frame.

Typical shapes: (C, H, W) or (1, C, H, W) or (B, C, H, W).
Default: channel-wise mean -> min-max normalize -> grayscale PNG.

Examples:
  python tools/visualize_bev_feat.py ./bev_out/frame_0000_decoder_neck.pt -o bev_view.png
  python tools/visualize_bev_feat.py feats.npy --mode first --cmap viridis -o bev_color.png
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import Any, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_feature(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        x = np.load(path)
    elif ext in (".pt", ".pth"):
        import torch

        obj: Any = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            if len(obj) == 1:
                obj = next(iter(obj.values()))
            else:
                raise ValueError(
                    f"Checkpoint dict has multiple keys {list(obj.keys())}; pass a single-tensor .pt"
                )
        if hasattr(obj, "detach"):
            obj = obj.detach()
        if hasattr(obj, "cpu"):
            obj = obj.cpu()
        if hasattr(obj, "numpy"):
            x = obj.numpy()
        else:
            x = np.asarray(obj)
    else:
        raise ValueError(f"Unsupported extension {ext}; use .pt, .pth, or .npy")

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if x.dtype == np.float16:
        x = x.astype(np.float32)
    return x


def to_2d_heatmap(x: np.ndarray, mode: str) -> np.ndarray:
    """Return float32 (H, W) in [0, 1]."""
    if x.ndim == 4:
        x = x[0]  # first in batch
    if x.ndim != 3:
        raise ValueError(f"Expected (B,C,H,W) or (C,H,W); got shape {x.shape}")

    if mode == "mean":
        img = x.mean(axis=0)
    elif mode == "first":
        img = x[0].copy()
    elif mode == "l2":
        img = np.linalg.norm(x, axis=0)
    else:
        raise ValueError(f"Unknown mode {mode!r}; use mean, first, l2")

    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return img


def save_grayscale(img01: np.ndarray, out_path: str) -> None:
    from PIL import Image

    g = (img01 * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(g, mode="L").save(out_path)


def save_colormap(img01: np.ndarray, out_path: str, cmap_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(img01)  # (H,W,4) float
    rgb = (rgba[:, :, :3] * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    from PIL import Image

    Image.fromarray(rgb, mode="RGB").save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize BEV .pt / .npy as PNG")
    parser.add_argument("input", type=str, help="Path to .pt, .pth, or .npy")
    parser.add_argument("-o", "--output", type=str, default="bev_feat_vis.png")
    parser.add_argument(
        "--mode",
        type=str,
        default="mean",
        choices=("mean", "first", "l2"),
        help="How to collapse channels to 2D",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="",
        help='If set (e.g. viridis, plasma), save RGB heatmap; else grayscale',
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=1,
        help="Integer upscale for viewing (nearest)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(args.input)

    x = load_feature(args.input)
    h = to_2d_heatmap(x, args.mode)
    if args.upscale > 1:
        from PIL import Image

        pil = Image.fromarray((h * 255).astype(np.uint8), mode="L")
        w2, h2 = pil.size
        pil = pil.resize(
            (w2 * args.upscale, h2 * args.upscale), Image.Resampling.NEAREST
        )
        h = np.asarray(pil).astype(np.float32) / 255.0

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def do_save(path: str) -> None:
        if args.cmap:
            save_colormap(h, path, args.cmap)
        else:
            save_grayscale(h, path)

    try:
        do_save(args.output)
        print(f"Saved {args.output} (from {args.input}, shape {x.shape}, mode={args.mode})")
    except PermissionError:
        fb = os.path.join(tempfile.gettempdir(), os.path.basename(args.output))
        do_save(fb)
        print(
            f"Permission denied on {args.output!r} (e.g. directory owned by root from Docker). "
            f"Saved instead to: {fb}",
            file=sys.stderr,
        )
        print(f"Fix: sudo chown -R \"$USER:$USER\" {out_dir or '.'}")


if __name__ == "__main__":
    main()
