#!/usr/bin/env python3
"""
Visualize saved BEV tensors (.pt) with per-pixel channel statistics in one figure.

For each tensor (C, H, W), computes across the channel axis:
  mean, max, min, L2 norm (Euclidean norm per spatial location).

Default: first scene that has both *_vtransform.pt and *_decoder_neck.pt under --dir.

Example:
  python tools/visualize_bev_channel_stats.py \\
    --dir bev_gallery/pt/trainval \\
    -o bev_gallery/bev_channel_stats_overview.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from tools.visualize_bev_feat import load_feature  # noqa: E402


def _to_chw(x: np.ndarray) -> np.ndarray:
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected (C,H,W) or (1,C,H,W); got {x.shape}")
    return x.astype(np.float32)


def channel_stats(x: np.ndarray) -> Dict[str, np.ndarray]:
    """x: (C,H,W) -> maps stat name -> (H,W)."""
    x = _to_chw(x)
    return {
        "mean": x.mean(axis=0),
        "max": x.max(axis=0),
        "min": x.min(axis=0),
        "norm": np.linalg.norm(x, axis=0),
    }


def _norm01(img: np.ndarray) -> np.ndarray:
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        return (img - lo) / (hi - lo)
    return np.zeros_like(img, dtype=np.float32)


def find_first_pair(root: str) -> Tuple[str, str, str]:
    """Returns (scene_dir, vt_path, neck_path)."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    for dirpath, _, filenames in os.walk(root):
        necks = [f for f in filenames if f.endswith("_decoder_neck.pt")]
        for nf in necks:
            tok = nf.replace("_decoder_neck.pt", "")
            vf = f"{tok}_vtransform.pt"
            if vf in filenames:
                return dirpath, os.path.join(dirpath, vf), os.path.join(dirpath, nf)
    raise FileNotFoundError(f"No matching vtransform+decoder_neck pair under {root}")


def main() -> None:
    p = argparse.ArgumentParser(description="BEV .pt channel stats in one figure")
    p.add_argument(
        "--dir",
        type=str,
        default=os.path.join(_REPO_ROOT, "bev_gallery/pt/trainval"),
        help="Root folder to search (e.g. bev_gallery/pt/trainval or .../travel)",
    )
    p.add_argument(
        "--token",
        type=str,
        default="",
        help="Optional token prefix; if set, use scene dir containing this token",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=os.path.join(_REPO_ROOT, "bev_gallery/bev_channel_stats_overview.png"),
    )
    p.add_argument("--dpi", type=int, default=120)
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = os.path.abspath(args.dir)
    if args.token:
        vt_path = neck_path = None
        scene_dir = None
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.startswith(args.token) and f.endswith("_vtransform.pt"):
                    vt_path = os.path.join(dirpath, f)
                    neck_path = os.path.join(
                        dirpath, f.replace("_vtransform.pt", "_decoder_neck.pt")
                    )
                    scene_dir = dirpath
                    break
            if vt_path:
                break
        if not vt_path or not os.path.isfile(neck_path or ""):
            raise FileNotFoundError(f"No pair for token {args.token!r} under {root}")
    else:
        scene_dir, vt_path, neck_path = find_first_pair(root)

    vt = load_feature(vt_path)
    nk = load_feature(neck_path)
    stats_vt = channel_stats(vt)
    stats_nk = channel_stats(nk)

    cols = ["mean", "max", "min", "norm"]
    titles = ["mean (C)", "max (C)", "min (C)", "L2 norm (C)"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        f"BEV channel stats  |  {os.path.basename(scene_dir)}  |  token={os.path.basename(vt_path).replace('_vtransform.pt', '')}",
        fontsize=12,
    )

    for row, (name, st) in enumerate(
        [("vtransform", stats_vt), ("decoder_neck", stats_nk)]
    ):
        for j, c in enumerate(cols):
            ax = axes[row, j]
            im = ax.imshow(_norm01(st[c]), cmap="viridis", aspect="auto")
            ax.set_title(f"{name}: {titles[j]}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
