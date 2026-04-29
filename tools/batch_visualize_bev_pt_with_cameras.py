#!/usr/bin/env python3
"""
For each (vtransform, decoder_neck) pair under --pt-root, build one PNG:
  top = 6 RGB cameras in a 2x3 grid,
  middle/bottom = 2x4 BEV channel stats (mean / max / min / L2 norm) for vt and neck.

**NAVSIM:** ``--dataset-root`` is WoTE-style: ``navsim_logs/{split}/`` and
``sensor_blobs/{split}/``.

**nuScenes:** pass ``--nuscenes-info path/to/nuscenes_infos_val.pkl`` and
``--dataset-root`` = folder with ``samples/`` (e.g. ``data/nuscenes``).

Examples:

  # NAVSIM
  python tools/batch_visualize_bev_pt_with_cameras.py \\
    --pt-root bev_gallery/pt/trainval \\
    --dataset-root /path/to/WoTE/dataset \\
    --out-dir bev_gallery/bev_pt_viz_with_cam

  # nuScenes (flat ``{token}_*.pt`` under --pt-root); pass train+val pkls if needed
  python tools/batch_visualize_bev_pt_with_cameras.py \\
    --pt-root bev_gallery/nuscenes_mini \\
    --dataset-root data/nuscenes \\
    --nuscenes-info data/nuscenes/nuscenes_infos_train.pkl data/nuscenes/nuscenes_infos_val.pkl \\
    --out-dir bev_gallery/nuscenes_mini_viz

  # NAVSIM: same script; use ``--viz-tag navsim`` for ``navsim__{split}__{scene}__{token}.png``
  python tools/batch_visualize_bev_pt_with_cameras.py \\
    --pt-root bev_gallery/navsim_samples_20 \\
    --dataset-root /path/to/WoTE/dataset \\
    --out-dir bev_gallery/navsim_viz \\
    --viz-tag navsim --max-frames 20
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from tools.navsim_frame_utils import NAVSIM_CAMERAS_VISUAL_ORDER  # noqa: E402
from tools.nuscenes_bev_adapter import (  # noqa: E402
    NUSCENES_CAMERAS_VISUAL_ORDER,
    resolve_nuscenes_data_path,
)
from tools.visualize_bev_channel_stats import channel_stats  # noqa: E402
from tools.visualize_bev_feat import load_feature  # noqa: E402


def _norm01(img: np.ndarray) -> np.ndarray:
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        return (img - lo) / (hi - lo)
    return np.zeros_like(img, dtype=np.float32)


def build_camera_rgb_grid(
    frame: Dict[str, Any],
    sensor_blobs_root: str,
    max_width: int = 3000,
) -> np.ndarray:
    """Return uint8 (H, W, 3) RGB."""
    from PIL import Image

    tiles: List[np.ndarray] = []
    h0 = w0 = None
    for name in NAVSIM_CAMERAS_VISUAL_ORDER:
        cams = frame.get("cams") or {}
        if name not in cams:
            raise KeyError(f"missing camera {name}")
        rel = cams[name]["data_path"]
        path = os.path.join(sensor_blobs_root, rel)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        pil = Image.open(path).convert("RGB")
        rgb = np.asarray(pil)
        if h0 is None:
            h0, w0 = rgb.shape[0], rgb.shape[1]
        tiles.append(rgb)

    rows, cols = 2, 3
    canvas = np.zeros((h0 * rows, w0 * cols, 3), dtype=np.uint8)
    for i, im in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * h0 : (r + 1) * h0, c * w0 : (c + 1) * w0] = im

    if canvas.shape[1] > max_width:
        scale = max_width / float(canvas.shape[1])
        nh = max(1, int(round(canvas.shape[0] * scale)))
        nw = max_width
        pil = Image.fromarray(canvas)
        pil = pil.resize((nw, nh), Image.BILINEAR)
        canvas = np.asarray(pil)
    return canvas


def build_camera_rgb_grid_nuscenes(
    info: Dict[str, Any],
    dataset_root: str,
    max_width: int = 3000,
) -> np.ndarray:
    """6-camera RGB grid from nuScenes ``info`` dict (``cams``[*].``data_path``)."""
    from PIL import Image

    tiles: List[np.ndarray] = []
    h0 = w0 = None
    for name in NUSCENES_CAMERAS_VISUAL_ORDER:
        cams = info.get("cams") or {}
        if name not in cams:
            raise KeyError(f"missing camera {name}")
        path = resolve_nuscenes_data_path(dataset_root, cams[name]["data_path"])
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        pil = Image.open(path).convert("RGB")
        rgb = np.asarray(pil)
        if h0 is None:
            h0, w0 = rgb.shape[0], rgb.shape[1]
        tiles.append(rgb)

    rows, cols = 2, 3
    canvas = np.zeros((h0 * rows, w0 * cols, 3), dtype=np.uint8)
    for i, im in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * h0 : (r + 1) * h0, c * w0 : (c + 1) * w0] = im

    if canvas.shape[1] > max_width:
        scale = max_width / float(canvas.shape[1])
        nh = max(1, int(round(canvas.shape[0] * scale)))
        nw = max_width
        pil = Image.fromarray(canvas)
        pil = pil.resize((nw, nh), Image.BILINEAR)
        canvas = np.asarray(pil)
    return canvas


def load_nuscenes_token_index(info_pkl: str) -> Dict[str, Dict[str, Any]]:
    with open(info_pkl, "rb") as f:
        data = pickle.load(f)
    infos = data.get("infos") or []
    return {str(x["token"]): x for x in infos if x.get("token")}


def load_nuscenes_token_index_multi(info_pkls: List[str]) -> Dict[str, Dict[str, Any]]:
    """Merge token -> info from several pkls (e.g. train + val for mini)."""
    merged: Dict[str, Dict[str, Any]] = {}
    for p in info_pkls:
        merged.update(load_nuscenes_token_index(os.path.abspath(p)))
    return merged


def iter_pt_pairs_flat(pt_root: str) -> Iterator[Tuple[str, str, str, str, str]]:
    """
    Flat layout: ``pt_root/{token}_vtransform.pt`` and ``{token}_decoder_neck.pt``.

    Yields (split_tag, scene_tag, token, vt_path, neck_path) with split_tag=nuscenes, scene_tag=flat.
    """
    pt_root = os.path.abspath(pt_root)
    if not os.path.isdir(pt_root):
        raise FileNotFoundError(pt_root)
    for f in sorted(os.listdir(pt_root)):
        if not f.endswith("_decoder_neck.pt"):
            continue
        tok = f.replace("_decoder_neck.pt", "")
        vf = f"{tok}_vtransform.pt"
        vp = os.path.join(pt_root, vf)
        np_ = os.path.join(pt_root, f)
        if os.path.isfile(vp):
            yield ("nuscenes", "flat", tok, vp, np_)


def load_frame_for_token(
    dataset_root: str, split: str, scene: str, token: str
) -> Optional[Dict[str, Any]]:
    pkl_path = os.path.join(
        dataset_root, "navsim_logs", split, f"{scene}.pkl"
    )
    if not os.path.isfile(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        scene_list = pickle.load(f)
    if not isinstance(scene_list, list):
        return None
    for fr in scene_list:
        if fr.get("token") == token:
            return fr
    return None


def iter_pt_pairs(pt_root: str) -> Iterator[Tuple[str, str, str, str, str]]:
    """
    Yields (split, scene, token, vt_path, neck_path).

    Supports either:
      - pt_root/<split>/<scene>/*.pt  (e.g. .../pt/trainval/scene/)
      - pt_root/<scene>/*.pt when pt_root basename is ``trainval`` or ``test``
    """
    pt_root = os.path.abspath(pt_root)
    if not os.path.isdir(pt_root):
        raise FileNotFoundError(pt_root)
    base = os.path.basename(pt_root.rstrip(os.sep))

    def emit_for_split(split: str, scenes_root: str) -> Iterator[Tuple[str, str, str, str, str]]:
        for scene in sorted(os.listdir(scenes_root)):
            sdir = os.path.join(scenes_root, scene)
            if not os.path.isdir(sdir):
                continue
            names = os.listdir(sdir)
            for f in names:
                if not f.endswith("_decoder_neck.pt"):
                    continue
                tok = f.replace("_decoder_neck.pt", "")
                vf = f"{tok}_vtransform.pt"
                if vf not in names:
                    continue
                yield (
                    split,
                    scene,
                    tok,
                    os.path.join(sdir, vf),
                    os.path.join(sdir, f),
                )

    if base in ("trainval", "test"):
        yield from emit_for_split(base, pt_root)
    else:
        for split in sorted(os.listdir(pt_root)):
            sp = os.path.join(pt_root, split)
            if not os.path.isdir(sp):
                continue
            yield from emit_for_split(split, sp)


def render_one(
    split: str,
    scene: str,
    token: str,
    vt_path: str,
    neck_path: str,
    dataset_root: str,
    dpi: int,
    camera_max_width: int,
    nuscenes_info: Optional[Dict[str, Any]] = None,
) -> "matplotlib.figure.Figure":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    if nuscenes_info is not None:
        cam_rgb = build_camera_rgb_grid_nuscenes(
            nuscenes_info, dataset_root, max_width=camera_max_width
        )
    else:
        frame = load_frame_for_token(dataset_root, split, scene, token)
        if frame is None:
            raise RuntimeError(f"no frame token={token} in {scene}.pkl")

        sensor_root = os.path.join(dataset_root, "sensor_blobs", split)
        cam_rgb = build_camera_rgb_grid(
            frame, sensor_root, max_width=camera_max_width
        )

    vt = load_feature(vt_path)
    nk = load_feature(neck_path)
    stats_vt = channel_stats(vt)
    stats_nk = channel_stats(nk)
    cols = ["mean", "max", "min", "norm"]
    titles = ["mean (C)", "max (C)", "min (C)", "L2 norm (C)"]

    # Taller figure + larger height share for RGB row so cameras read clearly.
    fig = plt.figure(figsize=(22, 20))
    gs = gridspec.GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[2.2, 1.0, 1.0],
        hspace=0.22,
        wspace=0.22,
    )
    ax0 = fig.add_subplot(gs[0, :])
    # aspect="auto" avoids default equal-aspect letterboxing (white bars) around the RGB grid.
    ax0.imshow(cam_rgb, aspect="auto")
    ax0.set_title(
        f"6 cameras (RGB)  |  {split} / {scene}  |  {token}",
        fontsize=10,
        pad=6,
    )
    ax0.axis("off")

    for row, (name, st) in enumerate(
        [("vtransform", stats_vt), ("decoder_neck", stats_nk)]
    ):
        r = row + 1
        for j, c in enumerate(cols):
            ax = fig.add_subplot(gs[r, j])
            im = ax.imshow(_norm01(st[c]), cmap="viridis", aspect="auto")
            ax.set_title(f"{name}: {titles[j]}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Tight figure box; no fig.suptitle (y>1 leaves a white band when saving).
    gs.update(left=0.03, right=0.97, top=0.98, bottom=0.03, hspace=0.18, wspace=0.2)
    return fig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pt-root",
        type=str,
        default=os.path.join(_REPO_ROOT, "bev_gallery/pt/trainval"),
        help="NAVSIM: <split>/<scene>/*.pt ; nuScenes: flat {token}_*.pt",
    )
    p.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="NAVSIM: WoTE root ; nuScenes: folder with samples/ (see --nuscenes-info)",
    )
    p.add_argument(
        "--nuscenes-info",
        type=str,
        nargs="+",
        default=None,
        help="If set, load camera paths from these nuscenes_infos_*.pkl files (merged token index)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(_REPO_ROOT, "bev_gallery/bev_pt_viz_with_cam"),
        help="Output directory for PNGs",
    )
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument(
        "--camera-max-width",
        type=int,
        default=3000,
        help="Max width (px) for stitched 6-camera RGB grid before embedding in figure (larger = sharper).",
    )
    p.add_argument("--max-frames", type=int, default=0, help="If >0, only process first N pairs")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip pairs whose output PNG already exists (resume long batch runs)",
    )
    p.add_argument(
        "--viz-tag",
        type=str,
        default="",
        help=(
            "NAVSIM only: if set (e.g. 'navsim'), save as "
            "{viz_tag}__{split}__{scene}__{token}.png (parallel to nuScenes "
            "nuscenes__flat__{token}.png). If empty, legacy {split}__{scene}__{token}.png"
        ),
    )
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    dataset_root = os.path.abspath(args.dataset_root)

    nuscenes_index: Optional[Dict[str, Dict[str, Any]]] = None
    if args.nuscenes_info:
        nuscenes_index = load_nuscenes_token_index_multi(
            list(args.nuscenes_info)
        )
        pairs = list(iter_pt_pairs_flat(args.pt_root))
    else:
        pairs = list(iter_pt_pairs(args.pt_root))
    if args.max_frames > 0:
        pairs = pairs[: args.max_frames]

    ok, fail = 0, 0
    for i, (split, scene, token, vt_path, neck_path) in enumerate(pairs):
        if args.nuscenes_info:
            out_name = f"{split}__{scene}__{token}.png"
        elif args.viz_tag:
            out_name = f"{args.viz_tag}__{split}__{scene}__{token}.png"
        else:
            out_name = f"{split}__{scene}__{token}.png"
        out_path = os.path.join(out_dir, out_name)
        if args.skip_existing and os.path.isfile(out_path):
            ok += 1
            print(f"[{i+1}/{len(pairs)}] skip (exists) {out_name}")
            continue
        try:
            nu_info = None
            if nuscenes_index is not None:
                nu_info = nuscenes_index.get(token)
                if nu_info is None:
                    raise KeyError(f"token {token} not in nuScenes info pkl")
            fig = render_one(
                split,
                scene,
                token,
                vt_path,
                neck_path,
                dataset_root,
                args.dpi,
                args.camera_max_width,
                nuscenes_info=nu_info,
            )
            fig.savefig(
                out_path,
                dpi=args.dpi,
                bbox_inches="tight",
                pad_inches=0.08,
                facecolor="white",
            )
            plt.close(fig)
            ok += 1
            print(f"[{i+1}/{len(pairs)}] ok {out_name}")
        except Exception as e:
            fail += 1
            print(f"[{i+1}/{len(pairs)}] FAIL {out_name}: {e}")

    print(f"Done. ok={ok} fail={fail} -> {out_dir}")


if __name__ == "__main__":
    main()
