#!/usr/bin/env python3
"""
Export BEVFusion camera-only BEV features from one NAVSIM frame to **two** ``.pt`` files:

  * ``{prefix}_vtransform.pt`` — LSS / vtransform output (after ``bev_pool`` + vtransform downsample).
  * ``{prefix}_decoder_neck.pt`` — decoder backbone + neck.

Each file is a bare ``torch.Tensor`` on CPU (load with ``torch.load``).

Example (trainval; pkls live next to ``sensor_blobs/trainval``)::

  python tools/generate_bev_features.py \\
    --dataset-root ~/wm_ws/WoTE/dataset \\
    --pkl navsim_logs/trainval/2021.05.12.19.36.12_veh-35_00005_00204.pkl \\
    --output-prefix ./bev_out/frame_0000

Test split uses ``navsim_logs/test/*.pkl`` and ``sensor_blobs/test`` (auto-detected from ``--pkl``).

Visualize::

  python tools/visualize_bev_feat.py ./bev_out/frame_0000_decoder_neck.pt -o view.png --cmap viridis
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NAVSIM frame -> two independent BEV feature .pt files"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/home/wenzhe/wm_ws/WoTE/dataset",
        help="Root containing sensor_blobs/ and navsim_logs/",
    )
    parser.add_argument(
        "--pkl",
        type=str,
        required=True,
        help="Scene .pkl path (relative to dataset-root or absolute)",
    )
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nuscenes/seg/camera-bev256d2.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pretrained/camera-only-seg.pth",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Writes {prefix}_vtransform.pt and {prefix}_decoder_neck.pt (mkdir parent dirs as needed)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--save-vtransform",
        action="store_true",
        help="Also write {prefix}_vtransform.pt (default: only {prefix}_decoder_neck.pt).",
    )
    parser.add_argument(
        "--sensor-split",
        type=str,
        default="auto",
        choices=("auto", "trainval", "test"),
        help="Which sensor_blobs subdir to use (default: infer from --pkl path)",
    )
    args = parser.parse_args()

    from tools.navsim_bev_adapter import BEVFeatureExtractor, NavsimDataAdapter

    pkl_path = args.pkl if os.path.isabs(args.pkl) else os.path.join(args.dataset_root, args.pkl)
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"Missing pkl: {pkl_path}")

    if args.sensor_split == "auto":
        norm = os.path.normpath(pkl_path).replace("\\", "/")
        if f"/navsim_logs/test/" in norm or norm.rstrip("/").endswith("/navsim_logs/test"):
            split = "test"
        else:
            split = "trainval"
    else:
        split = args.sensor_split

    sensor_blobs_root = os.path.join(args.dataset_root, "sensor_blobs", split)
    if not os.path.isdir(sensor_blobs_root):
        raise FileNotFoundError(f"Missing sensor dir: {sensor_blobs_root}")
    print(f"sensor_blobs split: {split} -> {sensor_blobs_root}")

    scene = NavsimDataAdapter.load_scene_pkl(pkl_path)
    if args.frame_index < 0 or args.frame_index >= len(scene):
        raise IndexError(
            f"frame_index {args.frame_index} out of range [0, {len(scene)})"
        )
    frame = scene[args.frame_index]

    config_path = (
        args.config
        if os.path.isabs(args.config)
        else os.path.join(_REPO_ROOT, args.config)
    )
    ckpt = (
        args.checkpoint
        if os.path.isabs(args.checkpoint)
        else os.path.join(_REPO_ROOT, args.checkpoint)
    )
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    prefix = os.path.abspath(args.output_prefix)
    parent = os.path.dirname(prefix)
    if parent:
        os.makedirs(parent, exist_ok=True)

    print(f"Loading model: {ckpt}")
    extractor = BEVFeatureExtractor(
        config_path=config_path,
        checkpoint_path=ckpt,
        device=args.device,
    )

    print(f"Frame {args.frame_index} / {len(scene)} token={frame.get('token', '')}")
    feats = extractor.extract_from_navsim_frame(
        frame, sensor_blobs_root=sensor_blobs_root
    )

    p_neck = f"{prefix}_decoder_neck.pt"
    torch.save(feats["decoder_neck"].detach().cpu(), p_neck)
    print(f"Saved: {p_neck}  shape={tuple(feats['decoder_neck'].shape)}")
    if args.save_vtransform:
        p_vt = f"{prefix}_vtransform.pt"
        torch.save(feats["vtransform"].detach().cpu(), p_vt)
        print(f"Saved: {p_vt}  shape={tuple(feats['vtransform'].shape)}")

    for k in ("vtransform", "decoder_neck"):
        assert torch.isfinite(feats[k]).all(), k


if __name__ == "__main__":
    main()
