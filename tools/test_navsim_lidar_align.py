#!/usr/bin/env python
"""Run pretrained camera-only-seg on a NAVSIM frame with/without lidar-frame alignment.

Goal: verify that the pretrained model's BEV output on NAVSIM is rotated 90°
relative to NAVSIM GT due to the lidar frame convention difference:

  nuScenes lidar: X = right,   Y = forward
  NAVSIM   lidar: X = forward, Y = left       (lidar = ego, identity)

R_align = 90° CCW about Z = nuScenes convention expressed in NAVSIM frame.
Applied as ``lidar_aug_matrix``, it pre-rotates LSSTransform's point cloud
before bev_pool so the model operates in its training convention.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from tools.bev_seg_inference import BEVSegmentationInference
from tools.navsim_bev_adapter import NavsimDataAdapter
from tools.navsim_frame_utils import NAVSIM_CAMERAS_NUSCENES_ORDER


MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "ped_crossing":  (251, 154, 153),
    "walkway":       (227, 26, 28),
    "stop_line":     (253, 191, 111),
    "carpark_area":  (255, 127, 0),
    "divider":       (106, 61, 154),
}
DEFAULT_CLASSES = [
    "drivable_area", "ped_crossing", "walkway",
    "stop_line", "carpark_area", "divider",
]


def colorize_mask(mask_bool: np.ndarray, classes) -> np.ndarray:
    canvas = np.full((mask_bool.shape[1], mask_bool.shape[2], 3), 240, dtype=np.uint8)
    for c, name in enumerate(classes):
        if name not in MAP_PALETTE:
            continue
        canvas[mask_bool[c] > 0] = MAP_PALETTE[name]
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--frame_idx", type=int, required=True)
    ap.add_argument("--sensor_blobs_root", required=True)
    ap.add_argument("--gt_npy", default=None)
    ap.add_argument("--gt_index", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.45)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.pkl, "rb") as f:
        scene = pickle.load(f)
    frame = scene[args.frame_idx]
    token = str(frame.get("token", f"f{args.frame_idx}"))
    print(f"[info] token={token} frame_idx={args.frame_idx}")

    runner = BEVSegmentationInference(
        config_path=args.config,
        checkpoint_path=args.ckpt,
        device="cuda:0",
    )

    adapter = NavsimDataAdapter(camera_names=NAVSIM_CAMERAS_NUSCENES_ORDER)
    target_hw = tuple(runner.image_size)
    images_np, calib, _ = adapter.frame_to_numpy_batch(frame, args.sensor_blobs_root, target_hw)
    images = runner.preprocess_images(images_np, target_size=target_hw)

    def to_t(k): return torch.from_numpy(calib[k].astype(np.float32))

    # R_align: 90° CCW about Z applied in NAVSIM lidar frame
    # After applying to all points: new_x = -y_navsim = right, new_y = x_navsim = forward
    # -> now matches nuScenes lidar convention, which is what the pretrained model expects.
    R_cw90 = np.eye(4, dtype=np.float32)
    R_cw90[:3, :3] = np.array([
        [ 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [ 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    R_ccw90 = np.eye(4, dtype=np.float32)
    R_ccw90[:3, :3] = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=np.float32)

    experiments = [
        ("no_align",      np.eye(4, dtype=np.float32)),
        ("R_ccw90",       R_ccw90),
        ("R_cw90",        R_cw90),
    ]

    # Use BEVSegmentationInference.infer (probabilities, then threshold)
    for name, mat in experiments:
        lidar_aug = torch.from_numpy(mat[None, ...])
        # NOTE: BEVSegmentationHead already sigmoids in eval mode, so ``infer``'s
        # raw output is already in [0, 1]; we threshold directly (return_logits=False
        # would double-sigmoid due to a quirk in BEVSegmentationInference.infer).
        runner.map_score_threshold = args.threshold
        masks = runner.infer(
            images,
            to_t("camera_intrinsics"),
            to_t("camera2ego"),
            to_t("lidar2ego"),
            to_t("lidar2camera"),
            to_t("lidar2image"),
            to_t("camera2lidar"),
            img_aug_matrix=to_t("img_aug_matrix"),
            lidar_aug_matrix=lidar_aug,
            return_logits=False,
        )
        pred = masks[0].astype(np.uint8)
        img = colorize_mask(pred, DEFAULT_CLASSES)
        Image.fromarray(img).save(os.path.join(args.out_dir, f"pred_{name}.png"))
        means = np.round(pred.mean(axis=(1, 2)), 3)
        print(f"[{name}] per-class mean:", dict(zip(DEFAULT_CLASSES, means.tolist())))

    if args.gt_npy and args.gt_index:
        with open(args.gt_index, "r") as f:
            idx = json.load(f)
        token_to_row = {str(e["token"]): int(e["row"]) for e in idx["entries"]}
        if token in token_to_row:
            shape = tuple(idx["shape"])
            dtype = np.dtype(idx["dtype"])
            mm = np.memmap(args.gt_npy, dtype=dtype, mode="r", shape=shape)
            gt = np.asarray(mm[token_to_row[token]]).astype(np.uint8)
            gt_img = colorize_mask(gt, DEFAULT_CLASSES)
            Image.fromarray(gt_img).save(os.path.join(args.out_dir, "gt.png"))
            print("[gt] per-class mean:",
                  dict(zip(DEFAULT_CLASSES, np.round(gt.mean(axis=(1, 2)), 3).tolist())))
        else:
            print(f"[warn] token {token} not in gt index")


if __name__ == "__main__":
    main()
