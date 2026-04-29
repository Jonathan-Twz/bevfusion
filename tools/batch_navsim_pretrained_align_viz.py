#!/usr/bin/env python
"""Batch-run the pretrained camera-only-seg model on NAVSIM frames with lidar-frame
alignment (R_ccw90 in lidar_aug_matrix) and stitch GT vs PRED images.

We intentionally bypass ``tools/visualize.py`` / ``ImageAug3D`` because the NAVSIM
dataset provides ``camera_intrinsics`` already scaled to ``target_hw``, while
``ImageAug3D`` (designed for nuScenes) assumes intrinsics at ORIGINAL image size
and overrides ``img_aug_matrix`` with a scale-0.48 crop. That combination
silently double-scales the projection and makes the pretrained (zero-shot) model
produce a diagonally-tilted BEV output. Stage1/2/3 fine-tuning learned to absorb
the mismatch, but the pretrained checkpoint cannot.

This script reuses the same direct-resize + target-scale-intrinsics + identity
img_aug_matrix path that was validated in ``test_navsim_lidar_align.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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


def stitch_gt_pred(gt_img: np.ndarray, pred_img: np.ndarray, title: str) -> Image.Image:
    h, w, _ = gt_img.shape
    gap = 12
    header = 32
    canvas = np.full((header + h, w * 2 + gap, 3), 255, dtype=np.uint8)
    canvas[header : header + h, :w] = gt_img
    canvas[header : header + h, w + gap :] = pred_img
    out = Image.fromarray(canvas)
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((w // 2 - 10, 4), "GT", fill=(0, 0, 0), font=font)
    draw.text((w + gap + w // 2 - 50, 4), "PRED (aligned)", fill=(0, 0, 0), font=font)
    draw.text((6, h + header - 22), title, fill=(80, 80, 80), font=font)
    return out


def load_gt_from_memmap(gt_npy: str, gt_index: str) -> Tuple[Dict[str, int], Tuple, np.dtype, str]:
    with open(gt_index, "r") as f:
        idx = json.load(f)
    token_to_row = {str(e["token"]): int(e["row"]) for e in idx["entries"]}
    return token_to_row, tuple(idx["shape"]), np.dtype(idx["dtype"]), gt_npy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="yaml config used to build the model (e.g. configs/nuscenes/seg/camera-bev256d2.yaml)")
    ap.add_argument("--ckpt", default="pretrained/camera-only-seg.pth")
    ap.add_argument("--manifest", required=True,
                    help="json list of {pkl, frame_idx, token, split}")
    ap.add_argument("--pkl_root", default="/home/wenzhe/wm_ws/WoTE/dataset",
                    help="prefix for ``pkl`` entries in the manifest (e.g. dataset root)")
    ap.add_argument("--sensor_blobs_root_template", default="{dataset_root}/sensor_blobs/{split}",
                    help="template with {dataset_root} and {split}")
    ap.add_argument("--dataset_root", default="/home/wenzhe/wm_ws/WoTE/dataset")
    ap.add_argument("--gt_npy", required=True, help="e.g. logs/navsim_bev_gt_cache/stage3_masks.npy")
    ap.add_argument("--gt_index", required=True, help="e.g. logs/navsim_bev_gt_cache/stage3_index.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--max_frames", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pred_dir = os.path.join(args.out_dir, "pred_map")
    gt_dir = os.path.join(args.out_dir, "gt_map")
    stitch_dir = os.path.join(args.out_dir, "stitched")
    for d in (pred_dir, gt_dir, stitch_dir):
        os.makedirs(d, exist_ok=True)

    with open(args.manifest, "r") as f:
        manifest: List[Dict] = json.load(f)
    if args.max_frames is not None:
        manifest = manifest[: args.max_frames]

    token_to_row, gt_shape, gt_dtype, gt_npy = load_gt_from_memmap(args.gt_npy, args.gt_index)
    gt_mm = np.memmap(gt_npy, dtype=gt_dtype, mode="r", shape=gt_shape)

    runner = BEVSegmentationInference(
        config_path=args.config,
        checkpoint_path=args.ckpt,
        device="cuda:0",
    )
    runner.map_score_threshold = args.threshold

    # The correct ``lidar_aug_matrix`` here is a **reflection that swaps X and Y**
    # (det = -1), NOT a pure 90° rotation. Reason:
    #
    #   * ``mmdet3d/datasets/pipelines/loading.py::LoadBEVSegmentation`` applies
    #     ``masks = masks.transpose(0, 2, 1)`` to the nuScenes GT returned by
    #     ``get_map_mask``, so nuScenes GT has image-TOP = vehicle LEFT.
    #   * ``tools/data_converter/navsim_bev_seg_gt.py::NavsimMapRasterizer`` does
    #     NOT transpose, so NAVSIM GT has image-TOP = vehicle RIGHT.
    #   * The two display conventions are therefore mirrored along the ego-vehicle
    #     left/right axis. No pure rotation (det=+1) of the input point cloud can
    #     reconcile them: we need a reflection.
    #
    # Mapping derived from the image-axis analysis (xbound=ybound=[-51.2, 51.2]):
    #   grid_x = y_nav (image row 0 = TOP ↔ y_nav = -51.2 = vehicle RIGHT) ✓
    #   grid_y = x_nav (image col max = RIGHT ↔ x_nav = +51.2 = AHEAD)     ✓
    #
    # Effect on the nuScenes-pretrained decoder: the network is roughly
    # equivariant to the input reflection, so both the features and the
    # predictions get reflected together — the rendered image ends up visually
    # consistent with NAVSIM GT. Previously tried matrices:
    #   * identity           → ~90° diagonal tilt (NAVSIM X vs nuScenes X mismatch)
    #   * R_ccw90 (det=+1)   → top/bottom flipped relative to NAVSIM GT
    #   * R_cw90  (det=+1)   → left/right flipped relative to NAVSIM GT
    R_swap_xy = np.eye(4, dtype=np.float32)
    R_swap_xy[:3, :3] = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    lidar_aug_t = torch.from_numpy(R_swap_xy[None, ...])

    adapter = NavsimDataAdapter(camera_names=NAVSIM_CAMERAS_NUSCENES_ORDER)
    target_hw = tuple(runner.image_size)
    classes = getattr(runner, "map_classes", DEFAULT_CLASSES) or DEFAULT_CLASSES

    print(f"[batch] frames={len(manifest)} target_hw={target_hw} classes={classes}")

    ok = 0
    for i, entry in enumerate(manifest):
        token = str(entry["token"])
        pkl_path = os.path.join(args.pkl_root, entry["pkl"])
        split = entry.get("split", "trainval")
        sensor_blobs_root = args.sensor_blobs_root_template.format(
            dataset_root=args.dataset_root, split=split
        )

        if not os.path.isfile(pkl_path):
            print(f"[warn] {i}: missing {pkl_path}")
            continue
        with open(pkl_path, "rb") as f:
            scene = pickle.load(f)
        frame = scene[int(entry["frame_idx"])]

        try:
            images_np, calib, _ = adapter.frame_to_numpy_batch(frame, sensor_blobs_root, target_hw)
        except Exception as exc:
            print(f"[warn] {i} token={token}: adapter failed {exc}")
            continue
        images = runner.preprocess_images(images_np, target_size=target_hw)

        def to_t(k: str) -> torch.Tensor:
            return torch.from_numpy(calib[k].astype(np.float32))

        try:
            masks = runner.infer(
                images,
                to_t("camera_intrinsics"),
                to_t("camera2ego"),
                to_t("lidar2ego"),
                to_t("lidar2camera"),
                to_t("lidar2image"),
                to_t("camera2lidar"),
                img_aug_matrix=to_t("img_aug_matrix"),
                lidar_aug_matrix=lidar_aug_t,
                return_logits=False,
            )
        except Exception as exc:
            print(f"[warn] {i} token={token}: infer failed {exc}")
            continue

        pred = masks[0].astype(np.uint8)
        pred_img = colorize_mask(pred, classes)
        Image.fromarray(pred_img).save(os.path.join(pred_dir, f"{token}.png"))

        if token not in token_to_row:
            print(f"[warn] {i} token={token} not in gt index, skip stitch")
            continue
        gt = np.asarray(gt_mm[token_to_row[token]]).astype(np.uint8)
        gt_img = colorize_mask(gt, classes)
        Image.fromarray(gt_img).save(os.path.join(gt_dir, f"{token}.png"))

        stitched = stitch_gt_pred(gt_img, pred_img, title=f"{split} | {token}")
        stitched.save(os.path.join(stitch_dir, f"{token}.png"))
        ok += 1
        print(f"[ok] {i+1}/{len(manifest)} token={token}")

    print(f"[done] wrote {ok} stitched pairs to {stitch_dir}")


if __name__ == "__main__":
    main()
