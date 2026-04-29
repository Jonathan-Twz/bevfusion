"""Tiny smoke test for the aligned NAVSIM stage3 config.

Builds the train dataset from ``stage3_full_aug_aligned.yaml``, grabs one
sample, and prints the BEV calibration (especially ``lidar_aug_matrix``)
plus the GT shape so we can verify the new R_swap_xy wiring before spending
~10 h on a multi-GPU retrain.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

CONFIG_DEFAULT = "configs/navsim/seg/stage3_full_aug_aligned.yaml"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=CONFIG_DEFAULT)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    args = parser.parse_args()

    # torchpack + mmcv config load (mirrors tools/train.py).
    from torchpack.utils.config import configs
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.utils import recursive_eval
    import mmdet3d  # noqa: F401  # registers PIPELINES

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    ds_cfg = cfg.data[args.split]
    print(f"[cfg] data.{args.split}.type     = {ds_cfg.type}")
    print(f"[cfg] data.{args.split}.ann_file = {ds_cfg.ann_file}")
    print(f"[cfg] data.{args.split}.lidar_align_to_nuscenes = "
          f"{ds_cfg.get('lidar_align_to_nuscenes', '<unset>')}")

    ds = build_dataset(ds_cfg)
    print(f"[ds ] len = {len(ds)}")

    sample = ds[0]

    def _to_np(x):
        if hasattr(x, "numpy"):
            return x.numpy()
        if hasattr(x, "data") and hasattr(x.data, "numpy"):
            return x.data.numpy()
        return np.asarray(x)

    lidar_aug = _to_np(sample["lidar_aug_matrix"])
    img_aug = _to_np(sample["img_aug_matrix"])
    gt = _to_np(sample["gt_masks_bev"])

    print("[sample 0] lidar_aug_matrix =\n", lidar_aug)
    print("[sample 0] det(lidar_aug[:3, :3]) =",
          float(np.linalg.det(lidar_aug[:3, :3])))
    print("[sample 0] img_aug_matrix[0] =\n", img_aug[0])
    print("[sample 0] gt_masks_bev shape =", gt.shape, "dtype=", gt.dtype,
          "sum=", int(gt.sum()))

    expected = np.array(
        [[0.0, 1.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    if not np.allclose(lidar_aug, expected, atol=1e-6):
        print("[FAIL] lidar_aug_matrix is NOT the expected swap-XY reflection.")
        return 1
    print("[OK ] lidar_aug_matrix matches the swap-XY reflection (det=-1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
