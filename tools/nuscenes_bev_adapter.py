"""
nuScenes info.pkl -> BEVFusion camera-only calibration and BEV feature extraction.

Uses the same 6-camera order as training:
  CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
"""

from __future__ import annotations

import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.navsim_bev_adapter import (
    BEVFeatureExtractor,
    BEVSegmentationInference,
    build_camera2lidar_from_navsim_cam,
    intrinsics_3x3_to_4x4,
    scale_camera_intrinsics,
)

# nuScenes devkit / info.pkl keys (do not use NAVSIM CAM_F0-style names here)
NUSCENES_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

# Figure layout 2x3: front row FL–F–FR; back row swaps corners vs vehicle frame: BR–B–BL
NUSCENES_CAMERAS_VISUAL_ORDER = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
)


def resolve_nuscenes_data_path(dataset_root: str, data_path: str) -> str:
    """
    Join nuScenes ``data_path`` from info pkl with on-disk dataset root.

    Converter often stores paths like ``data/nuscenes/samples/CAM_FRONT/xxx.jpg``;
    ``dataset_root`` should be the folder containing ``samples/`` and ``v1.0-mini/``.
    """
    data_path = data_path.replace("\\", "/")
    for pref in ("data/nuscenes/", "./data/nuscenes/"):
        if data_path.startswith(pref):
            data_path = data_path[len(pref) :]
            break
    return os.path.normpath(os.path.join(os.path.abspath(dataset_root), data_path))


def build_lidar2ego_4x4(info: Dict[str, Any]) -> np.ndarray:
    """LiDAR -> ego 4x4 from ``lidar2ego_rotation`` (quaternion wxyz) + translation."""
    l2e_r = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    l2e_t = np.asarray(info["lidar2ego_translation"], dtype=np.float64).reshape(3)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = l2e_r.astype(np.float32)
    m[:3, 3] = l2e_t.astype(np.float32)
    return m


def check_nuscenes_images_exist(
    info: Dict[str, Any],
    dataset_root: str,
    camera_names: Sequence[str] = NUSCENES_CAMERAS,
) -> bool:
    for name in camera_names:
        if name not in info.get("cams", {}):
            return False
        p = resolve_nuscenes_data_path(dataset_root, info["cams"][name]["data_path"])
        if not os.path.isfile(p):
            return False
    return True


def nuscenes_bev_collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Same stacking rules as :func:`tools.navsim_bev_adapter.navsim_bev_collate_fn`."""
    from tools.navsim_bev_adapter import navsim_bev_collate_fn

    return navsim_bev_collate_fn(samples)


def preprocess_images_batch(
    runner: "BEVSegmentationInference",
    images_uint8: np.ndarray,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    from tools.navsim_bev_adapter import preprocess_images_batch as _p

    return _p(runner, images_uint8, target_hw)


class NuscenesDataAdapter:
    """
    Load nuScenes ``info`` dicts and produce batched numpy calibration + uint8 images.
    """

    def __init__(
        self,
        camera_names: Sequence[str] = NUSCENES_CAMERAS,
        target_hw: Optional[Tuple[int, int]] = None,
    ):
        self.camera_names = tuple(camera_names)
        self.target_hw = target_hw

    def _load_bgr_uint8(self, path: str) -> np.ndarray:
        pil = Image.open(path).convert("RGB")
        rgb = np.asarray(pil)
        return rgb[..., ::-1].copy()

    def load_multiview_bgr(
        self,
        info: Dict[str, Any],
        dataset_root: str,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Returns (N, H, W, 3) uint8 BGR resized to target_hw, and orig (H, W)."""
        cams = info["cams"]
        paths = []
        for name in self.camera_names:
            if name not in cams:
                raise KeyError(f"Camera {name} not in info cams: {list(cams.keys())}")
            paths.append(cams[name]["data_path"])

        first = resolve_nuscenes_data_path(dataset_root, paths[0])
        img0 = self._load_bgr_uint8(first)
        orig_hw = (img0.shape[0], img0.shape[1])

        target_hw = self.target_hw
        if target_hw is None:
            raise ValueError("target_hw must be set before loading images")

        out: List[np.ndarray] = []
        for p in paths:
            path = resolve_nuscenes_data_path(dataset_root, p)
            img = self._load_bgr_uint8(path)
            if (img.shape[0], img.shape[1]) != orig_hw:
                raise ValueError(
                    f"Inconsistent image size {img.shape} vs first {orig_hw} for {path}"
                )
            if (img.shape[0], img.shape[1]) != tuple(target_hw):
                pil = Image.fromarray(img[..., ::-1])
                pil = pil.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
                img = np.asarray(pil)[..., ::-1].copy()
            out.append(img)
        return np.stack(out, axis=0), orig_hw

    def build_calibration_np(
        self,
        info: Dict[str, Any],
        orig_hw: Tuple[int, int],
        target_hw: Tuple[int, int],
    ) -> Dict[str, np.ndarray]:
        """Batch size 1 calibration dict (same keys as NAVSIM adapter)."""
        cams = info["cams"]
        lidar2ego = build_lidar2ego_4x4(info)

        cam_intr = []
        cam2ego = []
        cam2lidar = []
        lidar2cam = []
        lidar2img = []

        for name in self.camera_names:
            cam = cams[name]
            k_orig = np.asarray(cam["cam_intrinsic"], dtype=np.float32).reshape(3, 3)
            k_tgt = scale_camera_intrinsics(k_orig, orig_hw, target_hw)
            k4 = intrinsics_3x3_to_4x4(k_tgt)

            c2l = build_camera2lidar_from_navsim_cam(cam)
            l2c = np.linalg.inv(c2l)
            c2e = lidar2ego @ c2l
            l2i = k4 @ l2c

            cam_intr.append(k4)
            cam2ego.append(c2e)
            cam2lidar.append(c2l)
            lidar2cam.append(l2c)
            lidar2img.append(l2i)

        n = len(self.camera_names)
        eye4 = np.eye(4, dtype=np.float32)
        img_aug = np.stack([eye4.copy() for _ in range(n)], axis=0)[None, ...]
        lidar_aug = np.eye(4, dtype=np.float32)[None, :, :]

        return {
            "camera_intrinsics": np.stack(cam_intr, axis=0)[None, ...],
            "camera2ego": np.stack(cam2ego, axis=0)[None, ...],
            "lidar2ego": lidar2ego[None, ...],
            "lidar2camera": np.stack(lidar2cam, axis=0)[None, ...],
            "lidar2image": np.stack(lidar2img, axis=0)[None, ...],
            "camera2lidar": np.stack(cam2lidar, axis=0)[None, ...],
            "img_aug_matrix": img_aug,
            "lidar_aug_matrix": lidar_aug,
        }

    def info_to_numpy_batch(
        self,
        info: Dict[str, Any],
        dataset_root: str,
        target_hw: Tuple[int, int],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[int, int]]:
        self.target_hw = target_hw
        views, orig_hw = self.load_multiview_bgr(info, dataset_root)
        images = views[None, ...]
        calib = self.build_calibration_np(info, orig_hw, target_hw)
        return images, calib, orig_hw


def load_nuscenes_infos_pkl(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "infos" not in data:
        raise ValueError(f"Expected dict with 'infos' in {path}")
    return data["infos"], data.get("metadata", {})


class NuscenesBEVDataset(Dataset):
    """One sample = one nuScenes keyframe from ``infos`` list."""

    def __init__(
        self,
        infos: List[Dict[str, Any]],
        dataset_root: str,
        target_hw: Tuple[int, int],
        camera_names: Sequence[str] = NUSCENES_CAMERAS,
    ):
        self.infos = infos
        self.dataset_root = os.path.abspath(dataset_root)
        self.target_hw = tuple(target_hw)
        self.adapter = NuscenesDataAdapter(
            camera_names=camera_names, target_hw=self.target_hw
        )

    def __len__(self) -> int:
        return len(self.infos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.infos[idx]
        images_np, calib, _ = self.adapter.info_to_numpy_batch(
            info, self.dataset_root, self.target_hw
        )
        return {
            "images_uint8": images_np,
            "calib": calib,
            "meta": {"token": info["token"]},
        }


__all__ = [
    "NUSCENES_CAMERAS",
    "NUSCENES_CAMERAS_VISUAL_ORDER",
    "NuscenesDataAdapter",
    "NuscenesBEVDataset",
    "BEVFeatureExtractor",
    "build_lidar2ego_4x4",
    "resolve_nuscenes_data_path",
    "check_nuscenes_images_exist",
    "nuscenes_bev_collate_fn",
    "preprocess_images_batch",
    "load_nuscenes_infos_pkl",
]
