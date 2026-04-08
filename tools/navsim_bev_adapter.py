"""
NAVSIM / OpenScene (nuPlan) -> BEVFusion camera-only calibration and BEV feature extraction.

Expects the pickle layout produced by common NAVSIM exports:
  - List[dict] per scene, with ``cams``, ``lidar2ego``, etc.
  - Images under ``sensor_blobs_root`` joined with each camera's ``data_path``.

Six cameras are selected to align with nuScenes training order:
  CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
  -> CAM_F0, CAM_R0, CAM_L0, CAM_B0, CAM_L2, CAM_R2
"""

from __future__ import annotations

import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Repo root for imports when running as script
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.bev_seg_inference import BEVSegmentationInference
from tools.navsim_frame_utils import (
    NAVSIM_CAMERAS_NUSCENES_ORDER,
    check_frame_images_exist as _check_frame_images_exist_base,
)


def scale_camera_intrinsics(
    k_3x3: np.ndarray,
    orig_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> np.ndarray:
    """Scale pinhole intrinsics when resizing image from orig (H,W) to target (H,W)."""
    h0, w0 = orig_hw
    h1, w1 = target_hw
    sx = w1 / float(w0)
    sy = h1 / float(h0)
    k = k_3x3.astype(np.float64).copy()
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] *= sx
    k[1, 2] *= sy
    return k.astype(np.float32)


def _as_homogeneous_4x4(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    t = np.asarray(trans, dtype=np.float64).reshape(3)
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = r.astype(np.float32)
    m[:3, 3] = t.astype(np.float32)
    return m


def build_camera2lidar_from_navsim_cam(cam: Dict[str, Any]) -> np.ndarray:
    """camera2lidar (4x4) from NAVSIM fields ``sensor2lidar_rotation`` / ``sensor2lidar_translation``."""
    return _as_homogeneous_4x4(
        cam["sensor2lidar_rotation"],
        cam["sensor2lidar_translation"],
    )


def intrinsics_3x3_to_4x4(k: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = k.astype(np.float32)
    return out


def check_frame_images_exist(
    frame: Dict[str, Any],
    sensor_blobs_root: str,
    camera_names: Sequence[str] = NAVSIM_CAMERAS_NUSCENES_ORDER,
) -> bool:
    """Re-export of :func:`tools.navsim_frame_utils.check_frame_images_exist`."""
    return _check_frame_images_exist_base(frame, sensor_blobs_root, camera_names)


def navsim_bev_collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate batch from :meth:`NavsimBEVDataset.__getitem__`.

    Stacks calibration numpy arrays to batch dimension; concatenates metadata list.
    """
    if len(samples) == 0:
        raise ValueError("empty batch")

    calib_keys = [
        "camera_intrinsics",
        "camera2ego",
        "lidar2ego",
        "lidar2camera",
        "lidar2image",
        "camera2lidar",
        "img_aug_matrix",
        "lidar_aug_matrix",
    ]
    batch_calib: Dict[str, np.ndarray] = {}
    for k in calib_keys:
        batch_calib[k] = np.concatenate([s["calib"][k] for s in samples], axis=0)

    return {
        "images_uint8": np.concatenate([s["images_uint8"] for s in samples], axis=0),
        "calib": batch_calib,
        "meta": [s["meta"] for s in samples],
    }


def preprocess_images_batch(
    runner: "BEVSegmentationInference",
    images_uint8: np.ndarray,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Preprocess a stacked batch ``(B, N, H, W, C)`` uint8 BGR on the runner's device.

    Same logic as :meth:`BEVSegmentationInference.preprocess_images` but for numpy batch.
    """
    if images_uint8.dtype != np.uint8:
        raise TypeError("images_uint8 must be uint8")
    # BGR -> RGB, (B,N,H,W,C) -> (B,N,C,H,W)
    x = images_uint8[..., ::-1].copy()
    x = x.transpose(0, 1, 4, 2, 3)
    x = x.astype(np.float32) / 255.0
    images = torch.from_numpy(x).to(runner.device)
    B, N, C, H, W = images.shape
    th, tw = target_hw
    if (H, W) != (th, tw):
        images = images.reshape(B * N, C, H, W)
        images = F.interpolate(
            images, size=target_hw, mode="bilinear", align_corners=False
        )
        images = images.view(B, N, C, th, tw)
    mean = runner.mean.expand(B, N, -1, -1, -1)
    std = runner.std.expand(B, N, -1, -1, -1)
    images = (images - mean) / std
    return images


class NavsimDataAdapter:
    """
    Load NAVSIM frames and produce batched numpy calibration + uint8 images for BEVFusion.

    Images are resized to ``target_hw`` (default model input ``[256, 704]`` as H, W).
    Intrinsics are scaled to match; ``img_aug_matrix`` is identity (augment baked into K).
    """

    def __init__(
        self,
        camera_names: Sequence[str] = NAVSIM_CAMERAS_NUSCENES_ORDER,
        target_hw: Optional[Tuple[int, int]] = None,
    ):
        self.camera_names = tuple(camera_names)
        self.target_hw = target_hw  # (H, W); if None, set from BEVSegmentationInference later

    @staticmethod
    def load_scene_pkl(pkl_path: str) -> List[Dict[str, Any]]:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list in pkl, got {type(data)}")
        return data

    def _resolve_image_path(self, sensor_blobs_root: str, data_path: str) -> str:
        return os.path.join(sensor_blobs_root, data_path)

    def load_multiview_bgr(
        self,
        frame: Dict[str, Any],
        sensor_blobs_root: str,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Returns:
            images: (N, H, W, 3) uint8 BGR, all resized to same size
            orig_hw: (H0, W0) of the first loaded image (before resize)
        """
        cams = frame["cams"]
        paths = []
        for name in self.camera_names:
            if name not in cams:
                raise KeyError(f"Camera {name} not in frame cams: {list(cams.keys())}")
            paths.append(cams[name]["data_path"])

        first = self._resolve_image_path(sensor_blobs_root, paths[0])
        img0 = self._load_bgr_uint8(first)
        orig_hw = (img0.shape[0], img0.shape[1])

        target_hw = self.target_hw
        if target_hw is None:
            raise ValueError("target_hw must be set (e.g. [256, 704]) before loading images")

        out: List[np.ndarray] = []
        for p in paths:
            path = self._resolve_image_path(sensor_blobs_root, p)
            img = self._load_bgr_uint8(path)
            if (img.shape[0], img.shape[1]) != orig_hw:
                raise ValueError(
                    f"Inconsistent image size {img.shape} vs first {orig_hw} for {path}"
                )
            if (img.shape[0], img.shape[1]) != tuple(target_hw):
                pil = Image.fromarray(img[..., ::-1])  # BGR -> RGB for PIL
                pil = pil.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
                img = np.asarray(pil)[..., ::-1].copy()  # RGB -> BGR
            out.append(img)
        return np.stack(out, axis=0), orig_hw

    @staticmethod
    def _load_bgr_uint8(path: str) -> np.ndarray:
        """Load image as uint8 BGR (H, W, 3)."""
        pil = Image.open(path).convert("RGB")
        rgb = np.asarray(pil)
        return rgb[..., ::-1].copy()

    def build_calibration_np(
        self,
        frame: Dict[str, Any],
        orig_hw: Tuple[int, int],
        target_hw: Tuple[int, int],
    ) -> Dict[str, np.ndarray]:
        """Batch size 1: each array has leading dimension 1 where applicable."""
        cams = frame["cams"]
        lidar2ego = np.asarray(frame["lidar2ego"], dtype=np.float32).reshape(4, 4)

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

    def frame_to_numpy_batch(
        self,
        frame: Dict[str, Any],
        sensor_blobs_root: str,
        target_hw: Tuple[int, int],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[int, int]]:
        """
        Returns:
            images: (1, N, H, W, 3) uint8 BGR
            calib: dict of arrays batch size 1
            orig_hw: original (H, W) before resize
        """
        self.target_hw = target_hw
        views, orig_hw = self.load_multiview_bgr(frame, sensor_blobs_root)
        # views already (N,H,W,3) at target size
        images = views[None, ...]
        calib = self.build_calibration_np(frame, orig_hw, target_hw)
        return images, calib, orig_hw


class BEVFeatureExtractor:
    """
    Loads BEVFusion and runs :meth:`BEVSegmentationInference.extract_bev_features`.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda:0",
    ):
        self.runner = BEVSegmentationInference(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self.device = self.runner.device

    @property
    def image_size(self) -> List[int]:
        return list(self.runner.image_size)

    def extract_bev_features(
        self,
        images: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera2ego: torch.Tensor,
        lidar2ego: torch.Tensor,
        lidar2camera: torch.Tensor,
        lidar2image: torch.Tensor,
        camera2lidar: torch.Tensor,
        img_aug_matrix: Optional[torch.Tensor] = None,
        lidar_aug_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.runner.extract_bev_features(
            images,
            camera_intrinsics,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera2lidar,
            img_aug_matrix=img_aug_matrix,
            lidar_aug_matrix=lidar_aug_matrix,
        )

    def extract_from_navsim_frame(
        self,
        frame: Dict[str, Any],
        sensor_blobs_root: str,
        adapter: Optional[NavsimDataAdapter] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full pipeline: NAVSIM frame dict -> BEV tensors (vtransform + decoder neck).
        """
        adapter = adapter or NavsimDataAdapter()
        target_hw = tuple(self.runner.image_size)
        images_np, calib, _ = adapter.frame_to_numpy_batch(
            frame, sensor_blobs_root, target_hw
        )
        images = self.runner.preprocess_images(images_np, target_size=target_hw)

        def to_t(key: str) -> torch.Tensor:
            return torch.from_numpy(calib[key].astype(np.float32))

        return self.extract_bev_features(
            images,
            to_t("camera_intrinsics"),
            to_t("camera2ego"),
            to_t("lidar2ego"),
            to_t("lidar2camera"),
            to_t("lidar2image"),
            to_t("camera2lidar"),
            img_aug_matrix=to_t("img_aug_matrix"),
            lidar_aug_matrix=to_t("lidar_aug_matrix"),
        )


def save_bev_features(
    feats: Dict[str, torch.Tensor],
    path_prefix: str,
    suffix: str = "",
) -> None:
    """Save each tensor as ``{path_prefix}_{key}{suffix}.pt`` (CPU)."""
    extra = suffix if suffix else ""
    for k, v in feats.items():
        p = f"{path_prefix}_{k}{extra}.pt"
        torch.save(v.detach().cpu(), p)


__all__ = [
    "NAVSIM_CAMERAS_NUSCENES_ORDER",
    "NavsimDataAdapter",
    "BEVFeatureExtractor",
    "BEVSegmentationInference",
    "scale_camera_intrinsics",
    "build_camera2lidar_from_navsim_cam",
    "check_frame_images_exist",
    "navsim_bev_collate_fn",
    "preprocess_images_batch",
    "save_bev_features",
]
