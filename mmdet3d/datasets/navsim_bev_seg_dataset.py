from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from mmdet.datasets import DATASETS

from .custom_3d import Custom3DDataset
from tools.navsim_bev_adapter import (
    build_camera2lidar_from_navsim_cam,
    intrinsics_3x3_to_4x4,
    scale_camera_intrinsics,
)
from tools.navsim_frame_utils import NAVSIM_CAMERAS_NUSCENES_ORDER


@DATASETS.register_module()
class NavsimBEVSegDataset(Custom3DDataset):
    """NAVSIM dataset for camera-only BEV map segmentation fine-tuning."""

    CLASSES: Tuple[str, ...] = ()

    def __init__(
        self,
        ann_file: str,
        dataset_root: str,
        pipeline: List[Dict[str, Any]],
        object_classes: Optional[Sequence[str]] = None,
        map_classes: Optional[Sequence[str]] = None,
        modality: Optional[Dict[str, Any]] = None,
        camera_names: Sequence[str] = NAVSIM_CAMERAS_NUSCENES_ORDER,
        image_size: Sequence[int] = (256, 704),  # H, W
        test_mode: bool = False,
        filter_empty_gt: bool = False,
        lidar_align_to_nuscenes: bool = False,
        **kwargs: Any,
    ) -> None:
        self.camera_names = tuple(camera_names)
        self.target_hw = (int(image_size[0]), int(image_size[1]))
        self.map_classes = list(map_classes or [])
        # When True, set ``lidar_aug_matrix`` to a swap-XY reflection (det=-1)
        # that bridges the NAVSIM-native BEV display convention and the
        # nuScenes-native one, so that the LSS-pooled feature grid lands in
        # the same row/col orientation as the NAVSIM GT and as the
        # nuScenes-pretrained decoder expects.
        #
        # Derivation (xbound=ybound=[-51.2, 51.2]):
        #   * NavsimMapRasterizer does NOT transpose its mask, so the cached
        #     GT layout is:
        #         row 0 (TOP)  = vehicle RIGHT  (y_nav = -51.2)
        #         col 0 (LEFT) = BEHIND         (x_nav = -51.2)
        #   * Under identity lidar_aug, LSSTransform pools NAVSIM points as
        #         row 0 (TOP)  = x_nav min  = BEHIND
        #         col 0 (LEFT) = y_nav min  = vehicle RIGHT
        #     i.e. the model output is 90° rotated relative to the GT.
        #   * Applying a swap-XY lidar_aug rewrites each point as
        #     (x, y, z) -> (y, x, z) before pooling, which yields
        #         row 0 (TOP)  = y_nav min = vehicle RIGHT
        #         col 0 (LEFT) = x_nav min = BEHIND
        #     → pred and GT become pixel-for-pixel aligned, matching the
        #     zero-shot fix verified in ``tools/navsim_bev_adapter.py``.
        #
        # Prior to this fix, ``R_ccw90`` was used here (det=+1 rotation),
        # which left a residual left/right mirror; see the ``R_swap_xy``
        # commentary in ``tools/batch_navsim_pretrained_align_viz.py``.
        #
        # Set to True when fine-tuning from the nuScenes-pretrained
        # ``camera-only-seg.pth`` so the optimiser does not have to relearn
        # an internal frame rotation. Legacy stage1/2/3 checkpoints were
        # trained with this flag False and must keep using False at eval
        # time for visualizations to look correct.
        self.lidar_align_to_nuscenes = bool(lidar_align_to_nuscenes)
        # Minimal per-frame calibration keyed by (pkl_rel, frame_idx). Populated
        # once in the main process before DataLoader forks workers, so every
        # worker inherits it via Linux copy-on-write instead of maintaining its
        # own growing ``_pkl_cache`` (which previously crashed the box at 18
        # workers because each pickle.load of a 10–25 MB pkl expanded 3–5x in
        # RAM).
        self._frames: Dict[Tuple[str, int], Dict[str, Any]] = {}

        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=object_classes or (),
            modality=modality,
            box_type_3d="LiDAR",
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

        self._preload_frames()

    def load_annotations(self, ann_file: str) -> List[Dict[str, Any]]:
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Manifest must be list, got {type(data)}")
        return data

    @staticmethod
    def _strip_frame(frame: Dict[str, Any], camera_names: Sequence[str]) -> Dict[str, Any]:
        """Keep only the fields the pipeline actually reads.

        Dropping LiDAR paths, ego velocity, extra metadata etc. keeps the
        preloaded dict small (~2 KB / frame, so 33k frames ≈ 70 MB).
        """
        cams = {}
        for name in camera_names:
            if name not in frame.get("cams", {}):
                continue
            src = frame["cams"][name]
            cams[name] = {
                "cam_intrinsic": np.asarray(src["cam_intrinsic"], dtype=np.float32),
                "sensor2lidar_rotation": np.asarray(
                    src["sensor2lidar_rotation"], dtype=np.float32
                ),
                "sensor2lidar_translation": np.asarray(
                    src["sensor2lidar_translation"], dtype=np.float32
                ),
                "data_path": str(src["data_path"]),
            }
        return {
            "cams": cams,
            "lidar2ego": np.asarray(frame["lidar2ego"], dtype=np.float32),
            "ego2global": np.asarray(frame["ego2global"], dtype=np.float32),
            "timestamp": int(frame.get("timestamp", 0)),
            "map_location": str(frame.get("map_location", "us-nv-las-vegas-strip")),
            "token": str(frame.get("token", "")),
        }

    def _preload_frames(self) -> None:
        """Load each needed frame's minimal calibration into ``_frames``.

        Groups manifest entries by pkl so each pkl is loaded exactly once,
        then freed. Peak extra RSS during init ≈ size of a single pkl.
        """
        from collections import defaultdict

        needed: Dict[str, List[int]] = defaultdict(list)
        for e in self.data_infos:
            needed[str(e["pkl"])].append(int(e["frame_idx"]))

        for pkl_rel, frame_idxs in needed.items():
            pkl_abs = os.path.join(self.dataset_root, pkl_rel)
            with open(pkl_abs, "rb") as f:
                scene = pickle.load(f)
            if not isinstance(scene, list):
                raise ValueError(f"Expected list scene in {pkl_abs}")
            for fi in set(frame_idxs):
                if fi < 0 or fi >= len(scene):
                    raise IndexError(
                        f"frame_idx {fi} out of range for {pkl_rel} (len={len(scene)})"
                    )
                self._frames[(pkl_rel, fi)] = self._strip_frame(scene[fi], self.camera_names)
            # Let the full ``scene`` list go before loading the next pkl so peak
            # RSS stays bounded.
            del scene

    def _resolve_frame(self, index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        e = self.data_infos[index]
        key = (str(e["pkl"]), int(e["frame_idx"]))
        frame = self._frames.get(key)
        if frame is None:
            raise KeyError(f"frame not preloaded: {key}. Manifest changed after __init__?")
        return e, frame

    def _build_calibration(self, frame: Dict[str, Any], orig_hw: Tuple[int, int]) -> Dict[str, Any]:
        cams = frame["cams"]
        lidar2ego = np.asarray(frame["lidar2ego"], dtype=np.float32).reshape(4, 4)

        camera_intrinsics = []
        camera2ego = []
        camera2lidar = []
        lidar2camera = []
        lidar2image = []
        for name in self.camera_names:
            cam = cams[name]
            k0 = np.asarray(cam["cam_intrinsic"], dtype=np.float32).reshape(3, 3)
            k = scale_camera_intrinsics(k0, orig_hw, self.target_hw)
            k4 = intrinsics_3x3_to_4x4(k)

            c2l = build_camera2lidar_from_navsim_cam(cam)
            l2c = np.linalg.inv(c2l)
            c2e = lidar2ego @ c2l
            l2i = k4 @ l2c

            camera_intrinsics.append(k4.astype(np.float32))
            camera2ego.append(c2e.astype(np.float32))
            camera2lidar.append(c2l.astype(np.float32))
            lidar2camera.append(l2c.astype(np.float32))
            lidar2image.append(l2i.astype(np.float32))

        n = len(self.camera_names)
        img_aug = [np.eye(4, dtype=np.float32) for _ in range(n)]
        lidar_aug = np.eye(4, dtype=np.float32)
        if getattr(self, "_lidar_align_logged", False) is False:
            print(f"[NavsimBEVSegDataset] lidar_align_to_nuscenes={self.lidar_align_to_nuscenes}")
            self._lidar_align_logged = True
        if self.lidar_align_to_nuscenes:
            # Swap-XY reflection (det=-1). See ``__init__`` docstring for the
            # full derivation. Kept in-sync with ``tools/navsim_bev_adapter``.
            lidar_aug[:3, :3] = np.array(
                [[0.0, 1.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )

        return {
            "camera_intrinsics": camera_intrinsics,
            "camera2ego": camera2ego,
            "camera2lidar": camera2lidar,
            "lidar2camera": lidar2camera,
            "lidar2image": lidar2image,
            "img_aug_matrix": img_aug,
            "lidar_aug_matrix": lidar_aug,
            "lidar2ego": lidar2ego.astype(np.float32),
        }

    def get_data_info(self, index: int) -> Dict[str, Any]:
        e, frame = self._resolve_frame(index)
        sensor_root = os.path.join(self.dataset_root, "sensor_blobs", e["split"])
        image_paths = [os.path.join(sensor_root, frame["cams"][k]["data_path"]) for k in self.camera_names]
        for p in image_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Missing image: {p}")

        with Image.open(image_paths[0]) as img0:
            orig_hw = (img0.size[1], img0.size[0])  # H, W

        calib = self._build_calibration(frame, orig_hw)
        ego2global = np.asarray(frame["ego2global"], dtype=np.float32).reshape(4, 4)

        token = str(e.get("token", frame.get("token", f"f{e['frame_idx']}")))
        return {
            "token": token,
            "sample_idx": token,
            "image_paths": image_paths,
            "timestamp": int(frame.get("timestamp", 0)),
            "location": str(frame.get("map_location", "us-nv-las-vegas-strip")),
            "ego2global": ego2global,
            "lidar2ego": calib["lidar2ego"],
            "camera_intrinsics": calib["camera_intrinsics"],
            "camera2ego": calib["camera2ego"],
            "camera2lidar": calib["camera2lidar"],
            "lidar2camera": calib["lidar2camera"],
            "lidar2image": calib["lidar2image"],
            "img_aug_matrix": calib["img_aug_matrix"],
            "lidar_aug_matrix": calib["lidar_aug_matrix"],
        }

    def evaluate_map(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        thresholds = torch.tensor([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)
        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)

        for result in results:
            pred = result["masks_bev"].detach().reshape(num_classes, -1)
            label = result["gt_masks_bev"].detach().bool().reshape(num_classes, -1)
            pred = pred[:, :, None] >= thresholds
            label = label[:, :, None]
            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

        ious = tp / (tp + fp + fn + 1e-7)
        metrics: Dict[str, float] = {}
        for i, name in enumerate(self.map_classes):
            metrics[f"map/{name}/iou@max"] = ious[i].max().item()
            for t, iou in zip(thresholds, ious[i]):
                metrics[f"map/{name}/iou@{t.item():.2f}"] = iou.item()
        metrics["map/mean/iou@max"] = ious.max(dim=1).values.mean().item()
        return metrics

    def evaluate(self, results: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, float]:
        if not results:
            return {}
        if "masks_bev" in results[0]:
            return self.evaluate_map(results)
        return {}
