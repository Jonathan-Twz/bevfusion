from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import get_points_type

from tools.data_converter.navsim_bev_seg_gt import (
    DEFAULT_CLASS_TO_LAYERS,
    NavsimMapRasterizer,
)


@PIPELINES.register_module()
class LoadEmptyPoints:
    """Create an empty LiDAR points tensor for camera-only training."""

    def __init__(self, coord_type: str = "LIDAR", load_dim: int = 5, use_dim: Sequence[int] = (0, 1, 2, 3, 4)):
        self.coord_type = coord_type
        self.load_dim = int(load_dim)
        self.use_dim = list(use_dim)

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        points_class = get_points_type(self.coord_type)
        points = np.zeros((0, self.load_dim), dtype=np.float32)
        results["points"] = points_class(points, points_dim=self.load_dim, attribute_dims=None)
        return results


@PIPELINES.register_module()
class LoadBEVSegmentationNavsim:
    """Generate `gt_masks_bev` from nuPlan map layers and ego pose."""

    def __init__(
        self,
        maps_root: str,
        map_version: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Sequence[str],
        class_to_layers: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        self.rasterizer = NavsimMapRasterizer(
            maps_root=maps_root,
            map_version=map_version,
            classes=classes,
            class_to_layers=class_to_layers or DEFAULT_CLASS_TO_LAYERS,
            xbound=xbound,
            ybound=ybound,
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        ego2global = np.asarray(data["ego2global"], dtype=np.float32).reshape(4, 4)
        location = str(data.get("location", "us-nv-las-vegas-strip"))
        labels = self.rasterizer.rasterize(
            map_location=location,
            ego2global=ego2global,
        )
        data["gt_masks_bev"] = labels.astype(np.int64)
        return data


@PIPELINES.register_module()
class LoadBEVSegmentationNavsimCached:
    """Load pre-rasterized BEV map GT from a single uint8 memmap file.

    Designed as a drop-in replacement for :class:`LoadBEVSegmentationNavsim`
    when the on-the-fly nuPlan rasterization becomes a memory/throughput
    bottleneck. Run ``tools/precompute_navsim_bev_gt.py`` once to produce
    ``masks.npy`` (raw uint8 memmap, shape ``(N, C, H, W)``) and a
    companion ``index.json`` mapping ``token -> row``. Then swap this
    pipeline in and set ``workers_per_gpu`` back to a healthy value.

    Each dataloader worker opens the memmap lazily (so we survive ``fork``)
    and all workers share the same Linux page cache, so the incremental
    RSS per worker is effectively zero beyond the first few touched rows.
    """

    def __init__(
        self,
        masks_npy: str,
        index_json: str,
        key: str = "token",
    ) -> None:
        self.masks_npy = os.path.abspath(masks_npy)
        self.index_json = os.path.abspath(index_json)
        self.key = key

        with open(self.index_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.shape = tuple(meta["shape"])  # (N, C, H, W)
        self.dtype = np.dtype(meta["dtype"])
        self.classes = tuple(meta["classes"])
        # Build token -> row lookup once at __init__ (~33k entries, ~3 MB).
        self._token_to_row: Dict[str, int] = {
            str(e["token"]): int(e["row"]) for e in meta["entries"]
        }

        # Do NOT open the memmap here: the pipeline is typically built in
        # the main process before workers fork; numpy memmap uses a mmap'd
        # file handle that survives fork but re-initialising per worker is
        # more robust (and trivially cheap: just a syscall).
        self._mm: Optional[np.memmap] = None
        self._mm_pid: Optional[int] = None
        self._lock = threading.Lock()

    def _get_mm(self) -> np.memmap:
        pid = os.getpid()
        if self._mm is None or self._mm_pid != pid:
            with self._lock:
                if self._mm is None or self._mm_pid != pid:
                    self._mm = np.memmap(
                        self.masks_npy,
                        dtype=self.dtype,
                        mode="r",
                        shape=self.shape,
                    )
                    self._mm_pid = pid
        return self._mm

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        key_val = str(data.get(self.key, ""))
        if not key_val:
            raise KeyError(
                f"LoadBEVSegmentationNavsimCached: data has no '{self.key}' key"
            )
        row = self._token_to_row.get(key_val)
        if row is None:
            raise KeyError(
                f"LoadBEVSegmentationNavsimCached: token '{key_val}' not in "
                f"index {self.index_json}. Did you regenerate the cache after "
                f"changing the manifest?"
            )
        # Copy out of the read-only memmap into an owned int64 array so the
        # downstream pipeline (augmentations, collate, cuda transfer) can
        # freely mutate without touching the shared mmap page.
        mask = np.asarray(self._get_mm()[row]).astype(np.int64, copy=True)
        data["gt_masks_bev"] = mask
        return data
