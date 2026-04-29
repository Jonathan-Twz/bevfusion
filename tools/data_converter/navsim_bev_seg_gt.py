#!/usr/bin/env python3
"""
Generate BEV segmentation labels for NAVSIM/nuPlan map data.

This module provides:
  - NavsimMapRasterizer: map patch -> multi-class BEV mask tensor.
  - A small smoke test utility for single-frame visualization.
  - A coverage utility to estimate class sparsity on sampled frames.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

if False:  # pragma: no cover - imported lazily in NavsimMapRasterizer.__init__
    from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
    from nuplan.database.maps_db.map_api import NuPlanMapWrapper
    from nuplan.database.maps_db.map_explorer import NuPlanMapExplorer


MAP_CLASSES: Tuple[str, ...] = (
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "divider",
)

# Mapping designed to stay close to nuScenes map classes.
DEFAULT_CLASS_TO_LAYERS: Mapping[str, Tuple[str, ...]] = {
    "drivable_area": (
        "lanes_polygons",
        "lane_connectors",
        "intersections",
        "generic_drivable_areas",
        "road_segments",
    ),
    "ped_crossing": ("crosswalks",),
    "walkway": ("walkways",),
    "stop_line": ("stop_polygons",),
    "carpark_area": ("carpark_areas",),
    # nuPlan does not expose road_divider/lane_divider exactly like nuScenes.
    # We use lane boundary lines as a practical proxy.
    "divider": ("boundaries",),
}


@dataclass
class FrameRef:
    pkl: str
    frame_idx: int
    split: str
    scene: str
    token: str


def _yaw_deg_from_4x4(transform: np.ndarray) -> float:
    rot = np.asarray(transform, dtype=np.float64)[:3, :3]
    v = rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _load_manifest(path: str) -> List[FrameRef]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: List[FrameRef] = []
    for e in raw:
        out.append(
            FrameRef(
                pkl=str(e["pkl"]),
                frame_idx=int(e["frame_idx"]),
                split=str(e["split"]),
                scene=str(e["scene"]),
                token=str(e.get("token", f"f{e['frame_idx']}")),
            )
        )
    return out


class NavsimMapRasterizer:
    """Rasterize nuPlan vector map layers into BEV class masks."""

    def __init__(
        self,
        maps_root: str,
        map_version: str = "nuplan-maps-v1.0",
        classes: Sequence[str] = MAP_CLASSES,
        class_to_layers: Optional[Mapping[str, Sequence[str]]] = None,
        xbound: Tuple[float, float, float] = (-50.0, 50.0, 0.5),
        ybound: Tuple[float, float, float] = (-50.0, 50.0, 0.5),
    ) -> None:
        self.maps_root = os.path.abspath(maps_root)
        self.map_version = map_version
        self.classes = tuple(classes)
        self.class_to_layers: Dict[str, Tuple[str, ...]] = {
            c: tuple(v) for c, v in (class_to_layers or DEFAULT_CLASS_TO_LAYERS).items()
        }

        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(round(patch_h / ybound[2]))
        canvas_w = int(round(patch_w / xbound[2]))
        self.patch_size = (float(patch_h), float(patch_w))
        self.canvas_size = (int(canvas_h), int(canvas_w))

        # Lazy import: nuplan-devkit is only required when actually rasterizing.
        # Allows the cached pipeline (LoadBEVSegmentationNavsimCached) to import this
        # module on environments without nuplan installed.
        from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
        from nuplan.database.maps_db.map_api import NuPlanMapWrapper  # noqa: F401
        from nuplan.database.maps_db.map_explorer import NuPlanMapExplorer  # noqa: F401

        self.maps_db = GPKGMapsDB(map_version=self.map_version, map_root=self.maps_root)
        self._map_cache: MutableMapping[str, Any] = {}
        self._explorer_cache: MutableMapping[str, Any] = {}

    def _get_explorer(self, map_location: str):
        from nuplan.database.maps_db.map_api import NuPlanMapWrapper
        from nuplan.database.maps_db.map_explorer import NuPlanMapExplorer

        name = str(np.asarray(map_location).item()).replace(".gpkg", "")
        if name not in self._map_cache:
            self._map_cache[name] = NuPlanMapWrapper(maps_db=self.maps_db, map_name=name)
            self._explorer_cache[name] = NuPlanMapExplorer(self._map_cache[name])
        return self._explorer_cache[name]

    def _available_layers(self, map_location: str) -> Sequence[str]:
        from nuplan.database.maps_db.map_api import NuPlanMapWrapper
        from nuplan.database.maps_db.map_explorer import NuPlanMapExplorer

        name = str(np.asarray(map_location).item()).replace(".gpkg", "")
        if name not in self._map_cache:
            self._map_cache[name] = NuPlanMapWrapper(maps_db=self.maps_db, map_name=name)
            self._explorer_cache[name] = NuPlanMapExplorer(self._map_cache[name])
        raw = self._map_cache[name].available_vector_layers
        arr = np.asarray(raw)
        if arr.ndim == 2 and arr.shape[1] >= 1:
            return [str(x) for x in arr[:, 0].tolist()]
        return [str(x) for x in arr.tolist()]

    def rasterize(
        self,
        map_location: str,
        ego2global: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            map_location: nuPlan location name, e.g. ``us-nv-las-vegas-strip``.
            ego2global: 4x4 transform matrix.
        Returns:
            labels: uint8 array with shape (num_classes, H, W), values in {0,1}.
        """
        ego2global = np.asarray(ego2global, dtype=np.float64).reshape(4, 4)
        map_pose = ego2global[:2, 3]
        patch_box = (
            float(map_pose[0]),
            float(map_pose[1]),
            self.patch_size[0],
            self.patch_size[1],
        )
        patch_angle = _yaw_deg_from_4x4(ego2global)

        explorer = self._get_explorer(map_location)
        available = set(self._available_layers(map_location))

        layer_names: List[str] = []
        for cls in self.classes:
            for layer in self.class_to_layers.get(cls, ()):
                if layer in available and layer not in layer_names:
                    layer_names.append(layer)

        labels = np.zeros((len(self.classes), *self.canvas_size), dtype=np.uint8)
        if not layer_names:
            return labels

        layer_to_mask: Dict[str, np.ndarray] = {}
        for layer_name in layer_names:
            try:
                one = explorer.get_map_mask(
                    patch_box=patch_box,
                    patch_angle=patch_angle,
                    layer_names=[layer_name],
                    output_size=self.canvas_size,
                )
            except Exception:
                continue
            if one.size == 0:
                continue
            layer_to_mask[layer_name] = one[0].astype(np.bool_)

        for cls_idx, cls_name in enumerate(self.classes):
            for layer_name in self.class_to_layers.get(cls_name, ()):
                if layer_name not in layer_to_mask:
                    continue
                labels[cls_idx, layer_to_mask[layer_name]] = 1
        return labels


def estimate_class_coverage(
    dataset_root: str,
    manifest_path: str,
    maps_root: str,
    map_version: str,
    sample_count: int,
    classes: Sequence[str] = MAP_CLASSES,
) -> Dict[str, Any]:
    refs = _load_manifest(manifest_path)
    refs = refs[: max(1, min(sample_count, len(refs)))]

    rasterizer = NavsimMapRasterizer(
        maps_root=maps_root,
        map_version=map_version,
        classes=classes,
    )
    pkl_cache: Dict[str, List[Dict[str, Any]]] = {}

    pixel_positive = np.zeros((len(classes),), dtype=np.int64)
    pixel_total = np.zeros((len(classes),), dtype=np.int64)
    sample_positive = np.zeros((len(classes),), dtype=np.int64)

    for ref in refs:
        pkl_abs = os.path.join(dataset_root, ref.pkl)
        if pkl_abs not in pkl_cache:
            with open(pkl_abs, "rb") as f:
                pkl_cache[pkl_abs] = pickle.load(f)
        frame = pkl_cache[pkl_abs][ref.frame_idx]
        labels = rasterizer.rasterize(
            map_location=str(frame["map_location"]),
            ego2global=np.asarray(frame["ego2global"], dtype=np.float32),
        )
        flat = labels.reshape(labels.shape[0], -1)
        pos = flat.sum(axis=1).astype(np.int64)
        pixel_positive += pos
        pixel_total += flat.shape[1]
        sample_positive += (pos > 0).astype(np.int64)

    out: Dict[str, Any] = {"num_samples": len(refs), "classes": {}}
    for i, c in enumerate(classes):
        out["classes"][c] = {
            "pixel_positive_ratio": float(pixel_positive[i] / max(pixel_total[i], 1)),
            "sample_non_empty_ratio": float(sample_positive[i] / max(len(refs), 1)),
            "pixel_positive_count": int(pixel_positive[i]),
            "pixel_total_count": int(pixel_total[i]),
        }
    return out


def run_smoke(
    dataset_root: str,
    manifest_path: str,
    maps_root: str,
    map_version: str,
    output_png: str,
) -> None:
    refs = _load_manifest(manifest_path)
    if not refs:
        raise RuntimeError("empty manifest for smoke test")
    ref = refs[0]

    pkl_abs = os.path.join(dataset_root, ref.pkl)
    with open(pkl_abs, "rb") as f:
        scene = pickle.load(f)
    frame = scene[ref.frame_idx]

    rasterizer = NavsimMapRasterizer(
        maps_root=maps_root,
        map_version=map_version,
        classes=MAP_CLASSES,
    )
    labels = rasterizer.rasterize(
        map_location=str(frame["map_location"]),
        ego2global=np.asarray(frame["ego2global"], dtype=np.float32),
    )

    # save drivable_area as a quick sanity PNG
    drivable = labels[0].astype(np.uint8) * 255
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    Image.fromarray(drivable).save(output_png)
    print(f"[smoke] saved: {output_png}")
    print(f"[smoke] sample: split={ref.split} scene={ref.scene} token={ref.token}")
    print(f"[smoke] labels shape: {labels.shape}, classes={MAP_CLASSES}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NAVSIM BEV segmentation GT utilities")
    p.add_argument(
        "--dataset-root",
        type=str,
        default=os.path.expanduser("~/wm_ws/WoTE/dataset"),
        help="NAVSIM dataset root containing navsim_logs/ and sensor_blobs/",
    )
    p.add_argument(
        "--maps-root",
        type=str,
        default=os.path.expanduser("~/wm_ws/WoTE/dataset/maps"),
        help="nuPlan maps root directory",
    )
    p.add_argument(
        "--map-version",
        type=str,
        default="nuplan-maps-v1.0",
        help="nuPlan map version name without .json",
    )
    p.add_argument(
        "--manifest",
        type=str,
        default="logs/manifest_navsim_test_only.json",
        help="Manifest JSON built from NAVSIM logs",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run single-frame smoke test and save drivable_area PNG",
    )
    p.add_argument(
        "--smoke-output",
        type=str,
        default="bev_gallery/navsim_seg_smoke/drivable_area_smoke.png",
        help="Output PNG path for smoke test",
    )
    p.add_argument(
        "--coverage",
        action="store_true",
        help="Estimate per-class coverage ratios on sampled manifest entries",
    )
    p.add_argument(
        "--coverage-samples",
        type=int,
        default=1000,
        help="Number of manifest samples to use for coverage stats",
    )
    p.add_argument(
        "--coverage-out",
        type=str,
        default="logs/navsim_map_class_coverage.json",
        help="Where to save coverage stats JSON",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.smoke and not args.coverage:
        raise ValueError("choose at least one mode: --smoke and/or --coverage")

    if args.smoke:
        run_smoke(
            dataset_root=args.dataset_root,
            manifest_path=args.manifest,
            maps_root=args.maps_root,
            map_version=args.map_version,
            output_png=args.smoke_output,
        )

    if args.coverage:
        stats = estimate_class_coverage(
            dataset_root=args.dataset_root,
            manifest_path=args.manifest,
            maps_root=args.maps_root,
            map_version=args.map_version,
            sample_count=args.coverage_samples,
            classes=MAP_CLASSES,
        )
        out_path = os.path.abspath(args.coverage_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"[coverage] saved: {out_path}")
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
