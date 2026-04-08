"""
Lightweight NAVSIM helpers (no OpenCV / torch / BEVFusion imports).

Used by manifest scanning and adapters.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Sequence, Tuple

# Match nuScenes converter order in tools/data_converter/nuscenes_converter.py
NAVSIM_CAMERAS_NUSCENES_ORDER: Tuple[str, ...] = (
    "CAM_F0",  # CAM_FRONT
    "CAM_R0",  # CAM_FRONT_RIGHT
    "CAM_L0",  # CAM_FRONT_LEFT
    "CAM_B0",  # CAM_BACK
    "CAM_L2",  # CAM_BACK_LEFT
    "CAM_R2",  # CAM_BACK_RIGHT
)


def check_frame_images_exist(
    frame: Dict[str, Any],
    sensor_blobs_root: str,
    camera_names: Sequence[str] = NAVSIM_CAMERAS_NUSCENES_ORDER,
) -> bool:
    """True if every ``camera_names`` image file exists under ``sensor_blobs_root``."""
    cams = frame.get("cams")
    if not cams:
        return False
    for name in camera_names:
        if name not in cams:
            return False
        rel = cams[name].get("data_path")
        if not rel:
            return False
        path = os.path.join(sensor_blobs_root, rel)
        if not os.path.isfile(path):
            return False
    return True


__all__ = [
    "NAVSIM_CAMERAS_NUSCENES_ORDER",
    "check_frame_images_exist",
]
