#!/usr/bin/env python3
"""
Build NAVSIM train/val manifests for BEV segmentation fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from multiprocessing import Pool
from typing import Any, Dict, List, Sequence, Tuple

from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.navsim_frame_utils import check_frame_images_exist


def _scan_scene(args: Tuple[str, str, str]) -> List[Dict[str, Any]]:
    dataset_root, rel_pkl, split = args
    abs_pkl = os.path.join(dataset_root, rel_pkl)
    if not os.path.isfile(abs_pkl):
        return []
    scene_name = os.path.splitext(os.path.basename(rel_pkl))[0]
    sensor_root = os.path.join(dataset_root, "sensor_blobs", split)
    if not os.path.isdir(os.path.join(sensor_root, scene_name)):
        return []

    try:
        with open(abs_pkl, "rb") as f:
            scene = pickle.load(f)
    except Exception:
        return []
    if not isinstance(scene, list):
        return []

    out: List[Dict[str, Any]] = []
    for i, frame in enumerate(scene):
        if not check_frame_images_exist(frame, sensor_root):
            continue
        token = str(frame.get("token", f"f{i}"))
        out.append(
            {
                "pkl": rel_pkl.replace("\\", "/"),
                "frame_idx": i,
                "token": token,
                "split": split,
                "scene": scene_name,
            }
        )
    return out


def build_manifest(dataset_root: str, split: str, workers: int = 32) -> List[Dict[str, Any]]:
    nav_dir = os.path.join(dataset_root, "navsim_logs", split)
    if not os.path.isdir(nav_dir):
        raise FileNotFoundError(f"Missing directory: {nav_dir}")
    tasks: List[Tuple[str, str, str]] = []
    for name in os.listdir(nav_dir):
        if name.endswith(".pkl"):
            rel = os.path.join("navsim_logs", split, name).replace("\\", "/")
            tasks.append((dataset_root, rel, split))
    tasks.sort(key=lambda x: x[1])
    if not tasks:
        return []

    workers = max(1, min(workers, len(tasks)))
    out: List[Dict[str, Any]] = []
    with Pool(workers) as pool:
        for entries in tqdm(
            pool.imap_unordered(_scan_scene, tasks, chunksize=4),
            total=len(tasks),
            desc=f"scan {split}",
        ):
            out.extend(entries)
    out.sort(key=lambda e: (e["scene"], e["frame_idx"]))
    return out


def scene_stratified_pick(
    entries: Sequence[Dict[str, Any]],
    target_count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if target_count <= 0:
        return []
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        by_scene.setdefault(e["scene"], []).append(e)
    scenes = sorted(by_scene.keys())
    for s in scenes:
        by_scene[s].sort(key=lambda x: x["frame_idx"])

    rng = random.Random(seed)
    scene_order = scenes[:]
    rng.shuffle(scene_order)

    picked: List[Dict[str, Any]] = []
    scene_cursor = {s: 0 for s in scenes}
    while len(picked) < target_count and scene_order:
        progressed = False
        for s in scene_order:
            idx = scene_cursor[s]
            if idx < len(by_scene[s]):
                picked.append(by_scene[s][idx])
                scene_cursor[s] += 1
                progressed = True
                if len(picked) >= target_count:
                    break
        if not progressed:
            break
    return picked[:target_count]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build NAVSIM finetune train/val manifests")
    p.add_argument(
        "--dataset-root",
        type=str,
        default=os.path.expanduser("~/wm_ws/WoTE/dataset"),
    )
    p.add_argument("--split", type=str, default="trainval")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--train-count", type=int, default=10000)
    p.add_argument("--val-count", type=int, default=1000)
    p.add_argument(
        "--all-manifest-out",
        type=str,
        default="logs/manifest_navsim_trainval_all.json",
    )
    p.add_argument(
        "--train-manifest-out",
        type=str,
        default="logs/navsim_finetune_train.json",
    )
    p.add_argument(
        "--val-manifest-out",
        type=str,
        default="logs/navsim_finetune_val.json",
    )
    return p.parse_args()


def _save_json(path: str, data: Any) -> None:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    args = parse_args()
    all_entries = build_manifest(
        dataset_root=args.dataset_root,
        split=args.split,
        workers=args.workers,
    )
    if not all_entries:
        raise RuntimeError("No valid entries found in dataset.")

    _save_json(args.all_manifest_out, all_entries)

    val_entries = scene_stratified_pick(
        all_entries,
        target_count=min(args.val_count, len(all_entries)),
        seed=args.seed,
    )
    val_keys = {(e["scene"], e["frame_idx"]) for e in val_entries}
    remain = [e for e in all_entries if (e["scene"], e["frame_idx"]) not in val_keys]

    train_entries = scene_stratified_pick(
        remain,
        target_count=min(args.train_count, len(remain)),
        seed=args.seed + 1,
    )

    _save_json(args.train_manifest_out, train_entries)
    _save_json(args.val_manifest_out, val_entries)

    print(
        json.dumps(
            {
                "all_count": len(all_entries),
                "train_count": len(train_entries),
                "val_count": len(val_entries),
                "train_out": os.path.abspath(args.train_manifest_out),
                "val_out": os.path.abspath(args.val_manifest_out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
