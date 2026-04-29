#!/usr/bin/env python3
"""
Precompute NAVSIM BEV segmentation GT masks and dump to a single uint8 memmap.

Why: rasterizing nuPlan map GT on-the-fly in dataloader workers is (1) slow
(~100 ms/frame → data_time dominates iter time at bs>=30) and (2) causes
unbounded host-RAM growth per worker because nuPlan's GPKGMapsDB + map
wrappers retain vector layers in memory per-process (18 workers × ~10-15 GB
reached 240 GB out of 251 GB on our box and triggered swap/OOM).

This script precomputes the masks once into a single ``(N, C, H, W) uint8``
memmap file that dataloader workers can mmap-read (sub-ms per sample, near
zero extra RSS thanks to shared page cache).

Usage (inside the project Docker env with nuPlan deps installed):

    python tools/precompute_navsim_bev_gt.py \
        --manifests logs/navsim_finetune_stage3_train.json \
                    logs/navsim_finetune_stage3_val.json \
        --dataset-root ~/wm_ws/WoTE/dataset \
        --maps-root ~/wm_ws/WoTE/dataset/maps \
        --out-npy logs/navsim_bev_gt_cache/stage3_masks.npy \
        --out-index logs/navsim_bev_gt_cache/stage3_index.json \
        --workers 8

Output:
  - ``stage3_masks.npy`` : raw ``(N, len(classes), H, W) uint8`` memmap dump
    (no ``.npy`` header; we store shape/dtype alongside in the index JSON
    so the training pipeline just ``np.memmap``s the file directly).
  - ``stage3_index.json`` : {"shape": [N, C, H, W], "dtype": "uint8",
    "classes": [...], "entries": [{"token": ..., "row": ..., "scene": ...,
    "frame_idx": ...}, ...]}.

The training pipeline looks up ``row`` by ``data["token"]`` and reads
``memmap[row]``.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.data_converter.navsim_bev_seg_gt import (  # noqa: E402
    DEFAULT_CLASS_TO_LAYERS,
    MAP_CLASSES,
    NavsimMapRasterizer,
)


# Per-worker globals (populated in _init_worker via fork).
_WORKER: Dict[str, Any] = {}


def _init_worker(
    maps_root: str,
    map_version: str,
    classes: Tuple[str, ...],
    xbound: Tuple[float, float, float],
    ybound: Tuple[float, float, float],
    memmap_path: str,
    mm_shape: Tuple[int, int, int, int],
) -> None:
    _WORKER["rasterizer"] = NavsimMapRasterizer(
        maps_root=maps_root,
        map_version=map_version,
        classes=classes,
        class_to_layers=DEFAULT_CLASS_TO_LAYERS,
        xbound=xbound,
        ybound=ybound,
    )
    _WORKER["mm"] = np.memmap(
        memmap_path, dtype=np.uint8, mode="r+", shape=mm_shape
    )
    _WORKER["pkl_cache"] = {}


def _worker_task(args: Tuple[int, str, int]) -> Tuple[int, int]:
    """Return (row, nonzero_pixels) so the driver can track progress + sanity."""
    row, pkl_abs, frame_idx = args
    cache = _WORKER["pkl_cache"]
    if pkl_abs not in cache:
        with open(pkl_abs, "rb") as f:
            cache[pkl_abs] = pickle.load(f)
    frame = cache[pkl_abs][frame_idx]
    mask = _WORKER["rasterizer"].rasterize(
        map_location=str(frame["map_location"]),
        ego2global=np.asarray(frame["ego2global"], dtype=np.float32),
    )
    _WORKER["mm"][row] = mask
    return row, int(mask.sum())


def _load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"manifest must be a list: {path}")
    return data


def build_entry_list(manifests: List[str]) -> List[Dict[str, Any]]:
    """Combine manifests, dedup by token, keep a stable (scene, frame_idx) order."""
    seen: Dict[str, Dict[str, Any]] = {}
    for m in manifests:
        entries = _load_manifest(m)
        for e in entries:
            token = str(e.get("token", f"{e['scene']}:{e['frame_idx']}"))
            if token in seen:
                continue
            seen[token] = {
                "token": token,
                "pkl": str(e["pkl"]),
                "frame_idx": int(e["frame_idx"]),
                "split": str(e.get("split", "")),
                "scene": str(e.get("scene", "")),
            }
    out = list(seen.values())
    out.sort(key=lambda x: (x["pkl"], x["frame_idx"]))  # locality for pkl cache
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--manifests", type=str, nargs="+", required=True,
                   help="One or more NAVSIM manifest JSONs to merge (deduped by token).")
    p.add_argument("--dataset-root", type=str,
                   default=os.path.expanduser("~/wm_ws/WoTE/dataset"))
    p.add_argument("--maps-root", type=str,
                   default=os.path.expanduser("~/wm_ws/WoTE/dataset/maps"))
    p.add_argument("--map-version", type=str, default="nuplan-maps-v1.0")
    p.add_argument("--classes", type=str, nargs="+", default=list(MAP_CLASSES))
    p.add_argument("--xbound", type=float, nargs=3, default=[-50.0, 50.0, 0.5])
    p.add_argument("--ybound", type=float, nargs=3, default=[-50.0, 50.0, 0.5])
    p.add_argument("--out-npy", type=str, required=True,
                   help="Output raw uint8 memmap path (no NPY header).")
    p.add_argument("--out-index", type=str, required=True,
                   help="Output index JSON path.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunksize", type=int, default=32)
    p.add_argument("--limit", type=int, default=0,
                   help="Only process first N entries (debug).")
    args = p.parse_args()

    classes = tuple(args.classes)
    H = int(round((args.ybound[1] - args.ybound[0]) / args.ybound[2]))
    W = int(round((args.xbound[1] - args.xbound[0]) / args.xbound[2]))
    C = len(classes)

    entries = build_entry_list(args.manifests)
    if args.limit > 0:
        entries = entries[: args.limit]
    N = len(entries)
    print(f"[precompute] {N} unique frames from manifests {args.manifests}")
    print(f"[precompute] shape=({N}, {C}, {H}, {W}) uint8 -> "
          f"{N * C * H * W / (1024**3):.2f} GiB")

    # Preallocate memmap file on disk.
    os.makedirs(os.path.dirname(os.path.abspath(args.out_npy)), exist_ok=True)
    mm = np.memmap(args.out_npy, dtype=np.uint8, mode="w+", shape=(N, C, H, W))
    mm[:] = 0
    mm.flush()
    del mm

    # Build task list with absolute pkl paths.
    tasks: List[Tuple[int, str, int]] = []
    for row, e in enumerate(entries):
        pkl_abs = os.path.join(args.dataset_root, e["pkl"])
        tasks.append((row, pkl_abs, e["frame_idx"]))

    t0 = time.time()
    nonempty = 0
    done = 0
    report_every = max(500, N // 60)
    init_args = (
        os.path.abspath(args.maps_root),
        args.map_version,
        classes,
        tuple(args.xbound),
        tuple(args.ybound),
        os.path.abspath(args.out_npy),
        (N, C, H, W),
    )
    with Pool(processes=args.workers, initializer=_init_worker, initargs=init_args) as pool:
        for row, nz in pool.imap_unordered(_worker_task, tasks, chunksize=args.chunksize):
            done += 1
            if nz > 0:
                nonempty += 1
            if done % report_every == 0 or done == N:
                dt = time.time() - t0
                eta = dt / done * (N - done)
                print(
                    f"[precompute] {done:6d}/{N} "
                    f"({100.0 * done / N:5.1f}%)  "
                    f"elapsed={dt / 60:5.1f}m  eta={eta / 60:5.1f}m  "
                    f"nonempty={nonempty}",
                    flush=True,
                )

    # Write index JSON (after workers finish).
    index = {
        "shape": [N, C, H, W],
        "dtype": "uint8",
        "classes": list(classes),
        "xbound": list(args.xbound),
        "ybound": list(args.ybound),
        "memmap_file": os.path.basename(args.out_npy),
        "entries": [
            {"token": e["token"], "row": i, "scene": e["scene"],
             "frame_idx": e["frame_idx"], "split": e["split"]}
            for i, e in enumerate(entries)
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_index)), exist_ok=True)
    with open(args.out_index, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    total = time.time() - t0
    print(f"[precompute] done in {total / 60:.1f} min, "
          f"{N} frames, {nonempty} non-empty (has>=1 positive pixel)")
    print(f"[precompute] wrote {args.out_npy}  (~{os.path.getsize(args.out_npy) / (1024**3):.2f} GiB)")
    print(f"[precompute] wrote {args.out_index}")


if __name__ == "__main__":
    main()
