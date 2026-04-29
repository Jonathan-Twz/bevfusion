#!/usr/bin/env python3
"""
Batch nuScenes (info.pkl) -> BEVFusion BEV features (vtransform + decoder_neck as .pt).

Output layout (flat by sample token):
  ``{output_root}/{token}_vtransform.pt`` and ``{token}_decoder_neck.pt``

Resume: skips frames where ``{token}_decoder_neck.pt`` already exists.

Example (full v1.0-mini: train + val):
  python tools/generate_bev_features_nuscenes.py \\
    --info-pkl data/nuscenes/nuscenes_infos_train.pkl data/nuscenes/nuscenes_infos_val.pkl \\
    --dataset-root data/nuscenes \\
    --output-root bev_gallery/nuscenes_mini \\
    --num-gpus 1 --batch-size 4 --num-workers 4
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from tools.nuscenes_bev_adapter import (  # noqa: E402
    BEVFeatureExtractor,
    NuscenesBEVDataset,
    check_nuscenes_images_exist,
    load_nuscenes_infos_pkl,
    nuscenes_bev_collate_fn,
    preprocess_images_batch,
)


def filter_pending_nuscenes(
    infos: List[Dict[str, Any]],
    dataset_root: str,
    output_root: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    output_root = os.path.abspath(output_root)
    for info in infos:
        tok = info["token"]
        neck = os.path.join(output_root, f"{tok}_decoder_neck.pt")
        if os.path.isfile(neck):
            continue
        if not check_nuscenes_images_exist(info, dataset_root):
            continue
        out.append(info)
    return out


@dataclass
class WorkerArgs:
    dataset_root: str
    output_root: str
    config_path: str
    checkpoint_path: str
    batch_size: int
    num_workers: int
    target_hw: Tuple[int, int]


def gpu_worker_entry(rank: int, chunks: List[List[Dict[str, Any]]], wa: WorkerArgs) -> None:
    chunk = chunks[rank]
    if not chunk:
        return
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    extractor = BEVFeatureExtractor(
        config_path=wa.config_path,
        checkpoint_path=wa.checkpoint_path,
        device=device,
    )
    runner = extractor.runner
    th, tw = wa.target_hw
    if tuple(runner.image_size) != (th, tw):
        raise ValueError(
            f"Model image_size {runner.image_size} != worker {wa.target_hw}"
        )

    ds = NuscenesBEVDataset(
        chunk,
        dataset_root=wa.dataset_root,
        target_hw=(th, tw),
    )
    loader_kwargs: Dict[str, Any] = {
        "batch_size": wa.batch_size,
        "shuffle": False,
        "num_workers": wa.num_workers,
        "pin_memory": True,
        "collate_fn": nuscenes_bev_collate_fn,
        "drop_last": False,
    }
    if wa.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["multiprocessing_context"] = mp.get_context("spawn")

    loader = DataLoader(ds, **loader_kwargs)
    os.makedirs(wa.output_root, exist_ok=True)

    for batch in tqdm(loader, desc=f"GPU{rank}", position=rank):
        meta_list = batch["meta"]
        images = preprocess_images_batch(
            runner,
            batch["images_uint8"],
            (th, tw),
        )
        calib = batch["calib"]

        def tot(key: str) -> torch.Tensor:
            return torch.from_numpy(calib[key].astype(np.float32)).to(
                device, non_blocking=True
            )

        feats = extractor.extract_bev_features(
            images,
            tot("camera_intrinsics"),
            tot("camera2ego"),
            tot("lidar2ego"),
            tot("lidar2camera"),
            tot("lidar2image"),
            tot("camera2lidar"),
            img_aug_matrix=tot("img_aug_matrix"),
            lidar_aug_matrix=tot("lidar_aug_matrix"),
        )

        bsz = images.shape[0]
        vt = feats["vtransform"].detach().cpu()
        nk = feats["decoder_neck"].detach().cpu()
        for i in range(bsz):
            tok = meta_list[i]["token"]
            pfx = os.path.join(wa.output_root, tok)
            torch.save(vt[i].clone(), f"{pfx}_vtransform.pt")
            torch.save(nk[i].clone(), f"{pfx}_decoder_neck.pt")


def _merge_infos_from_pkls(paths: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"sources": paths}
    seen_tokens: set = set()
    for p in paths:
        infos, m = load_nuscenes_infos_pkl(p)
        meta[os.path.abspath(p)] = m
        for inf in infos:
            tok = inf.get("token")
            if tok is None:
                continue
            if tok in seen_tokens:
                continue
            seen_tokens.add(tok)
            merged.append(inf)
    return merged, meta


def run_multi_gpu(args: argparse.Namespace) -> None:
    info_paths = list(args.info_pkl)
    infos_all, meta = _merge_infos_from_pkls(info_paths)
    if args.max_frames > 0:
        infos_all = infos_all[: args.max_frames]
    dataset_root = os.path.abspath(args.dataset_root)
    pending = filter_pending_nuscenes(infos_all, dataset_root, args.output_root)
    if not pending:
        print("All frames already processed or missing images. Nothing to do.")
        return

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    if num_gpus < 1:
        raise RuntimeError("No CUDA devices available")

    config_path = (
        args.config
        if os.path.isabs(args.config)
        else os.path.join(_REPO_ROOT, args.config)
    )
    ckpt = (
        args.checkpoint
        if os.path.isabs(args.checkpoint)
        else os.path.join(_REPO_ROOT, args.checkpoint)
    )
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    probe = BEVFeatureExtractor(
        config_path=config_path,
        checkpoint_path=ckpt,
        device="cuda:0",
    )
    target_hw = tuple(probe.runner.image_size)
    del probe
    torch.cuda.empty_cache()

    chunks = [pending[i::num_gpus] for i in range(num_gpus)]
    print(
        f"metadata: {meta}, pending={len(pending)} / {len(infos_all)} infos, "
        f"{num_gpus} GPU(s), batch_size={args.batch_size}, chunks={[len(c) for c in chunks]}"
    )

    wa = WorkerArgs(
        dataset_root=dataset_root,
        output_root=os.path.abspath(args.output_root),
        config_path=config_path,
        checkpoint_path=ckpt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_hw=target_hw,
    )

    mp.spawn(
        gpu_worker_entry,
        args=(chunks, wa),
        nprocs=num_gpus,
        join=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch BEV feature export for nuScenes (info.pkl)")
    p.add_argument(
        "--info-pkl",
        type=str,
        nargs="+",
        required=True,
        help="One or more nuscenes_infos_*.pkl (e.g. train + val for full mini set)",
    )
    p.add_argument(
        "--dataset-root",
        type=str,
        default="data/nuscenes",
        help="Folder containing samples/, sweeps/, v1.0-*/ (not the repo root)",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=os.path.join(_REPO_ROOT, "bev_gallery/nuscenes_out"),
        help="Directory for flat {token}_*.pt files",
    )
    p.add_argument("--config", type=str, default="configs/nuscenes/seg/camera-bev256d2.yaml")
    p.add_argument("--checkpoint", type=str, default="pretrained/camera-only-seg.pth")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If >0, only use first N entries from the info pkl (for debugging)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_multi_gpu(args)


if __name__ == "__main__":
    main()
