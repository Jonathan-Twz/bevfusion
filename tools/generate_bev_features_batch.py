#!/usr/bin/env python3
"""
Batch NAVSIM -> BEVFusion BEV features. By default writes only ``*_decoder_neck.pt``;
use ``--save-vtransform`` to also write ``*_vtransform.pt`` (LSS / vtransform output).

Phases:
  1) --build-manifest  Parallel scan pkls, emit manifest.json of frames with all 6 images.
  2) --trial N          Small run + camera grid + BEV heatmaps for sanity check.
  3) default            Multi-GPU inference with resume.

Default export root: ``/media/T5/bev_features/{split}/{scene}/{token}_*.pt``
(override with ``--output-root``).

**Resume** still keys off ``*_decoder_neck.pt`` only, so turning off vtransform
exports does not affect skipping completed frames.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BEV_OUTPUT_ROOT = "/media/T5/bev_features"
sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tools.navsim_frame_utils import (
    NAVSIM_CAMERAS_NUSCENES_ORDER,
    NAVSIM_CAMERAS_VISUAL_ORDER,
    check_frame_images_exist,
)


def _auto_data_loader_workers_per_gpu(num_gpus: int) -> int:
    """Spread CPU cores across GPU worker processes; cap to limit RAM from dataloader children."""
    c = os.cpu_count() or 8
    per = max(2, (c - 2) // max(1, num_gpus))
    return min(16, per)


def _auto_scan_pool_workers() -> int:
    return min(64, max(4, (os.cpu_count() or 32)))


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _scan_pkl_task(args: Tuple[str, str, str]) -> List[Dict[str, Any]]:
    """(dataset_root, rel_pkl, split) -> list of manifest entries."""
    dataset_root, rel_pkl, split = args
    pkl_path = os.path.join(dataset_root, rel_pkl)
    if not os.path.isfile(pkl_path):
        return []
    scene_name = os.path.splitext(os.path.basename(rel_pkl))[0]
    sensor_root = os.path.join(dataset_root, "sensor_blobs", split)
    blob_scene = os.path.join(sensor_root, scene_name)
    if not os.path.isdir(blob_scene):
        return []

    try:
        with open(pkl_path, "rb") as f:
            scene = pickle.load(f)
    except Exception:
        return []
    if not isinstance(scene, list):
        return []

    out: List[Dict[str, Any]] = []
    for i, frame in enumerate(scene):
        if not check_frame_images_exist(frame, sensor_root):
            continue
        tok = frame.get("token")
        if not tok:
            tok = f"f{i}"
        out.append(
            {
                "pkl": rel_pkl.replace("\\", "/"),
                "frame_idx": i,
                "token": str(tok),
                "split": split,
                "scene": scene_name,
            }
        )
    return out


def build_manifest(
    dataset_root: str,
    splits: Sequence[str],
    pool_workers: int = 32,
) -> List[Dict[str, Any]]:
    tasks: List[Tuple[str, str, str]] = []
    for sp in splits:
        nav_dir = os.path.join(dataset_root, "navsim_logs", sp)
        if not os.path.isdir(nav_dir):
            raise FileNotFoundError(f"Missing {nav_dir}")
        for name in os.listdir(nav_dir):
            if not name.endswith(".pkl"):
                continue
            rel = os.path.join("navsim_logs", sp, name)
            rel = rel.replace("\\", "/")
            tasks.append((dataset_root, rel, sp))

    if not tasks:
        return []

    workers = max(1, min(pool_workers, len(tasks)))
    merged: List[Dict[str, Any]] = []
    with Pool(workers) as pool:
        for entries in tqdm(
            pool.imap_unordered(_scan_pkl_task, tasks, chunksize=4),
            total=len(tasks),
            desc="scan pkls",
        ):
            merged.extend(entries)

    merged.sort(key=lambda e: (e["split"], e["scene"], e["frame_idx"]))
    return merged


def load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("manifest must be a JSON list")
    return data


def save_manifest(path: str, entries: List[Dict[str, Any]]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=0)


# ---------------------------------------------------------------------------
# Resume filter
# ---------------------------------------------------------------------------


def filter_pending(
    entries: List[Dict[str, Any]],
    output_root: str,
) -> List[Dict[str, Any]]:
    pending: List[Dict[str, Any]] = []
    for e in entries:
        scene_dir = os.path.join(output_root, e["split"], e["scene"])
        neck = os.path.join(scene_dir, f"{e['token']}_decoder_neck.pt")
        if os.path.isfile(neck):
            continue
        pending.append(e)
    return pending


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class NavsimBEVDataset(Dataset):
    """One sample = one frame from manifest (numpy uint8 + calib + meta)."""

    def __init__(
        self,
        entries: List[Dict[str, Any]],
        dataset_root: str,
        target_hw: Tuple[int, int],
        camera_names: Sequence[str] = NAVSIM_CAMERAS_NUSCENES_ORDER,
        lidar_align_to_nuscenes: bool = False,
    ):
        from tools.navsim_bev_adapter import NavsimDataAdapter

        self.entries = entries
        self.dataset_root = dataset_root
        self.target_hw = tuple(target_hw)
        self.camera_names = tuple(camera_names)
        self.adapter = NavsimDataAdapter(
            camera_names=camera_names,
            target_hw=self.target_hw,
            lidar_align_to_nuscenes=lidar_align_to_nuscenes,
        )
        self._pkl_cache: Dict[str, List[Dict[str, Any]]] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def _get_scene(self, pkl_abs: str) -> List[Dict[str, Any]]:
        # Lazy import: DataLoader workers unpickle Dataset without re-running __init__.
        from tools.navsim_bev_adapter import NavsimDataAdapter

        if pkl_abs not in self._pkl_cache:
            self._pkl_cache[pkl_abs] = NavsimDataAdapter.load_scene_pkl(pkl_abs)
        return self._pkl_cache[pkl_abs]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        e = self.entries[idx]
        pkl_abs = os.path.join(self.dataset_root, e["pkl"])
        scene = self._get_scene(pkl_abs)
        frame = scene[e["frame_idx"]]
        sensor_root = os.path.join(self.dataset_root, "sensor_blobs", e["split"])

        images_np, calib, _ = self.adapter.frame_to_numpy_batch(
            frame, sensor_root, self.target_hw
        )

        return {
            "images_uint8": images_np,
            "calib": calib,
            "meta": {
                "token": e["token"],
                "scene": e["scene"],
                "split": e["split"],
                "pkl": e["pkl"],
                "frame_idx": e["frame_idx"],
            },
        }


# ---------------------------------------------------------------------------
# Trial: diverse picks + visualization
# ---------------------------------------------------------------------------


def pick_trial_entries(manifest: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    by_scene: Dict[str, deque] = defaultdict(deque)
    for e in manifest:
        by_scene[e["scene"]].append(e)
    keys_sorted = sorted(by_scene.keys())
    queues = [by_scene[k] for k in keys_sorted]
    out: List[Dict[str, Any]] = []
    ptr = 0
    # Strict round-robin across scenes: one frame per step for diversity.
    while len(out) < n and any(queues):
        moved = False
        max_try = max(len(queues), 1)
        for _ in range(max_try):
            q = queues[ptr % len(queues)]
            ptr += 1
            if q:
                out.append(q.popleft())
                moved = True
                break
        if not moved:
            break
    if len(out) < n:
        seen = {(x["split"], x["scene"], x["frame_idx"]) for x in out}
        for e in manifest:
            if len(out) >= n:
                break
            key = (e["split"], e["scene"], e["frame_idx"])
            if key not in seen:
                seen.add(key)
                out.append(e)
    return out[:n]


def _bev_to_heatmap_hw(feat: torch.Tensor) -> np.ndarray:
    """(C,H,W) or (1,C,H,W) -> float (H,W) in [0,1]."""
    x = feat.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    img = x.mean(dim=0).numpy()
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return img


def save_camera_grid_bgr(
    frame: Dict[str, Any],
    sensor_blobs_root: str,
    out_path: str,
    max_cell_w: int = 480,
) -> None:
    """Stitch 6 raw-resolution BGR images in nuScenes order; scale down for PNG."""
    from PIL import Image

    tiles: List[np.ndarray] = []
    h0 = w0 = None
    for name in NAVSIM_CAMERAS_VISUAL_ORDER:
        rel = frame["cams"][name]["data_path"]
        path = os.path.join(sensor_blobs_root, rel)
        pil = Image.open(path).convert("RGB")
        rgb = np.asarray(pil)
        bgr = rgb[..., ::-1].copy()
        if h0 is None:
            h0, w0 = bgr.shape[0], bgr.shape[1]
        tiles.append(bgr)

    rows = 2
    cols = 3
    canvas_h = h0 * rows
    canvas_w = w0 * cols
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    for i, im in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * h0 : (r + 1) * h0, c * w0 : (c + 1) * w0] = im

    scale = min(1.0, max_cell_w / float(w0 * cols))
    if scale < 1.0:
        nh = int(round(canvas_h * scale))
        nw = int(round(canvas_w * scale))
        pil = Image.fromarray(canvas[..., ::-1])  # BGR->RGB for PIL
        pil = pil.resize((nw, nh), Image.BILINEAR)
        canvas = np.asarray(pil)[..., ::-1]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    pil = Image.fromarray(canvas[..., ::-1])
    pil.save(out_path)


def save_trial_comparison(
    camera_grid_path: str,
    vt_map: np.ndarray,
    neck_map: np.ndarray,
    out_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
    ax_cam = fig.add_subplot(gs[0, :])
    ax_cam.axis("off")
    if os.path.isfile(camera_grid_path):
        cam = plt.imread(camera_grid_path)
        ax_cam.imshow(cam)
    ax_cam.set_title("6 cameras (RGB)")

    ax_vt = fig.add_subplot(gs[1, 0])
    ax_vt.imshow(vt_map, cmap="viridis")
    ax_vt.set_title("vtransform (mean channels)")
    ax_vt.axis("off")

    ax_nk = fig.add_subplot(gs[1, 1])
    ax_nk.imshow(neck_map, cmap="viridis")
    ax_nk.set_title("decoder_neck (mean channels)")
    ax_nk.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_trial(args: argparse.Namespace) -> None:
    from tools.navsim_bev_adapter import BEVFeatureExtractor, NavsimDataAdapter

    manifest = load_manifest(args.manifest)
    picks = pick_trial_entries(manifest, args.trial)
    dataset_root = os.path.abspath(args.dataset_root)
    trial_dir = os.path.abspath(args.trial_viz_dir)
    os.makedirs(trial_dir, exist_ok=True)

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

    device = args.device or "cuda:0"
    extractor = BEVFeatureExtractor(
        config_path=config_path,
        checkpoint_path=ckpt,
        device=device,
    )
    runner = extractor.runner
    target_hw = tuple(runner.image_size)
    adapter = NavsimDataAdapter(
        target_hw=target_hw,
        lidar_align_to_nuscenes=bool(args.lidar_align_to_nuscenes),
    )

    out_root = os.path.abspath(args.output_root)
    for k, e in enumerate(tqdm(picks, desc="trial")):
        pkl_abs = os.path.join(dataset_root, e["pkl"])
        scene = NavsimDataAdapter.load_scene_pkl(pkl_abs)
        frame = scene[e["frame_idx"]]
        sensor_root = os.path.join(dataset_root, "sensor_blobs", e["split"])

        prefix = os.path.join(out_root, e["split"], e["scene"], e["token"])
        parent = os.path.dirname(prefix)
        if parent:
            os.makedirs(parent, exist_ok=True)

        images_np, calib, _ = adapter.frame_to_numpy_batch(
            frame, sensor_root, target_hw
        )
        images = runner.preprocess_images(images_np, target_size=list(target_hw))

        def to_t(key: str) -> torch.Tensor:
            return torch.from_numpy(calib[key].astype(np.float32))

        feats = extractor.extract_bev_features(
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
        if args.save_vtransform:
            torch.save(feats["vtransform"].detach().cpu(), f"{prefix}_vtransform.pt")
        torch.save(feats["decoder_neck"].detach().cpu(), f"{prefix}_decoder_neck.pt")

        tag = f"trial_{k:03d}_{e['token']}"
        grid_png = os.path.join(trial_dir, f"{tag}_cameras.png")
        save_camera_grid_bgr(frame, sensor_root, grid_png)

        vt = _bev_to_heatmap_hw(feats["vtransform"])
        nk = _bev_to_heatmap_hw(feats["decoder_neck"])
        cmp_path = os.path.join(trial_dir, f"{tag}_compare.png")
        save_trial_comparison(grid_png, vt, nk, cmp_path)
        pt_info = f"{prefix}_decoder_neck.pt" + (
            f", {prefix}_vtransform.pt" if args.save_vtransform else ""
        )
        print(f"Saved trial: {pt_info}, {grid_png}, {cmp_path}")


# ---------------------------------------------------------------------------
# GPU worker (one process per GPU)
# ---------------------------------------------------------------------------


@dataclass
class WorkerArgs:
    dataset_root: str
    output_root: str
    config_path: str
    checkpoint_path: str
    batch_size: int
    num_workers: int
    target_hw: Tuple[int, int]
    lidar_align_to_nuscenes: bool = False
    save_vtransform: bool = False


def gpu_worker_entry(rank: int, chunks: List[List[Dict[str, Any]]], wa: WorkerArgs) -> None:
    from tools.navsim_bev_adapter import (
        BEVFeatureExtractor,
        navsim_bev_collate_fn,
        preprocess_images_batch,
    )

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

    ds = NavsimBEVDataset(
        chunk,
        dataset_root=wa.dataset_root,
        target_hw=(th, tw),
        lidar_align_to_nuscenes=wa.lidar_align_to_nuscenes,
    )
    loader_kwargs: Dict[str, Any] = {
        "batch_size": wa.batch_size,
        "shuffle": False,
        "num_workers": wa.num_workers,
        "pin_memory": True,
        "collate_fn": navsim_bev_collate_fn,
        "drop_last": False,
    }
    if wa.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 3
        loader_kwargs["multiprocessing_context"] = mp.get_context("spawn")

    loader = DataLoader(ds, **loader_kwargs)

    for batch in tqdm(loader, desc=f"GPU{rank}", position=rank):
        meta_list = batch["meta"]
        images = preprocess_images_batch(
            runner,
            batch["images_uint8"],
            (th, tw),
        )
        calib = batch["calib"]

        def tot(key: str) -> torch.Tensor:
            return torch.from_numpy(calib[key].astype(np.float32)).to(device, non_blocking=True)

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
        nk = feats["decoder_neck"].detach().cpu()
        vt = None
        if wa.save_vtransform:
            vt = feats["vtransform"].detach().cpu()
        for i in range(bsz):
            m = meta_list[i]
            scene_dir = os.path.join(wa.output_root, m["split"], m["scene"])
            os.makedirs(scene_dir, exist_ok=True)
            pfx = os.path.join(scene_dir, m["token"])
            # Slices share storage with the batch tensor; torch.save would serialize
            # the whole batch storage (~batch_size x larger). Clone to a compact file.
            if wa.save_vtransform and vt is not None:
                torch.save(vt[i].clone(), f"{pfx}_vtransform.pt")
            torch.save(nk[i].clone(), f"{pfx}_decoder_neck.pt")


def run_multi_gpu(args: argparse.Namespace) -> None:
    from tools.navsim_bev_adapter import BEVFeatureExtractor

    manifest = load_manifest(args.manifest)
    manifest = filter_pending(manifest, args.output_root)
    if not manifest:
        print("All frames already processed (decoder_neck present). Nothing to do.")
        return

    avail = torch.cuda.device_count()
    req = args.num_gpus
    if req < 0:
        req = avail
    num_gpus = min(req, avail)
    if num_gpus < 1:
        raise RuntimeError("No CUDA devices available")

    num_workers = args.num_workers
    if num_workers < 0:
        num_workers = _auto_data_loader_workers_per_gpu(num_gpus)
    args.num_workers = num_workers

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

    # Infer target_hw from config (lightweight: build extractor on cuda:0 briefly)
    probe = BEVFeatureExtractor(
        config_path=config_path,
        checkpoint_path=ckpt,
        device="cuda:0",
    )
    target_hw = tuple(probe.runner.image_size)
    del probe
    torch.cuda.empty_cache()

    chunks = [manifest[i::num_gpus] for i in range(num_gpus)]
    print(
        f"Processing {len(manifest)} frames on {num_gpus} GPU(s), "
        f"batch_size={args.batch_size}, num_workers/GPU={num_workers}, "
        f"save_vtransform={args.save_vtransform}, chunks={[len(c) for c in chunks]}"
    )

    wa = WorkerArgs(
        dataset_root=os.path.abspath(args.dataset_root),
        output_root=os.path.abspath(args.output_root),
        config_path=config_path,
        checkpoint_path=ckpt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_hw=target_hw,
        lidar_align_to_nuscenes=bool(args.lidar_align_to_nuscenes),
        save_vtransform=bool(args.save_vtransform),
    )

    mp.spawn(
        gpu_worker_entry,
        args=(chunks, wa),
        nprocs=num_gpus,
        join=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch BEV feature export for NAVSIM")
    p.add_argument(
        "--dataset-root",
        type=str,
        default=os.path.join(os.path.expanduser("~"), "wm_ws/WoTE/dataset"),
    )
    p.add_argument(
        "--build-manifest",
        action="store_true",
        help="Only scan pkls and write --manifest-out",
    )
    p.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["trainval", "test"],
    )
    p.add_argument(
        "--manifest-out",
        type=str,
        default=os.path.join(_REPO_ROOT, "outputs/manifest.json"),
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=os.path.join(_REPO_ROOT, "outputs/manifest.json"),
        help="Input manifest JSON for trial / full run",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_BEV_OUTPUT_ROOT,
        help=f"BEV .pt root: {{split}}/{{scene}}/{{token}}_* (default: {DEFAULT_BEV_OUTPUT_ROOT})",
    )
    p.add_argument("--config", type=str, default="configs/nuscenes/seg/camera-bev256d2.yaml")
    p.add_argument("--checkpoint", type=str, default="pretrained/camera-only-seg.pth")
    p.add_argument(
        "--save-vtransform",
        action="store_true",
        help="Also write *_vtransform.pt (default: only *_decoder_neck.pt).",
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=-1,
        help="GPU processes to spawn; -1 = all visible devices (from CUDA_VISIBLE_DEVICES / torch).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Per-GPU batch (raise if VRAM allows; lower if OOM).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers per GPU process; -1 = auto from CPU count (capped at 16).",
    )
    p.add_argument(
        "--scan-workers",
        type=int,
        default=-1,
        help="Multiprocessing pool for --build-manifest; -1 = auto from CPU count (capped at 64).",
    )
    p.add_argument("--trial", type=int, default=0, help="If >0, trial mode on N diverse frames")
    p.add_argument(
        "--trial-viz-dir",
        type=str,
        default=os.path.join(_REPO_ROOT, "outputs/bev_trial"),
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device for trial mode",
    )
    p.add_argument(
        "--lidar-align-to-nuscenes",
        action="store_true",
        help=(
            "Inject a swap-XY reflection into lidar_aug_matrix so that the "
            "nuScenes-pretrained BEV features align with the NAVSIM GT display "
            "convention. Only use this with pretrained/nuScenes-trained "
            "checkpoints. Leave OFF for stage1/2/3 fine-tuned checkpoints."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.build_manifest:
        sw = args.scan_workers
        if sw < 0:
            sw = _auto_scan_pool_workers()
        m = build_manifest(
            os.path.abspath(args.dataset_root),
            args.splits,
            pool_workers=sw,
        )
        out_path = os.path.abspath(args.manifest_out)
        save_manifest(out_path, m)
        print(f"Wrote manifest with {len(m)} entries -> {out_path}")
        return

    args.output_root = os.path.abspath(args.output_root)

    if args.trial > 0:
        if not os.path.isfile(args.manifest):
            raise FileNotFoundError(f"Missing manifest: {args.manifest}")
        run_trial(args)
        return

    if not os.path.isfile(args.manifest):
        raise FileNotFoundError(f"Missing manifest: {args.manifest}")
    run_multi_gpu(args)


if __name__ == "__main__":
    main()
