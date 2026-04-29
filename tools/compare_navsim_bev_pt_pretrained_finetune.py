#!/usr/bin/env python3
"""
并排对比「原始 pretrained」与「finetune 后」导出的 BEV 中间特征 (.pt)。

与 ``batch_visualize_bev_pt_with_cameras.py`` 类似：上行 6 路相机 RGB；
下面一行只绘制 decoder_neck 的 channel 统计（mean/max/min/L2 norm），
左半为 pretrained，右半为 finetune。vtransform 不再绘制，避免生成不必要的
大体积中间特征。

**前置**：对**同一批帧**、用同一 ``configs/navsim/seg/camera-bev256d2-finetune.yaml``（或等价结构），
分别导出两套 ``{token}_decoder_neck.pt`` 到两个目录，目录结构须一致，例如::

  python tools/generate_bev_features_batch.py \\
    --manifest logs/bev_compare_frames.json \\
    --dataset-root /path/to/WoTE/dataset \\
    --output-root /path/to/bev_pt_pretrained \\
    --config configs/navsim/seg/camera-bev256d2-finetune.yaml \\
    --checkpoint pretrained/camera-only-seg.pth \\
    --num-gpus 1 --batch-size 4

  python tools/generate_bev_features_batch.py \\
    --manifest logs/bev_compare_frames.json \\
    --dataset-root /path/to/WoTE/dataset \\
    --output-root /path/to/bev_pt_stage1 \\
    --config configs/navsim/seg/camera-bev256d2-finetune.yaml \\
    --checkpoint runs/navsim_seg_stage1/latest.pth \\
    --num-gpus 1 --batch-size 4

**对比出图**::

  python tools/compare_navsim_bev_pt_pretrained_finetune.py \\
    --pt-pretrained /path/to/bev_pt_pretrained \\
    --pt-finetune /path/to/bev_pt_stage1 \\
    --dataset-root /path/to/WoTE/dataset \\
    --out-dir bev_gallery/compare_pretrained_vs_stage1 \\
    --max-frames 20
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Any, Dict, Iterator, List, Set, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from tools.navsim_frame_utils import NAVSIM_CAMERAS_VISUAL_ORDER  # noqa: E402
from tools.visualize_bev_channel_stats import channel_stats  # noqa: E402
from tools.visualize_bev_feat import load_feature  # noqa: E402


def _norm01(img: np.ndarray) -> np.ndarray:
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        return (img - lo) / (hi - lo)
    return np.zeros_like(img, dtype=np.float32)


def iter_pt_pairs(pt_root: str) -> Iterator[Tuple[str, str, str, str]]:
    """Same layout as ``batch_visualize_bev_pt_with_cameras.iter_pt_pairs`` (NAVSIM tree)."""
    pt_root = os.path.abspath(pt_root)
    if not os.path.isdir(pt_root):
        raise FileNotFoundError(pt_root)
    base = os.path.basename(pt_root.rstrip(os.sep))

    def emit_for_split(split: str, scenes_root: str) -> Iterator[Tuple[str, str, str, str, str]]:
        for scene in sorted(os.listdir(scenes_root)):
            sdir = os.path.join(scenes_root, scene)
            if not os.path.isdir(sdir):
                continue
            names = os.listdir(sdir)
            for f in names:
                if not f.endswith("_decoder_neck.pt"):
                    continue
                tok = f.replace("_decoder_neck.pt", "")
                yield (
                    split,
                    scene,
                    tok,
                    os.path.join(sdir, f),
                )

    if base in ("trainval", "test"):
        yield from emit_for_split(base, pt_root)
    else:
        for split in sorted(os.listdir(pt_root)):
            sp = os.path.join(pt_root, split)
            if not os.path.isdir(sp):
                continue
            yield from emit_for_split(split, sp)


def load_frame_for_token(
    dataset_root: str, split: str, scene: str, token: str
) -> Any:
    pkl_path = os.path.join(
        dataset_root, "navsim_logs", split, f"{scene}.pkl"
    )
    if not os.path.isfile(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        scene_list = pickle.load(f)
    if not isinstance(scene_list, list):
        return None
    for fr in scene_list:
        if fr.get("token") == token:
            return fr
    return None


def build_camera_rgb_grid(
    frame: Dict[str, Any],
    sensor_blobs_root: str,
    max_width: int = 3000,
) -> np.ndarray:
    from PIL import Image

    tiles: List[np.ndarray] = []
    h0 = w0 = None
    for name in NAVSIM_CAMERAS_VISUAL_ORDER:
        cams = frame.get("cams") or {}
        if name not in cams:
            raise KeyError(f"missing camera {name}")
        rel = cams[name]["data_path"]
        path = os.path.join(sensor_blobs_root, rel)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        pil = Image.open(path).convert("RGB")
        rgb = np.asarray(pil)
        if h0 is None:
            h0, w0 = rgb.shape[0], rgb.shape[1]
        tiles.append(rgb)

    rows, cols = 2, 3
    canvas = np.zeros((h0 * rows, w0 * cols, 3), dtype=np.uint8)
    for i, im in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * h0 : (r + 1) * h0, c * w0 : (c + 1) * w0] = im

    if canvas.shape[1] > max_width:
        scale = max_width / float(canvas.shape[1])
        nh = max(1, int(round(canvas.shape[0] * scale)))
        nw = max_width
        pil = Image.fromarray(canvas)
        pil = pil.resize((nw, nh), Image.BILINEAR)
        canvas = np.asarray(pil)
    return canvas


def _index_pairs(pt_root: str) -> Dict[Tuple[str, str, str], str]:
    """(split, scene, token) -> decoder_neck_path."""
    out: Dict[Tuple[str, str, str], str] = {}
    for split, scene, token, np_ in iter_pt_pairs(pt_root):
        out[(split, scene, token)] = np_
    return out


def render_compare(
    split: str,
    scene: str,
    token: str,
    neck_pre: str,
    neck_ft: str,
    dataset_root: str,
    camera_max_width: int,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    frame = load_frame_for_token(dataset_root, split, scene, token)
    if frame is None:
        raise RuntimeError(f"no frame token={token} in {scene}.pkl")
    sensor_root = os.path.join(dataset_root, "sensor_blobs", split)
    cam_rgb = build_camera_rgb_grid(frame, sensor_root, max_width=camera_max_width)

    nk_a = load_feature(neck_pre)
    nk_b = load_feature(neck_ft)

    stats_nk_a = channel_stats(nk_a)
    stats_nk_b = channel_stats(nk_b)

    cols = ["mean", "max", "min", "norm"]
    titles = ["mean (C)", "max (C)", "min (C)", "L2 norm (C)"]

    fig = plt.figure(figsize=(28, 12))
    gs = gridspec.GridSpec(
        2,
        8,
        figure=fig,
        height_ratios=[2.0, 1.0],
        hspace=0.25,
        wspace=0.22,
    )

    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(cam_rgb, aspect="auto")
    ax0.set_title(
        f"6 cameras  |  {split} / {scene}  |  {token}\n"
        f"LEFT: pretrained BEV  |  RIGHT: finetuned BEV",
        fontsize=11,
        pad=6,
    )
    ax0.axis("off")

    for j, c in enumerate(cols):
        ax_l = fig.add_subplot(gs[1, j])
        ax_l.imshow(_norm01(stats_nk_a[c]), cmap="viridis", aspect="auto")
        ax_l.set_title(f"decoder_neck (pre)\n{titles[j]}", fontsize=8)
        ax_l.axis("off")

        ax_r = fig.add_subplot(gs[1, j + 4])
        ax_r.imshow(_norm01(stats_nk_b[c]), cmap="viridis", aspect="auto")
        ax_r.set_title(f"decoder_neck (finetune)\n{titles[j]}", fontsize=8)
        ax_r.axis("off")

    gs.update(left=0.02, right=0.98, top=0.97, bottom=0.03)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(
        description="Side-by-side PNG: pretrained vs finetune BEV .pt (same frames)"
    )
    p.add_argument("--pt-pretrained", type=str, required=True)
    p.add_argument("--pt-finetune", type=str, required=True)
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(_REPO_ROOT, "bev_gallery/compare_pretrained_finetune"),
    )
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--camera-max-width", type=int, default=3000)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pre_root = os.path.abspath(args.pt_pretrained)
    ft_root = os.path.abspath(args.pt_finetune)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    idx_pre = _index_pairs(pre_root)
    idx_ft = _index_pairs(ft_root)
    keys: Set[Tuple[str, str, str]] = set(idx_pre.keys()) & set(idx_ft.keys())
    pairs: List[Tuple[str, str, str]] = sorted(keys)
    if not pairs:
        raise SystemExit(
            f"No overlapping (split,scene,token) between\n  {pre_root}\n  {ft_root}"
        )
    if args.max_frames > 0:
        pairs = pairs[: args.max_frames]

    ok, fail = 0, 0
    for i, key in enumerate(pairs):
        split, scene, token = key
        nk_p = idx_pre[key]
        nk_f = idx_ft[key]
        out_name = f"compare__{split}__{scene}__{token}.png"
        out_path = os.path.join(out_dir, out_name)
        if args.skip_existing and os.path.isfile(out_path):
            ok += 1
            print(f"[{i+1}/{len(pairs)}] skip {out_name}")
            continue
        try:
            fig = render_compare(
                split,
                scene,
                token,
                nk_p,
                nk_f,
                os.path.abspath(args.dataset_root),
                args.camera_max_width,
            )
            fig.savefig(
                out_path,
                dpi=args.dpi,
                bbox_inches="tight",
                pad_inches=0.06,
                facecolor="white",
            )
            plt.close(fig)
            ok += 1
            print(f"[{i+1}/{len(pairs)}] ok {out_name}")
        except Exception as e:
            fail += 1
            print(f"[{i+1}/{len(pairs)}] FAIL {out_name}: {e}")

    print(f"Done. ok={ok} fail={fail} -> {out_dir}")


if __name__ == "__main__":
    main()
