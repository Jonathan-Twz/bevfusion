#!/usr/bin/env python3
"""Print max abs diff between matching *_vtransform.pt / *_decoder_neck.pt in two trees."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterator, List, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from tools.visualize_bev_feat import load_feature  # noqa: E402


def iter_pt_pairs(pt_root: str) -> Iterator[Tuple[str, str, str, str, str]]:
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
                vf = f"{tok}_vtransform.pt"
                if vf not in names:
                    continue
                yield (
                    split,
                    scene,
                    tok,
                    os.path.join(sdir, vf),
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


def _to_chw(x: np.ndarray) -> np.ndarray:
    if x.ndim == 4:
        x = x[0]
    return x.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pt-a", type=str, required=True, help="e.g. pretrained export root")
    p.add_argument("--pt-b", type=str, required=True, help="e.g. stage1 export root")
    args = p.parse_args()

    def index(root: str) -> Dict[Tuple[str, str, str], Tuple[str, str]]:
        d: Dict[Tuple[str, str, str], Tuple[str, str]] = {}
        for s, sc, t, vp, np_ in iter_pt_pairs(root):
            d[(s, sc, t)] = (vp, np_)
        return d

    ia, ib = index(args.pt_a), index(args.pt_b)
    keys = sorted(set(ia.keys()) & set(ib.keys()))
    if not keys:
        raise SystemExit("no overlapping keys")

    vt_max_all = 0.0
    nk_max_all = 0.0
    rows: List[str] = []
    for k in keys:
        vp_a, nk_a = ia[k]
        vp_b, nk_b = ib[k]
        a_v, a_n = _to_chw(load_feature(vp_a)), _to_chw(load_feature(nk_a))
        b_v, b_n = _to_chw(load_feature(vp_b)), _to_chw(load_feature(nk_b))
        if a_v.shape != b_v.shape:
            rows.append(f"{k} VT SHAPE {a_v.shape} vs {b_v.shape}")
            continue
        if a_n.shape != b_n.shape:
            rows.append(f"{k} NK SHAPE {a_n.shape} vs {b_n.shape}")
            continue
        dv = float(np.abs(a_v - b_v).max())
        dn = float(np.abs(a_n - b_n).max())
        vt_max_all = max(vt_max_all, dv)
        nk_max_all = max(nk_max_all, dn)
        rows.append(
            f"{k[0]}/{k[1][:24]}.. /{k[2]}  max|d_vt|={dv:.6e}  max|d_neck|={dn:.6e}"
        )

    print(f"Overlapping frames: {len(keys)}")
    print(f"GLOBAL max|vtransform_a - vtransform_b|: {vt_max_all:.6e}")
    print(f"GLOBAL max|neck_a - neck_b|: {nk_max_all:.6e}")
    print("--- per frame ---")
    for line in rows:
        print(line)


if __name__ == "__main__":
    main()
