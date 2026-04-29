#!/usr/bin/env python3
"""
Run validation on NavsimBEVSegDataset and write map IoU metrics to JSON.

Usage (inside Docker with mmcv + nuPlan deps, same as tools/test.py):

  torchpack dist-run -np 1 python tools/navsim_seg_eval_metrics.py \\
    configs/navsim/seg/stage2_freeze_swin.yaml \\
    work_dirs/your_exp/latest.pth \\
    --out-json logs/navsim_finetune_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torchpack.utils.config import configs

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mmdet3d.apis import single_gpu_test  # noqa: E402
from mmdet3d.datasets import build_dataloader, build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from mmdet3d.utils import recursive_eval  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument(
        "--out-json",
        type=str,
        default="logs/navsim_finetune_metrics.json",
        help="Where to save metric dict (rank 0 only in distributed)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model.cuda()
    outputs = single_gpu_test(model, data_loader)

    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ("interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"):
        eval_kwargs.pop(key, None)
    metrics = dataset.evaluate(outputs, **eval_kwargs)

    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
