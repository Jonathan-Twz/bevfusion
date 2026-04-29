---
name: bevfusion-navsim-bev-compare
description: >-
  Exports BEVFusion vtransform/decoder_neck .pt from NAVSIM (WoTE) for two checkpoints,
  runs numeric max-abs diff, and builds side-by-side PNGs (cameras + channel stats).
  Use in bevfusion repo when comparing pretrained/camera-only-seg.pth vs finetuned
  checkpoints, or when the user asks for BEV feature visualization, quick 20-frame
  sanity check, or pretrained vs finetune comparison.
---

# NAVSIM BEV 特征对比（pretrained vs 微调）

## 何时用

- 验证 **Stage1 只训 head** 时中间特征是否与 `pretrained/camera-only-seg.pth` 一致（或差多少）。
- 生成与 **batch_visualize** 同风格的 **拼图 PNG**（6 路相机 + vtransform/neck 的 mean/max/min/L2 norm）。
- 需要 **同一批帧**、两套权重各导出一遍 `.pt` 再对比。

## 环境

- **导出 `.pt`**：在 **`bevfusion:nucarla` Docker**（或已编译 `mmcv-full` 的 GPU 环境）里跑；宿主机缺 `mmcv._ext` 时不可用。
- **数值 diff / 拼 PNG**：一般只需 `python3` + `numpy` + `torch` + `matplotlib` + `PIL`，可在宿主机跑（读已导出的 `.pt`）。

工作目录设为 **bevfusion 仓库根**（下文 `$ROOT`）。

## 1. 准备 manifest（例如 val 前 20 条）

与 `make_navsim_split` / 训练 val 相同格式：`pkl`, `frame_idx`, `token`, `split`, `scene`。

```bash
python3 -c "
import json
src='\$ROOT/logs/navsim_finetune_stage1_val.json'
with open(src) as f: d=json.load(f)
out='\$ROOT/logs/bev_compare_stage1_vs_pre_20.json'
with open(out,'w') as f: json.dump(d[:20], f, indent=0)
print(out)
"
```

按需改 `d[:20]` 与输出路径。

## 2. 导出两套 BEV `.pt`（Docker 示例）

**同一 manifest、同一 `config`（与微调结构一致，如 navsim finetune yaml），只换 checkpoint 与 output-root。**

目录约定：`{output_root}/{split}/{scene}/{token}_vtransform.pt` 与 `{token}_decoder_neck.pt`。

```bash
# 容器内：先 pip install -e .，再执行
export CUDA_VISIBLE_DEVICES=0

python tools/generate_bev_features_batch.py \
  --manifest logs/bev_compare_stage1_vs_pre_20.json \
  --dataset-root /path/to/WoTE/dataset \
  --output-root bev_quick_cmp/pretrained \
  --config configs/navsim/seg/camera-bev256d2-finetune.yaml \
  --checkpoint pretrained/camera-only-seg.pth \
  --num-gpus 1 --batch-size 4 --num-workers 4

python tools/generate_bev_features_batch.py \
  --manifest logs/bev_compare_stage1_vs_pre_20.json \
  --dataset-root /path/to/WoTE/dataset \
  --output-root bev_quick_cmp/stage1 \
  --config configs/navsim/seg/camera-bev256d2-finetune.yaml \
  --checkpoint runs/navsim_seg_stage1/latest.pth \
  --num-gpus 1 --batch-size 4 --num-workers 4
```

挂载：代码仓库、`WoTE/dataset`，必要时大硬盘路径与 `launch_docker.sh` 一致。

## 3. 数值对比（是否逐点接近）

```bash
python tools/verify_bev_pt_numeric_diff.py \
  --pt-a bev_quick_cmp/pretrained \
  --pt-b bev_quick_cmp/stage1
```

输出：`GLOBAL max|vtransform_a-b|`、`GLOBAL max|neck_a-b|` 及每帧一行。

## 4. 拼图 PNG（左 pretrained，右 finetune）

```bash
python tools/compare_navsim_bev_pt_pretrained_finetune.py \
  --pt-pretrained bev_quick_cmp/pretrained \
  --pt-finetune bev_quick_cmp/stage1 \
  --dataset-root /path/to/WoTE/dataset \
  --out-dir bev_gallery/compare_stage1_vs_pretrained_20 \
  --max-frames 20 \
  --dpi 90
```

**输出文件命名**：`compare__{split}__{scene}__{token}.png`（ scene 名较长）。

**图面布局**：上行 6 相机 RGB；下面两行分别为 **vtransform** 与 **decoder_neck**，每行 **左四格 pretrained**（mean/max/min/norm）、**右四格 finetune**。

## 5. 相关脚本（仓库内）

| 脚本 | 作用 |
|------|------|
| `tools/generate_bev_features_batch.py` | 按 manifest 批量导出 `*_vtransform.pt` / `*_decoder_neck.pt` |
| `tools/verify_bev_pt_numeric_diff.py` | 两目录树对齐 token，打印 max 绝对误差 |
| `tools/compare_navsim_bev_pt_pretrained_finetune.py` | 读两套 `.pt` + 数据集相机，写对比 PNG |
| `tools/batch_visualize_bev_pt_with_cameras.py` | 单套 `.pt` + 相机，多面板可视化（非左右对比） |

更完整的 NAVSIM BEV 导出说明见仓库 `docs/navisim_bev_feature_generation.md`。

## 6. 注意

- **Stage1 仅训分割头** 时，若中间特征与 pretrained **差异仍大**，需排查：优化器是否对冻结层施加了 weight decay、BN 统计是否被更新等；以 `verify_bev_pt_numeric_diff` 为准。
- 两套导出必须用 **同一 `configs/navsim/seg/...yaml`**（与 checkpoint 结构一致），否则形状可能对不齐。
- **容器要设 `HOME` 为 workspace 下的可写目录**（如 `-e HOME=/workspace/bevfusion/.cache_docker`），否则 `--user` 模式下 `torch.hub` 下载 Swin 权重会 `PermissionError: '/.cache'`。
- **跳过 `pip install -e .`**：如果仓库里留下 root 所有的 `mmdet3d.egg-info` 目录，user 模式装不进去。改用 `-e PYTHONPATH=/workspace/bevfusion` 即可直接 `import mmdet3d`。
- **`nuplan` import 已延迟**：`tools/data_converter/navsim_bev_seg_gt.py` 里 nuplan 只在 `NavsimMapRasterizer.__init__` 里 lazy import，所以纯可视化容器（没装 nuplan）仍然能 `import mmdet3d.datasets`。

## 7. 多阶段对比模板（pretrained / stage2 / stage3）

已有 `bev_gallery/compare_stage1_vs_pretrained_20/bev_quick_cmp/pretrained/` 和
`bev_gallery/compare_stage2_vs_pretrained_20/bev_quick_cmp/stage2/` 两套 `.pt`，新
checkpoint（如 stage3/stage4）只需新导出一份，然后分别和前两者做 2-way 对比即可：

```bash
# 导出 stage3 的 20 帧 .pt
python tools/generate_bev_features_batch.py \
  --manifest logs/bev_compare_stage1_vs_pre_20.json \
  --dataset-root /dataset/navsim \
  --output-root bev_gallery/compare_stage3_vs_pretrained_20/bev_quick_cmp/stage3 \
  --config configs/navsim/seg/camera-bev256d2-finetune.yaml \
  --checkpoint runs/navsim_seg_stage3/latest.pth \
  --num-gpus 1 --batch-size 4

# pretrained vs stage3
python tools/compare_navsim_bev_pt_pretrained_finetune.py \
  --pt-pretrained bev_gallery/compare_stage1_vs_pretrained_20/bev_quick_cmp/pretrained \
  --pt-finetune bev_gallery/compare_stage3_vs_pretrained_20/bev_quick_cmp/stage3 \
  --dataset-root /home/wenzhe/wm_ws/WoTE/dataset \
  --out-dir bev_gallery/compare_stage3_vs_pretrained_20/plots --dpi 90

# stage2 vs stage3
python tools/compare_navsim_bev_pt_pretrained_finetune.py \
  --pt-pretrained bev_gallery/compare_stage2_vs_pretrained_20/bev_quick_cmp/stage2 \
  --pt-finetune bev_gallery/compare_stage3_vs_pretrained_20/bev_quick_cmp/stage3 \
  --dataset-root /home/wenzhe/wm_ws/WoTE/dataset \
  --out-dir bev_gallery/compare_stage3_vs_stage2_20/plots --dpi 90
```

## 8. 同批 token 的 GT vs Pred 可视化

为了让特征对比和分割图用**同一批 20 帧**，复制主 yaml，仅改 `val/test.ann_file`：

```bash
cp configs/navsim/seg/stage3_full_aug.yaml configs/navsim/seg/stage3_viz_20.yaml
sed -i 's|logs/navsim_finetune_stage3_val.json|logs/bev_compare_stage1_vs_pre_20.json|g' \
  configs/navsim/seg/stage3_viz_20.yaml
sed -i 's|samples_per_gpu: 10|samples_per_gpu: 1|' configs/navsim/seg/stage3_viz_20.yaml
sed -i 's|workers_per_gpu: 6|workers_per_gpu: 2|' configs/navsim/seg/stage3_viz_20.yaml
```

然后在 docker 里跑（注意数据集挂载到**与 `dataset_root` 完全一致**的容器路径）：

```bash
torchpack dist-run -np 1 python tools/visualize.py \
  configs/navsim/seg/stage3_viz_20.yaml --mode gt --split val \
  --out-dir bev_gallery/stage3_viz_val_20/gt

torchpack dist-run -np 1 python tools/visualize.py \
  configs/navsim/seg/stage3_viz_20.yaml --mode pred \
  --checkpoint runs/navsim_seg_stage3/latest.pth --split val \
  --out-dir bev_gallery/stage3_viz_val_20/pred
```

宿主机上用 Pillow 拼左右图：见仓库 `docs/navsim_bev_seg_finetune.md` 里的 Stage 3 章节。
