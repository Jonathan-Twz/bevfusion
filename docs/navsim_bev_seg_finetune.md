# NAVSIM Camera-Only BEV 分割微调

## 依赖

- **训练**：请在 `bevfusion:nucarla`（或已编译 `mmcv-full` 的环境）中运行；宿主机若缺 `mmcv._ext` 无法 `import mmdet3d`。
- **地图 GT**：需安装 `nuplan-devkit`（可编辑安装），以及 Geo 栈。Docker 内 Python 3.8 示例：

```bash
pip install -e /path/to/nuplan-devkit \
  shapely==2.0.7 geopandas==0.13.2 pyogrio==0.7.2 rasterio pyproj aioboto3 retry
```

（`nuscenes-devkit` 可能提示与 `shapely>=2` 冲突，地图管线以 nuPlan 为准，可忽略该警告或仅在训练容器内安装。）

## 数据与 manifest

- 数据根：`WoTE/dataset`（含 `navsim_logs/trainval`、`sensor_blobs/trainval`、`maps/`）。
- 生成 train/val 列表：

```bash
python tools/make_navsim_split.py \
  --dataset-root ~/wm_ws/WoTE/dataset \
  --split trainval \
  --train-count 10000 --val-count 1000 \
  --train-manifest-out logs/navsim_finetune_train.json \
  --val-manifest-out logs/navsim_finetune_val.json
```

- Stage1 小数据（约 2k/200）可用 `logs/navsim_finetune_stage1_train.json` 等（由同脚本另存）。

## 地图 GT 冒烟 / 覆盖率

```bash
python tools/data_converter/navsim_bev_seg_gt.py --smoke --coverage --coverage-samples 200 \
  --dataset-root ~/wm_ws/WoTE/dataset --maps-root ~/wm_ws/WoTE/dataset/maps \
  --manifest logs/manifest_navsim_test_only.json \
  --coverage-out logs/navsim_map_class_coverage.json
```

## 预训练权重

下载 `camera-only-seg.pth` 到仓库 `pretrained/`（见主 `README` Dropbox 链接）。

## 训练命令（Docker 示例）

挂载代码、WoTE 数据集、nuplan-devkit，安装上述 pip 依赖后：

```bash
# Stage1：仅训练分割头（冻结 backbone/neck/LSS/decoder）
torchpack dist-run -np 8 python tools/train.py configs/navsim/seg/stage1_head_only.yaml

# Stage2：冻结 Swin+FPN neck，训练 LSS + decoder + head（10k/1k manifest）
torchpack dist-run -np 8 python tools/train.py configs/navsim/seg/stage2_freeze_swin.yaml

# Stage3：30k/3k manifest + 20 epoch + ImageAug3D + 预计算 GT memmap
torchpack dist-run -np 3 python tools/train.py configs/navsim/seg/stage3_full_aug.yaml
```

Stage 3 使用离线 GT 缓存（`tools/precompute_navsim_bev_gt.py` 预先把 33k 帧的 map
mask 写进单个 `logs/navsim_bev_gt_cache/stage3_masks.npy` memmap，~7.4 GB），worker
只做 mmap read，`data_time` 从 ~2s 降到 ~0.03s，解决了 nuPlan map API 每 worker
独立缓存导致的 RAM 泄漏 + 数据加载瓶颈。

## 配置说明

| 文件 | 作用 |
|------|------|
| `configs/navsim/seg/camera-bev256d2-finetune.yaml` | 通用微调基配置（6 类与 nuScenes 预训练 head 对齐） |
| `configs/navsim/seg/stage1_head_only.yaml` | Head-only，2k 小 manifest，3 epoch |
| `configs/navsim/seg/stage2_freeze_swin.yaml` | 冻结 Swin，训 LSS+decoder+head，10k/1k manifest，10 epoch |
| `configs/navsim/seg/stage3_full_aug.yaml` | 30k/3k manifest，20 epoch，开 `ImageAug3D`，`vtransform.lr_mult` 升到 0.3，`decoder.lr_mult` 升到 1.0；搭配 `LoadBEVSegmentationNavsimCached` |

## 评估与可视化

- 验证阶段会输出 `map/<class>/iou@*`（与 `NuScenesDataset.evaluate_map` 相同逻辑）。
- 仅导出指标 JSON（单卡）：`python tools/navsim_seg_eval_metrics.py <config> <checkpoint.pth> --out-json logs/navsim_finetune_metrics.json`
- BEV 中间特征对比：可用 `tools/generate_bev_features_batch.py` / `tools/batch_visualize_bev_pt_with_cameras.py` 对微调前后 checkpoint 分别导出 `.pt` 再拼图对比。

## Config 对照：`stage2_freeze_swin.yaml` vs nuScenes upstream

用于对比的 upstream 配置是 `configs/nuscenes/seg/camera-bev256d2.yaml`，经 torchpack 沿路径 merge 后的**实际生效**配置（即
`configs/default.yaml` → `configs/nuscenes/default.yaml` → `configs/nuscenes/seg/default.yaml` → `camera-bev256d2.yaml`）。模型结构两
者完全一致（Swin-Tiny → GeneralizedLSSFPN → LSSTransform → GeneralizedResNet → LSSFPN → BEVSegmentationHead，6 类），所以 pretrained
checkpoint 可以直接 `load_from` 到 NAVSIM 侧；差异都集中在**训练范式 / 数据 / 增强 / 优化器**。

### 1) 训练范式

| 维度 | nuScenes upstream | NAVSIM stage2 |
|---|---|---|
| 起点权重 | `load_from: null`（Swin 走 ImageNet，其他随机） | `load_from: pretrained/camera-only-seg.pth` |
| 任务 | **map seg + object det 6 类物体联合训练** | **只训 map seg**（`object_classes: []`） |
| 总 epoch | 20 | 10（`step=[6, 9]`） |
| Checkpoint 保留 | `max_keep_ckpts: 1` | `max_keep_ckpts: 3` |

### 2) 数据：数据集、模态、GT

| 维度 | nuScenes upstream | NAVSIM stage2 |
|---|---|---|
| Dataset 类 | `NuScenesDataset`，外套 `CBGSDataset`（类别均衡） | `NavsimBEVSegDataset`（无 CBGS） |
| 样本量（train/val） | 官方 `nuscenes_infos_{train,val}.pkl`（~28k / ~6k） | 自拆 `logs/navsim_finetune_{train,val}.json`（10k / 1k） |
| `use_lidar` | **true**（`LoadPointsFromFile` + `LoadPointsFromMultiSweeps` 9 sweeps） | **false**（`LoadEmptyPoints` 占位） |
| 物体标注 | `LoadAnnotations3D`（with_bbox_3d / with_label_3d） | ❌ 不加载 |
| 地图 GT | `LoadBEVSegmentation`（nuScenes 官方 maps） | `LoadBEVSegmentationNavsim`（nuPlan `map.gpkg` 在线栅格化） |
| Map classes | 6 类 | 同样 6 类，顺序一致 |

> nuPlan 的 `stop_line / carpark_area` 层稀疏甚至部分场景全空，和 nuScenes 不等价；这是 fine-tune 后这两类 IoU 个位数的主要数据原因。

### 3) 数据增强（这是最容易忽略的差异）

| 步骤 | nuScenes train | NAVSIM stage2 train |
|---|---|---|
| `ImageAug3D` `resize_lim` | **`[0.38, 0.55]`**（每帧随机） | `[0.48, 0.48]` |
| `ImageAug3D` `rot_lim` | `[-5.4°, +5.4°]` | `[0°, 0°]` |
| `ImageAug3D` `rand_flip` | true | false |
| `ImageAug3D` `is_train` | **true** | **false** |
| `GlobalRotScaleTrans` | scale `[0.9, 1.1]`、rot ±45°、trans 0.5 | ❌ |
| `RandomFlip3D` | ✅ | ❌ |
| `GridMask` | ✅（`prob=0` 但挂载） | ❌ |
| `PointShuffle` | ✅ | ❌（也无意义，点云是空的） |

当前 NAVSIM train pipeline 和 test pipeline **完全一致**——等于没做任何增强。

### 4) 优化器与学习率

| 维度 | nuScenes upstream | NAVSIM stage2 |
|---|---|---|
| 优化器 | AdamW, lr=1e-4, wd=0.01 | AdamW, lr=1e-4, wd=0.01 |
| `paramwise_cfg` `lr_mult` | 无（全模型同 lr） | Swin 0 / neck 0 / vtransform 0.1 / decoder 0.5 / head 1.0 |
| 特殊 decay_mult | `absolute_pos_embed`、`relative_position_bias_table` 置 0 | 同 |
| LR policy | `cyclic`（CosineAnnealing，2 周期） | `step, [6, 9], gamma=0.1` + `linear warmup 500 iter` |
| Momentum schedule | `cyclic`（AdamW β1 也跟随循环） | ❌ 固定动量 |
| `grad_clip` max_norm | 35 | 35 |
| FP16 | 继承自 `configs/default.yaml` | 本文件显式声明，参数相同 |
| Batch | `samples_per_gpu=8 × 8 GPU = 64`（CBGS 放大） | `samples_per_gpu=4 × 4 GPU = 16` |

## 可改进方向分析

下面按「可能影响 mIoU 的大小」从高到低列出。结论：**LSSTransform 和数据增强是优先改动点，Swin 基本可保持冻结**。

### A. LSSTransform 和深度预测头（最关键 —— domain gap 源头）

LSSTransform 里最"看数据"的两个模块是 `depthnet`（每像素的深度分布 softmax）和随后的 BEV splat。它们学习的是"相机成像 → 深度分布 → BEV 栅格"的几何映射。相机内参、外参、装车高度、标定方式只要和 pretrain 不同，深度分布就会漂移。

**两个数据集的实际差异：**

| | nuScenes | NAVSIM (nuPlan) |
|---|---|---|
| 相机个数 | 6 | 6（但布置不同） |
| 相机型号/畸变 | 特定 | 不同 |
| 装车高度 / 俯仰 | 特定 | 不同 |
| 典型场景距离 | 城市 / 郊区、距离分布偏近 | Pittsburgh / Boston / Las Vegas / Singapore，场景尺度差异更大 |

**建议改动：**

1. **把 `encoders.camera.vtransform` 的 `lr_mult` 从 `0.1` 提到 `0.3 ~ 0.5`**，让 depthnet 在 NAVSIM 上更充分地重学深度分布。代价极小（几 M 参数）。
2. **增加 image-level 的尺度抖动**（见 B 部分的 `resize_lim`）——这是从数据端直接攻 LSS 的办法，比调 lr 更有效。

### B. 数据增强（和 A 协同最强 —— 强烈建议开）

按预期增益排序：

1. **`resize_lim: [0.38, 0.55]`（P0，强烈推荐）**
   - nuScenes 把每张原图按随机比例缩放到 `256×704`，等价于在 2D 像素域模拟不同焦距 / FOV / 物体尺寸。
   - 对 LSS 里的 depthnet 是**最直接**的增强信号：在训练里它被迫学会"同一场景、不同像素尺度下，深度分布该怎么推"。
   - 这一条和 A.1 合起来，就是专门针对你说的 domain gap 的主武器。

2. **`GlobalRotScaleTrans` + `RandomFlip3D`（P1，建议开但要小幅度）**
   - BEV 级随机小角度旋转 + 轻微缩放 + 左右翻。模拟 ego heading / 位置噪声，直接增加 map head 对局部位姿扰动的鲁棒性。
   - 建议起始值：`scale: [0.95, 1.05]`，`rot: [-0.26, 0.26]`（±15°），`trans: 0.0` 或 `0.2`。nuScenes 原版太激进（±45°），fine-tune 不必到那么大。
   - ⚠️ `rand_flip` 需要 `ImageAug3D.rand_flip` 与 `RandomFlip3D` 同步；`LoadBEVSegmentationNavsim` 也要在 flip 时把 GT 掩码左右翻转。当前我们的实现**应该**已经通过 `img_aug_matrix` / `lidar_aug_matrix` 元数据把一致性传下去了，但第一次打开的时候建议先小规模 val 一下确认 GT 不错位。

3. **`rot_lim: [-5°, 5°]`（P2）**
   - 图像小幅 roll 扰动，增益有限，打开也无妨。

4. **`GridMask`、`PointShuffle`（不必要）**
   - GridMask 默认 `prob=0` 本来就不生效，对 seg 没什么用；PointShuffle 对空点云无意义。

### C. Swin backbone（保持冻结即可 —— 但要"真"冻结）

- 低层特征（边缘、纹理、颜色）在 ImageNet + nuScenes 已学得很充分，NAVSIM 两端的相机图像在视觉上并没有颠覆性差异。**继续冻结是合理的**。
- 现在的 `lr_mult=0` 只是把梯度步置零，**BN running_mean/var 仍在 forward 时被更新**，这就是为什么之前对比 stage1 vs pretrained 的 vtransform 输出也有几个数量级的差——BN 漂移。
- **建议加一个 ~20 行的 `FreezeModuleHook`**：在 `before_train_epoch` 给 `encoders.camera.backbone` 和 `encoders.camera.neck` 调 `.eval() + requires_grad_(False)`。顺带省 2–3GB 显存（不用存 activation 反向）。
- 如果真想"动一点 Swin"，最温和的选择是 LoRA（`r=8, α=16`，只改 attn 的 qkv/proj）。此前讨论过的方案。

### D. Decoder（GenResNet + LSSFPN）

- 作用于 BEV 特征；输入分布受 LSS 影响（我们已经观测到 `max|ΔBEV| ≈ 33`）。decoder 要跟着学新的输入分布，这是它的本职。
- 当前 `lr_mult=0.5` 偏保守，可以尝试 `1.0`（和 head 一致）。收益不大但也无害，配合 A.1 + B.1 一起改更稳。

### E. Seg head

- `lr_mult=1.0` 已经最大。唯一剩下的优化点是**损失函数层面**——见 F。

### F. 损失函数 & 类别平衡（可能是 `stop_line / ped_crossing / carpark` 的救命稻草）

当前用的是默认 **focal loss**，6 类共用同一组超参。问题：

- `carpark_area` 在 NAVSIM 的正样本像素占比 < 0.1%（冒烟时覆盖率统计过）；focal 对极稀有类依旧容易被主类淹没。
- `ped_crossing / stop_line` 像素少且细，IoU 对轻微位移非常敏感。

可尝试：

1. **per-class focal alpha**（加大稀有类权重，比如 `[1, 5, 2, 5, 5, 2]`）。
2. **加 Dice loss**（`loss: combo`，focal + dice，dice 对稀有类 IoU 最直接友好）——需要在 `BEVSegmentationHead` 里扩实现，代码改动约 30 行。
3. **class-balanced sampling / CBGS over map**：现状每个 train sample 等权重，可以按帧上是否出现稀有类分桶上采样。

### G. Batch size / epoch / schedule

- 现在 bs=4/GPU × 4 GPU = 16，和 nuScenes 上游（8×8=64）差 4×，AdamW 的 gradient 噪声更大。
- 我们当前显存只用 20GB/49GB，**bs 可以直接升到 6 或 8**。加了增强后，一个 epoch 步数会减少，正好抵消额外开销。
- 当前 `step=[6, 9]`、epoch 8 已基本收敛（8→10 mean IoU 从 21.85 到 21.90）；加了增强 + 上调 lr_mult 后，可能需要 `max_epochs=12`、`step=[8, 11]` 才能再挤出一些。

## 下一步建议优先级

| 级别 | 改动 | 预期收益 | 实现代价 |
|---|---|---|---|
| P0 | 打开 `resize_lim=[0.38, 0.55]` + `is_train: true`（ImageAug3D） | 中 ~ 大 | 只改 yaml |
| P0 | `vtransform.lr_mult: 0.1 → 0.3` | 中 | 只改 yaml |
| P1 | 加 `GlobalRotScaleTrans`（小幅度）+ `RandomFlip3D` | 小 ~ 中 | 只改 yaml，+ 第一次跑前做一次 GT 对齐可视化 |
| P1 | 加 `FreezeModuleHook`（backbone/neck 真正冻结） | 小（稳定性 + 显存） | ~20 行 hook + 注册 |
| P2 | per-class focal alpha | 小 ~ 中（针对稀有类） | 改 head 配置（若 head 支持） 或改 head 代码 |
| P2 | `samples_per_gpu: 4 → 6/8` | 小（训练更稳） | 只改 yaml |
| P3 | focal + dice 组合损失 | 小 ~ 中 | ~30 行代码 |
| P3 | `decoder.lr_mult: 0.5 → 1.0` | 极小 | 只改 yaml |

## Stage 3 实测结果（2026-04-18）

Stage 3 一次性落实了上文 A/B/D 三组 P0 改动：`ImageAug3D` 打开（`resize_lim [0.38, 0.55]` + `rot_lim ±5.4°` + `is_train: true`；`rand_flip` / `GlobalRotScaleTrans` 暂缓），`vtransform.lr_mult 0.1 → 0.3`，`decoder.{backbone,neck}.lr_mult 0.5 → 1.0`；数据从 10k/1k 扩到 30k/3k，`max_epochs` 10 → 20，全局 bs=30（3 × 10），`step=[14, 18]`，warmup 1500 iter。

### 训练情况

- 3 × H100 (GPU 1/2/3) 稳定跑满；ETA 5h10，实际约 5h15 完成 20 epoch。
- `data_time` 稳定 0.027s，`iter_time` ~0.92s，容器 RAM 稳定 ~43 GiB（对比之前不用 cached pipeline 时工作集 >200 GiB，见下文）。
- `train loss` 平滑下降：ep1 2.44 → ep20 0.23，grad_norm 后期稳定 0.15 左右。

### 验证集 IoU@max 每两个 epoch 轨迹（`logs/navsim_finetune_stage3_val.json`，3000 帧）

| ep | mean | drivable | ped_cross | walkway | stop_line | carpark | divider |
|---:|-----:|---------:|----------:|--------:|----------:|--------:|--------:|
|  2 | 0.171 | 0.557 | 0.017 | 0.133 | 0.004 | 0.005 | 0.307 |
|  6 | 0.278 | 0.663 | 0.203 | 0.233 | 0.069 | 0.142 | 0.360 |
| 10 | 0.328 | 0.705 | 0.269 | 0.292 | 0.117 | 0.202 | 0.380 |
| 14 | 0.359 | 0.727 | 0.312 | 0.321 | 0.155 | 0.244 | 0.392 |
| 18 | 0.365 | 0.733 | 0.321 | 0.330 | 0.159 | 0.251 | 0.395 |
| **20** | **0.366** | **0.734** | **0.322** | **0.330** | **0.160** | **0.252** | **0.395** |

ep16 以后 mean IoU 基本饱和（ep16→ep20 仅 +0.0015），20 epoch 选得比较合适。

### 对比前两阶段

| 指标 | Stage1 (2k, 3ep) | Stage2 (10k, 10ep) | **Stage3 (30k, 20ep)** | 提升 vs S2 |
|:--|:--|:--|:--|:--|
| mean IoU | 0.123 | 0.219 | **0.366** | **+0.147 (+67%)** |
| drivable_area | — | 0.620 | **0.734** | +0.114 |
| ped_crossing | — | 0.104 | **0.322** | **3.1×** |
| walkway | — | 0.176 | **0.330** | **1.9×** |
| stop_line | — | 0.030 | **0.160** | **5.4×** |
| carpark_area | — | 0.047 | **0.252** | **5.4×** |
| divider | — | 0.337 | **0.395** | +0.058 |

稀有类（`stop_line` / `carpark_area` / `ped_crossing`）涨幅最大，主因是数据量翻 3 倍 +
`ImageAug3D` 的尺度抖动让 LSS 的 depthnet 真正看到了足够的 domain 内分布。

### Checkpoint / 输出文件

- `runs/navsim_seg_stage3/latest.pth`（epoch_20.pth，594 MB）
- 训练日志：`runs/navsim_seg_stage3/20260418_064104.log{,.json}`
- 预计算 GT：`logs/navsim_bev_gt_cache/stage3_masks.npy` (7.4 GB) + `stage3_index.json`（33k 条，stage3 train+val 全覆盖）

### BEV 中间特征对比（pretrained / stage2 / stage3，20 帧）

复用 `.cursor/skills/bevfusion-navsim-bev-compare/SKILL.md` 中的 workflow：

```bash
python tools/generate_bev_features_batch.py \
  --manifest logs/bev_compare_stage1_vs_pre_20.json \
  --dataset-root /dataset/navsim \
  --output-root bev_gallery/compare_stage3_vs_pretrained_20/bev_quick_cmp/stage3 \
  --config configs/navsim/seg/camera-bev256d2-finetune.yaml \
  --checkpoint runs/navsim_seg_stage3/latest.pth \
  --num-gpus 1 --batch-size 4
```

数值 diff（`tools/verify_bev_pt_numeric_diff.py`，GLOBAL max abs 在 20 帧上取最大）：

| 对比 | max \|Δvtransform\| | max \|Δdecoder_neck\| |
|---|---:|---:|
| pretrained vs stage2 | 32.80 | 6.25 |
| pretrained vs **stage3** | **29.86** | **8.41** |
| stage2 vs **stage3** | 19.66 | 6.41 |

Stage3 的 vtransform 偏移比 stage2 略小一些（因为我们把 `vtransform.lr_mult` 从 0.1 涨到 0.3，depthnet 收敛得更"贴紧" NAVSIM，但起点仍来自 pretrained 初始化），decoder neck 端差异则进一步拉大（decoder 真正跟着新 BEV 分布学了）。

拼图 PNG：

- `bev_gallery/compare_stage3_vs_pretrained_20/plots/` — 20 张 `compare__*.png`（上行 6 相机 RGB；下两行 vtransform / decoder_neck 的 mean/max/min/L2 norm，左 pretrained、右 stage3）
- `bev_gallery/compare_stage3_vs_stage2_20/plots/` — 20 张同布局（左 stage2、右 stage3）

### GT vs Pred 可视化（stage3 模型，val 前 20 帧）

```bash
# 在 Docker 内（挂载 WoTE/dataset 到同名路径）
torchpack dist-run -np 1 python tools/visualize.py \
  configs/navsim/seg/stage3_viz_20.yaml --mode gt --split val \
  --out-dir bev_gallery/stage3_viz_val_20/gt

torchpack dist-run -np 1 python tools/visualize.py \
  configs/navsim/seg/stage3_viz_20.yaml --mode pred \
  --checkpoint runs/navsim_seg_stage3/latest.pth --split val \
  --out-dir bev_gallery/stage3_viz_val_20/pred
```

`stage3_viz_20.yaml` 是 `stage3_full_aug.yaml` 的轻量副本，仅把 `val/test.ann_file` 指向 `logs/bev_compare_stage1_vs_pre_20.json`，并把 `samples_per_gpu=1`、`workers_per_gpu=2`，便于和 BEV 特征对比共用同一批 token。拼图脚本内联在仓库（20 张 `bev_gallery/stage3_viz_val_20_gt_vs_pred/*.png`，左 GT、右 Pred）。

### 过程中踩到的坑

1. **`NavsimMapRasterizer` 每 worker 独立缓存 nuPlan map 层** → 18 worker × ~13 GB = 240 GB RAM，会把宿主机打进 swap 后 OOM。
   - Fix：离线预计算所有帧的 BEV GT 写进单个 uint8 memmap，新增
     `LoadBEVSegmentationNavsimCached`（`mmdet3d/datasets/pipelines/loading_navsim_bev.py`）只做 `mmap[row]` 读取，worker 内存增量 ≈ 0。
2. **`NavsimBEVSegDataset._pkl_cache` 让每 worker 重新解序列化大 pkl** → 14 GB 磁盘的 pkl 解开后 ~60 GB，workers × N 又一次拉爆。
   - Fix：dataset `__init__` 阶段把每个 pkl **load-once**，只留下每帧最小标定信息（intrinsics / transforms / paths）存进 `self._frames`，完毕后立刻 `del scene`；worker 通过 fork 共享小结构。
3. **BN running_stats 漂移（即所谓"hard freeze 问题"）**：Swin 的 `lr_mult=0` 只停了 optimizer step，BN buffer 仍在 forward 时更新。Stage3 沿用 Stage2 的软冻结，结果并未因此崩坏（甚至可能帮助 backbone BN 轻微适应 NAVSIM 图像分布）；但严格语义的 hard freeze 可以之后做消融（`frozen_stages: 4` + 一个 ~10 行 `FreezeNeckHook`）。
4. **Docker `--user "$(id -u):$(id -g)"` + `pip install -e .` 冲突**：之前留下的 root 所有的 `mmdet3d.egg-info` 目录让 user 模式无法重装；离线可视化 / 导出时改用 `PYTHONPATH=/workspace/bevfusion` 跳过 `pip install`。
5. **`nuplan` 只在栅格化时需要**，但之前在 `tools/data_converter/navsim_bev_seg_gt.py` 顶层 import，导致纯缓存环境（可视化容器）无法 `import mmdet3d.datasets`。
   - Fix：把 nuplan 相关 import 延迟到 `NavsimMapRasterizer.__init__` 里（lazy import），缓存 pipeline 完全不依赖 nuplan。
