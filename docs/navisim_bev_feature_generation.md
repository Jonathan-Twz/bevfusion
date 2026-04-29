# NAVSIM / OpenScene BEV feature generation

This document describes how to export **camera-only** BEV intermediate features from NAVSIM-style datasets using this fork’s tooling. It is the canonical reference for manifest build, Docker runs, batch multi-GPU export, outputs, and resume behavior.

For operational issues (slow runs, disk, Git, permissions), see [navisim_bev_feature_troubleshooting.md](navisim_bev_feature_troubleshooting.md).

---

## What gets exported

For each frame the pipeline writes **two** PyTorch tensor files (CPU, `torch.load`):

| File suffix | Meaning |
|-------------|---------|
| `*_vtransform.pt` | LSS / vtransform output after BEV pooling (shape roughly `(C_vt, H_bev, W_bev)`). |
| `*_decoder_neck.pt` | Output after `decoder.backbone` + `decoder.neck` (e.g. 256 channels). |

Implementation: `tools/bev_seg_inference.py` → `BEVSegmentationInference.extract_bev_features`, driven by `tools/navsim_bev_adapter.py` and batch entry `tools/generate_bev_features_batch.py`.

---

## Dataset layout (WoTE / NAVSIM)

Under **dataset root** (e.g. `WoTE/dataset` on the host, `/dataset/navsim` in Docker):

- `navsim_logs/{split}/*.pkl` — scene pickles (`split` is usually `trainval` or `test`).
- `sensor_blobs/{split}/<scene>/...` — camera images referenced by each frame.

The manifest builder only keeps frames where **all six** cameras have image files on disk. Camera names follow nuScenes-style ordering (see `NAVSIM_CAMERAS_NUSCENES_ORDER` in `tools/navsim_frame_utils.py`).

---

## Prerequisites

1. **Docker image** `bevfusion:nucarla` (build per `docker/Dockerfile` / `docker/build_docker.sh`).
2. **Checkpoint** `pretrained/camera-only-seg.pth` on disk inside the repo (or path passed via `--checkpoint`). This file is **not** stored in Git (see `.gitignore`); download from upstream BEVFusion instructions / release assets.
3. **CUDA** for trial and full batch runs. Manifest build is CPU-only (Python + `tqdm`; uses `multiprocessing` pool).
4. **Disk**: full splits are large (order of **~1.5+ TB** for both branches for ~75k test frames — plan free space on the output filesystem).

---

## Manifest format

`--build-manifest` writes a JSON **array** of objects. Each entry includes at least:

- `split`, `scene`, `frame_idx`, `token`, `pkl` (path relative to dataset root under `navsim_logs/...`).

Full run and trial modes read this list with `--manifest`.

To restrict to **test only**, build with `--splits test` or post-filter the JSON; alternatively keep a dedicated file (e.g. `logs/manifest_navsim_test_only.json`).

---

## Phase 1: Build manifest

Scans all scene pkls under chosen splits and filters frames with complete six-camera coverage.

```bash
python tools/generate_bev_features_batch.py --build-manifest \
  --dataset-root /dataset/navsim \
  --manifest-out /path/to/manifest.json \
  --splits trainval test \
  --scan-workers 32
```

- **`--scan-workers`**: size of the multiprocessing pool (default 32).
- Output path parent directories are created if missing.

---

## Phase 2: Trial (sanity check)

Runs **N** diverse frames (round-robin across scenes), writes `.pt` files under `--output-root`, and writes camera grids + BEV comparison PNGs under `--trial-viz-dir`.

```bash
python tools/generate_bev_features_batch.py --trial 8 \
  --dataset-root /dataset/navsim \
  --manifest /path/to/manifest.json \
  --output-root /media/hdd/wenzhe/bev_features \
  --trial-viz-dir /tmp/bev_trial \
  --device cuda:0
```

Use this before a long batch job to verify paths, checkpoint, and visuals.

---

## Phase 3: Multi-GPU batch export

Default output root in code: **`/media/hdd/wenzhe/bev_features`**. Override with `--output-root`.

Layout:

```text
{output_root}/{split}/{scene}/{token}_vtransform.pt
{output_root}/{split}/{scene}/{token}_decoder_neck.pt
```

**Resume**: before processing, entries whose `{token}_decoder_neck.pt` already exist are **skipped** (`filter_pending` in `generate_bev_features_batch.py`). If you need to regenerate a frame, delete **both** `.pt` files for that token (or at least the `*_decoder_neck.pt` marker).

**Important**: each saved tensor is **`clone()`’d** before `torch.save` so files are not bloated by shared storage across batch slices.

Example **inside Docker** (adjust mounts and manifest path):

```bash
docker run --gpus all --rm --shm-size 64g \
  -v /path/to/WoTE/dataset:/dataset/navsim \
  -v /path/to/bevfusion:/home/bevfusion \
  -v /path/to/manifest.json:/manifest.json:ro \
  -v /media/hdd/wenzhe:/media/hdd/wenzhe \
  bevfusion:nucarla \
  bash -lc 'cd /home/bevfusion && python tools/generate_bev_features_batch.py \
    --dataset-root /dataset/navsim \
    --manifest /manifest.json \
    --output-root /media/hdd/wenzhe/bev_features \
    --num-gpus 4 --batch-size 8 --num-workers 8'
```

Interactive shell with the same mounts: see [launch_docker.sh](../launch_docker.sh) (WoTE + bevfusion + optional HDD mount, `shm-size 64g`).

### Common flags

| Flag | Role |
|------|------|
| `--dataset-root` | NAVSIM dataset root. |
| `--manifest` | Input JSON list for trial / batch. |
| `--output-root` | Root for `{split}/{scene}/{token}_*.pt`. |
| `--config` | Default `configs/nuscenes/seg/camera-bev256d2.yaml`. |
| `--checkpoint` | Default `pretrained/camera-only-seg.pth` (relative to repo root). |
| `--num-gpus` | Capped by visible `torch.cuda.device_count()`. |
| `--batch-size` | Per-GPU batch size. |
| `--num-workers` | `DataLoader` workers **per GPU process** (4 GPUs ⇒ 4 processes each with this many workers). |

---

## Trainval BEV export (OpenScene navtrain / HDD2 layout)

OpenScene **navtrain** assets (e.g. `navtrain_history_*.tgz` from HuggingFace) are typically merged into **`trainval`** paths: scene pkls under `navsim_logs/trainval` and images under `sensor_blobs/trainval`. In the batch tool, the manifest **`split` field is `trainval`**, not the string `navtrain`.

### Dataset layout (non-standard top-level names)

If your tree looks like this (example under `/media/hdd2/wenzhe`):

- `trainval_navsim_logs/trainval/*.pkl`
- `trainval_sensor_blobs/trainval/<scene>/...`

it is equivalent to a standard root `DATASET_ROOT` with `navsim_logs/trainval` and `sensor_blobs/trainval`, only the **parent folder names** differ.

### 1) Build manifest (host or any env with Python + tqdm)

Point `--dataset-root` at a directory whose **children** are exactly `navsim_logs` and `sensor_blobs`. Two options:

**A — Standard layout** (e.g. WoTE):

```bash
python tools/generate_bev_features_batch.py --build-manifest \
  --dataset-root /path/to/WoTE/dataset \
  --manifest-out /path/to/manifest_navsim_trainval.json \
  --splits trainval \
  --scan-workers 32
```

**B — Symlink “flat” root** (when pkls and blobs live under names like `trainval_navsim_logs/`):

```bash
mkdir -p /tmp/navtrain_dataset/navsim_logs /tmp/navtrain_dataset/sensor_blobs
ln -sfn /media/hdd2/wenzhe/trainval_navsim_logs/trainval /tmp/navtrain_dataset/navsim_logs/trainval
ln -sfn /media/hdd2/wenzhe/trainval_sensor_blobs/trainval /tmp/navtrain_dataset/sensor_blobs/trainval

python tools/generate_bev_features_batch.py --build-manifest \
  --dataset-root /tmp/navtrain_dataset \
  --manifest-out /path/to/manifest_navsim_trainval.json \
  --splits trainval \
  --scan-workers 32
```

A full **trainval** manifest is on the order of **~7×10⁴** frames (exact count depends on how many frames pass the six-camera check).

### 2) Docker: mount real directories (avoid broken symlinks)

Do **not** mount only a symlinked tree whose targets point to **`/media/hdd2/...`** unless you also mount **`/media/hdd2`** into the container — symlinks will **break** inside Docker and you will get `FileNotFoundError` on `.pkl` paths.

**Recommended:** bind-mount the **actual** `trainval` folders into the standard paths under `/dataset/navsim`:

```text
HOST .../trainval_navsim_logs/trainval   -> /dataset/navsim/navsim_logs/trainval
HOST .../trainval_sensor_blobs/trainval -> /dataset/navsim/sensor_blobs/trainval
```

### 3) Multi-GPU batch (trainval)

Output goes under **`{output_root}/trainval/{scene}/`** next to any existing **`test/`** split.

```bash
docker run -d --gpus all --shm-size 64g \
  -v /media/hdd2/wenzhe/trainval_navsim_logs/trainval:/dataset/navsim/navsim_logs/trainval:ro \
  -v /media/hdd2/wenzhe/trainval_sensor_blobs/trainval:/dataset/navsim/sensor_blobs/trainval:ro \
  -v /path/to/bevfusion:/home/bevfusion \
  -v /path/to/manifest_navsim_trainval.json:/manifest_trainval.json:ro \
  -v /media/hdd/wenzhe:/media/hdd/wenzhe \
  bevfusion:nucarla \
  bash -lc 'cd /home/bevfusion && python tools/generate_bev_features_batch.py \
    --dataset-root /dataset/navsim \
    --manifest /manifest_trainval.json \
    --output-root /media/hdd/wenzhe/bev_features \
    --num-gpus 4 --batch-size 8 --num-workers 4'
```

- Adjust host paths (`/media/hdd2/wenzhe/...`, `bevfusion`, manifest) to your machine.
- **`--num-workers`**: start with **4** if you previously saw `DataLoader worker ... Killed` (host **OOM**); try **8** only if RAM is comfortable. Total RAM pressure scales roughly with **(number of GPUs) × (`num_workers` + 1)** per process group.

### 4) Resume

Same command as above. The script skips frames that already have **`{token}_decoder_neck.pt`** under the output tree. To force a redo for specific tokens, delete both `*_vtransform.pt` and `*_decoder_neck.pt` for those tokens (or at least `*_decoder_neck.pt`).

### 5) Disk and time

- Trainval + test BEV tensors together can reach **multiple terabytes** on the output volume; keep **hundreds of GB to 1+ TB free** before starting long runs.
- Wall time is similar in order of magnitude to the **test** split full pass when using the same GPUs and batch settings; trainval frame counts are typically **larger** than test.

### 6) Optional: downloading / extracting `navtrain_history_*.tgz`

Fetching and unpacking archives is **outside** this script; use parallel download tools and **`pigz`** for faster `.tgz` extraction where appropriate. See discussion in [navisim_bev_feature_troubleshooting.md](navisim_bev_feature_troubleshooting.md) or your own `download_navtrain.sh`.

---

## Single-frame debugging

```bash
python tools/generate_bev_features.py \
  --dataset-root /dataset/navsim \
  --pkl navsim_logs/test/<scene>.pkl \
  --frame-index 0 \
  --output-prefix /tmp/bev_out/frame000
```

Produces `/tmp/bev_out/frame000_vtransform.pt` and `frame000_decoder_neck.pt`.

---

## Visualizing `.pt` features

```bash
python tools/visualize_bev_feat.py /path/to/token_decoder_neck.pt -o view.png --cmap viridis
```

---

## Related source files

| File | Role |
|------|------|
| `tools/generate_bev_features_batch.py` | Manifest, trial, multi-GPU worker, resume. |
| `tools/generate_bev_features.py` | Single-frame export. |
| `tools/navsim_bev_adapter.py` | NAVSIM → model inputs, `BEVFeatureExtractor`, collate / preprocess. |
| `tools/navsim_frame_utils.py` | Camera list, lightweight image existence checks (no OpenCV required for manifest scan). |
| `tools/bev_seg_inference.py` | Model load, `extract_bev_features`. |
| `tools/visualize_bev_feat.py` | Heatmap PNG from a saved tensor. |

---

## Quick reference checklist

1. Dataset has `navsim_logs/{split}/*.pkl` and `sensor_blobs/{split}/...`.
2. Place `pretrained/camera-only-seg.pth` locally (not in Git).
3. Build manifest → optional `--trial` → full batch with HDD (or fast disk) mounted at `--output-root`.
4. **Trainval on a second disk:** use direct Docker `-v` binds to `.../navsim_logs/trainval` and `.../sensor_blobs/trainval` (see [Trainval BEV export](#trainval-bev-export-openscene-navtrain--hdd2-layout)); avoid symlink-only mounts that point outside mounted volumes.
5. Monitor with `docker logs` or tqdm in terminal; ensure **terabytes** free on output volume for large splits.
6. To fix bad exports: remove oversized or wrong `*_decoder_neck.pt` / `*_vtransform.pt` for affected tokens, then rerun (resume fills gaps).

---

## See also

- README “NAVSIM / OpenScene batch BEV features” section for a shorter copy-paste block.
- [navisim_bev_feature_troubleshooting.md](navisim_bev_feature_troubleshooting.md) for bottlenecks, GitHub file limits, and Docker file ownership.
