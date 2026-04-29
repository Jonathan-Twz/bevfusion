#!/usr/bin/env bash
# Export BEVFusion camera-only BEV features (vtransform + decoder_neck as .pt)
# from NAVSIM / OpenScene data, inside the bevfusion:nucarla docker container.
#
# Three phases, selected by the first positional arg:
#
#   manifest   scan navsim_logs/{split}/*.pkl -> write MANIFEST (CPU only)
#   trial      run ${TRIAL_N} diverse frames, write .pt + visualizations
#   batch      full multi-GPU export, auto-resume (default)
#
# All knobs are overridable via environment variables. Examples:
#
#   # 1) build manifest for test split only
#   SPLITS="test" MANIFEST=/workspace/bevfusion/logs/manifest_test.json \
#     ./tools/export_bev_features.sh manifest
#
#   # 2) 8-frame trial using stage3 checkpoint (no lidar-align flip)
#   CHECKPOINT=runs/navsim_seg_stage3_aligned/latest.pth \
#   LIDAR_ALIGN=0 TRIAL_N=8 \
#     ./tools/export_bev_features.sh trial
#
#   # 3) 全量 trainval+test BEV 特征（需数 TB 空闲盘；可续跑）
#   CHECKPOINT=pretrained/camera-only-seg.pth LIDAR_ALIGN=1 \
#   CONTAINER=bev_export_trainval_test MANIFEST=logs/manifest_navsim_trainval_and_test.json \
#   OUTPUT_ROOT=/media/T5/bev_features \
#     ./tools/export_bev_features.sh batch --no-tail
#
#   # 4) 同一 manifest 先后用 pretrained + stage3 各导一份（两套输出目录）
#   CKPT_STAGE3=runs/navsim_seg_stage3_aligned/latest.pth \
#     ./tools/export_bev_features_two_models.sh
#
# Run modes (second-style flags):
#   --fg        foreground (exec docker run --rm)
#   --no-tail   detach and return, don't tail logs
#   --dry-run   print the docker + python command, don't execute
#   --kill      remove the container and exit
# Default: detach + tail logs (Ctrl-C detaches, container keeps running).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# Phase selection
# ---------------------------------------------------------------------------
PHASE="batch"
MODE="detach_tail"
POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    manifest|trial|batch) PHASE="$arg" ;;
    --no-tail) MODE="detach" ;;
    --fg)      MODE="fg" ;;
    --dry-run) MODE="dryrun" ;;
    --kill)    MODE="kill" ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "[export_bev_features] unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Tunables (override via env)
# ---------------------------------------------------------------------------
# Model / checkpoint
CONFIG="${CONFIG:-configs/nuscenes/seg/camera-bev256d2.yaml}"
CHECKPOINT="${CHECKPOINT:-pretrained/camera-only-seg.pth}"
# LIDAR_ALIGN=1 injects an XY swap into lidar_aug_matrix.
# Required for the nuScenes pretrained ckpt when exporting on NAVSIM.
# MUST be 0 for stage2/stage3 fine-tuned checkpoints.
LIDAR_ALIGN="${LIDAR_ALIGN:-1}"

# Data (MANIFEST 必须存在)
# 全量 trainval+test: logs/manifest_navsim_trainval_and_test.json（~22.7 万帧，仅 decoder_neck 约 3.5+ TiB）
# 仅 test: logs/manifest_navsim_test_only.json
DATASET_ROOT="${DATASET_ROOT:-/home/wenzhe/wm_ws/WoTE/dataset}"
MANIFEST="${MANIFEST:-logs/manifest_navsim_trainval_and_test.json}"
SPLITS="${SPLITS:-trainval test}"   # used only by phase=manifest

# Output (decoder_neck only by default; set SAVE_VTRANSFORM=1 to also write vtransform)
OUTPUT_ROOT="${OUTPUT_ROOT:-/media/T5/bev_features}"
SAVE_VTRANSFORM="${SAVE_VTRANSFORM:-0}"
TRIAL_VIZ_DIR="${TRIAL_VIZ_DIR:-outputs/bev_trial}"
TRIAL_N="${TRIAL_N:-8}"

# 监控: 4×RTX 6000 Ada, bs=10, nw=4 时约 14GB/GPU; 见 logs/bev_export_runs/ 下采样
# Set CUDA_DEVICES explicitly (e.g. 0,1,2,3) to pin; leave unset to use 0..N-1 for all nvidia GPUs.
if [[ -n "${CUDA_DEVICES:-}" ]]; then
  N_GPU="$(echo "$CUDA_DEVICES" | tr ',' '\n' | sed '/^$/d' | wc -l)"
  N_GPU="${N_GPU//[[:space:]]/}"
else
  N_GPU="$(command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
  N_GPU="${N_GPU//[[:space:]]/}"
  if ! [[ "$N_GPU" =~ ^[0-9]+$ ]] || [[ "$N_GPU" -lt 1 ]]; then
    N_GPU=1
  fi
  if command -v python3 &>/dev/null; then
    CUDA_DEVICES="$(python3 -c "print(','.join(str(i) for i in range(int('${N_GPU}'))))")"
  else
    CUDA_DEVICES="0"
  fi
fi
NP="${NP:-$N_GPU}"
# 4×RTX 6000 Ada, tools/bench_bev_export_inner.sh: bs10≈14GB/卡, nw4 优于 nw8; 多机 shared load 时勿盲目 auto 拉满
BATCH_SIZE="${BATCH_SIZE:-10}"
# 4 = 与 bench/线上采样一致; 想自动按 CPU 核数可设 -1
NUM_WORKERS="${NUM_WORKERS:-4}"
SCAN_WORKERS="${SCAN_WORKERS:--1}"

# Docker
IMAGE="${IMAGE:-bevfusion:nucarla}"
CONTAINER="${CONTAINER:-bev_export_${PHASE}}"
# 64g 在 nw=4、prefetch=3 时足够; 需更大 dataloader 队列时改为 128g
SHM_SIZE="${SHM_SIZE:-64g}"
NUPLAN_DEVKIT="${NUPLAN_DEVKIT:-/home/wenzhe/wm_ws/DrivoR/nuplan-devkit}"

# Optional extra mount for OUTPUT_ROOT parent (skip if already under DATASET_ROOT
# or REPO_DIR). Set to empty to disable.
OUTPUT_MOUNT="${OUTPUT_MOUNT:-$(dirname "${OUTPUT_ROOT}")}"

# ---------------------------------------------------------------------------
# Kill mode
# ---------------------------------------------------------------------------
if [[ "$MODE" == "kill" ]]; then
  docker rm -f "$CONTAINER" 2>&1 || true
  exit 0
fi

# ---------------------------------------------------------------------------
# Build the python command per phase
# ---------------------------------------------------------------------------
ALIGN_FLAG=""
if [[ "$LIDAR_ALIGN" == "1" ]]; then
  ALIGN_FLAG="--lidar-align-to-nuscenes"
fi
SAVE_VT_FLAG=""
if [[ "$SAVE_VTRANSFORM" == "1" ]]; then
  SAVE_VT_FLAG="--save-vtransform"
fi

# One line each: trailing "\\" + optional empty ${SAVE_VT_FLAG} used to break argparse.
case "$PHASE" in
  manifest)
    PY_CMD="python tools/generate_bev_features_batch.py --build-manifest --dataset-root ${DATASET_ROOT} --manifest-out ${MANIFEST} --splits ${SPLITS} --scan-workers ${SCAN_WORKERS}"
    NEED_GPU=0
    ;;
  trial)
    PY_CMD="python tools/generate_bev_features_batch.py --trial ${TRIAL_N} --dataset-root ${DATASET_ROOT} --manifest ${MANIFEST} --output-root ${OUTPUT_ROOT} --trial-viz-dir ${TRIAL_VIZ_DIR} --config ${CONFIG} --checkpoint ${CHECKPOINT} --device cuda:0 ${ALIGN_FLAG} ${SAVE_VT_FLAG}"
    NEED_GPU=1
    ;;
  batch)
    PY_CMD="python tools/generate_bev_features_batch.py --dataset-root ${DATASET_ROOT} --manifest ${MANIFEST} --output-root ${OUTPUT_ROOT} --config ${CONFIG} --checkpoint ${CHECKPOINT} --num-gpus ${NP} --batch-size ${BATCH_SIZE} --num-workers ${NUM_WORKERS} ${ALIGN_FLAG} ${SAVE_VT_FLAG}"
    NEED_GPU=1
    ;;
esac

# ---------------------------------------------------------------------------
# Assemble docker run args
# ---------------------------------------------------------------------------
RUN_ARGS=(
  --name "$CONTAINER"
  --shm-size "$SHM_SIZE"
  -v "${REPO_DIR}:/workspace/bevfusion"
  -v "${DATASET_ROOT}:${DATASET_ROOT}"
  -v "${NUPLAN_DEVKIT}:${NUPLAN_DEVKIT}"
  -w /workspace/bevfusion
  -e PYTHONUNBUFFERED=1
)
if [[ "$NEED_GPU" == "1" ]]; then
  RUN_ARGS+=( --gpus "\"device=${CUDA_DEVICES}\"" )
fi
if [[ -n "$OUTPUT_MOUNT" && -d "$OUTPUT_MOUNT" ]]; then
  RUN_ARGS+=( -v "${OUTPUT_MOUNT}:${OUTPUT_MOUNT}" )
fi
RUN_ARGS+=( "$IMAGE" )

mkdir -p "${REPO_DIR}/$(dirname "${MANIFEST}")" 2>/dev/null || true
mkdir -p "${OUTPUT_ROOT}" 2>/dev/null || true
mkdir -p "${REPO_DIR}/${TRIAL_VIZ_DIR}" 2>/dev/null || true

read -r -d '' INNER_CMD <<INNER || true
set -e
pip install -q -e ${NUPLAN_DEVKIT} shapely==2.0.7 geopandas==0.13.2 pyogrio==0.7.2 rasterio pyproj aioboto3 retry
pip install -q -e /workspace/bevfusion
echo "=== export_bev_features phase=${PHASE} ==="
echo "    dataset_root = ${DATASET_ROOT}"
echo "    manifest     = ${MANIFEST}"
echo "    output_root  = ${OUTPUT_ROOT}"
echo "    config       = ${CONFIG}"
echo "    checkpoint   = ${CHECKPOINT}"
echo "    lidar_align  = ${LIDAR_ALIGN}"
echo "    save_vtransform = ${SAVE_VTRANSFORM} (1 also writes *_vtransform.pt)"
if [[ "${NEED_GPU}" == "1" ]]; then
  echo "    gpus=${CUDA_DEVICES}  n_gpu=${N_GPU}  np=${NP}  bs=${BATCH_SIZE}  workers=${NUM_WORKERS}"
fi
if [[ "$PHASE" == "manifest" ]]; then
  echo "    scan_workers = ${SCAN_WORKERS} (use -1 for auto CPU pool)"
fi
${PY_CMD}
INNER

RUN_ARGS+=( bash -lc "$INNER_CMD" )

if [[ "$MODE" == "dryrun" ]]; then
  echo "[dry-run] docker run ${RUN_ARGS[*]}"
  exit 0
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

if [[ "$MODE" == "fg" ]]; then
  exec docker run --rm "${RUN_ARGS[@]}"
fi

docker run -d "${RUN_ARGS[@]}" >/dev/null
sleep 2
STATE="$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo 'missing')"
if [[ "$STATE" == "created" ]]; then
  docker start "$CONTAINER" >/dev/null
fi

echo "[export_bev_features] phase     : $PHASE"
echo "[export_bev_features] container : $CONTAINER"
echo "[export_bev_features] manifest  : $MANIFEST"
echo "[export_bev_features] output    : $OUTPUT_ROOT"
echo "[export_bev_features] logs      : docker logs -f $CONTAINER"
echo "[export_bev_features] stop      : CONTAINER=$CONTAINER $0 --kill"

if [[ "$MODE" == "detach_tail" ]]; then
  echo "--- tailing logs (Ctrl-C to detach, container keeps running) ---"
  exec docker logs -f "$CONTAINER"
fi
