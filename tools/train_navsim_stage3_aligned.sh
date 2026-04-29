#!/usr/bin/env bash
# Launch the Stage-3 "aligned" (lidar_aug = R_swap_xy) NAVSIM BEV-seg
# fine-tune inside a disposable docker container.
#
# Defaults:
#   * config     configs/navsim/seg/stage3_full_aug_aligned.yaml
#   * run-dir    runs/navsim_seg_stage3_aligned
#   * GPUs       0,1,2,3     (torchpack dist-run -np 4)
#   * container  navsim_seg_stage3_aligned
#   * wandb      project=bevfusion-navsim-seg  (set WANDB_API_KEY to log
#                online; otherwise falls back to WANDB_MODE=offline and
#                writes a local ``wandb/`` run you can sync later.)
#
# Override via env vars, e.g.:
#   CUDA_DEVICES=1,2,3 NP=3 WANDB_RUN_NAME=stage3_aligned_r2 \
#     ./tools/train_navsim_stage3_aligned.sh
#
# Usage:
#   ./tools/train_navsim_stage3_aligned.sh             # detached + tail logs
#   ./tools/train_navsim_stage3_aligned.sh --no-tail   # detach, return
#   ./tools/train_navsim_stage3_aligned.sh --fg        # run in foreground
#   ./tools/train_navsim_stage3_aligned.sh --dry-run   # print cmd, don't run
#   ./tools/train_navsim_stage3_aligned.sh --kill      # kill + rm container
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${CFG:-configs/navsim/seg/stage3_full_aug_aligned.yaml}"
RUN_DIR="${RUN_DIR:-runs/navsim_seg_stage3_aligned}"
CONTAINER="${CONTAINER:-navsim_seg_stage3_aligned}"
IMAGE="${IMAGE:-bevfusion:nucarla}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NP="${NP:-4}"
SHM_SIZE="${SHM_SIZE:-32g}"

DATASET_ROOT="${DATASET_ROOT:-/home/wenzhe/wm_ws/WoTE/dataset}"
NUPLAN_DEVKIT="${NUPLAN_DEVKIT:-/home/wenzhe/wm_ws/DrivoR/nuplan-devkit}"

WANDB_PROJECT="${WANDB_PROJECT:-bevfusion-navsim-seg}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-stage3_aligned_$(date +%Y%m%d_%H%M%S)}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

MODE="detach_tail"     # detach_tail | detach | fg | dryrun | kill
for arg in "$@"; do
  case "$arg" in
    --no-tail) MODE="detach" ;;
    --fg)      MODE="fg" ;;
    --dry-run) MODE="dryrun" ;;
    --kill)    MODE="kill" ;;
    -h|--help)
      sed -n '2,26p' "$0"; exit 0 ;;
    *)
      echo "[train_navsim_stage3_aligned] unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$MODE" == "kill" ]]; then
  docker rm -f "$CONTAINER" 2>&1 || true
  nvidia-smi --query-gpu=index,memory.used --format=csv | head -6 || true
  exit 0
fi

# Pick wandb online vs offline based on whether an API key is visible to the
# container. We forward *either* WANDB_API_KEY or the user's ~/.netrc if
# present; otherwise the run is created in offline mode.
WANDB_ENV_ARGS=(
  -e "WANDB_PROJECT=${WANDB_PROJECT}"
  -e "WANDB_NAME=${WANDB_RUN_NAME}"
  # WandbLoggerHook writes to /workspace/bevfusion/wandb inside the container,
  # which maps back to ./wandb on the host.
  -e "WANDB_DIR=/workspace/bevfusion/wandb"
)
if [[ -n "$WANDB_API_KEY" ]]; then
  WANDB_ENV_ARGS+=(-e "WANDB_API_KEY=${WANDB_API_KEY}")
elif [[ -f "$HOME/.netrc" ]] && grep -q "api.wandb.ai" "$HOME/.netrc"; then
  WANDB_ENV_ARGS+=(-v "$HOME/.netrc:/root/.netrc:ro")
else
  echo "[train_navsim_stage3_aligned] no WANDB_API_KEY / ~/.netrc → running in offline mode" >&2
  WANDB_ENV_ARGS+=(-e "WANDB_MODE=offline")
fi

mkdir -p "${REPO_DIR}/${RUN_DIR}" "${REPO_DIR}/wandb"

# The in-container bash script. Keep it as a heredoc so arg quoting stays
# sane even if paths contain spaces.
read -r -d '' INNER_CMD <<INNER || true
set -e
pip install -q -e ${NUPLAN_DEVKIT} shapely==2.0.7 geopandas==0.13.2 pyogrio==0.7.2 rasterio pyproj aioboto3 retry
pip install -q -e /workspace/bevfusion
pip install -q wandb
echo "=== Stage3 aligned (R_swap_xy lidar_aug): np=${NP}, global bs = ${NP}*samples_per_gpu ==="
echo "=== wandb project=${WANDB_PROJECT} name=${WANDB_RUN_NAME} ==="
torchpack dist-run -np ${NP} python tools/train.py \\
  ${CFG} \\
  --run-dir ${RUN_DIR}
INNER

RUN_ARGS=(
  --name "$CONTAINER"
  --gpus "\"device=${CUDA_DEVICES}\""
  --shm-size "$SHM_SIZE"
  -v "${REPO_DIR}:/workspace/bevfusion"
  -v "${DATASET_ROOT}:${DATASET_ROOT}"
  -v "${NUPLAN_DEVKIT}:${NUPLAN_DEVKIT}"
  -w /workspace/bevfusion
  -e PYTHONUNBUFFERED=1
  "${WANDB_ENV_ARGS[@]}"
  "$IMAGE"
  bash -lc "$INNER_CMD"
)

if [[ "$MODE" == "dryrun" ]]; then
  echo "[dry-run] docker run ${RUN_ARGS[*]}"
  exit 0
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

if [[ "$MODE" == "fg" ]]; then
  exec docker run --rm "${RUN_ARGS[@]}"
fi

docker run -d "${RUN_ARGS[@]}" >/dev/null
# ``docker run -d`` occasionally returns before the container is actually
# "running" (containerd races); give it a beat, then ``docker start`` if the
# state is still ``Created``.
sleep 3
STATE="$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo 'missing')"
if [[ "$STATE" == "created" ]]; then
  echo "[train_navsim_stage3_aligned] container in 'Created' state, starting explicitly..." >&2
  docker start "$CONTAINER" >/dev/null
fi

echo "[train_navsim_stage3_aligned] container: $CONTAINER"
echo "[train_navsim_stage3_aligned] run dir  : ${RUN_DIR}"
echo "[train_navsim_stage3_aligned] wandb    : project=${WANDB_PROJECT} name=${WANDB_RUN_NAME}"
echo "[train_navsim_stage3_aligned] logs     : docker logs -f $CONTAINER"
echo "[train_navsim_stage3_aligned] stop     : $0 --kill"

if [[ "$MODE" == "detach_tail" ]]; then
  echo "--- tailing logs (Ctrl-C to detach, container keeps running) ---"
  exec docker logs -f "$CONTAINER"
fi
