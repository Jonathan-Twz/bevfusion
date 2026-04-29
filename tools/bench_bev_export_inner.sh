#!/usr/bin/env bash
# Run inside bevfusion:nucarla (one pip install). Args: none; uses /tmp paths.
set -euo pipefail
cd /workspace/bevfusion
pip install -q -e /workspace/bevfusion
MINI=/tmp/bev_bench_manifest/mini32.json
DATASET=/home/wenzhe/wm_ws/WoTE/dataset
CKPT=pretrained/camera-only-seg.pth

bench() {
  local name=$1 ng=$2 bs=$3 nw=$4
  local out=/tmp/bev_bench2_${name}
  rm -rf "$out" && mkdir -p "$out"
  echo ""
  echo "========== ${name}  gpus=${ng} batch=${bs} workers=${nw} =========="
  local t0
  t0=$(date +%s)
  python tools/generate_bev_features_batch.py \
    --dataset-root "$DATASET" \
    --manifest "$MINI" \
    --output-root "$out" \
    --checkpoint "$CKPT" \
    --lidar-align-to-nuscenes \
    --num-gpus "$ng" \
    --batch-size "$bs" \
    --num-workers "$nw"
  echo "wall_s $(( $(date +%s) - t0 ))"
}

bench "1g_b08_nw4" 1 8 4
bench "1g_b12_nw4" 1 12 4
bench "1g_b08_nw8" 1 8 8
bench "4g_b10_nw4" 4 10 4
bench "4g_b12_nw4" 4 12 4
echo "done"
