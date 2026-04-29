#!/usr/bin/env bash
# Micro-benchmark: 32 frames from MINI manifest, report wall time and peak GPU mem.
# Usage (host, from repo root; needs docker + bevfusion:nucarla):
#   ./tools/bench_bev_export.sh
set -euo pipefail
REPO="/home/wenzhe/wm_ws/bevfusion"
DATASET="/home/wenzhe/wm_ws/WoTE/dataset"
MINI="${MINI:-/tmp/bev_bench_manifest/mini32.json}"
CKPT="${CKPT:-pretrained/camera-only-seg.pth}"

if [[ ! -f "$MINI" ]]; then
  echo "Missing $MINI — run: python3 -c \"import json; m=json.load(open('...')); open('$MINI','w').write(json.dumps(m[:32]))\""
  exit 1
fi

echo "========== nvidia-smi (before) =========="
nvidia-smi --query-gpu=index,memory.total --format=csv 2>/dev/null || true

# rows: name num_gpus batch_size num_workers
CONFIGS=(
  "1g_bs08_nw04 1 8 4"
  "1g_bs10_nw04 1 10 4"
  "1g_bs12_nw04 1 12 4"
  "1g_bs16_nw04 1 16 4"
  "1g_bs08_nw08 1 8 8"
  "4g_bs08_nw04 4 8 4"
  "4g_bs10_nw04 4 10 4"
  "4g_bs12_nw04 4 12 4"
)

for row in "${CONFIGS[@]}"; do
  read -r name ng bs nw <<< "$row"
  out="/tmp/bev_bench_${name}"
  rm -rf "$out"
  mkdir -p "$out"
  log="/tmp/bev_bench_${name}.log"
  echo ""
  echo "========== $name  GPUs=$ng  batch=$bs  workers=$nw =========="
  t0=$(date +%s)
  if [[ "$ng" == "1" ]]; then
    GPU_arg=(--gpus '"device=0"')
  else
    GPU_arg=(--gpus all)
  fi
  if ! docker run --rm "${GPU_arg[@]}" --shm-size 64g \
    -v "$REPO:/workspace/bevfusion" \
    -v "$DATASET:$DATASET" \
    -v /tmp:/tmp \
    -w /workspace/bevfusion \
    bevfusion:nucarla bash -lc "set -e
pip install -q -e /workspace/bevfusion
python tools/generate_bev_features_batch.py \
  --dataset-root $DATASET \
  --manifest $MINI \
  --output-root $out \
  --checkpoint $CKPT \
  --lidar-align-to-nuscenes \
  --num-gpus $ng \
  --batch-size $bs \
  --num-workers $nw
" 2>&1 | tee "$log"; then
    t1=$(date +%s)
    echo "OK  wall_s=$((t1 - t0))"
  else
    echo "FAIL (likely OOM or error)  — see $log"
  fi
done

echo ""
echo "Done. Suggested: pick fastest row with OK; if OOM, lower batch-size."
