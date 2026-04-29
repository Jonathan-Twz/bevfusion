#!/usr/bin/env bash
# While a bevfusion export container runs, log host + GPU + docker stats to a file.
# Usage:
#   ./tools/monitor_bev_export.sh bev_export_batch
#   INTERVAL=5 SAMPLES=36 ./tools/monitor_bev_export.sh bev_export_batch
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:-bev_export_batch}"
OUT="${OUT:-${REPO}/logs/bev_export_runs/monitor_$(date +%Y%m%d_%H%M%S).log}"
INTERVAL="${INTERVAL:-5}"
SAMPLES="${SAMPLES:-120}"
mkdir -p "$(dirname "$OUT")"
{
  echo "=== $(date -Iseconds)  container=$NAME  interval=${INTERVAL}s  samples=${SAMPLES} ==="
  nproc
  free -h | head -3
  for i in $(seq 1 "$SAMPLES"); do
    echo
    echo "----- $i @ $(date -Iseconds) -----"
    uptime
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv 2>/dev/null || true
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" "$NAME" 2>/dev/null || echo "(no such container: $NAME)"
    sleep "$INTERVAL"
  done
} | tee "$OUT"
echo "Wrote $OUT"
