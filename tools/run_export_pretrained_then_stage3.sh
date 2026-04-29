#!/usr/bin/env bash
# 先 detached 起 pretrained 容器，docker wait 结束后起 stage3（3 万帧、各一套输出目录）。
# 整段可 nohup 到 logs。
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
LOG="${LOG:-$REPO/logs/bev_export_runs/chain_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

export MANIFEST="${MANIFEST:-bev_feature/manifest_trainval_30k.json}"
export DATASET_ROOT="${DATASET_ROOT:-/home/wenzhe/wm_ws/WoTE/dataset}"

echo "======== $(date -Iseconds) 1/2 pretrained ========="
docker rm -f bev_export_2m_pretrained 2>/dev/null || true
export CONTAINER=bev_export_2m_pretrained
export OUTPUT_ROOT="${OUTPUT_ROOT_PRETRAINED:-/media/T5/bev_feature/exports_pretrained}"
export CHECKPOINT="${CKPT_PRETRAINED:-pretrained/camera-only-seg.pth}"
export LIDAR_ALIGN=1
./tools/export_bev_features.sh batch --no-tail
echo "Waiting for $CONTAINER ..."
ec=$(docker wait "$CONTAINER" 2>/dev/null || echo 1)
echo "pretrained container exit: $ec"

echo "======== $(date -Iseconds) 2/2 stage3 ========="
docker rm -f bev_export_2m_stage3 2>/dev/null || true
export CONTAINER=bev_export_2m_stage3
export OUTPUT_ROOT="${OUTPUT_ROOT_STAGE3:-/media/T5/bev_feature/exports_stage3}"
export CHECKPOINT="${CKPT_STAGE3:-runs/navsim_seg_stage3_aligned/latest.pth}"
export LIDAR_ALIGN=0
./tools/export_bev_features.sh batch --no-tail
echo "Waiting for $CONTAINER (stage3) ..."
ec2=$(docker wait bev_export_2m_stage3 2>/dev/null || echo 1)
echo "stage3 container exit: $ec2"
echo "All two exports finished at $(date -Iseconds). Log: $LOG"
