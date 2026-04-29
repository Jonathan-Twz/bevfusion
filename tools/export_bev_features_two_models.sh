#!/usr/bin/env bash
# 顺序用两个 checkpoint 各导出一次 BEV 特征（同一份 manifest、两套输出目录）。
#   1) pretrained/camera-only-seg.pth  +  --lidar-align（nuScenes→NAVSIM 外参约定）
#   2) stage3 latest  +  不 align（微调权重已在 NAVSIM 约定下训练）
#
# 需 Docker bevfusion:nucarla；默认按顺序前台跑完第一轮再跑第二轮（--fg）。若只跑某一轮可
# 仍用 tools/export_bev_features.sh 单跑。
#
# 示例（3 万条 manifest、输出到 T5 下两子目录）:
#   MANIFEST=bev_feature/manifest_trainval_30k.json \
#   CKPT_STAGE3=runs/navsim_seg_stage3_full_aug/latest.pth \
#     ./tools/export_bev_features_two_models.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# 与单模型脚本保持一致的 data / manifest
DATASET_ROOT="${DATASET_ROOT:-/home/wenzhe/wm_ws/WoTE/dataset}"
MANIFEST="${MANIFEST:-bev_feature/manifest_trainval_30k.json}"

# 两路输出（勿共用同一根目录，否则会覆盖/混淆）
OUTPUT_ROOT_PRETRAINED="${OUTPUT_ROOT_PRETRAINED:-/media/T5/bev_feature/exports_pretrained}"
OUTPUT_ROOT_STAGE3="${OUTPUT_ROOT_STAGE3:-/media/T5/bev_feature/exports_stage3}"

CKPT_PRETRAINED="${CKPT_PRETRAINED:-pretrained/camera-only-seg.pth}"
CKPT_STAGE3="${CKPT_STAGE3:-runs/navsim_seg_stage3_aligned/latest.pth}"
# 结构一致时可用同一 config；若 stage3 与 nuScenes yaml 不兼容，再设 CONFIG_STAGE3
CONFIG_BASE="${CONFIG:-configs/nuscenes/seg/camera-bev256d2.yaml}"
CONFIG_PRETRAINED="${CONFIG_PRETRAINED:-$CONFIG_BASE}"
CONFIG_STAGE3="${CONFIG_STAGE3:-$CONFIG_BASE}"

RUN_MODE="${RUN_MODE:-both}"   # both | pretrained | stage3

if [[ -f "$REPO/$MANIFEST" ]]; then
  :
else
  echo "missing manifest: $REPO/$MANIFEST" >&2
  exit 1
fi
if [[ ! -f "$REPO/$CKPT_PRETRAINED" ]]; then
  echo "missing pretrained: $REPO/$CKPT_PRETRAINED" >&2
  exit 1
fi
if [[ "$RUN_MODE" != "pretrained" && ! -f "$REPO/$CKPT_STAGE3" ]]; then
  echo "missing stage3 checkpoint: $REPO/$CKPT_STAGE3" >&2
  echo "Set CKPT_STAGE3=... to your latest.pth" >&2
  exit 1
fi

run_one() {
  local name=$1
  case "$name" in
    pretrained)
      echo "======== (1/2) pretrained: $CKPT_PRETRAINED -> $OUTPUT_ROOT_PRETRAINED"
      CONFIG="$CONFIG_PRETRAINED" \
      CHECKPOINT="$CKPT_PRETRAINED" \
      LIDAR_ALIGN=1 \
      CONTAINER=bev_export_2m_pretrained \
      OUTPUT_ROOT="$OUTPUT_ROOT_PRETRAINED" \
      DATASET_ROOT="$DATASET_ROOT" \
      MANIFEST="$MANIFEST" \
        "$REPO/tools/export_bev_features.sh" batch --fg
      ;;
    stage3)
      echo "======== (2/2) stage3: $CKPT_STAGE3 -> $OUTPUT_ROOT_STAGE3"
      CONFIG="$CONFIG_STAGE3" \
      CHECKPOINT="$CKPT_STAGE3" \
      LIDAR_ALIGN=0 \
      CONTAINER=bev_export_2m_stage3 \
      OUTPUT_ROOT="$OUTPUT_ROOT_STAGE3" \
      DATASET_ROOT="$DATASET_ROOT" \
      MANIFEST="$MANIFEST" \
        "$REPO/tools/export_bev_features.sh" batch --fg
      ;;
  esac
}

case "$RUN_MODE" in
  both)       run_one pretrained; run_one stage3 ;;
  pretrained) run_one pretrained ;;
  stage3)     run_one stage3 ;;
  *) echo "RUN_MODE must be both|pretrained|stage3" >&2; exit 2 ;;
esac
echo "done: RUN_MODE=$RUN_MODE"
