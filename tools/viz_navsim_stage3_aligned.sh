#!/usr/bin/env bash
# One-shot visualization pipeline for a NAVSIM Stage-3 "aligned"
# (lidar_aug = R_swap_xy) checkpoint on the fixed 20-frame benchmark manifest
# (``logs/bev_compare_stage1_vs_pre_20.json``):
#
#   (1) BEV intermediate features (.pt)  via tools/generate_bev_features_batch.py
#   (2) BEV seg prediction PNGs          via tools/visualize.py
#   (3) GT | Pred side-by-side stitch    via inline python3 + PIL
#   (4) pretrained-aligned vs this CKPT  feature compare plots
#                                        via tools/compare_navsim_bev_pt_pretrained_finetune.py
#
# Outputs go under ``${OUT_ROOT}``:
#   ${OUT_ROOT}/features/trainval/<scene>/<token>_{vtransform,decoder_neck}.pt
#   ${OUT_ROOT}/pred/{camera-*,lidar,map}/<token>.png
#   ${OUT_ROOT}/gt_vs_pred/<token>.png
#   ${OUT_ROOT}/feature_plots/compare__<split>__<scene>__<token>.png
#
# Defaults assume the checkpoint was trained with
# ``lidar_align_to_nuscenes=True`` (i.e. Stage-3 "aligned" branch). For a
# *legacy* Stage-3 checkpoint, unset the flag with ``LIDAR_ALIGN=0`` and swap
# ``VIZ_CONFIG`` to the identity-lidar_aug variant.
#
# Common overrides (env vars):
#   CKPT          checkpoint .pth to visualize
#                 (default: runs/navsim_seg_stage3_aligned/latest.pth)
#   OUT_TAG       output folder suffix, final dir is
#                 bev_gallery/stage3_aligned_viz_val_20_${OUT_TAG}
#                 (default: from CKPT basename without .pth, e.g. ``latest`` or ``epoch_5``)
#   OUT_ROOT      full output root; overrides OUT_TAG
#                 (default: bev_gallery/stage3_aligned_viz_val_20_${OUT_TAG})
#   MANIFEST      20-frame JSON manifest
#                 (default: logs/bev_compare_stage1_vs_pre_20.json)
#   VIZ_CONFIG    yaml for tools/visualize.py (must have lidar_align_to_nuscenes
#                 matching the CKPT; default already does)
#                 (default: configs/navsim/seg/stage3_viz_20_pretrained_aligned.yaml)
#   FEAT_CONFIG   yaml for tools/generate_bev_features_batch.py (model arch only)
#                 (default: configs/nuscenes/seg/camera-bev256d2.yaml)
#   GT_MAP_DIR    existing GT BEV seg PNGs to stitch against
#                 (default: bev_gallery/stage3_viz_val_20/gt/map)
#   REF_FEAT_DIR  reference BEV .pt root for the "pre" column of the compare
#                 plot (typically pretrained-aligned, but can be any stage)
#                 (default: bev_gallery/compare_stage3_vs_pretrained_aligned_20/bev_quick_cmp/pretrained_aligned)
#   LIDAR_ALIGN   pass --lidar-align-to-nuscenes to feature extractor (1/0)
#                 (default: 1  — matches aligned CKPTs; set to 0 for legacy)
#   CUDA_DEVICES  GPU indices the viz container is allowed to use
#                 (default: 0   — single GPU is plenty for 20-frame inference)
#   CONTAINER     viz container name
#                 (default: navsim_seg_aligned_viz)
#   IMAGE         docker image
#                 (default: bevfusion:nucarla)
#   BATCH_SIZE    feature extractor batch
#                 (default: 4)
#   SKIP_FEATURES set to 1 to skip step (1) + (4)
#   SKIP_PRED     set to 1 to skip step (2) + (3)
#   FORCE         set to 1 to delete existing ${OUT_ROOT} before running
#
# Usage:
#   ./tools/viz_navsim_stage3_aligned.sh                    # full pipeline on latest.pth
#   CKPT=runs/.../epoch_10.pth ./tools/viz_navsim_stage3_aligned.sh
#   OUT_TAG=ep10_preview ./tools/viz_navsim_stage3_aligned.sh
#   SKIP_FEATURES=1 ./tools/viz_navsim_stage3_aligned.sh    # only seg pred + stitch
#   ./tools/viz_navsim_stage3_aligned.sh --kill             # kill + rm viz container
#
# IMPORTANT: by design this script opens a second GPU-enabled container. If a
# training run is also using the same GPU, the training will almost certainly
# OOM — explicitly set CUDA_DEVICES to an idle GPU, or stop training first.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${CKPT:-runs/navsim_seg_stage3_aligned/latest.pth}"
# Derive a default OUT_TAG from the checkpoint filename (``epoch_5`` / ``latest``).
_ckpt_base="$(basename "${CKPT}" .pth)"
OUT_TAG="${OUT_TAG:-${_ckpt_base}}"
OUT_ROOT="${OUT_ROOT:-bev_gallery/stage3_aligned_viz_val_20_${OUT_TAG}}"

MANIFEST="${MANIFEST:-logs/bev_compare_stage1_vs_pre_20.json}"
VIZ_CONFIG="${VIZ_CONFIG:-configs/navsim/seg/stage3_viz_20_pretrained_aligned.yaml}"
FEAT_CONFIG="${FEAT_CONFIG:-configs/nuscenes/seg/camera-bev256d2.yaml}"
GT_MAP_DIR="${GT_MAP_DIR:-bev_gallery/stage3_viz_val_20/gt/map}"
REF_FEAT_DIR="${REF_FEAT_DIR:-bev_gallery/compare_stage3_vs_pretrained_aligned_20/bev_quick_cmp/pretrained_aligned}"

LIDAR_ALIGN="${LIDAR_ALIGN:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
CONTAINER="${CONTAINER:-navsim_seg_aligned_viz}"
IMAGE="${IMAGE:-bevfusion:nucarla}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SHM_SIZE="${SHM_SIZE:-16g}"

DATASET_ROOT="${DATASET_ROOT:-/home/wenzhe/wm_ws/WoTE/dataset}"
NUPLAN_DEVKIT="${NUPLAN_DEVKIT:-/home/wenzhe/wm_ws/DrivoR/nuplan-devkit}"

SKIP_FEATURES="${SKIP_FEATURES:-0}"
SKIP_PRED="${SKIP_PRED:-0}"
FORCE="${FORCE:-0}"

MODE="run"
for arg in "$@"; do
  case "$arg" in
    --kill)     MODE="kill" ;;
    --dry-run)  MODE="dryrun" ;;
    -h|--help)  sed -n '2,70p' "$0"; exit 0 ;;
    *)
      echo "[viz_navsim_stage3_aligned] unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$MODE" == "kill" ]]; then
  docker rm -f "$CONTAINER" 2>&1 || true
  exit 0
fi

# Resolve paths relative to REPO_DIR so the script is runnable from anywhere.
_abs_or_rel() {
  # Absolute -> return as-is. Relative -> prepend REPO_DIR.
  local p="$1"
  if [[ "$p" = /* ]]; then echo "$p"; else echo "${REPO_DIR}/$p"; fi
}
CKPT_ABS="$(_abs_or_rel "$CKPT")"
MANIFEST_ABS="$(_abs_or_rel "$MANIFEST")"
GT_MAP_DIR_ABS="$(_abs_or_rel "$GT_MAP_DIR")"
REF_FEAT_DIR_ABS="$(_abs_or_rel "$REF_FEAT_DIR")"
OUT_ROOT_ABS="$(_abs_or_rel "$OUT_ROOT")"

# Fail fast with readable errors rather than deep inside docker.
if [[ ! -f "$CKPT_ABS" ]]; then
  echo "[viz_navsim_stage3_aligned] ERROR: CKPT not found: $CKPT_ABS" >&2; exit 1
fi
if [[ ! -f "$MANIFEST_ABS" ]]; then
  echo "[viz_navsim_stage3_aligned] ERROR: MANIFEST not found: $MANIFEST_ABS" >&2; exit 1
fi
if [[ "$SKIP_PRED" != "1" && ! -d "$GT_MAP_DIR_ABS" ]]; then
  echo "[viz_navsim_stage3_aligned] WARN : GT_MAP_DIR not found, will skip stitch: $GT_MAP_DIR_ABS" >&2
fi
if [[ "$SKIP_FEATURES" != "1" && ! -d "$REF_FEAT_DIR_ABS" ]]; then
  echo "[viz_navsim_stage3_aligned] WARN : REF_FEAT_DIR not found, will skip feature_plots: $REF_FEAT_DIR_ABS" >&2
fi

# Warn (but don't block) if the same GPU is currently busy with training.
if command -v nvidia-smi >/dev/null 2>&1; then
  IFS=',' read -ra _gpus <<< "$CUDA_DEVICES"
  for gi in "${_gpus[@]}"; do
    used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gi" 2>/dev/null || echo 0)
    if [[ "$used_mb" =~ ^[0-9]+$ ]] && (( used_mb > 4096 )); then
      echo "[viz_navsim_stage3_aligned] WARN : GPU ${gi} already using ${used_mb} MiB." >&2
      echo "       Running viz here will likely OOM a concurrent training. Stop training or pick a free GPU." >&2
    fi
  done
fi

if [[ "$FORCE" == "1" && -d "$OUT_ROOT_ABS" ]]; then
  echo "[viz_navsim_stage3_aligned] FORCE=1, removing existing $OUT_ROOT_ABS"
  rm -rf "$OUT_ROOT_ABS"
fi
mkdir -p "$OUT_ROOT_ABS"/{features,pred,gt_vs_pred,feature_plots}

LIDAR_FLAG=""
if [[ "$LIDAR_ALIGN" == "1" ]]; then
  LIDAR_FLAG="--lidar-align-to-nuscenes"
fi

echo "[viz_navsim_stage3_aligned] ckpt     : $CKPT_ABS"
echo "[viz_navsim_stage3_aligned] out      : $OUT_ROOT_ABS"
echo "[viz_navsim_stage3_aligned] manifest : $MANIFEST_ABS ($(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$MANIFEST_ABS" 2>/dev/null || echo '?') frames)"
echo "[viz_navsim_stage3_aligned] GPU      : ${CUDA_DEVICES}  lidar_align=${LIDAR_ALIGN}"

if [[ "$MODE" == "dryrun" ]]; then
  echo "[dry-run] skipping docker / python execution"
  exit 0
fi

# ---------------------------------------------------------------------------
# (1) + (2): single docker container does BOTH feature extraction and seg viz.
# Mounting REPO_DIR read-write so outputs land in ${OUT_ROOT_ABS}.
# ---------------------------------------------------------------------------
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# Build the inside-container command conditionally. Heredoc avoids quoting hell.
INNER_FEATURES=""
if [[ "$SKIP_FEATURES" != "1" ]]; then
  INNER_FEATURES="
echo '=== (1) BEV features .pt ==='
torchpack dist-run -np 1 python tools/generate_bev_features_batch.py \\
  --manifest ${MANIFEST} \\
  --dataset-root ${DATASET_ROOT} \\
  --output-root ${OUT_ROOT}/features \\
  --config ${FEAT_CONFIG} \\
  --checkpoint ${CKPT} \\
  --num-gpus 1 --batch-size ${BATCH_SIZE} --num-workers 4 \\
  ${LIDAR_FLAG}
"
fi

INNER_PRED=""
if [[ "$SKIP_PRED" != "1" ]]; then
  INNER_PRED="
echo '=== (2) BEV seg predictions PNG ==='
torchpack dist-run -np 1 python tools/visualize.py \\
  ${VIZ_CONFIG} \\
  --mode pred \\
  --checkpoint ${CKPT} \\
  --split val \\
  --out-dir ${OUT_ROOT}/pred
"
fi

read -r -d '' INNER_CMD <<INNER || true
set -e
pip install -q -e ${NUPLAN_DEVKIT} shapely==2.0.7 geopandas==0.13.2 pyogrio==0.7.2 rasterio pyproj aioboto3 retry >/dev/null 2>&1
pip install -q -e /workspace/bevfusion >/dev/null 2>&1
${INNER_FEATURES}
${INNER_PRED}
echo '=== done (inside container) ==='
INNER

if [[ -n "${INNER_FEATURES}${INNER_PRED}" ]]; then
  docker run --rm --name "$CONTAINER" \
    --gpus "\"device=${CUDA_DEVICES}\"" \
    --shm-size "$SHM_SIZE" \
    -v "${REPO_DIR}:/workspace/bevfusion" \
    -v "${DATASET_ROOT}:${DATASET_ROOT}" \
    -v "${NUPLAN_DEVKIT}:${NUPLAN_DEVKIT}" \
    -w /workspace/bevfusion \
    -e PYTHONUNBUFFERED=1 \
    "$IMAGE" \
    bash -lc "$INNER_CMD"
else
  echo "[viz_navsim_stage3_aligned] nothing to do in docker (both SKIP_* set)"
fi

# ---------------------------------------------------------------------------
# (3) Stitch GT | Pred side-by-side — host python3 + PIL, no docker / GPU.
# ---------------------------------------------------------------------------
if [[ "$SKIP_PRED" != "1" && -d "$GT_MAP_DIR_ABS" ]]; then
  echo "=== (3) stitch GT | Pred ==="
  GT_MAP_DIR="$GT_MAP_DIR_ABS" \
  PRED_MAP_DIR="$OUT_ROOT_ABS/pred/map" \
  OUT_DIR="$OUT_ROOT_ABS/gt_vs_pred" \
  PRED_LABEL="Pred ($(basename "$OUT_TAG"))" \
  python3 - <<'PYEOF'
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

gt_dir = Path(os.environ["GT_MAP_DIR"])
pred_dir = Path(os.environ["PRED_MAP_DIR"])
out_dir = Path(os.environ["OUT_DIR"])
pred_label = os.environ.get("PRED_LABEL", "Pred")
out_dir.mkdir(parents=True, exist_ok=True)

pairs = sorted([f for f in gt_dir.glob("*.png") if (pred_dir / f.name).exists()])
if not pairs:
    print(f"[stitch] no matched pairs between {gt_dir} and {pred_dir}")
    raise SystemExit(0)
print(f"[stitch] {len(pairs)} matched pairs -> {out_dir}")

pad, label_h = 12, 40
try:
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
    )
except Exception:
    font = ImageFont.load_default()

for gt_path in pairs:
    pred_path = pred_dir / gt_path.name
    gt = Image.open(gt_path).convert("RGB")
    pr = Image.open(pred_path).convert("RGB")
    h = max(gt.height, pr.height)
    if gt.height != h:
        gt = gt.resize((int(gt.width * h / gt.height), h))
    if pr.height != h:
        pr = pr.resize((int(pr.width * h / pr.height), h))
    canvas = Image.new("RGB", (gt.width + pr.width + pad, h + label_h), (255, 255, 255))
    canvas.paste(gt, (0, label_h))
    canvas.paste(pr, (gt.width + pad, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "GT", fill=(0, 0, 0), font=font)
    draw.text((gt.width + pad + 10, 8), pred_label, fill=(0, 0, 0), font=font)
    canvas.save(out_dir / gt_path.name)
PYEOF
else
  echo "[viz_navsim_stage3_aligned] skip stitch (SKIP_PRED or GT_MAP_DIR missing)"
fi

# ---------------------------------------------------------------------------
# (4) Feature compare plots. Uses docker for a clean matplotlib env; no GPU.
# ---------------------------------------------------------------------------
if [[ "$SKIP_FEATURES" != "1" && -d "$REF_FEAT_DIR_ABS" ]]; then
  echo "=== (4) feature compare plots ==="
  docker run --rm --shm-size 4g \
    -v "${REPO_DIR}:/workspace/bevfusion" \
    -v "${DATASET_ROOT}:${DATASET_ROOT}" \
    -w /workspace/bevfusion \
    "$IMAGE" \
    bash -lc "
python tools/compare_navsim_bev_pt_pretrained_finetune.py \
  --pt-pretrained ${REF_FEAT_DIR} \
  --pt-finetune ${OUT_ROOT}/features \
  --dataset-root ${DATASET_ROOT} \
  --out-dir ${OUT_ROOT}/feature_plots \
  --max-frames 20
" 2>&1 | tail -25
else
  echo "[viz_navsim_stage3_aligned] skip feature compare (SKIP_FEATURES or REF_FEAT_DIR missing)"
fi

echo ""
echo "[viz_navsim_stage3_aligned] DONE"
echo "  features     : $OUT_ROOT_ABS/features            ($(find "$OUT_ROOT_ABS/features" -name '*.pt' 2>/dev/null | wc -l) pt)"
echo "  pred PNGs    : $OUT_ROOT_ABS/pred/map            ($(find "$OUT_ROOT_ABS/pred/map" -name '*.png' 2>/dev/null | wc -l) png)"
echo "  gt_vs_pred   : $OUT_ROOT_ABS/gt_vs_pred          ($(find "$OUT_ROOT_ABS/gt_vs_pred" -name '*.png' 2>/dev/null | wc -l) png)"
echo "  feature_plots: $OUT_ROOT_ABS/feature_plots       ($(find "$OUT_ROOT_ABS/feature_plots" -name '*.png' 2>/dev/null | wc -l) png)"
