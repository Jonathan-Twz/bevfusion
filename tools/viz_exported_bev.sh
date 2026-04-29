#!/usr/bin/env bash
# Render one exported *_decoder_neck.pt (or vtransform) to PNG.
# Usage: ./tools/viz_exported_bev.sh /path/to/token_decoder_neck.png [out.png]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IN="${1:?path to .pt}"
OUT="${2:-${IN%.pt}_viz.png}"
PY="${PYTHON:-/home/wenzhe/miniconda/envs/wote/bin/python3}"
if [[ ! -f "$IN" ]]; then
  echo "not found: $IN" >&2
  exit 1
fi
exec "$PY" "$REPO/tools/visualize_bev_feat.py" "$IN" -o "$OUT" --mode l2 --cmap viridis
