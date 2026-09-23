#!/bin/bash
# Full evaluation pipeline: image-similarity (n_clip + pl), GPT scoring,
# CSV export, and per-scene side-by-side comparison figures.
#
# Run from anywhere. Outputs in evaluation/.
#
# Set the dataset / baseline paths via env vars (or edit the placeholders below):
#
#   export OPENAI_API_KEY=sk-...
#   export REF_DIR=/path/to/testing_first1
#   export REGEN_DIR=/path/to/3D-RE-GEN/results_batch
#   export GEN3DSR_DIR=/path/to/Gen3DSR/out/testing_first1
#   export HARMONY_ROOTS=/path/to/harmony/outputs/Demo/Artifact,/path/to/harmony/outputs/Demo/_processed
#   bash evaluation.sh

set -e

# ─── Config ───────────────────────────────────────────────────────────────────

PY=${PY:-python}                                   # python with torch + transformers + openai
SCENEWEAVE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)   # repo root (parent of evaluation/)

# Input photos (flat dir of {scene}.{jpg,png,avif,...})
REF_DIR=${REF_DIR:-/path/to/testing_first1}
# Baseline outputs (see loaders.py for the expected per-method layout)
REGEN_DIR=${REGEN_DIR:-/path/to/3D-RE-GEN/results_batch}
GEN3DSR_DIR=${GEN3DSR_DIR:-/path/to/Gen3DSR/out/testing_first1}
# HARMONY outputs: comma-separated roots, earlier roots win
ARTIFACT_DIR=${ARTIFACT_DIR:-/path/to/harmony/outputs/Demo/Artifact}
PROCESSED_DIR=${PROCESSED_DIR:-/path/to/harmony/outputs/Demo/_processed}
HARMONY_ROOTS=${HARMONY_ROOTS:-"$ARTIFACT_DIR,$PROCESSED_DIR"}
export REF_DIR REGEN_DIR GEN3DSR_DIR HARMONY_ROOTS

# Eval outputs (CLIP + photometric)
PIXEL_JSON=$SCENEWEAVE/evaluation/results_walltex.json
PIXEL_CSV=$SCENEWEAVE/evaluation/results_walltex.csv

# GPT outputs
GPT_JSON=$SCENEWEAVE/evaluation/gpt_compare_walltex.json
GPT_CSV=$SCENEWEAVE/evaluation/gpt_compare_walltex.csv

# GPU to use (eval_runner + GPT both small)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Skip user-site so torch/CLIP load from the intended env cleanly
export PYTHONNOUSERSITE=1

# OpenAI key for the GPT-as-judge step
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set (needed for gpt_compare_3methods.py)" >&2
    exit 1
fi

for d in "$REF_DIR" "$REGEN_DIR" "$GEN3DSR_DIR"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: directory not found: $d  (set REF_DIR / REGEN_DIR / GEN3DSR_DIR / HARMONY_ROOTS)" >&2
        exit 1
    fi
done

cd "$SCENEWEAVE"

# ─── 1. Image-similarity eval (CLIP + photometric MSE) ────────────────────────

echo "===== 1. eval_runner.py — CLIP + photometric =====               $(date +%H:%M:%S)"
$PY evaluation/eval_runner.py \
    --ref  "flat:$REF_DIR" \
    --pred "Gen3DSR=gen3dsr:$GEN3DSR_DIR" \
    --pred "3DREGEN=3dregen:$REGEN_DIR" \
    --pred "Harmony=harmony:$HARMONY_ROOTS" \
    --out  "$PIXEL_JSON" 2>&1 | grep -vE "size mismatch|^  [A-Z]" | tail -12

echo
echo "===== 2. json_to_csv (pixel metrics) =====                       $(date +%H:%M:%S)"
$PY evaluation/json_to_csv.py "$PIXEL_JSON" "$PIXEL_CSV"

# ─── 2. GPT-as-a-judge eval ───────────────────────────────────────────────────

echo
echo "===== 3. gpt_compare_3methods.py — GPT scoring =====             $(date +%H:%M:%S)"
echo "(uses gpt-4o, costs ~\$2 for 30 scenes × 3 methods)"
$PY evaluation/gpt_compare_3methods.py 2>&1 | tail -5

# ─── 3. Per-scene comparison figures ──────────────────────────────────────────

echo
echo "===== 4. make_comparison_figures.py — side-by-side panels =====  $(date +%H:%M:%S)"
$PY evaluation/make_comparison_figures.py 2>&1 | tail -5

# ─── Summary ──────────────────────────────────────────────────────────────────

echo
echo "===== DONE $(date) ====="
echo
echo "Outputs:"
echo "  - $PIXEL_JSON / $PIXEL_CSV       (CLIP + photometric)"
echo "  - $GPT_JSON / $GPT_CSV           (GPT 4-criteria scoring)"
echo "  - $SCENEWEAVE/evaluation/comparison_figures/      (per-scene PNGs + index.html)"

# Print key averages
$PY -c "
import json
d = json.load(open('$PIXEL_JSON'))
methods = list(d['methods'].keys())
def valid(m): return {k for k,v in d['methods'][m]['per_scene'].items() if 'n_clip' in v}
inter = set.intersection(*(valid(m) for m in methods))
print()
print(f'=== INTERSECT ({len(inter)} scenes) ===')
for m in methods:
    per = d['methods'][m]['per_scene']
    nc = [per[k]['n_clip'] for k in inter]
    pl = [per[k]['pl'] for k in inter]
    print(f'  {m:<10}  n_clip={sum(nc)/len(nc):.4f}  pl={sum(pl)/len(pl):.4f}')
"
