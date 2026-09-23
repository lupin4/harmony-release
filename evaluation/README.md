# HARMONY/evaluation — HARMONY300 eval pipeline

Scoring code for the **HARMONY300** benchmark
([dataset](https://huggingface.co/datasets/ShufanSun/harmony) ·
[project page](https://cwchenwang.github.io/harmony/)), plus the GPT-as-judge
comparison used in the paper.

Two families of metrics:

- **Appearance**, over all 300 scenes — N-CLIP (1 − CLIP cosine), PL
  (photometric MSE) and LPIPS, each rendered from the scene's own camera
  (`camera_vggt.json`) and compared to the reference photograph.
- **Geometry**, over the 100 Front3D scenes that ship a ground-truth mesh —
  Chamfer distance and F-score at 0.1 / 0.01 / 0.001, after Sim(3)
  normalisation and ICP alignment (methods do not share the GT world frame).

Also: GPT 4-criteria scoring and side-by-side comparison figures.

## Files

| File | Role |
|---|---|
| `evaluation.sh` | one-shot pipeline: pixel eval → CSV → GPT eval → figures |
| `eval_runner.py` | N-CLIP + photometric MSE + LPIPS, multi-method |
| `gpt_compare_3methods.py` | GPT-4o 4-criteria scoring across 3 methods |
| `make_comparison_figures.py` | per-scene 4-panel JPGs (input + 3 methods) |
| `json_to_csv.py` | flatten eval JSON into wide CSV (one row per scene) |
| `loaders.py` | per-method file lookup (`flat`, `3dregen`, `gen3dsr`, `harmony`, …) |
| `metrics.py` | `clip_similarity`, `photometric_loss`, `lpips_distance` primitives |
| `chamfer_eval.py` | Chamfer + F-score vs. Front3D GT meshes (ICP-aligned) |
| `gpt_eval.py` | OpenAI client + criteria/prompt + `evaluate_pair()` |

## Outputs (current — keep these)

| File | Source |
|---|---|
| `results_walltex.json` / `.csv` | `eval_runner.py` → `json_to_csv.py` |
| `gpt_compare_walltex.json` / `.csv` | `gpt_compare_3methods.py` |
| `comparison_figures/{scene}.jpg` + `index.html` | `make_comparison_figures.py` |

Anything not listed here can probably be deleted (stale snapshots from
earlier configs — old `gpt_compare_22*`, `results_all*`, etc.).

## Run

All dataset / baseline locations are placeholders (`/path/to/...`) read from
env vars — set them once, then run the one-shot script:

```bash
export OPENAI_API_KEY=sk-...
export REF_DIR=/path/to/testing_first1                      # input photos
export REGEN_DIR=/path/to/3D-RE-GEN/results_batch           # 3D-RE-GEN outputs
export GEN3DSR_DIR=/path/to/Gen3DSR/out/testing_first1      # Gen3DSR outputs
export HARMONY_ROOTS=/path/to/harmony/outputs/Demo/Artifact,/path/to/harmony/outputs/Demo/_processed
export PY=python          # interpreter with torch + transformers + openai (optional)

bash evaluation.sh        # everything, ~10 min, ~$2 GPT cost
```

Or individual steps (must include `PYTHONNOUSERSITE=1` — see footguns):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python eval_runner.py \
    --ref  flat:$REF_DIR \
    --pred Gen3DSR=gen3dsr:$GEN3DSR_DIR \
    --pred 3DREGEN=3dregen:$REGEN_DIR \
    --pred Harmony=harmony:$HARMONY_ROOTS \
    --out  results_walltex.json
```

`gpt_compare_3methods.py` and `make_comparison_figures.py` read the same
`REF_DIR` / `REGEN_DIR` / `GEN3DSR_DIR` / `HARMONY_ROOTS` env vars (and
`GPT_EVAL_MODEL`, default `gpt-4o`).

## Loader registry (`loaders.py:LOADERS`)

Each loader takes a root path → returns `{scene_key: image_path}` map.
Keys are normalized (`,` → `_`, spaces → `_`).

| name | file pattern | notes |
|---|---|---|
| `flat` | `{root}/{stem}.{ext}` | for input photos, accepts jpg/png/avif/webp |
| `3dregen` | `{root}/{scene}/render_cam1_white_bg.png` | one file per scene dir |
| `gen3dsr` | `{root}/{scene}/render_bproc.png` (preferred) → `render_input_view.png` (fallback) | BlenderProc Cycles for paper quality |
| `harmony` | `{root}/{scene}/decorations/placements/render_decorations_placed_with_new_wall_texture.png` (preferred) → 3 fallbacks | comma-separated multi-root supported (earlier roots win) |
| `scenegen` | `{root}/{stem}.png` | (legacy, unused currently) |
| `gen3dsr_gt` | `{root}/gt_{scene_id}.png` | strips `gt_` prefix |

## How to add a new method

1. Add a function `load_xxx(root: str) -> dict[str, str]` in `loaders.py`.
2. Register it in the `LOADERS` dict at the bottom.
3. Pass `--pred NewName=xxx:/path/to/dir` to `eval_runner.py`.
4. For GPT eval, edit `gpt_compare_3methods.py:methods` dict (and
   `make_comparison_figures.py:methods` for the figures).

## Pixel metric semantics

| Metric | Direction | What it measures |
|---|---|---|
| **n_clip** | ↓ lower is better | `1 - cosine(CLIP(pred), CLIP(ref))`. CLIP-ViT-B/32 |
| **pl** | ↓ lower is better | mean pixel MSE in normalized RGB ∈ [0,1] |

If pred and ref have different sizes, `pl` is computed on the resized pair
(warning printed). Aspect ratio is preserved.

## GPT eval

Single API call per (scene, method) pair. JSON response covers 4 criteria
at once. Defined in `gpt_eval.py`:

| Criterion | Scale | What |
|---|---|---|
| `visual_quality` | 0-5 | rendering quality, lighting, materials |
| `object_identity` | 0-5 | are TARGET objects present in correct grid cell + wall |
| `object_orientation` | 0-5 | objects facing the right direction (per wall + camera-facing rules) |
| `spatial_accuracy` | 0-5 | floor / wall-mounted / decoration placement correct |

Model: `gpt-4o`. Cost: ~$0.02 per pair (2 images + JSON output). 30 scenes
× 3 methods = ~$2 / 7 min total.

## CSV format (`results_walltex.csv`)

```
scene, Gen3DSR_n_clip, Gen3DSR_pl, 3DREGEN_n_clip, 3DREGEN_pl,
Harmony_n_clip, Harmony_pl, best_n_clip_method, best_pl_method
```

Followed by:
- blank row
- `AVERAGE` (full-set means per method)
- `NUM_SCORED` (full-set counts)
- blank row
- `AVERAGE_INTERSECT(N)` (means on 3-method intersection)
- `INTERSECT_NUM`

`gpt_compare_walltex.csv` is similar but with 4 criterion columns +
`<method>_avg` per method.

## Footguns

- **Always export `PYTHONNOUSERSITE=1`** when running anything here.
  Without it, `~/.local/lib/python3.10/site-packages` torch / transformers
  can shadow the intended env and either crash (`libcudart.so.12` missing)
  or produce wrong shapes (`get_image_features` returns weird object).
- **CLIP feature collapse**: random / unrelated images get cosine ≈ 0.9+
  on CLIP-B/32. `n_clip = 1 - cos` will be 0.05-0.3 across all method/ref
  pairs even when they look very different. Compare deltas, not absolutes.
- **AVIF support**: `livingroom1.avif`, `office8.avif`. Requires
  `pillow-avif-plugin` in the env. `gpt_eval.encode_image` transcodes AVIF → PNG
  before uploading to OpenAI (which doesn't accept AVIF).

## Comparison figures

`make_comparison_figures.py` reads the same loaders + intersects. Outputs
640px-wide per-method panels stitched horizontally with 8 px white gaps
into `comparison_figures/{scene}.jpg`. `index.html` shows all of them
stacked for quick browsing.

Adjust `PANEL_W`, `GAP`, `LABEL_H`, `BG` at the top of that file.

## Current state (intersection of 30 scenes)

```
              n_clip    pl     GPT_avg
Gen3DSR        0.2125  0.0493    3.99
3DREGEN        0.1535  0.0527    3.10
Harmony        0.1249  0.0446    3.33
```

Harmony wins pixel metrics + visual_quality; Gen3DSR wins GPT identity /
orientation / spatial (per-object reconstruction preserves photo content);
3DREGEN is in between.
