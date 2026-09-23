"""GPT eval over the 3-method intersection (Harmony ∩ 3DREGEN ∩ Gen3DSR ∩ ref).

For every scene with results in all 3 methods, run evaluate_pair(pred, target=input_photo).
Saves a JSON with full per-criterion details + a wide CSV (one row per scene).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpt_eval import EVALUATION_CRITERIA, evaluate_pair
from loaders import load_3dregen, load_flat, load_gen3dsr, load_harmony

_HERE = os.path.dirname(os.path.abspath(__file__))

# Dataset / baseline paths: override via env vars (evaluation.sh exports them)
# or replace the placeholders below.
REF_DIR = os.environ.get("REF_DIR", "/path/to/testing_first1")
REGEN_DIR = os.environ.get("REGEN_DIR", "/path/to/3D-RE-GEN/results_batch")
GEN3DSR_DIR = os.environ.get("GEN3DSR_DIR", "/path/to/Gen3DSR/out/testing_first1")
# Comma-separated roots; earlier roots take precedence (curated Artifact/ first,
# raw batch _processed/ second).
HARMONY_DIR = os.environ.get(
    "HARMONY_ROOTS",
    "/path/to/harmony/outputs/Demo/Artifact,/path/to/harmony/outputs/Demo/_processed",
)
OUT_JSON = os.path.join(_HERE, "gpt_compare_walltex.json")
OUT_CSV = os.path.join(_HERE, "gpt_compare_walltex.csv")
MODEL = os.environ.get("GPT_EVAL_MODEL", "gpt-4o")


def main():
    ref = load_flat(REF_DIR)
    methods = {
        "Harmony": load_harmony(HARMONY_DIR),
        "3DREGEN": load_3dregen(REGEN_DIR),
        "Gen3DSR": load_gen3dsr(GEN3DSR_DIR),
    }

    # Intersection across all methods + ref
    shared = set(ref)
    for m, idx in methods.items():
        shared &= set(idx)
    scenes = sorted(shared)
    print(f"3-method intersection: {len(scenes)} scenes")
    for s in scenes:
        print(f"  {s}")
    if not scenes:
        return

    criteria_names = list(EVALUATION_CRITERIA.keys())

    # Run all (scene, method) pairs
    results: dict = {}
    for i, scene in enumerate(scenes, 1):
        target = ref[scene]
        print(f"\n[{i}/{len(scenes)}] {scene}")
        results[scene] = {}
        for method, idx in methods.items():
            pred = idx[scene]
            t0 = time.time()
            r = evaluate_pair(pred_path=pred, target_path=target, model_name=MODEL)
            dt = time.time() - t0
            results[scene][method] = r
            line = "  ".join(f"{c}={r[c]['score']:>3.1f}" for c in criteria_names)
            err = f"  ERR: {r.get('_error')}" if r.get("_error") else ""
            print(f"  {method:<8}  {line}  avg={r['average_score']:.2f}  ({dt:.1f}s){err}")

    # Save JSON
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved → {OUT_JSON}")

    # Build wide CSV
    import csv
    methods_order = ["Harmony", "3DREGEN", "Gen3DSR"]
    header = ["scene"]
    for m in methods_order:
        for c in criteria_names:
            header.append(f"{m}_{c}")
        header.append(f"{m}_avg")
    header.append("best_avg_method")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for scene in scenes:
            row = [scene]
            avgs = {}
            for m in methods_order:
                r = results[scene][m]
                for c in criteria_names:
                    row.append(f"{r[c]['score']:.1f}")
                row.append(f"{r['average_score']:.2f}")
                avgs[m] = r["average_score"]
            row.append(max(avgs, key=avgs.get))
            w.writerow(row)

        # AVERAGE row
        w.writerow([])
        avg_row = ["AVERAGE"]
        for m in methods_order:
            for c in criteria_names:
                vals = [results[s][m][c]["score"] for s in scenes]
                avg_row.append(f"{sum(vals)/len(vals):.3f}")
            vals = [results[s][m]["average_score"] for s in scenes]
            avg_row.append(f"{sum(vals)/len(vals):.3f}")
        avg_row.append("")
        w.writerow(avg_row)

        # WIN counts per method (best_avg_method)
        from collections import Counter
        wins = Counter(row_best for row_best in (
            max({m: results[s][m]["average_score"] for m in methods_order},
                key=lambda m: results[s][m]["average_score"]) for s in scenes
        ))
        win_row = ["WINS"]
        for m in methods_order:
            for _ in criteria_names:
                win_row.append("")
            win_row.append(str(wins.get(m, 0)))
        win_row.append(f"total={sum(wins.values())}")
        w.writerow(win_row)

    print(f"saved → {OUT_CSV}")
    print("\nDONE")


if __name__ == "__main__":
    main()
