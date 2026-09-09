"""
AASE batch evaluation runner.

Runs the Council (Prosecutor/Defender/Coroner) against N real failure
traces from data/injected/, compares each verdict to the trace's ground
truth label, and reports accuracy. This is what produces your actual
results-section numbers.

Usage:
    python src/council/batch_runner.py --n 20
    python src/council/batch_runner.py --n 100 --types hallucination,tool_misuse
"""

import os
import sys
import json
import argparse
import random
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from .council import run_council, parse_verdict
except ImportError:
    from council import run_council, parse_verdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INJECTED_DIR = os.path.join(PROJECT_ROOT, "data", "injected")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "data", "council_batch_results.json")


def load_sample(n: int, types: list = None, seed: int = 42) -> list:
    files = sorted(os.listdir(INJECTED_DIR))
    files = [f for f in files if f.endswith(".json")]

    if types:
        files = [f for f in files if any(f.startswith(t) for t in types)]

    random.seed(seed)
    random.shuffle(files)
    files = files[:n]

    traces = []
    for f in files:
        traces.append(json.load(open(os.path.join(INJECTED_DIR, f))))
    return traces


def run_batch(n: int = 20, types: list = None):
    traces = load_sample(n, types)
    print(f"Running Council against {len(traces)} real failure traces...\n")

    results = []
    correct = 0
    per_type_total = Counter()
    per_type_correct = Counter()

    for i, trace in enumerate(traces, 1):
        true_label = trace["injected_failure"]
        print(f"[{i}/{len(traces)}] trace={trace['id']} true_label={true_label} ...")

        t0 = time.time()
        try:
            council_result = run_council(trace)
            verdict = parse_verdict(council_result["coroner"])
            predicted_label = (verdict.get("VERDICT") or "").strip().lower()
        except Exception as e:
            predicted_label = f"ERROR: {e}"
            verdict = {}
        elapsed = time.time() - t0

        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1
        per_type_total[true_label] += 1
        if is_correct:
            per_type_correct[true_label] += 1

        print(f"    predicted={predicted_label} correct={is_correct} ({elapsed:.1f}s)\n")

        results.append({
            "trace_id": trace["id"],
            "true_label": true_label,
            "predicted_label": predicted_label,
            "correct": is_correct,
            "confidence": verdict.get("CONFIDENCE"),
            "final_fix": verdict.get("FINAL_FIX"),
            "elapsed_seconds": round(elapsed, 2),
        })

    accuracy = correct / len(traces) if traces else 0.0

    summary = {
        "total_traces": len(traces),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "per_type_accuracy": {
            t: round(per_type_correct[t] / per_type_total[t], 4)
            for t in per_type_total
        },
        "results": results,
    }

    json.dump(summary, open(RESULTS_PATH, "w"), indent=2)

    print("=" * 60)
    print(f"TOTAL: {correct}/{len(traces)} correct  ->  accuracy = {accuracy:.1%}")
    print("Per-type accuracy:")
    for t, acc in summary["per_type_accuracy"].items():
        print(f"  {t}: {acc:.1%} ({per_type_correct[t]}/{per_type_total[t]})")
    print(f"\nFull results saved to: {RESULTS_PATH}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="number of traces to test")
    parser.add_argument("--types", type=str, default=None,
                         help="comma-separated failure types to filter, e.g. hallucination,tool_misuse")
    args = parser.parse_args()

    types_list = args.types.split(",") if args.types else None
    run_batch(n=args.n, types=types_list)