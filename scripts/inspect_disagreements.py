"""
scripts/inspect_disagreements.py
Task 3 — evidence for the label-boundary claim.
Reads saved run files. No API calls.

Run: python scripts/inspect_disagreements.py
"""

import os, sys, json, glob
from collections import defaultdict, Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_DIR = os.path.join(BASE, "data", "real_failures")
MANIFEST = os.path.join(BASE, "data", "real_failures_manifest.json")


def load_runs():
    runs = {}
    for fp in sorted(glob.glob(os.path.join(BASE, "data", "real_run*.json"))) + \
              [os.path.join(BASE, "data", "accuracy_results_real.json")]:
        if not os.path.exists(fp):
            continue
        try:
            d = json.load(open(fp))
            runs[os.path.basename(fp)] = d
        except Exception:
            pass
    return runs


def trace_lookup():
    """trace_id → (annotation text, current label, question)"""
    out = {}
    if not os.path.exists(MANIFEST):
        return out
    for e in json.load(open(MANIFEST)):
        ft = e["failure_type"]
        for cand in (os.path.join(BASE, e["file"]),
                     os.path.join(REAL_DIR, ft, os.path.basename(e["file"]))):
            if os.path.exists(cand):
                try:
                    d = json.load(open(cand))
                    out[e["trace_id"]] = {
                        "label": ft,
                        "original_keyword_label": d.get("original_keyword_label"),
                        "annotation": (d.get("mistake_reason") or "").strip(),
                        "question": (d.get("question") or "")[:200],
                        "mistake_agent": d.get("mistake_agent", ""),
                        "steps": len(d.get("trace", [])),
                    }
                except Exception:
                    pass
                break
    return out


def main():
    runs = load_runs()
    traces = trace_lookup()

    if not runs:
        print("No run files found. Expected data/real_run*.json")
        return

    print("=" * 78)
    print("  RUN INVENTORY")
    print("=" * 78)
    for name, d in runs.items():
        print(f"  {name:<32} N={d.get('scored','?'):<4} "
              f"correct={d.get('correct','?'):<4} acc={d.get('accuracy_pct','?')}%  "
              f"excluded={d.get('excluded_api',0)}")

    # ── per-trace verdicts across runs ────────────────────────────────
    per_trace = defaultdict(list)
    for name, d in runs.items():
        for r in d.get("all_results", []):
            if r.get("excluded"):
                continue
            per_trace[r["trace_id"]].append((name, r["actual"], r["diagnosed"]))

    unstable = {tid: v for tid, v in per_trace.items()
                if len(v) > 1 and len({dx for _, _, dx in v}) > 1}

    print()
    print("=" * 78)
    print("  A. RUN-TO-RUN INSTABILITY")
    print("=" * 78)
    seen_multi = [t for t, v in per_trace.items() if len(v) > 1]
    print(f"  Traces judged in more than one run : {len(seen_multi)}")
    print(f"  Of those, judged differently       : {len(unstable)}")
    if seen_multi:
        print(f"  Instability rate                   : "
              f"{len(unstable)/len(seen_multi)*100:.1f}%")

    for tid, v in sorted(unstable.items())[:12]:
        info = traces.get(tid, {})
        gt = v[0][1]
        verdicts = " / ".join(dx for _, _, dx in v)
        print(f"\n  {tid}   label={gt}")
        print(f"    verdicts across runs : {verdicts}")
        if info.get("annotation"):
            print(f"    human annotation     : {info['annotation'][:150]}")

    # ── systematic confusions ────────────────────────────────────────
    pairs = Counter()
    for tid, v in per_trace.items():
        for _, gt, dx in v:
            if gt != dx:
                pairs[(gt, dx)] += 1

    print()
    print("=" * 78)
    print("  B. SYSTEMATIC CONFUSIONS (all runs pooled)")
    print("=" * 78)
    for (gt, dx), n in pairs.most_common(10):
        print(f"  {n:>3}×   {gt}  →  {dx}")

    # ── the memory_overflow case, 6/6 every run ──────────────────────
    print()
    print("=" * 78)
    print("  C. EVIDENCE: memory_overflow labelled traces")
    print("=" * 78)
    print("  These were relabelled memory_overflow by the annotation-only labeller.")
    print("  The Coroner, reading the trace, consistently says otherwise.")
    print()

    mem = [tid for tid, info in traces.items() if info["label"] == "memory_overflow"]
    for tid in sorted(mem):
        info = traces[tid]
        verdicts = [dx for _, _, dx in per_trace.get(tid, [])]
        print(f"  ── {tid} ──")
        print(f"     keyword label   : {info.get('original_keyword_label') or '(unchanged)'}")
        print(f"     relabelled to   : {info['label']}")
        print(f"     coroner said    : {', '.join(verdicts) if verdicts else '(not judged)'}")
        print(f"     failing agent   : {info['mistake_agent']}")
        print(f"     human annotation:")
        ann = info["annotation"]
        for i in range(0, min(len(ann), 400), 92):
            print(f"       {ann[i:i+92]}")
        print()

    # ── save ─────────────────────────────────────────────────────────
    out = {
        "runs": {k: {kk: v.get(kk) for kk in
                     ("scored", "correct", "accuracy_pct", "excluded_api", "generated")}
                 for k, v in runs.items()},
        "traces_judged_multiple_times": len(seen_multi),
        "unstable_traces": len(unstable),
        "instability_rate_pct": round(len(unstable)/len(seen_multi)*100, 1) if seen_multi else None,
        "unstable_detail": [
            {"trace_id": tid,
             "ground_truth": v[0][1],
             "verdicts": [dx for _, _, dx in v],
             "annotation": traces.get(tid, {}).get("annotation", "")[:300]}
            for tid, v in sorted(unstable.items())
        ],
        "systematic_confusions": [{"from": gt, "to": dx, "count": n}
                                  for (gt, dx), n in pairs.most_common()],
        "memory_overflow_cases": [
            {"trace_id": tid,
             "keyword_label": traces[tid].get("original_keyword_label"),
             "relabelled_to": traces[tid]["label"],
             "coroner_verdicts": [dx for _, _, dx in per_trace.get(tid, [])],
             "annotation": traces[tid]["annotation"]}
            for tid in sorted(mem)
        ],
    }
    fp = os.path.join(BASE, "data", "disagreement_analysis.json")
    json.dump(out, open(fp, "w"), indent=2)
    print("=" * 78)
    print(f"  Saved → data/disagreement_analysis.json")
    print("=" * 78)


if __name__ == "__main__":
    main()