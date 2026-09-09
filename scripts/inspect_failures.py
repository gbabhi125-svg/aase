"""
scripts/inspect_failures.py
Pulls RAW Coroner output for the failing categories.
Does not fix anything. Shows exactly what the Coroner said and how it was parsed.

Run: python scripts/inspect_failures.py
"""

import os, sys, json, random, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from src.council.prosecutor import ProsecutorAgent
from src.council.defender   import DefenderAgent
from src.council.coroner    import CoronerAgent

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The categories scoring 0% / 10% / 40%
TARGETS = ["prompt_injection", "memory_overflow", "goal_drift", "context_collapse"]
PER_TYPE = 5

FAILURE_TYPES = ["hallucination","tool_misuse","reasoning_loop","context_collapse",
                 "goal_drift","prompt_injection","memory_overflow"]


def parse_failure_type_from_verdict(verdict):
    """EXACT copy of the parser used in run_accuracy_test.py"""
    v = (verdict or "").lower()
    if "failure_type:" in v:
        for line in verdict.split("\n"):
            if "failure_type:" in line.lower():
                val = line.split(":", 1)[-1].strip().lower()
                for ft in FAILURE_TYPES:
                    if ft in val or ft.replace("_", " ") in val:
                        return ft, f"LABEL LINE matched '{ft}'"
                return "unknown", f"LABEL LINE FOUND but no type matched. Raw value: '{val}'"
    kw = {
        "prompt_injection": ["prompt injection","injection","injected","ignore all previous","hijack","exfiltration"],
        "memory_overflow":  ["memory overflow","token limit","context_length_exceeded","hard crash","process killed","step limit"],
        "reasoning_loop":   ["reasoning loop","repeated the same","same action","same query","no progress"],
        "context_collapse": ["context collapse","forgot","informal","context warning","truncat","failed to recall"],
        "goal_drift":       ["goal drift","drifted","original goal","deviated","constraint ignor","premature"],
        "tool_misuse":      ["tool misuse","wrong tool","incorrect tool","invalid action","parameter error"],
        "hallucination":    ["hallucin","fabricat","made up","no records","invented","assumed"],
    }
    for ft, words in kw.items():
        for w in words:
            if w in v:
                return ft, f"KEYWORD fallback matched '{w}' → {ft}"
    return "unknown", "NO MATCH — neither label line nor any keyword"


def load(ftype, n):
    out = []
    for mname, folder in [("injected_manifest.json","injected"),
                          ("real_failures_manifest.json","real_failures")]:
        mp = os.path.join(BASE, "data", mname)
        if not os.path.exists(mp):
            continue
        for e in json.load(open(mp)):
            if e.get("failure_type") != ftype:
                continue
            for cand in (os.path.join(BASE, e["file"]),
                         os.path.join(BASE, "data", folder, ftype, os.path.basename(e["file"]))):
                if os.path.exists(cand):
                    try:
                        d = json.load(open(cand))
                        out.append({"id": e["trace_id"],
                                    "trace": d.get("trace", []),
                                    "task": d.get("task") or d.get("question","")})
                    except Exception:
                        pass
                    break
            if len(out) >= n:
                return out
    return out


def main():
    prosecutor = ProsecutorAgent()
    defender   = DefenderAgent()
    coroner    = CoronerAgent()

    report = []
    print("=" * 78)
    print("  RAW CORONER OUTPUT INSPECTION")
    print(f"  {PER_TYPE} traces each from: {', '.join(TARGETS)}")
    print("=" * 78)

    for ftype in TARGETS:
        traces = load(ftype, PER_TYPE)
        if not traces:
            print(f"\n### {ftype.upper()} — NO TRACES FOUND")
            continue

        print(f"\n\n{'#'*78}")
        print(f"###  {ftype.upper()}  —  {len(traces)} traces")
        print(f"{'#'*78}")

        for i, t in enumerate(traces, 1):
            trace_str = json.dumps(t["trace"], indent=2)[:3000]
            unknown = "unknown — determine from trace"

            print(f"\n{'-'*78}")
            print(f"[{i}/{len(traces)}] {t['id']}   GROUND TRUTH = {ftype}")
            print(f"{'-'*78}")

            try:
                pa = prosecutor.analyze(trace_str, unknown, t["task"])
                da = defender.analyze(trace_str, unknown, t["task"])
                vd = coroner.decide(trace_str, pa, da, unknown)
            except Exception as e:
                print(f"API ERROR: {e}")
                continue

            dx, why = parse_failure_type_from_verdict(vd)

            print("\n>>> RAW CORONER OUTPUT (verbatim, untouched):")
            print("┌" + "─"*76)
            for line in vd.split("\n"):
                print("│ " + line)
            print("└" + "─"*76)

            print(f"\n>>> PARSER SAID:     {dx}")
            print(f">>> PARSER REASON:   {why}")
            print(f">>> GROUND TRUTH:    {ftype}")
            print(f">>> SCORED AS:       {'CORRECT' if dx == ftype else 'WRONG'}")

            first = vd.strip().split("\n")[0] if vd.strip() else ""
            print(f">>> FIRST LINE:      {first!r}")
            print(f">>> HAS 'FAILURE_TYPE:' ANYWHERE: {'failure_type:' in vd.lower()}")

            report.append({
                "trace_id": t["id"],
                "ground_truth": ftype,
                "parsed_as": dx,
                "parse_reason": why,
                "correct": dx == ftype,
                "first_line": first,
                "has_label_line": "failure_type:" in vd.lower(),
                "raw_coroner": vd,
                "raw_prosecutor": pa,
                "raw_defender": da,
            })

    out_path = os.path.join(BASE, "data", "failure_inspection.json")
    json.dump(report, open(out_path, "w"), indent=2)

    print("\n\n" + "="*78)
    print("  SUMMARY")
    print("="*78)
    by_type = defaultdict(lambda: {"n":0, "correct":0, "no_label":0, "unknown":0})
    for r in report:
        s = by_type[r["ground_truth"]]
        s["n"] += 1
        if r["correct"]: s["correct"] += 1
        if not r["has_label_line"]: s["no_label"] += 1
        if r["parsed_as"] == "unknown": s["unknown"] += 1

    print(f"\n{'TYPE':<20}{'N':<5}{'OK':<5}{'NO LABEL LINE':<16}{'PARSED UNKNOWN'}")
    print("-"*70)
    for ft, s in by_type.items():
        print(f"{ft:<20}{s['n']:<5}{s['correct']:<5}{s['no_label']:<16}{s['unknown']}")

    print("\nWHAT THE CORONER ACTUALLY SAID (parsed vs truth):")
    for r in report:
        flag = "OK  " if r["correct"] else "WRONG"
        print(f"  [{flag}] {r['ground_truth']:<18} → {r['parsed_as']:<18} | {r['parse_reason']}")

    print(f"\nFull raw text saved to: data/failure_inspection.json")


if __name__ == "__main__":
    main()