"""
scripts/run_baseline.py
Baseline vs AASE comparison. This produces the delta that is AASE's actual claim.

Three conditions, same traces:
  A  no_repair      — failure detected, nothing done. What happens today.
  B  generic_repair — a single fixed generic clause applied to every failure,
                      regardless of type. Controls for "does ANY prompt edit help?"
  C  aase           — full council diagnoses the type, surgeon applies a targeted fix.

Scoring is repair correctness, not diagnosis correctness:
  a repair COUNTS only if the applied clause actually addresses the
  ground-truth failure type. Condition A can never score. Condition B scores
  only when the generic clause happens to cover that type.

  python scripts/run_baseline.py --n 5
  python scripts/run_baseline.py --n 5 --source real
  python scripts/run_baseline.py --n 10 --conditions A B C
"""

import os, sys, json, time, argparse
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from src.parser import FAILURE_TYPES, parse_failure_type, extract_fix, is_api_failure
from src.council.prosecutor import ProsecutorAgent
from src.council.defender   import DefenderAgent
from src.council.coroner    import CoronerAgent
from src.surgeon.surgeon    import SurgeonAgent
from src.llm import provider, model_name

console = Console()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANIFESTS = {
    "injected": ("injected_manifest.json", "injected"),
    "real":     ("real_failures_manifest.json", "real_failures"),
}

BASE_PROMPT = """You are an enterprise pipeline agent.
Process tasks step by step using available tools.
Always verify tool outputs before using them.
Maintain formal professional tone at all times."""

# Condition B — one generic clause for every failure, no diagnosis involved.
GENERIC_CLAUSE = (
    "\nGENERAL GUARD: Be careful. Double-check your work before responding, "
    "verify your outputs, and follow the task instructions closely."
)

# What a correct repair must actually contain, per failure type.
# A repair counts only if it addresses the real mechanism, not just any edit.
REPAIR_REQUIREMENTS = {
    "hallucination": {
        "must_mention": [["tool", "returned", "retriev", "source", "confirm", "verif"],
                         ["never state", "do not state", "not fabricat", "unavailable",
                          "could not", "no data", "never generate", "never invent"]],
        "desc": "must forbid stating values not returned by a tool",
    },
    "tool_misuse": {
        "must_mention": [["tool", "api"],
                         ["select", "choos", "classif", "internal", "public", "match", "correct tool"]],
        "desc": "must constrain tool selection",
    },
    "reasoning_loop": {
        "must_mention": [["repeat", "same", "again", "retry", "attempt"],
                         ["stop", "do not", "halt", "refine", "change", "abort", "escalat"]],
        "desc": "must break repetition",
    },
    "context_collapse": {
        "must_mention": [["constraint", "instruction", "tone", "format", "context", "re-read", "restate"],
                         ["every", "each", "maintain", "re-read", "restate", "invariant", "persist", "pinned"]],
        "desc": "must re-anchor constraints during long runs",
    },
    "goal_drift": {
        "must_mention": [["original task", "original goal", "constraint", "scope",
                          "recipient", "length", "requirement"],
                         ["verify", "re-read", "check", "confirm", "before", "align", "abort"]],
        "desc": "must verify output against original constraints",
    },
    "prompt_injection": {
        "must_mention": [["external", "document", "file", "retriev", "untrusted", "content"],
                         ["data", "never", "ignore", "not instruction", "do not execute",
                          "do not follow", "inert"]],
        "desc": "must treat external content as data only",
    },
    "memory_overflow": {
        "must_mention": [["batch", "chunk", "context", "token", "memory", "state"],
                         ["clear", "summar", "limit", "checkpoint", "persist", "discard", "flush"]],
        "desc": "must bound memory growth",
    },
}


def repair_is_valid(applied_text, ftype):
    """
    Does the applied clause actually address this failure type?
    Requires at least one term from EVERY required group.
    """
    if not applied_text:
        return False
    t = applied_text.lower()
    req = REPAIR_REQUIREMENTS.get(ftype)
    if not req:
        return False
    for group in req["must_mention"]:
        if not any(term in t for term in group):
            return False
    return True


def load_traces(source, per_type):
    picked = []
    mfile, folder = MANIFESTS[source]
    mp = os.path.join(BASE, "data", mfile)
    if not os.path.exists(mp):
        console.print(f"[red]{mfile} not found[/red]")
        return []

    by_type = defaultdict(list)
    for e in json.load(open(mp)):
        if e.get("failure_type") in FAILURE_TYPES:
            by_type[e["failure_type"]].append(e)

    for ft in FAILURE_TYPES:
        for e in by_type.get(ft, [])[:per_type]:
            for cand in (os.path.join(BASE, e["file"]),
                         os.path.join(BASE, "data", folder, ft, os.path.basename(e["file"]))):
                if os.path.exists(cand):
                    try:
                        d = json.load(open(cand))
                        picked.append({
                            "trace_id": e["trace_id"],
                            "failure_type": ft,
                            "trace": d.get("trace", []),
                            "task": d.get("task") or d.get("question", ""),
                        })
                    except Exception:
                        pass
                    break
    return picked


def run(source="injected", per_type=5, conditions=("A", "B", "C"), delay=0.0):
    console.print(Panel(
        f"[bold cyan]AASE Baseline Comparison[/bold cyan]\n"
        f"[dim]provider={provider()} · model={model_name()}\n"
        f"source={source} · {per_type} per type · conditions={' '.join(conditions)}[/dim]",
        title="AASE"))

    traces = load_traces(source, per_type)
    if not traces:
        console.print("[red]No traces loaded.[/red]")
        return

    dist = defaultdict(int)
    for t in traces:
        dist[t["failure_type"]] += 1
    console.print(f"[green]{len(traces)} traces[/green] — " +
                  ", ".join(f"{k}={v}" for k, v in dist.items()))

    calls = len(traces) * (4 if "C" in conditions else 0)
    console.print(f"[dim]≈{calls} API calls · ~{calls*6//60} min[/dim]\n")

    prosecutor = ProsecutorAgent()
    defender   = DefenderAgent()
    coroner    = CoronerAgent()
    surgeon    = SurgeonAgent()

    stats = {c: defaultdict(lambda: {"ok": 0, "n": 0}) for c in conditions}
    totals = {c: {"ok": 0, "n": 0, "api": 0} for c in conditions}
    rows = []
    consecutive_api = 0

    for i, item in enumerate(traces):
        ft   = item["failure_type"]
        tid  = item["trace_id"]
        tstr = json.dumps(item["trace"], indent=2)[:3000]
        unk  = "unknown — determine from trace"

        console.print(f"[dim][{i+1}/{len(traces)}] {tid} — {ft}[/dim]")
        row = {"trace_id": tid, "failure_type": ft}

        # ── A: no repair ─────────────────────────────────────────────
        if "A" in conditions:
            stats["A"][ft]["n"] += 1
            totals["A"]["n"] += 1
            row["A_repaired"] = False
            row["A_applied"] = ""
            console.print("  [dim]A no_repair      → [red]not repaired[/red][/dim]")

        # ── B: generic repair, no diagnosis ─────────────────────────
        if "B" in conditions:
            ok = repair_is_valid(GENERIC_CLAUSE, ft)
            stats["B"][ft]["n"] += 1
            stats["B"][ft]["ok"] += ok
            totals["B"]["n"] += 1
            totals["B"]["ok"] += ok
            row["B_repaired"] = ok
            row["B_applied"] = GENERIC_CLAUSE.strip()
            console.print(f"  [dim]B generic_repair → "
                          f"{'[green]valid[/green]' if ok else '[red]does not address failure[/red]'}[/dim]")

        # ── C: full AASE ────────────────────────────────────────────
        if "C" in conditions:
            t0 = time.time()
            pa = prosecutor.analyze(tstr, unk, item["task"])
            da = defender.analyze(tstr, unk, item["task"])
            vd = coroner.decide(tstr, pa, da, unk)

            fail = is_api_failure(vd) or is_api_failure(pa) or is_api_failure(da)
            if fail:
                consecutive_api += 1
                totals["C"]["api"] += 1
                row["C_repaired"] = None
                row["C_excluded"] = fail
                console.print(f"  [red]C aase           → {fail} — EXCLUDED[/red]")
                rows.append(row)
                if consecutive_api >= 5:
                    console.print(Panel(
                        f"[bold red]{consecutive_api} consecutive API failures — stopping.[/bold red]",
                        title="[red]ABORTED[/red]"))
                    break
                if delay:
                    time.sleep(delay)
                continue

            consecutive_api = 0
            dx, _ = parse_failure_type(vd)
            fix = extract_fix(vd)
            after = surgeon.apply_fix(BASE_PROMPT, dx, fix)
            applied = after.replace(BASE_PROMPT, "").strip() or fix

            ok = repair_is_valid(applied, ft)
            el = round(time.time() - t0, 1)

            stats["C"][ft]["n"] += 1
            stats["C"][ft]["ok"] += ok
            totals["C"]["n"] += 1
            totals["C"]["ok"] += ok

            row["C_repaired"]  = ok
            row["C_diagnosed"] = dx
            row["C_correct_dx"] = (dx == ft)
            row["C_applied"]   = applied[:400]
            row["C_time"]      = el

            console.print(f"  C aase           → dx=[cyan]{dx}[/cyan] "
                          f"{'[green]valid repair[/green]' if ok else '[yellow]repair misses[/yellow]'} ({el}s)")
            if delay:
                time.sleep(delay)

        rows.append(row)

    # ── Results ──────────────────────────────────────────────────────
    NAMES = {"A": "No repair (today)", "B": "Generic repair", "C": "AASE (full council)"}

    table = Table(title="Repair Validity by Condition")
    table.add_column("Condition", style="bold")
    table.add_column("Description")
    table.add_column("Valid repairs", style="green")
    table.add_column("Scored", style="white")
    table.add_column("Rate", style="cyan")
    for c in conditions:
        t = totals[c]
        rate = f"{t['ok']/t['n']*100:.1f}%" if t["n"] else "—"
        table.add_row(c, NAMES[c], str(t["ok"]), str(t["n"]), rate)
    console.print(table)

    pt = Table(title="Per Failure Type")
    pt.add_column("Failure Type", style="bold")
    for c in conditions:
        pt.add_column(c, style="cyan")
    for ft in FAILURE_TYPES:
        if not any(stats[c][ft]["n"] for c in conditions):
            continue
        cells = []
        for c in conditions:
            s = stats[c][ft]
            cells.append(f"{s['ok']}/{s['n']}" if s["n"] else "—")
        pt.add_row(ft, *cells)
    console.print(pt)

    if "A" in conditions and "C" in conditions and totals["C"]["n"]:
        a_rate = totals["A"]["ok"] / max(totals["A"]["n"], 1) * 100
        c_rate = totals["C"]["ok"] / totals["C"]["n"] * 100
        lines = [f"[bold]A → C:[/bold]  {a_rate:.1f}%  →  {c_rate:.1f}%   "
                 f"[green]+{c_rate - a_rate:.1f} points[/green]"]
        if "B" in conditions and totals["B"]["n"]:
            b_rate = totals["B"]["ok"] / totals["B"]["n"] * 100
            lines.append(f"[bold]B → C:[/bold]  {b_rate:.1f}%  →  {c_rate:.1f}%   "
                         f"[green]+{c_rate - b_rate:.1f} points[/green]")
            lines.append("")
            lines.append("[dim]B → C isolates the council's contribution: both edit the\n"
                         "prompt, only C diagnoses the failure type first.[/dim]")
        console.print(Panel("\n".join(lines), title="[bold green]Contribution Delta[/bold green]"))

    if totals.get("C", {}).get("api"):
        console.print(f"[red]C excluded for API failures: {totals['C']['api']}[/red]")

    out = {
        "provider": provider(), "model": model_name(),
        "source": source, "per_type": per_type,
        "conditions": list(conditions),
        "condition_names": NAMES,
        "totals": {c: dict(totals[c]) for c in conditions},
        "per_type_stats": {c: {k: dict(v) for k, v in stats[c].items()} for c in conditions},
        "generic_clause": GENERIC_CLAUSE.strip(),
        "scoring": "a repair counts only if the applied clause addresses the ground-truth failure mechanism",
        "requirements": {k: v["desc"] for k, v in REPAIR_REQUIREMENTS.items()},
        "rows": rows,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    fp = os.path.join(BASE, "data", f"baseline_results_{source}.json")
    json.dump(out, open(fp, "w"), indent=2)
    console.print(f"[green]Saved → data/baseline_results_{source}.json[/green]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="injected", choices=["injected", "real"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--conditions", nargs="*", default=["A", "B", "C"], choices=["A", "B", "C"])
    ap.add_argument("--delay", type=float, default=0.0)
    a = ap.parse_args()
    run(a.source, a.n, tuple(a.conditions), a.delay)