"""
scripts/run_accuracy_test.py
Evaluates the Diagnosis Council against ground-truth labels.
Uses the shared parser. API failures are EXCLUDED from accuracy, never scored as wrong.

  python scripts/run_accuracy_test.py
  python scripts/run_accuracy_test.py --n 30
  python scripts/run_accuracy_test.py --source real
  python scripts/run_accuracy_test.py --all
  python scripts/run_accuracy_test.py --n 5 --delay 2
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
from src.ledger.ledger      import RepairLedger
from src.llm import provider, model_name

console = Console()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANIFESTS = {
    "injected": ("injected_manifest.json", "injected"),
    "real":     ("real_failures_manifest.json", "real_failures"),
}


def load_traces(source="injected", per_type=10, all_of_them=False):
    picked = []
    sources = MANIFESTS.items() if source == "any" else [(source, MANIFESTS[source])]

    for src_name, (mfile, folder) in sources:
        mp = os.path.join(BASE, "data", mfile)
        if not os.path.exists(mp):
            console.print(f"[yellow]skip {mfile} — not found[/yellow]")
            continue
        by_type = defaultdict(list)
        for e in json.load(open(mp)):
            if e.get("failure_type") in FAILURE_TYPES:
                by_type[e["failure_type"]].append(e)

        for ft in FAILURE_TYPES:
            entries = by_type.get(ft, [])
            if not all_of_them:
                entries = entries[:per_type]
            for e in entries:
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
                                "src": src_name,
                            })
                        except Exception:
                            pass
                        break
    return picked


def run(source="injected", per_type=10, all_of_them=False, delay=0.0):
    console.print(Panel(
        f"[bold cyan]AASE Council Evaluation[/bold cyan]\n"
        f"[dim]provider={provider()} · model={model_name()}\n"
        f"source={source} · {'ALL' if all_of_them else str(per_type)+' per type'}"
        f"{f' · {delay}s pacing' if delay else ''}[/dim]",
        title="AASE"))

    traces = load_traces(source, per_type, all_of_them)
    if not traces:
        console.print("[red]No traces loaded.[/red]")
        return

    dist = defaultdict(int)
    for t in traces:
        dist[t["failure_type"]] += 1
    console.print(f"[green]{len(traces)} traces[/green] — " +
                  ", ".join(f"{k}={v}" for k, v in dist.items()) + "\n")

    prosecutor = ProsecutorAgent()
    defender   = DefenderAgent()
    coroner    = CoronerAgent()
    ledger     = RepairLedger()

    results  = []
    per      = defaultdict(lambda: {"ok": 0, "n": 0, "api": 0})
    conf     = defaultdict(lambda: defaultdict(int))
    correct  = unknown = 0
    api_fail = defaultdict(int)
    consecutive_api = 0

    for i, item in enumerate(traces):
        ft   = item["failure_type"]
        tid  = item["trace_id"]
        tstr = json.dumps(item["trace"], indent=2)[:3000]
        unk  = "unknown — determine from trace"

        console.print(f"[dim][{i+1}/{len(traces)}] {tid} — {ft}[/dim]")
        t0 = time.time()
        pa = prosecutor.analyze(tstr, unk, item["task"])
        da = defender.analyze(tstr, unk, item["task"])
        vd = coroner.decide(tstr, pa, da, unk)
        el = round(time.time() - t0, 1)

        fail = is_api_failure(vd) or is_api_failure(pa) or is_api_failure(da)
        if fail:
            consecutive_api += 1
            api_fail[fail] += 1
            per[ft]["api"] += 1
            console.print(f"  → [red]{fail} — EXCLUDED[/red] ({el}s)")
            results.append({"trace_id": tid, "actual": ft, "diagnosed": None,
                            "correct": False, "time": el, "how": fail,
                            "excluded": True, "src": item["src"]})
            if consecutive_api >= 5:
                console.print(Panel(
                    f"[bold red]{consecutive_api} consecutive API failures — stopping.[/bold red]\n"
                    f"[dim]Most recent: {fail}. Wait for quota reset, switch LLM_PROVIDER "
                    f"in .env, or reduce --n.[/dim]",
                    title="[red]ABORTED[/red]"))
                break
            if delay:
                time.sleep(delay)
            continue

        consecutive_api = 0
        dx, how = parse_failure_type(vd)
        ok = (dx == ft)

        correct += ok
        unknown += (dx == "unknown")
        per[ft]["n"]  += 1
        per[ft]["ok"] += ok
        conf[ft][dx]  += 1

        console.print(f"  → [cyan]{dx}[/cyan] "
                      f"{'[green]CORRECT[/green]' if ok else '[yellow]WRONG[/yellow]'} "
                      f"({el}s) [dim]{how}[/dim]")

        results.append({"trace_id": tid, "actual": ft, "diagnosed": dx,
                        "correct": ok, "time": el, "how": how,
                        "excluded": False, "src": item["src"]})

        try:
            ledger.store_repair(f"failure_type:{ft}|task:{item['task'][:60]}",
                                dx if dx != "unknown" else ft,
                                extract_fix(vd), "success" if ok else "partial")
        except Exception:
            pass

        if delay:
            time.sleep(delay)

    scored   = [r for r in results if not r["excluded"]]
    excluded = [r for r in results if r["excluded"]]
    total    = len(scored)
    acc      = correct / total * 100 if total else 0

    table = Table(title=f"Per-Type Accuracy — scored N={total}")
    for col, style in [("Failure Type","bold"), ("Correct","green"),
                       ("Scored","white"), ("Accuracy","cyan"), ("API fails","red")]:
        table.add_column(col, style=style)
    for ft in FAILURE_TYPES:
        s = per[ft]
        if s["n"] == 0 and s["api"] == 0:
            continue
        a = f"{s['ok']/s['n']*100:.0f}%" if s["n"] else "—"
        table.add_row(ft, str(s["ok"]), str(s["n"]), a, str(s["api"]) if s["api"] else "")
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{correct}[/bold]",
                  f"[bold]{total}[/bold]", f"[bold]{acc:.1f}%[/bold]",
                  f"[bold]{len(excluded)}[/bold]" if excluded else "")
    console.print(table)

    if conf:
        console.print("\n[bold]Confusion — actual → diagnosed[/bold]")
        for ft in FAILURE_TYPES:
            if ft not in conf:
                continue
            row = ", ".join(f"{k}:{v}" for k, v in sorted(conf[ft].items(), key=lambda x: -x[1]))
            console.print(f"  [cyan]{ft:<18}[/cyan] {row}")

    if excluded:
        console.print("\n[bold red]API failures (excluded from accuracy)[/bold red]")
        for k, v in api_fail.items():
            console.print(f"  {k}: {v}")

    lat = sorted(r["time"] for r in scored if r["time"])
    console.print(Panel(
        f"[bold]Provider:[/bold]          {provider()} / {model_name()}\n"
        f"[bold]Traces attempted:[/bold]  {len(results)}\n"
        f"[bold]Scored:[/bold]            {total}\n"
        f"[bold]Excluded (API):[/bold]    {len(excluded)}\n"
        f"[bold]Correct:[/bold]           {correct}\n"
        f"[bold]Unknown label:[/bold]     {unknown}\n"
        f"[bold]Accuracy:[/bold]          {acc:.1f}%  (of scored only)\n"
        f"[bold]Median latency:[/bold]    {lat[len(lat)//2] if lat else 0}s",
        title="[bold green]Results[/bold green]"))

    out = {
        "provider": provider(),
        "model": model_name(),
        "source": source,
        "attempted": len(results),
        "scored": total,
        "excluded_api": len(excluded),
        "api_failures": dict(api_fail),
        "correct": correct,
        "accuracy_pct": round(acc, 1),
        "accuracy_basis": "scored traces only; API failures excluded",
        "unknown": unknown,
        "median_time_sec": lat[len(lat)//2] if lat else 0,
        "per_type": {k: dict(v) for k, v in per.items()},
        "confusion": {k: dict(v) for k, v in conf.items()},
        "all_results": results,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parser": "src/parser.py (shared)",
    }
    fname = f"accuracy_results_{source}.json"
    json.dump(out, open(os.path.join(BASE, "data", fname), "w"), indent=2)
    console.print(f"[green]Saved → data/{fname}[/green]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="injected", choices=["injected","real","any"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--delay", type=float, default=0.0)
    a = ap.parse_args()
    run(a.source, a.n, a.all, a.delay)