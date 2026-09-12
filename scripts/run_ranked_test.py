"""
scripts/run_ranked_test.py

The real-world number sits at 34.2% pooled with a 19.4-50.0% spread, and the
disagreements cluster on traces whose human annotation describes a mistake AND
its consequence in one sentence. A single forced label cannot represent that.

This harness asks the Coroner for a primary label, a runner-up, a confidence
level and an ambiguity flag — over the SAME decision procedure — and scores
four ways:

  top-1        the existing metric, unchanged, so numbers stay comparable
  top-2        ground truth appears as primary OR alternative
  calibration  top-1 accuracy split by the confidence the Coroner stated
  separation   top-1 when it flagged ambiguity vs when it did not

The last two matter more than the first two. If high-confidence verdicts are
far more accurate than low-confidence ones, the system knows what it does not
know — and that is reportable in a way a flat accuracy number is not.

  python scripts/run_ranked_test.py --source real --n 10
  python scripts/run_ranked_test.py --source injected --n 10
  python scripts/run_ranked_test.py --source real --n 10 --delay 1
"""

import os, sys, json, time, argparse
from collections import defaultdict, Counter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from src.parser import FAILURE_TYPES, parse_ranked_verdict, extract_fix, is_api_failure
from src.council.prosecutor import ProsecutorAgent
from src.council.defender import DefenderAgent
from src.council.coroner import CoronerAgent
from src.llm import provider, model_name

console = Console()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANIFESTS = {
    "injected": ("injected_manifest.json", "injected"),
    "real":     ("real_failures_manifest.json", "real_failures"),
}


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
                            "trace_id": e["trace_id"], "failure_type": ft,
                            "trace": d.get("trace", []),
                            "task": d.get("task") or d.get("question", ""),
                            "annotation": d.get("mistake_reason", ""),
                        })
                    except Exception:
                        pass
                    break
    return picked


def run(source="real", per_type=10, delay=0.0):
    console.print(Panel(
        f"[bold cyan]Ambiguity-aware diagnosis[/bold cyan]\n"
        f"[dim]provider={provider()} · model={model_name()}\n"
        f"source={source} · {per_type} per type · same decision procedure, ranked output[/dim]",
        title="AASE"))

    traces = load_traces(source, per_type)
    if not traces:
        console.print("[red]No traces loaded.[/red]")
        return

    dist = Counter(t["failure_type"] for t in traces)
    console.print(f"[green]{len(traces)} traces[/green] — " +
                  ", ".join(f"{k}={v}" for k, v in dist.items()) + "\n")

    prosecutor, defender, coroner = ProsecutorAgent(), DefenderAgent(), CoronerAgent()

    rows, api_fail = [], Counter()
    consecutive = 0

    for i, item in enumerate(traces):
        ft, tid = item["failure_type"], item["trace_id"]
        tstr = json.dumps(item["trace"], indent=2)[:3000]
        unk = "unknown — determine from trace"

        console.print(f"[dim][{i+1}/{len(traces)}] {tid} — {ft}[/dim]")
        t0 = time.time()
        pa = prosecutor.analyze(tstr, unk, item["task"])
        da = defender.analyze(tstr, unk, item["task"])
        vd = coroner.decide_ranked(tstr, pa, da, unk)
        el = round(time.time() - t0, 1)

        fail = is_api_failure(vd) or is_api_failure(pa) or is_api_failure(da)
        if fail:
            consecutive += 1
            api_fail[fail] += 1
            console.print(f"  → [red]{fail} — EXCLUDED[/red] ({el}s)")
            rows.append({"trace_id": tid, "gt": ft, "excluded": True, "reason": fail})
            if consecutive >= 5:
                console.print(Panel(f"[bold red]{consecutive} consecutive API failures — "
                                    f"stopping.[/bold red]", title="[red]ABORTED[/red]"))
                break
            if delay:
                time.sleep(delay)
            continue
        consecutive = 0

        r = parse_ranked_verdict(vd)
        top1 = (r["primary"] == ft)
        top2 = ft in {r["primary"], r["alternative"]}

        mark = "[green]TOP-1[/green]" if top1 else \
               ("[cyan]TOP-2[/cyan]" if top2 else "[yellow]MISS[/yellow]")
        alt = r["alternative"] or "—"
        flag = " [magenta]AMB[/magenta]" if r["ambiguous"] else ""
        console.print(f"  → {mark} {r['primary']} / {alt} · conf={r['confidence']}{flag} ({el}s)")

        rows.append({"trace_id": tid, "gt": ft, "excluded": False,
                     "primary": r["primary"], "alternative": r["alternative"],
                     "confidence": r["confidence"], "ambiguous": r["ambiguous"],
                     "top1": top1, "top2": top2, "time": el,
                     "annotation": item["annotation"][:200],
                     "fix": extract_fix(vd)[:200]})
        if delay:
            time.sleep(delay)

    scored = [r for r in rows if not r["excluded"]]
    excl = [r for r in rows if r["excluded"]]
    n = len(scored)
    if n == 0:
        console.print("[red]Nothing scored.[/red]")
        return

    top1 = sum(1 for r in scored if r["top1"])
    top2 = sum(1 for r in scored if r["top2"])

    # ── headline ─────────────────────────────────────────────────────
    t = Table(title=f"Ranked diagnosis — {source} (N={n} scored)")
    t.add_column("Metric", style="bold")
    t.add_column("Count", style="green")
    t.add_column("Rate", style="cyan")
    t.add_row("top-1  (primary correct)", f"{top1}/{n}", f"{top1/n*100:.1f}%")
    t.add_row("top-2  (primary or alternative)", f"{top2}/{n}", f"{top2/n*100:.1f}%")
    t.add_row("lift from the runner-up", f"+{top2-top1}", f"+{(top2-top1)/n*100:.1f} pts")
    console.print(t)

    # ── calibration ──────────────────────────────────────────────────
    by_conf = defaultdict(lambda: {"n": 0, "t1": 0, "t2": 0})
    for r in scored:
        c = by_conf[r["confidence"]]
        c["n"] += 1
        c["t1"] += r["top1"]
        c["t2"] += r["top2"]

    ct = Table(title="Calibration — accuracy by the confidence the Coroner stated")
    ct.add_column("Confidence", style="bold")
    ct.add_column("N", style="white")
    ct.add_column("top-1", style="green")
    ct.add_column("top-2", style="cyan")
    for c in ("high", "medium", "low", "unknown"):
        s = by_conf.get(c)
        if not s or s["n"] == 0:
            continue
        ct.add_row(c, str(s["n"]),
                   f"{s['t1']}/{s['n']} = {s['t1']/s['n']*100:.0f}%",
                   f"{s['t2']}/{s['n']} = {s['t2']/s['n']*100:.0f}%")
    console.print(ct)

    # ── separation ───────────────────────────────────────────────────
    amb = [r for r in scored if r["ambiguous"]]
    una = [r for r in scored if not r["ambiguous"]]
    st = Table(title="Separation — does flagging ambiguity isolate the hard cases?")
    st.add_column("Group", style="bold")
    st.add_column("N", style="white")
    st.add_column("top-1", style="green")
    st.add_column("top-2", style="cyan")
    for label, g in (("flagged ambiguous", amb), ("not flagged", una)):
        if not g:
            continue
        a1 = sum(1 for r in g if r["top1"])
        a2 = sum(1 for r in g if r["top2"])
        st.add_row(label, str(len(g)),
                   f"{a1}/{len(g)} = {a1/len(g)*100:.0f}%",
                   f"{a2}/{len(g)} = {a2/len(g)*100:.0f}%")
    console.print(st)

    # ── per type ─────────────────────────────────────────────────────
    per = defaultdict(lambda: {"n": 0, "t1": 0, "t2": 0})
    for r in scored:
        p = per[r["gt"]]
        p["n"] += 1
        p["t1"] += r["top1"]
        p["t2"] += r["top2"]
    pt = Table(title="By failure type")
    pt.add_column("Type", style="bold")
    pt.add_column("N", style="white")
    pt.add_column("top-1", style="green")
    pt.add_column("top-2", style="cyan")
    for ft in FAILURE_TYPES:
        s = per.get(ft)
        if not s or s["n"] == 0:
            continue
        pt.add_row(ft, str(s["n"]),
                   f"{s['t1']}/{s['n']}", f"{s['t2']}/{s['n']}")
    console.print(pt)

    # ── recovered pairs ──────────────────────────────────────────────
    rec = [r for r in scored if r["top2"] and not r["top1"]]
    if rec:
        console.print("\n[bold]Recovered by the runner-up — the confusable pairs[/bold]")
        pairs = Counter(f"{r['gt']} ↔ {r['primary']}" for r in rec)
        for k, v in pairs.most_common(8):
            console.print(f"  {v:>3}×  {k}")

    if excl:
        console.print(f"\n[red]Excluded for API failures: {len(excl)}[/red] — " +
                      ", ".join(f"{k}={v}" for k, v in api_fail.items()))

    lat = sorted(r["time"] for r in scored)
    console.print(Panel(
        f"[bold]Scored:[/bold]        {n}\n"
        f"[bold]Excluded:[/bold]      {len(excl)}\n"
        f"[bold]top-1:[/bold]         {top1}/{n} = {top1/n*100:.1f}%\n"
        f"[bold]top-2:[/bold]         {top2}/{n} = {top2/n*100:.1f}%\n"
        f"[bold]Flagged amb.:[/bold]  {len(amb)}/{n}\n"
        f"[bold]Median:[/bold]        {lat[len(lat)//2]}s\n\n"
        f"[dim]Report top-1 as the headline. top-2 is a secondary measure and must "
        f"be labelled as such — it is not a replacement for top-1.[/dim]",
        title="[bold green]Result[/bold green]"))

    out = {
        "provider": provider(), "model": model_name(), "source": source,
        "scored": n, "excluded": len(excl), "api_failures": dict(api_fail),
        "top1": top1, "top1_pct": round(top1 / n * 100, 1),
        "top2": top2, "top2_pct": round(top2 / n * 100, 1),
        "calibration": {k: dict(v) for k, v in by_conf.items()},
        "ambiguous_flagged": len(amb),
        "separation": {
            "flagged": {"n": len(amb),
                        "top1": sum(1 for r in amb if r["top1"]),
                        "top2": sum(1 for r in amb if r["top2"])},
            "not_flagged": {"n": len(una),
                            "top1": sum(1 for r in una if r["top1"]),
                            "top2": sum(1 for r in una if r["top2"])},
        },
        "per_type": {k: dict(v) for k, v in per.items()},
        "recovered_pairs": dict(Counter(f"{r['gt']}->{r['primary']}" for r in rec)),
        "rows": rows,
        "note": ("top-1 is the headline metric and is directly comparable with "
                 "run_accuracy_test.py. top-2 is reported as a secondary measure."),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    fp = os.path.join(BASE, "data", f"ranked_results_{source}.json")
    json.dump(out, open(fp, "w"), indent=2)
    console.print(f"[green]Saved → data/ranked_results_{source}.json[/green]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="real", choices=["real", "injected"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--delay", type=float, default=0.0)
    a = ap.parse_args()
    run(a.source, a.n, a.delay)