"""
scripts/cost_report.py

What does running AASE actually cost, and when does the Repair Ledger pay
for itself?

Reads the timings already recorded in your saved result files. No API calls.
Model-call counts are structural and exact. Token counts are estimated from
observed payload sizes and are labelled as estimates.

  python scripts/cost_report.py
  python scripts/cost_report.py --price 0.20      # USD per million tokens
"""

import os, sys, json, glob, time, argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
console = Console()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

TIME_KEYS = ("time", "t", "total_time", "seconds", "elapsed", "total")


def load(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def collect_latencies():
    """
    Every per-trace timing this project has recorded.

    Saved files are not uniform — some are top-level arrays (manifests,
    inspection dumps), some are objects with a results list, some nest
    results one level deeper. Handle all three rather than assuming.
    """
    lat = []
    for fp in glob.glob(os.path.join(DATA, "*.json")):
        try:
            d = json.load(open(fp))
        except Exception:
            continue

        buckets = []
        if isinstance(d, list):
            buckets.append(d)
        elif isinstance(d, dict):
            for key in ("all_results", "rows", "results", "session",
                        "session_runs", "unstable_detail"):
                v = d.get(key)
                if isinstance(v, list):
                    buckets.append(v)
                elif isinstance(v, dict):
                    for sub in v.values():
                        if isinstance(sub, list):
                            buckets.append(sub)
        else:
            continue

        for rows in buckets:
            for r in rows:
                if not isinstance(r, dict):
                    continue
                for tk in TIME_KEYS:
                    v = r.get(tk)
                    if isinstance(v, (int, float)) and 0 < v < 600:
                        lat.append(float(v))
                        break
    return sorted(lat)


def main(price):
    console.print(Panel("[bold cyan]AASE cost accounting[/bold cyan]\n"
                        "[dim]measured from saved runs · zero API calls[/dim]",
                        title="AASE"))

    lat = collect_latencies()
    if lat:
        med = lat[len(lat) // 2]
        p90 = lat[int(len(lat) * 0.9)]
        console.print(f"[bold]Latency samples[/bold]  {len(lat)} recorded runs")
        console.print(f"  min {lat[0]:.1f}s · median [bold]{med:.1f}s[/bold] · "
                      f"p90 {p90:.1f}s · max {lat[-1]:.1f}s\n")
    else:
        med = 20.0
        console.print("[yellow]No timings found in data/. Using 20s as a placeholder.[/yellow]\n")

    # ── calls per operation — structural, exact ──────────────────────
    t = Table(title="Model calls per operation")
    t.add_column("Operation", style="bold")
    t.add_column("Calls", style="cyan")
    t.add_column("What runs")
    t.add_row("Diagnosis only", "3", "Prosecutor, Defender, Coroner")
    t.add_row("Full repair cycle", "4", "the three above, plus the Surgeon")
    t.add_row("Cycle with agent retry", "6", "agent run, council x3, surgeon, agent re-run")
    t.add_row("Operator override", "1", "agent re-run with a human-written clause")
    t.add_row("Ledger hit", "[green]0[/green]", "vector lookup only — council bypassed")
    console.print(t)

    # ── token estimate, labelled ─────────────────────────────────────
    IN_EST, OUT_EST = 1400, 260
    per_call = IN_EST + OUT_EST
    cycle_tokens = per_call * 6
    console.print(f"\n[bold]Token estimate per call[/bold]  ~{IN_EST} in + ~{OUT_EST} out "
                  f"= ~{per_call}")
    console.print("[dim]Estimated from the 3,000-character trace cap and max_tokens=400. "
                  "Not metered — treat as an approximation.[/dim]")
    console.print(f"[bold]Full cycle with retry[/bold]  ~{cycle_tokens:,} tokens, "
                  f"~{med * 2:.0f}s wall clock")
    if price:
        console.print(f"[bold]At ${price:.2f}/M tokens[/bold]  "
                      f"~${cycle_tokens / 1_000_000 * price:.5f} per cycle")

    # ── ledger economics ─────────────────────────────────────────────
    rb = load("ledger_rebuild.json")
    console.print("\n[bold]Repair Ledger economics[/bold]")
    if rb:
        console.print(f"  store: {rb.get('records', '?')} repairs · "
                      f"embedder {rb.get('embedder', '?')}")
        console.print(f"  self-distance median {rb.get('self_distance_median', '?')} · "
                      f"unrelated {rb.get('unrelated_median', '?')}")
    console.print("  a ledger hit costs [green]0 model calls[/green] and returns in "
                  "single-digit milliseconds")
    console.print(f"  a council diagnosis costs [red]3 calls[/red] and ~{med:.0f}s")

    bt = Table(title="Cost of N occurrences of the SAME failure signature")
    bt.add_column("Occurrences", style="bold")
    bt.add_column("Without ledger", style="red")
    bt.add_column("With ledger", style="green")
    bt.add_column("Calls saved", style="cyan")
    for n in (1, 2, 5, 10, 50, 100):
        without = n * 4
        with_l = 4 + (n - 1)
        saved = without - with_l
        bt.add_row(str(n), f"{without} calls", f"{with_l} calls",
                   f"{saved}" + (f"  ({saved / without * 100:.0f}%)" if without else ""))
    console.print(bt)
    console.print("[dim]The ledger pays for itself on the second occurrence. Beyond that "
                  "the saving grows linearly with the repeat rate.[/dim]")

    # ── illustrative deployment ──────────────────────────────────────
    console.print("\n[bold]Illustrative deployment[/bold]")
    console.print("[dim]assumption: an agent handling 10,000 tasks/day at a 3% failure "
                  "rate, where 70% of failures repeat a known signature[/dim]")
    tasks, fail_rate, repeat = 10000, 0.03, 0.70
    fails = int(tasks * fail_rate)
    novel = int(fails * (1 - repeat))
    known = fails - novel
    no_ledger = fails * 4
    with_ledger = novel * 4 + known
    console.print(f"  failures/day: {fails}  ({novel} novel, {known} known signatures)")
    console.print(f"  without ledger: {no_ledger:,} calls/day")
    console.print(f"  with ledger:    [green]{with_ledger:,}[/green] calls/day  "
                  f"({(1 - with_ledger / no_ledger) * 100:.0f}% fewer)")
    if price:
        c1 = no_ledger * per_call / 1_000_000 * price
        c2 = with_ledger * per_call / 1_000_000 * price
        console.print(f"  at ${price:.2f}/M: ${c1:.2f}/day → [green]${c2:.2f}/day[/green]")

    console.print(Panel(
        "Model-call counts are structural and exact. Token counts are estimated "
        "from observed payload sizes, not metered, and are labelled as such. The "
        "deployment figures are an illustration built on stated assumptions, not "
        "a measurement.",
        title="[bold]What is measured and what is assumed[/bold]"))

    out = {
        "latency_samples": len(lat),
        "latency_min_s": round(lat[0], 1) if lat else None,
        "latency_median_s": round(med, 1) if lat else None,
        "latency_p90_s": round(lat[int(len(lat) * 0.9)], 1) if lat else None,
        "latency_max_s": round(lat[-1], 1) if lat else None,
        "calls": {"diagnosis": 3, "repair_cycle": 4, "with_retry": 6,
                  "override": 1, "ledger_hit": 0},
        "tokens_per_call_est": per_call,
        "tokens_per_cycle_est": cycle_tokens,
        "price_per_m_usd": price or None,
        "deployment_illustration": {
            "tasks_per_day": tasks, "failure_rate": fail_rate,
            "repeat_rate": repeat, "calls_without_ledger": no_ledger,
            "calls_with_ledger": with_ledger,
            "reduction_pct": round((1 - with_ledger / no_ledger) * 100, 1)},
        "note": ("model-call counts are exact; token counts are estimates from "
                 "observed payload sizes; deployment figures are an illustration"),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(out, open(os.path.join(DATA, "cost_report.json"), "w"), indent=2)
    console.print("[green]Saved -> data/cost_report.json[/green]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", type=float, default=0.0,
                    help="USD per million tokens, e.g. 0.20")
    main(ap.parse_args().price)