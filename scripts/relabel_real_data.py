"""
scripts/relabel_real_data.py

The real-data ground-truth labels were produced by keyword-matching the Who&When
annotators' free-text mistake_reason onto our 7 classes. That mapping is crude —
"used the wrong search query" got labelled hallucination when it is tool_misuse.
The 39.1% figure was therefore measuring label quality, not Coroner accuracy.

This script relabels using an LLM applied to the ANNOTATOR'S TEXT ONLY.
It never sees the trace, so it cannot leak into the Coroner's later judgement —
it is a labeller, not a second diagnosis.

Every change is logged. Nothing is silently overwritten.

  python scripts/relabel_real_data.py --dry-run     # preview, no writes
  python scripts/relabel_real_data.py               # apply
  python scripts/relabel_real_data.py --limit 30    # partial run
"""

import os, sys, json, time, argparse, shutil
from collections import defaultdict, Counter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from src.parser import FAILURE_TYPES, normalise
from src.llm import complete, LLMError, provider, model_name

console = Console()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(BASE, "data", "real_failures_manifest.json")
REAL_DIR = os.path.join(BASE, "data", "real_failures")

VALID = ", ".join(FAILURE_TYPES)

LABELLER_PROMPT = f"""You are a data annotator. You are given a human expert's
free-text description of why an AI agent failed. Map it onto exactly one of these
seven categories.

You are NOT diagnosing the failure. You are translating an existing human
judgement into our vocabulary. Trust what the human wrote.

CATEGORIES:
hallucination — the agent stated facts, values or results that were not returned
  by any tool, or claimed to have done something it never did. Includes: assumed,
  fabricated, invented, unverified claim, incorrect value with no source.

tool_misuse — the agent used the wrong tool, wrong API, wrong search query, wrong
  URL, wrong parameters, or malformed action syntax. Includes: bad query
  formulation, failed to use an available tool, navigated to the wrong place.

reasoning_loop — the agent repeated the same action or query multiple times with
  no variation and no progress.

context_collapse — the agent lost or oversimplified information it had earlier,
  or stopped following an instruction given at the start. Gradual degradation
  while still running.

goal_drift — the agent's final output did not satisfy the original request:
  wrong scope, wrong recipient, incomplete answer, concluded prematurely,
  ignored a stated constraint, or answered a different question.

prompt_injection — content from an external document, page or tool contained
  instructions and the agent obeyed them.

memory_overflow — a hard limit stopped execution: token limit, context length
  exceeded, step limit reached, timeout, process killed.

If the description fits more than one, choose the one describing the ROOT cause
— the earliest thing that went wrong, not its downstream effect.

Reply with ONE line only, nothing else:
LABEL: <one of: {VALID}>"""


def load_entries():
    if not os.path.exists(MANIFEST):
        console.print(f"[red]{MANIFEST} not found[/red]")
        return []
    out = []
    for e in json.load(open(MANIFEST)):
        ft = e.get("failure_type")
        for cand in (os.path.join(BASE, e["file"]),
                     os.path.join(REAL_DIR, ft, os.path.basename(e["file"]))):
            if os.path.exists(cand):
                try:
                    out.append({"entry": e, "path": cand, "data": json.load(open(cand))})
                except Exception:
                    pass
                break
    return out


def label_one(reason, question):
    user = (f"Task the agent was given: {question[:300]}\n\n"
            f"Human expert's description of the failure:\n{reason[:900]}\n\n"
            f"LABEL:")
    out = complete(LABELLER_PROMPT, user, 40)
    for line in out.split("\n"):
        if "label" in line.lower():
            v = normalise(line.split(":", 1)[-1])
            if v:
                return v
    return normalise(out)


def run(dry_run=False, limit=None, delay=0.0):
    console.print(Panel(
        f"[bold cyan]Real-data relabelling[/bold cyan]\n"
        f"[dim]provider={provider()} · model={model_name()}\n"
        f"labeller sees ONLY the human annotation, never the trace[/dim]",
        title="AASE"))

    items = load_entries()
    if not items:
        return
    if limit:
        items = items[:limit]

    with_reason = [i for i in items if (i["data"].get("mistake_reason") or "").strip()]
    console.print(f"[green]{len(items)} traces[/green] · "
                  f"{len(with_reason)} carry a human annotation\n")

    before = Counter(i["entry"]["failure_type"] for i in items)
    changes, unchanged, failed = [], 0, 0
    new_counts = Counter()
    consecutive_api = 0

    for n, it in enumerate(items):
        e, d = it["entry"], it["data"]
        old = e["failure_type"]
        reason = (d.get("mistake_reason") or "").strip()
        tid = e["trace_id"]

        if not reason:
            new_counts[old] += 1
            unchanged += 1
            console.print(f"[dim][{n+1}/{len(items)}] {tid} — no annotation, keeping {old}[/dim]")
            continue

        try:
            new = label_one(reason, d.get("question", ""))
            consecutive_api = 0
        except LLMError as ex:
            failed += 1
            consecutive_api += 1
            new_counts[old] += 1
            console.print(f"[red][{n+1}/{len(items)}] {tid} — {ex.tag}, keeping {old}[/red]")
            if consecutive_api >= 5:
                console.print(Panel("[bold red]5 consecutive API failures — stopping.[/bold red]",
                                    title="[red]ABORTED[/red]"))
                break
            if delay:
                time.sleep(delay)
            continue

        if not new:
            new = old
        new_counts[new] += 1

        if new != old:
            changes.append({"trace_id": tid, "old": old, "new": new,
                            "reason": reason[:220], "path": it["path"],
                            "entry": e, "data": d})
            console.print(f"[{n+1}/{len(items)}] {tid} — "
                          f"[yellow]{old}[/yellow] → [green]{new}[/green]")
            console.print(f"    [dim]{reason[:110]}[/dim]")
        else:
            unchanged += 1
            console.print(f"[dim][{n+1}/{len(items)}] {tid} — {old} confirmed[/dim]")

        if delay:
            time.sleep(delay)

    # ── Report ───────────────────────────────────────────────────────
    t = Table(title="Label distribution")
    t.add_column("Failure Type", style="bold")
    t.add_column("Before (keyword)", style="yellow")
    t.add_column("After (LLM)", style="green")
    t.add_column("Delta", style="cyan")
    for ft in FAILURE_TYPES:
        b, a = before.get(ft, 0), new_counts.get(ft, 0)
        if b == 0 and a == 0:
            continue
        d_ = a - b
        t.add_row(ft, str(b), str(a), f"{d_:+d}" if d_ else "—")
    console.print(t)

    if changes:
        mt = Counter(f"{c['old']} → {c['new']}" for c in changes)
        console.print("\n[bold]Most common corrections[/bold]")
        for k, v in mt.most_common(10):
            console.print(f"  {v:>3}×  {k}")

    console.print(Panel(
        f"[bold]Examined:[/bold]   {len(items)}\n"
        f"[bold]Relabelled:[/bold] {len(changes)}\n"
        f"[bold]Confirmed:[/bold]  {unchanged}\n"
        f"[bold]API failed:[/bold] {failed}\n"
        f"[bold]Agreement with original keyword labels:[/bold] "
        f"{unchanged/max(len(items),1)*100:.1f}%",
        title="[bold green]Summary[/bold green]"))

    audit = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": provider(), "model": model_name(),
        "method": "LLM labeller applied to human annotation text only; trace never shown",
        "examined": len(items), "relabelled": len(changes),
        "confirmed": unchanged, "api_failed": failed,
        "before": dict(before), "after": dict(new_counts),
        "changes": [{k: c[k] for k in ("trace_id", "old", "new", "reason")} for c in changes],
    }
    json.dump(audit, open(os.path.join(BASE, "data", "relabel_audit.json"), "w"), indent=2)
    console.print("[green]Audit → data/relabel_audit.json[/green]")

    if dry_run:
        console.print("\n[yellow]DRY RUN — nothing written. Re-run without --dry-run to apply.[/yellow]")
        return

    if not changes:
        console.print("\n[green]No changes needed.[/green]")
        return

    # Back up before touching anything
    bak = os.path.join(BASE, "data", "real_failures_manifest.backup.json")
    if not os.path.exists(bak):
        shutil.copy(MANIFEST, bak)
        console.print(f"[dim]Manifest backed up → real_failures_manifest.backup.json[/dim]")

    manifest = json.load(open(MANIFEST))
    by_id = {c["trace_id"]: c for c in changes}
    moved = 0

    for m in manifest:
        c = by_id.get(m["trace_id"])
        if not c:
            continue
        old_path = c["path"]
        new_dir = os.path.join(REAL_DIR, c["new"])
        os.makedirs(new_dir, exist_ok=True)
        new_path = os.path.join(new_dir, os.path.basename(old_path))

        d = c["data"]
        d["failure_type"] = c["new"]
        d["original_keyword_label"] = c["old"]
        d["relabelled_by"] = f"llm:{model_name()}"
        json.dump(d, open(new_path, "w"), indent=2)
        if os.path.abspath(old_path) != os.path.abspath(new_path):
            try:
                os.remove(old_path)
            except OSError:
                pass

        m["failure_type"] = c["new"]
        m["file"] = f"data/real_failures/{c['new']}/{os.path.basename(new_path)}"
        m["original_keyword_label"] = c["old"]
        moved += 1

    json.dump(manifest, open(MANIFEST, "w"), indent=2)
    console.print(f"[green]Applied {moved} relabels. Manifest updated.[/green]")
    console.print("\n[bold]Next:[/bold] python scripts/run_accuracy_test.py --n 10 --source real")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--delay", type=float, default=0.0)
    a = ap.parse_args()
    run(a.dry_run, a.limit, a.delay)