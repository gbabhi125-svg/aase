"""
scripts/probe_ledger.py

Does the Repair Ledger actually retrieve anything?

Zero API calls. Reads the existing ChromaDB store, probes it with its own
stored signatures and with near-variants, sweeps the similarity threshold,
and reports the distance distribution.

Three outcomes, all worth knowing:
  A  exact signatures hit at ~0 distance and variants hit too   -> works, threshold fine
  B  exact hit but variants miss                                -> threshold too tight
  C  even exact signatures miss                                 -> retrieval is broken

  python scripts/probe_ledger.py
"""

import os, sys, json, time, random
from collections import Counter, defaultdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from src.ledger.ledger import RepairLedger, SIMILARITY_THRESHOLD

console = Console()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def variants(sig):
    """Realistic near-misses of a stored signature."""
    out = []
    out.append(("identical", sig))
    out.append(("trailing space", sig + " "))
    out.append(("case changed", sig.upper()))
    if "|" in sig:
        head, _, tail = sig.partition("|")
        out.append(("same failure_type, different task", head + "|task:a completely different request"))
    out.append(("failure_type only", sig.split("|")[0]))
    words = sig.split()
    if len(words) > 4:
        out.append(("one word dropped", " ".join(words[:-1])))
    out.append(("unrelated", "agent:Nothing|check:none|task:the weather in Bengaluru"))
    return out


def main():
    console.print(Panel("[bold cyan]Repair Ledger diagnostic[/bold cyan]\n"
                        "[dim]zero API calls — reads the existing store only[/dim]",
                        title="AASE"))

    led = RepairLedger()
    size = led.stats()["total_repairs"]
    mode = getattr(led, "mode", "unknown")

    console.print(f"[bold]Store size[/bold]      {size} repairs")
    console.print(f"[bold]Embedder[/bold]        {mode}")
    console.print(f"[bold]Threshold[/bold]       {SIMILARITY_THRESHOLD}  "
                  f"[dim](a hit requires distance < this)[/dim]\n")

    if size == 0:
        console.print("[red]Ledger is empty. Run a cycle or seed it first.[/red]")
        return

    # ── pull stored records ──────────────────────────────────────────
    try:
        raw = led.collection.get(include=["metadatas", "documents"])
    except Exception as e:
        console.print(f"[red]Could not read the collection: {e}[/red]")
        return

    metas = raw.get("metadatas", []) or []
    summaries = [(m or {}).get("trace_summary", "") for m in metas]
    summaries = [s for s in summaries if s.strip()]

    console.print(f"[bold]Signatures with text[/bold]  {len(summaries)} of {size}")
    if not summaries:
        console.print("[red]No stored signature text. Nothing can ever match.[/red]")
        console.print("[dim]store_repair() is being called with an empty trace_summary.[/dim]")
        return

    by_type = Counter((m or {}).get("failure_type", "?") for m in metas)
    console.print("[dim]by type: " + ", ".join(f"{k}={v}" for k, v in by_type.most_common()) + "[/dim]\n")

    # ── A. probe with exact stored signatures ────────────────────────
    console.print("[bold]A. Exact stored signatures[/bold]")
    console.print("[dim]These are already in the store. They must match, or retrieval is broken.[/dim]\n")

    sample = random.sample(summaries, min(8, len(summaries)))
    exact_hits, exact_dists = 0, []

    t = Table(show_header=True)
    t.add_column("signature", style="dim", max_width=44)
    t.add_column("distance", style="cyan")
    t.add_column("hit?", style="bold")
    for sig in sample:
        t0 = time.time()
        try:
            hit = led.find_similar_repair(sig)
        except Exception as e:
            t.add_row(sig[:44], "error", f"[red]{str(e)[:24]}[/red]")
            continue
        ms = (time.time() - t0) * 1000
        if hit:
            exact_hits += 1
            d = hit["similarity_distance"]
            exact_dists.append(d)
            t.add_row(sig[:44], f"{d:.4f}", f"[green]YES[/green] ({ms:.1f}ms)")
        else:
            # find out how far the nearest actually was
            d = nearest_distance(led, sig)
            exact_dists.append(d if d is not None else -1)
            t.add_row(sig[:44],
                      f"{d:.4f}" if d is not None else "—",
                      f"[red]NO[/red] ({ms:.1f}ms)")
    console.print(t)
    console.print(f"\n  exact-match hit rate: [bold]{exact_hits}/{len(sample)}[/bold]\n")

    # ── B. near variants ─────────────────────────────────────────────
    console.print("[bold]B. Near variants of one stored signature[/bold]")
    base_sig = sample[0]
    console.print(f"[dim]base: {base_sig[:96]}[/dim]\n")

    t2 = Table(show_header=True)
    t2.add_column("variant", style="bold")
    t2.add_column("nearest distance", style="cyan")
    t2.add_column("under threshold?", style="bold")
    for label, v in variants(base_sig):
        d = nearest_distance(led, v)
        if d is None:
            t2.add_row(label, "—", "[red]no result[/red]")
        else:
            ok = d < SIMILARITY_THRESHOLD
            t2.add_row(label, f"{d:.4f}",
                       "[green]YES[/green]" if ok else "[yellow]no[/yellow]")
    console.print(t2)

    # ── C. threshold sweep ───────────────────────────────────────────
    console.print("\n[bold]C. Threshold sweep on exact signatures[/bold]")
    console.print("[dim]what fraction would hit at each threshold value[/dim]\n")

    probe = random.sample(summaries, min(30, len(summaries)))
    dists = [d for d in (nearest_distance(led, s) for s in probe) if d is not None]

    if not dists:
        console.print("[red]No distances could be computed. Retrieval is returning nothing.[/red]")
        return

    t3 = Table(show_header=True)
    t3.add_column("threshold", style="bold")
    t3.add_column("would hit", style="green")
    t3.add_column("rate", style="cyan")
    for th in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00):
        n = sum(1 for d in dists if d < th)
        mark = "  <- current" if abs(th - SIMILARITY_THRESHOLD) < 1e-9 else ""
        t3.add_row(f"{th:.2f}{mark}", f"{n}/{len(dists)}", f"{n/len(dists)*100:.0f}%")
    console.print(t3)

    lo, hi = min(dists), max(dists)
    med = sorted(dists)[len(dists)//2]
    console.print(f"\n  distance on exact signatures — min {lo:.4f} · median {med:.4f} · max {hi:.4f}")

    # ── verdict ──────────────────────────────────────────────────────
    console.print()
    if exact_hits == len(sample) and med < 0.05:
        verdict = ("[bold green]A — retrieval works.[/bold green]\n"
                   "Exact signatures match at near-zero distance. If the console still "
                   "reports no match on a repeat run, the signature string is being built "
                   "differently between the write and the probe. Compare signature_of() in "
                   "live.py against the string passed to store_repair().")
    elif exact_hits == 0 and med >= SIMILARITY_THRESHOLD:
        verdict = (f"[bold yellow]B — threshold too tight.[/bold yellow]\n"
                   f"Exact signatures sit at median distance {med:.4f}, above the "
                   f"threshold of {SIMILARITY_THRESHOLD}. Nothing can ever hit. "
                   f"Set SIMILARITY_THRESHOLD in .env above {med:.2f} and re-run this script.")
    elif exact_hits == 0:
        verdict = ("[bold red]C — retrieval is broken.[/bold red]\n"
                   "Signatures stored in the collection do not come back when queried. "
                   "Check that _embed() is deterministic and that store_repair() and "
                   "find_similar_repair() embed the same field.")
    else:
        verdict = (f"[bold yellow]Partial.[/bold yellow]\n"
                   f"{exact_hits} of {len(sample)} exact signatures matched, median "
                   f"distance {med:.4f} against a threshold of {SIMILARITY_THRESHOLD}. "
                   f"Retrieval works but the threshold is marginal.")

    console.print(Panel(verdict, title="[bold]Verdict[/bold]"))

    out = {
        "store_size": size, "embedder": mode, "threshold": SIMILARITY_THRESHOLD,
        "signatures_with_text": len(summaries),
        "by_type": dict(by_type),
        "exact_hit_rate": f"{exact_hits}/{len(sample)}",
        "distance_min": round(lo, 4), "distance_median": round(med, 4),
        "distance_max": round(hi, 4),
        "sweep": {str(th): sum(1 for d in dists if d < th) / len(dists)
                  for th in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00)},
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(out, open(os.path.join(BASE, "data", "ledger_diagnostic.json"), "w"), indent=2)
    console.print("[green]Saved -> data/ledger_diagnostic.json[/green]")


def nearest_distance(led, text):
    """Distance to the closest stored record, ignoring the threshold."""
    try:
        if led.collection.count() == 0:
            return None
        res = led.collection.query(
            query_embeddings=[led._embed(text)],
            n_results=1,
            include=["distances"],
        )
        ds = res.get("distances") or []
        if ds and ds[0]:
            return float(ds[0][0])
    except Exception:
        pass
    return None


if __name__ == "__main__":
    main()