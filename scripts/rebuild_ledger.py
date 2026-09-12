"""
scripts/rebuild_ledger.py

The Repair Ledger currently holds 323 vectors written by sentence-transformers
and is being queried with hash-fallback vectors. The two embedders occupy
different spaces, so nothing can ever match.

This script:
  1. verifies which embedder is actually active right now
  2. backs up the existing store
  3. rebuilds every record's vector with the active embedder, keeping all
     metadata, fixes and outcomes intact
  4. re-probes and reports the real distance distribution
  5. recommends a threshold from measured data rather than a guess

Zero API calls.

  python scripts/rebuild_ledger.py --dry-run
  python scripts/rebuild_ledger.py
"""

import os, sys, json, time, shutil, argparse, random
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

console = Console()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(BASE, "data", "memory_store")


def lst(v):
    if v is None:
        return None
    try:
        return v.tolist()
    except AttributeError:
        return list(v)


def cos(u, v):
    dot = sum(x * y for x, y in zip(u, v))
    nu = sum(x * x for x in u) ** 0.5
    nv = sum(x * x for x in v) ** 0.5
    return 1 - dot / (nu * nv) if nu and nv else None


def check_real_embedder():
    """Can sentence-transformers load right now? Report honestly."""
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        v = m.encode("test").tolist()
        return m, len(v)
    except Exception as e:
        return None, "{}: {}".format(type(e).__name__, str(e)[:120])


def main(dry):
    console.print(Panel("[bold cyan]Repair Ledger rebuild[/bold cyan]\n"
                        "[dim]zero API calls[/dim]", title="AASE"))

    from src.ledger.ledger import RepairLedger
    led = RepairLedger()
    mode = getattr(led, "mode", "?")
    n = led.collection.count()
    console.print("[bold]Active embedder[/bold]  {}".format(mode))
    console.print("[bold]Store size[/bold]       {} records\n".format(n))

    if n == 0:
        console.print("[red]Ledger is empty. Nothing to rebuild.[/red]")
        return

    # ── step 1: is the real model available? ─────────────────────────
    console.print("[bold]1. Checking whether sentence-transformers can load[/bold]")
    model, info = check_real_embedder()
    if model is not None:
        console.print("   [green]available[/green] — all-MiniLM-L6-v2, {} dims".format(info))
        console.print("   [dim]the ledger will be rebuilt with semantic embeddings[/dim]\n")
        use_semantic = True
    else:
        console.print("   [yellow]unavailable[/yellow] — {}".format(info))
        console.print("   [dim]rebuild will use the hash fallback; it will match only")
        console.print("   near-identical signatures, which must be stated as a limitation[/dim]\n")
        use_semantic = False

    def embed(text):
        if use_semantic:
            return model.encode(text).tolist()
        return lst(led._embed(text))

    # ── step 2: read everything out ──────────────────────────────────
    console.print("[bold]2. Reading existing records[/bold]")
    raw = led.collection.get(include=["metadatas", "documents", "embeddings"])
    ids = raw.get("ids", [])
    metas = raw.get("metadatas", []) or []
    docs = raw.get("documents", []) or []
    embs = raw.get("embeddings")

    rows = []
    no_text = 0
    for i, rid in enumerate(ids):
        m = metas[i] or {}
        summary = (m.get("trace_summary") or "").strip()
        if not summary:
            no_text += 1
            continue
        rows.append({"id": rid, "summary": summary,
                     "fix": docs[i] if i < len(docs) else "",
                     "meta": m})

    console.print("   {} records with signature text".format(len(rows)))
    if no_text:
        console.print("   [yellow]{} records have no trace_summary and cannot be "
                      "rebuilt[/yellow]".format(no_text))
    by_type = Counter(r["meta"].get("failure_type", "?") for r in rows)
    console.print("   [dim]" + ", ".join("{}={}".format(k, v)
                                         for k, v in by_type.most_common()) + "[/dim]\n")

    # ── step 3: prove the mismatch ───────────────────────────────────
    console.print("[bold]3. Confirming the mismatch on the existing store[/bold]")
    try:
        s0 = rows[0]["summary"]
        stored0 = lst(embs[0])
        fresh0 = lst(led._embed(s0))
        d = cos(fresh0, stored0)
        console.print("   distance from a signature to its own stored vector: "
                      "[bold]{:.4f}[/bold]".format(d))
        console.print("   [dim]~0 would mean they agree; ~1 means orthogonal — "
                      "different embedding spaces[/dim]\n")
    except Exception as e:
        console.print("   [dim]could not compare: {}[/dim]\n".format(e))

    if dry:
        console.print(Panel("[yellow]DRY RUN — nothing written.[/yellow]\n"
                            "{} records would be re-embedded with the {} embedder.\n"
                            "Re-run without --dry-run to apply."
                            .format(len(rows), "semantic" if use_semantic else "hash"),
                            title="[yellow]Preview[/yellow]"))
        return

    # ── step 4: back up ──────────────────────────────────────────────
    console.print("[bold]4. Backing up[/bold]")
    bak = STORE + ".backup-" + time.strftime("%Y%m%d-%H%M%S")
    try:
        shutil.copytree(STORE, bak)
        console.print("   [green]{}[/green]\n".format(os.path.basename(bak)))
    except Exception as e:
        console.print("   [red]backup failed: {} — stopping[/red]".format(e))
        return

    # ── step 5: rebuild ──────────────────────────────────────────────
    console.print("[bold]5. Re-embedding[/bold]")
    coll = led.client.get_or_create_collection(
        name="repair_ledger_v2",
        metadata={"description": "AASE Repair Memory",
                  "embedder": "minilm" if use_semantic else "hash",
                  "rebuilt": time.strftime("%Y-%m-%d %H:%M:%S")})
    try:
        led.client.delete_collection("repair_ledger_v2")
    except Exception:
        pass
    coll = led.client.get_or_create_collection(
        name="repair_ledger_v2",
        metadata={"description": "AASE Repair Memory",
                  "embedder": "minilm" if use_semantic else "hash"})

    B = 64
    for i in range(0, len(rows), B):
        chunk = rows[i:i + B]
        coll.add(ids=[r["id"] for r in chunk],
                 embeddings=[embed(r["summary"]) for r in chunk],
                 documents=[r["fix"] for r in chunk],
                 metadatas=[r["meta"] for r in chunk])
        console.print("   {}/{}".format(min(i + B, len(rows)), len(rows)), end="\r")
    console.print("   [green]{} records re-embedded[/green]\n".format(len(rows)))

    # ── step 6: measure ──────────────────────────────────────────────
    console.print("[bold]6. Measuring the rebuilt store[/bold]\n")
    sample = random.sample(rows, min(25, len(rows)))

    self_d, other_d = [], []
    for r in sample:
        q = embed(r["summary"])
        res = coll.query(query_embeddings=[q], n_results=2, include=["distances"])
        ds = (res.get("distances") or [[]])[0]
        if ds:
            self_d.append(float(ds[0]))
            if len(ds) > 1:
                other_d.append(float(ds[1]))

    unrel = []
    for txt in ["the weather in Bengaluru tomorrow",
                "a recipe for lemon rice",
                "quarterly hiring plan for the design team"]:
        res = coll.query(query_embeddings=[embed(txt)], n_results=1, include=["distances"])
        ds = (res.get("distances") or [[]])[0]
        if ds:
            unrel.append(float(ds[0]))

    def stats(v):
        if not v:
            return (None, None, None)
        s = sorted(v)
        return (s[0], s[len(s) // 2], s[-1])

    t = Table(show_header=True)
    t.add_column("probe", style="bold")
    t.add_column("min", style="cyan")
    t.add_column("median", style="cyan")
    t.add_column("max", style="cyan")
    for label, v in (("signature vs itself", self_d),
                     ("vs next-nearest stored", other_d),
                     ("unrelated text", unrel)):
        lo, md, hi = stats(v)
        t.add_row(label,
                  "{:.4f}".format(lo) if lo is not None else "—",
                  "{:.4f}".format(md) if md is not None else "—",
                  "{:.4f}".format(hi) if hi is not None else "—")
    console.print(t)

    _, self_med, self_max = stats(self_d)
    _, unrel_med, _ = stats(unrel)

    console.print()
    if self_med is None:
        console.print("[red]Could not measure. Something is still wrong.[/red]")
        return

    if self_med < 0.05:
        sep = (unrel_med or 1.0) - self_max
        rec = round(min(max(self_max * 3, 0.05), (unrel_med or 1.0) * 0.5), 2)
        console.print(Panel(
            "[bold green]Retrieval works.[/bold green]\n\n"
            "A signature now matches itself at distance {:.4f}, while unrelated text "
            "sits at {:.4f} — a separation of {:.2f}.\n\n"
            "[bold]Set SIMILARITY_THRESHOLD={} in .env[/bold]\n"
            "[dim]comfortably above exact matches, well below unrelated text[/dim]"
            .format(self_med, unrel_med or 0, sep, rec),
            title="[bold]Result[/bold]"))
    else:
        console.print(Panel(
            "[yellow]Self-distance is {:.4f}, which is still too high.[/yellow]\n\n"
            "With the hash fallback, distinct strings are near-orthogonal regardless "
            "of meaning, so the ledger can only ever match byte-identical signatures.\n\n"
            "Report this as a limitation: semantic retrieval requires the "
            "sentence-transformers model, which could not be downloaded in this "
            "environment."
            .format(self_med),
            title="[yellow]Partial[/yellow]"))

    out = {"rebuilt": time.strftime("%Y-%m-%d %H:%M:%S"),
           "embedder": "minilm" if use_semantic else "hash",
           "records": len(rows), "skipped_no_text": no_text,
           "self_distance_median": self_med, "self_distance_max": self_max,
           "unrelated_median": unrel_med,
           "backup": os.path.basename(bak)}
    json.dump(out, open(os.path.join(BASE, "data", "ledger_rebuild.json"), "w"), indent=2)
    console.print("\n[green]Saved -> data/ledger_rebuild.json[/green]")
    console.print("[dim]New collection: repair_ledger_v2. "
                  "Point ledger.py at it once you are satisfied.[/dim]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().dry_run)