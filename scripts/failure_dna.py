"""
scripts/failure_dna.py

The Repair Ledger holds real embeddings of every failure signature. Project
them to 2D and you get a map of the failure space.

The question it answers: do the seven types occupy separate regions, or do
tool_misuse and memory_overflow overlap exactly where the diagnosis errors
are? If they overlap in embedding space, the confusion is a property of the
data, not a defect in the Coroner.

Zero API calls. Writes an SVG you can drop straight into the report.

  python scripts/failure_dna.py
  python scripts/failure_dna.py --method pca      # no extra dependencies
"""

import os, sys, json, math, argparse
from collections import defaultdict, Counter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

console = Console()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLOURS = {
    "hallucination":    "#E0555A",
    "tool_misuse":      "#4C8DF5",
    "reasoning_loop":   "#3FBF8F",
    "context_collapse": "#E0A03F",
    "goal_drift":       "#A96BE0",
    "prompt_injection": "#E066B8",
    "memory_overflow":  "#4ECDC4",
    "unknown":          "#8C94A3",
}


def pca2(X):
    """Two-component PCA in pure Python. No sklearn needed."""
    n, d = len(X), len(X[0])
    mean = [sum(r[j] for r in X) / n for j in range(d)]
    C = [[r[j] - mean[j] for j in range(d)] for r in X]

    def norm(v):
        m = math.sqrt(sum(x * x for x in v))
        return [x / m for x in v] if m else v

    comps = []
    for _ in range(2):
        v = norm([1.0 / math.sqrt(d)] * d)
        for _ in range(60):                       # power iteration
            w = [0.0] * d
            for row in C:
                dot = sum(row[j] * v[j] for j in range(d))
                for j in range(d):
                    w[j] += dot * row[j]
            for c in comps:                       # deflate
                dot = sum(w[j] * c[j] for j in range(d))
                for j in range(d):
                    w[j] -= dot * c[j]
            v = norm(w)
        comps.append(v)
        # remove this component from the data
        for row in C:
            dot = sum(row[j] * v[j] for j in range(d))
            for j in range(d):
                row[j] -= dot * v[j]

    return [[sum((X[i][j] - mean[j]) * comps[k][j] for j in range(d))
             for k in range(2)] for i in range(n)]


def centroid(pts):
    return [sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)]


def spread(pts, c):
    return sum(math.dist(p, c) for p in pts) / len(pts)


def main(method):
    console.print(Panel("[bold cyan]Failure DNA — the embedding space of stored repairs"
                        "[/bold cyan]\n[dim]zero API calls[/dim]", title="AASE"))

    from src.ledger.ledger import RepairLedger
    led = RepairLedger()
    if not getattr(led, "compatible", True):
        console.print("[red]Ledger embedder mismatch. Run scripts/rebuild_ledger.py first.[/red]")
        return

    raw = led.collection.get(include=["metadatas", "embeddings"])
    ids = raw.get("ids", [])
    metas = raw.get("metadatas", []) or []
    embs = raw.get("embeddings")

    try:
        n_emb = len(embs)
    except TypeError:
        n_emb = 0
    if n_emb == 0:
        console.print("[red]No embeddings in the store.[/red]")
        return

    X, labels, sigs = [], [], []
    for i in range(min(len(ids), n_emb)):
        v = embs[i]
        try:
            v = v.tolist()
        except AttributeError:
            v = list(v)
        m = metas[i] or {}
        X.append(v)
        labels.append(m.get("failure_type", "unknown"))
        sigs.append((m.get("trace_summary") or "")[:70])

    console.print(f"[bold]{len(X)}[/bold] vectors · {len(X[0])} dimensions · "
                  f"embedder {getattr(led,'mode','?')}\n")
    counts = Counter(labels)
    for k, v in counts.most_common():
        console.print(f"  {k:<18} {v}")

    console.print(f"\n[dim]projecting with {method}…[/dim]")
    if method == "umap":
        try:
            import umap
            P = umap.UMAP(n_components=2, random_state=42).fit_transform(X).tolist()
        except Exception as e:
            console.print(f"[yellow]umap unavailable ({type(e).__name__}) — using PCA[/yellow]")
            method = "pca"
            P = pca2(X)
    else:
        P = pca2(X)

    # ── cluster geometry ─────────────────────────────────────────────
    by = defaultdict(list)
    for p, l in zip(P, labels):
        by[l].append(p)

    cents = {k: centroid(v) for k, v in by.items() if len(v) >= 2}
    spreads = {k: spread(by[k], cents[k]) for k in cents}

    t = Table(title="Cluster geometry")
    t.add_column("Failure type", style="bold")
    t.add_column("N", style="white")
    t.add_column("Spread", style="cyan")
    t.add_column("Nearest other cluster", style="yellow")
    t.add_column("Separation", style="green")
    for k in sorted(cents, key=lambda x: -len(by[x])):
        others = [(math.dist(cents[k], cents[o]), o) for o in cents if o != k]
        if not others:
            continue
        d, o = min(others)
        ratio = d / spreads[k] if spreads[k] else 0
        flag = "[red]OVERLAP[/red]" if ratio < 1.0 else f"{ratio:.2f}×"
        t.add_row(k, str(len(by[k])), f"{spreads[k]:.3f}", o, flag)
    console.print(t)
    console.print("[dim]Separation is centroid distance divided by this cluster's own "
                  "spread. Below 1.0 means the clusters overlap more than they separate "
                  "— the two failure types are not distinguishable in embedding space.[/dim]")

    # ── SVG ──────────────────────────────────────────────────────────
    xs = [p[0] for p in P]
    ys = [p[1] for p in P]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    W, H, PAD = 900, 620, 60

    def sx(x):
        return PAD + (x - x0) / (x1 - x0 + 1e-9) * (W - 2 * PAD)

    def sy(y):
        return H - PAD - (y - y0) / (y1 - y0 + 1e-9) * (H - 2 * PAD - 90)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="IBM Plex Mono, monospace">',
           f'<rect width="{W}" height="{H}" fill="#0D1117"/>']
    for gx in range(PAD, W - PAD, 60):
        svg.append(f'<line x1="{gx}" y1="{PAD}" x2="{gx}" y2="{H-PAD-70}" '
                   f'stroke="#1A2029" stroke-width="1"/>')
    for gy in range(PAD, H - PAD - 70, 60):
        svg.append(f'<line x1="{PAD}" y1="{gy}" x2="{W-PAD}" y2="{gy}" '
                   f'stroke="#1A2029" stroke-width="1"/>')

    svg.append(f'<text x="{PAD}" y="34" fill="#E8EAEE" font-size="16" font-weight="700">'
               f'AASE Failure DNA — {len(X)} stored repair signatures</text>')
    svg.append(f'<text x="{PAD}" y="52" fill="#5A6472" font-size="11">'
               f'{method.upper()} projection of {len(X[0])}-dimensional embeddings '
               f'· embedder {getattr(led,"mode","?")}</text>')

    for p, l in zip(P, labels):
        svg.append(f'<circle cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="4" '
                   f'fill="{COLOURS.get(l,"#8C94A3")}" opacity="0.62"/>')
    for k, c in cents.items():
        svg.append(f'<circle cx="{sx(c[0]):.1f}" cy="{sy(c[1]):.1f}" r="9" fill="none" '
                   f'stroke="{COLOURS.get(k,"#8C94A3")}" stroke-width="2.5"/>')
        svg.append(f'<text x="{sx(c[0]):.1f}" y="{sy(c[1])-15:.1f}" fill="{COLOURS.get(k)}" '
                   f'font-size="11" font-weight="700" text-anchor="middle">{k}</text>')

    ly = H - 52
    lx = PAD
    for k, v in counts.most_common():
        svg.append(f'<circle cx="{lx+5}" cy="{ly-4}" r="4.5" fill="{COLOURS.get(k,"#8C94A3")}"/>')
        svg.append(f'<text x="{lx+16}" y="{ly}" fill="#8C94A3" font-size="11">'
                   f'{k} ({v})</text>')
        lx += 16 + len(k) * 7 + 42
        if lx > W - 190:
            lx = PAD
            ly += 20
    svg.append('</svg>')

    fp = os.path.join(BASE, "data", "failure_dna.svg")
    open(fp, "w", encoding="utf-8").write("\n".join(svg))
    console.print(f"\n[green]Figure → data/failure_dna.svg[/green]")

    overlaps = []
    for k in cents:
        others = [(math.dist(cents[k], cents[o]), o) for o in cents if o != k]
        if not others:
            continue
        d, o = min(others)
        if spreads[k] and d / spreads[k] < 1.0:
            overlaps.append((k, o, round(d / spreads[k], 3)))

    if overlaps:
        console.print(Panel(
            "[bold yellow]Overlapping clusters[/bold yellow]\n" +
            "\n".join(f"  {a} ↔ {b}  (separation {r}×)" for a, b, r in overlaps) +
            "\n\n[dim]These pairs are not separable in embedding space. If your "
            "diagnosis errors concentrate on the same pairs, the confusion is a "
            "property of the data rather than a defect in the decision procedure.[/dim]",
            title="[bold]Finding[/bold]"))

    json.dump({"n": len(X), "method": method, "dims": len(X[0]),
               "counts": dict(counts),
               "spreads": {k: round(v, 4) for k, v in spreads.items()},
               "overlaps": [{"a": a, "b": b, "separation": r} for a, b, r in overlaps],
               "points": [{"x": round(p[0], 4), "y": round(p[1], 4), "label": l}
                          for p, l in zip(P, labels)]},
              open(os.path.join(BASE, "data", "failure_dna.json"), "w"), indent=2)
    console.print("[green]Saved → data/failure_dna.json[/green]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="pca", choices=["pca", "umap"])
    main(ap.parse_args().method)