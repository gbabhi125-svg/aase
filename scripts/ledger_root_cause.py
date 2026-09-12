"""
scripts/ledger_root_cause.py
Why does an identical string not match itself?
Zero API calls.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from src.ledger.ledger import RepairLedger

def lst(v):
    """ChromaDB may hand back numpy arrays."""
    if v is None:
        return None
    try:
        return v.tolist()
    except AttributeError:
        return list(v)

def cos_dist(u, v):
    dot = sum(x*y for x, y in zip(u, v))
    nu = sum(x*x for x in u) ** 0.5
    nv = sum(x*x for x in v) ** 0.5
    return 1 - dot/(nu*nv) if nu and nv else None

led = RepairLedger()
print("embedder mode :", getattr(led, "mode", "?"))

s = "failure_type: hallucination | task: test"
a = lst(led._embed(s))
b = lst(led._embed(s))
print("deterministic :", a == b)
print("query dim     :", len(a))
print("query norm    :", round(sum(x*x for x in a) ** 0.5, 4))
print("query first5  :", [round(x, 4) for x in a[:5]])
print("self-distance :", round(cos_dist(a, a), 6), " (must be ~0)")

print()
raw = led.collection.get(include=["embeddings", "metadatas"], limit=3)
embs = raw.get("embeddings")
metas = raw.get("metadatas") or []

n = 0
try:
    n = len(embs)
except TypeError:
    n = 0

if n == 0:
    print(">>> collection returned NO embeddings.")
else:
    stored = lst(embs[0])
    print("stored dim    :", len(stored))
    print("stored norm   :", round(sum(x*x for x in stored) ** 0.5, 4))
    print("stored first5 :", [round(x, 4) for x in stored[:5]])
    print("stored zeros  :", sum(1 for x in stored if x == 0), "of", len(stored))

    sig = (metas[0] or {}).get("trace_summary", "") if metas else ""
    print()
    print("stored summary:", sig[:80])

    if sig.strip():
        re_emb = lst(led._embed(sig))
        d = cos_dist(re_emb, stored)
        print("re-embed that summary, distance to its own stored vector:",
              round(d, 6) if d is not None else "n/a")
        print()
        if d is not None and d < 0.01:
            print(">>> Vectors agree. The mismatch is elsewhere — check that")
            print("    find_similar_repair embeds the SAME field it stores.")
        else:
            print(">>> CONFIRMED: the stored vector does not match a fresh")
            print("    embedding of its own text. The store was written by a")
            print("    different embedder than the one querying it now.")
    else:
        print(">>> stored record has no trace_summary text — cannot compare.")

print()
print("collection    :", led.collection.count(), "records")
print("metadata      :", led.collection.metadata)