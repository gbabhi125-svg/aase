"""
Repair Ledger — AASE's immune memory.

Stores a failure signature, the fix that resolved it, and the outcome.
A future failure whose signature is semantically close is resolved from
memory without convening the council.

The collection records which embedder wrote it. If the active embedder
differs, the ledger refuses to query rather than returning garbage —
a silent embedder switch previously made 323 stored repairs unreachable
while every probe reported "no match".
"""

import os, json, hashlib
import chromadb
from dotenv import load_dotenv

load_dotenv()

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.05))
COLLECTION = os.getenv("LEDGER_COLLECTION", "repair_ledger_v2")

os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")


class _HashEmbedder:
    """Deterministic offline fallback. Near-orthogonal on distinct strings —
    matches only near-identical signatures, never semantic neighbours."""
    DIM = 384
    name = "hash"

    def encode(self, text):
        vec = [0.0] * self.DIM
        for tok in str(text).lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec


class RepairLedger:

    def __init__(self):
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(base, "data", "memory_store")
        os.makedirs(path, exist_ok=True)

        self.client = chromadb.PersistentClient(path=path)

        # ── choose an embedder and name it ───────────────────────────
        self.embedder = None
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.mode = "minilm"
        except Exception as e:
            self.embedder = _HashEmbedder()
            self.mode = "hash"
            self.embedder_error = "{}: {}".format(type(e).__name__, str(e)[:140])

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION,
            metadata={"description": "AASE Repair Memory", "embedder": self.mode},
        )

        # ── refuse to mix embedding spaces ───────────────────────────
        written_by = (self.collection.metadata or {}).get("embedder")
        self.compatible = (written_by is None or written_by == self.mode
                           or self.collection.count() == 0)
        self.written_by = written_by or self.mode

        self._counter = self.collection.count()
        if self.compatible:
            print("[Ledger] {} repairs | embedder {} | threshold {}".format(
                self._counter, self.mode, SIMILARITY_THRESHOLD))
        else:
            print("[Ledger] DISABLED — store written by '{}', active embedder is '{}'. "
                  "Run scripts/rebuild_ledger.py".format(written_by, self.mode))

    # ── embedding ────────────────────────────────────────────────────
    def _embed(self, text):
        v = self.embedder.encode(str(text))
        return v.tolist() if hasattr(v, "tolist") else list(v)

    # ── write ────────────────────────────────────────────────────────
    def store_repair(self, trace_summary, failure_type, fix_applied, outcome):
        if not str(trace_summary).strip():
            return None
        self._counter += 1
        rid = "repair_{:06d}".format(self._counter)
        self.collection.add(
            ids=[rid],
            embeddings=[self._embed(trace_summary)],
            documents=[fix_applied or ""],
            metadatas=[{
                "failure_type": failure_type,
                "outcome": outcome,
                "trace_summary": str(trace_summary)[:400],
                "embedder": self.mode,
            }],
        )
        print("[Ledger] stored #{} ({}, {})".format(self._counter, failure_type, outcome))
        return rid

    # ── read ─────────────────────────────────────────────────────────
    def find_similar_repair(self, trace_summary):
        if not self.compatible or self.collection.count() == 0:
            return None
        try:
            res = self.collection.query(
                query_embeddings=[self._embed(trace_summary)],
                n_results=min(3, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return None
        if not res["ids"][0]:
            return None
        d = float(res["distances"][0][0])
        if d >= SIMILARITY_THRESHOLD:
            return None
        return {
            "fix": res["documents"][0][0],
            "failure_type": res["metadatas"][0][0].get("failure_type"),
            "outcome": res["metadatas"][0][0].get("outcome"),
            "trace_summary": res["metadatas"][0][0].get("trace_summary"),
            "similarity_distance": d,
        }

    def probe(self, text, n=5):
        """Nearest neighbours regardless of threshold — for inspection."""
        if self.collection.count() == 0:
            return []
        try:
            res = self.collection.query(
                query_embeddings=[self._embed(text)],
                n_results=min(n, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []
        out = []
        for i in range(len(res["ids"][0])):
            m = res["metadatas"][0][i] or {}
            d = float(res["distances"][0][i])
            out.append({
                "id": res["ids"][0][i],
                "distance": round(d, 4),
                "hit": d < SIMILARITY_THRESHOLD,
                "failure_type": m.get("failure_type"),
                "outcome": m.get("outcome"),
                "trace_summary": (m.get("trace_summary") or "")[:220],
                "fix": (res["documents"][0][i] or "")[:400],
            })
        return out

    def browse(self, q="", ftype="", outcome="", limit=200, dedupe=True):
        """Every stored repair, filterable. For the console's LEDGER tab."""
        try:
            raw = self.collection.get(include=["metadatas", "documents"])
        except Exception:
            return {"rows": [], "total": 0, "by_type": {}, "by_outcome": {}}

        ids = raw.get("ids", [])
        metas = raw.get("metadatas", []) or []
        docs = raw.get("documents", []) or []

        rows, seen = [], set()
        for i, rid in enumerate(ids):
            m = metas[i] or {}
            summary = m.get("trace_summary") or ""
            fix = docs[i] if i < len(docs) else ""
            if dedupe:
                key = (summary, fix[:120])
                if key in seen:
                    continue
                seen.add(key)
            rows.append({
                "id": rid,
                "failure_type": m.get("failure_type", "?"),
                "outcome": m.get("outcome", "?"),
                "trace_summary": summary[:220],
                "fix": fix[:500],
            })

        by_type, by_out = {}, {}
        for r in rows:
            by_type[r["failure_type"]] = by_type.get(r["failure_type"], 0) + 1
            by_out[r["outcome"]] = by_out.get(r["outcome"], 0) + 1

        if ftype:
            rows = [r for r in rows if r["failure_type"] == ftype]
        if outcome:
            rows = [r for r in rows if r["outcome"] == outcome]
        if q:
            ql = q.lower()
            rows = [r for r in rows
                    if ql in r["trace_summary"].lower() or ql in r["fix"].lower()
                    or ql in r["failure_type"]]

        rows.reverse()
        return {"rows": rows[:limit], "total": len(rows),
                "by_type": by_type, "by_outcome": by_out,
                "raw_count": len(ids), "deduped": dedupe}

    # ── seeding ──────────────────────────────────────────────────────
    def seed_from_dataset(self, manifest_path, base_dir, max_per_type=20):
        if not os.path.exists(manifest_path):
            print("[Ledger] manifest not found: {}".format(manifest_path))
            return 0
        manifest = json.load(open(manifest_path))
        from collections import defaultdict
        by_type = defaultdict(list)
        for e in manifest:
            by_type[e["failure_type"]].append(e)

        seeded = 0
        for ftype, entries in by_type.items():
            for entry in entries[:max_per_type]:
                raw = entry["file"]
                for cand in (os.path.join(base_dir, raw),
                             os.path.join(base_dir, "data", "injected", ftype,
                                          os.path.basename(raw)),
                             os.path.join(base_dir, "data", "real_failures", ftype,
                                          os.path.basename(raw))):
                    if os.path.exists(cand):
                        try:
                            d = json.load(open(cand))
                        except Exception:
                            break
                        summary = "failure_type: {} | task: {} | steps: {}".format(
                            ftype, (d.get("task") or d.get("question", ""))[:110],
                            len(d.get("trace", [])))
                        fix = d.get("ground_truth_fix") or d.get("mistake_reason") or ""
                        if fix:
                            self.store_repair(summary, ftype, fix, "success")
                            seeded += 1
                        break
        print("[Ledger] seeded {} repairs".format(seeded))
        return seeded

    def stats(self):
        return {
            "total_repairs": self.collection.count(),
            "embedder": self.mode,
            "written_by": self.written_by,
            "compatible": self.compatible,
            "threshold": SIMILARITY_THRESHOLD,
            "collection": COLLECTION,
        }