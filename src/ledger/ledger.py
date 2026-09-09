"""
Repair Ledger — AASE's memory.
Fully offline. No HuggingFace, no network, no SSL issues.
Uses deterministic hash-based embeddings computed locally.
"""

import os, json, hashlib, math
import chromadb
from dotenv import load_dotenv

load_dotenv()
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.35))
DIM = 384


def embed(text: str):
    """
    Local deterministic embedding. No model download required.
    Uses character n-grams + token hashing into a fixed vector.
    """
    text = str(text).lower()
    vec = [0.0] * DIM

    # word-level features
    for tok in text.split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0

    # character trigram features (captures partial matches)
    for i in range(len(text) - 2):
        tri = text[i:i+3]
        h = int(hashlib.md5(tri.encode()).hexdigest(), 16)
        vec[h % DIM] += 0.3

    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


class RepairLedger:

    def __init__(self):
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(base, "data", "memory_store")
        os.makedirs(path, exist_ok=True)

        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name="repair_ledger",
            metadata={"description": "AASE Repair Memory"}
        )
        self.mode = "local-hash"
        self._counter = self.collection.count()
        print(f"[Ledger] {self._counter} repairs loaded | embedder: {self.mode} (offline)")

    def _embed(self, text):
        return embed(text)

    def store_repair(self, trace_summary, failure_type, fix_applied, outcome):
        self._counter += 1
        try:
            self.collection.add(
                ids=[f"repair_{self._counter:06d}"],
                embeddings=[self._embed(trace_summary)],
                documents=[str(fix_applied)[:2000]],
                metadatas=[{
                    "failure_type": failure_type,
                    "outcome": outcome,
                    "trace_summary": str(trace_summary)[:300],
                    "times_applied": 1
                }]
            )
            print(f"[Ledger] Stored #{self._counter} ({failure_type}, {outcome})")
        except Exception as e:
            print(f"[Ledger] Store failed: {e}")

    def find_similar_repair(self, trace_summary):
        if self.collection.count() == 0:
            return None
        try:
            res = self.collection.query(
                query_embeddings=[self._embed(trace_summary)],
                n_results=min(3, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            return None

        if not res["ids"] or not res["ids"][0]:
            return None

        dist = res["distances"][0][0]
        if dist < SIMILARITY_THRESHOLD:
            return {
                "fix": res["documents"][0][0],
                "failure_type": res["metadatas"][0][0].get("failure_type"),
                "outcome": res["metadatas"][0][0].get("outcome"),
                "similarity_distance": round(dist, 4)
            }
        return None

    def seed_from_dataset(self, manifest_path, base_dir, max_per_type=20):
        if not os.path.exists(manifest_path):
            print(f"[Ledger] Manifest not found: {manifest_path}")
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
                candidates = [
                    os.path.join(base_dir, raw),
                    os.path.join(base_dir, "data", "injected", ftype, os.path.basename(raw)),
                    os.path.join(base_dir, "data", "real_failures", ftype, os.path.basename(raw)),
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        try:
                            d = json.load(open(cand))
                        except Exception:
                            break
                        task = d.get("task") or d.get("question", "")
                        summary = (f"failure_type: {ftype} | "
                                   f"task: {task[:100]} | "
                                   f"steps: {len(d.get('trace', []))}")
                        fix = d.get("ground_truth_fix") or d.get("mistake_reason", "")
                        if fix:
                            self.store_repair(summary, ftype, fix, "success")
                            seeded += 1
                        break

        print(f"[Ledger] Seeded {seeded} repairs from dataset")
        return seeded

    def stats(self):
        return {"total_repairs": self.collection.count()}

    def show_all(self, limit=20):
        res = self.collection.get(include=["metadatas", "documents"], limit=limit)
        print(f"\n[Ledger] Showing {len(res['ids'])} repairs:")
        for i, (m, doc) in enumerate(zip(res["metadatas"], res["documents"])):
            print(f"  {i+1}. [{m.get('failure_type')}] {m.get('outcome')} — {doc[:70]}...")