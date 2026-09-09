"""
server.py — AASE backend.
Every number this serves is computed from your real files or a real model call.
Run:  python server.py     →  http://127.0.0.1:8000
"""

import os, sys, json, time, random, threading, uuid
from collections import defaultdict, Counter
from flask import Flask, jsonify, request, send_from_directory, Response

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.council.prosecutor import ProsecutorAgent
from src.council.defender   import DefenderAgent
from src.council.coroner    import CoronerAgent
from src.surgeon.surgeon    import SurgeonAgent
from src.ledger.ledger      import RepairLedger

BASE = os.path.dirname(os.path.abspath(__file__))
app  = Flask(__name__, static_folder=BASE)

FAILURE_TYPES = ["hallucination", "tool_misuse", "reasoning_loop", "context_collapse",
                 "goal_drift", "prompt_injection", "memory_overflow"]

BASE_PROMPT = """You are an enterprise pipeline agent.
Process tasks step by step using available tools.
Always verify tool outputs before using them.
Maintain formal professional tone at all times."""

prosecutor = ProsecutorAgent()
defender   = DefenderAgent()
coroner    = CoronerAgent()
surgeon    = SurgeonAgent()
ledger     = RepairLedger()

SESSION_START = time.time()

# ──────────────────────────────────────────────────────────────
# Trace pool
# ──────────────────────────────────────────────────────────────
def load_pool():
    pool = defaultdict(list)
    sources = [
        ("real_failures_manifest.json", "real_failures", "real"),
        ("injected_manifest.json",      "injected",      "injected"),
    ]
    for mname, folder, tag in sources:
        mp = os.path.join(BASE, "data", mname)
        if not os.path.exists(mp):
            continue
        try:
            entries = json.load(open(mp))
        except Exception:
            continue
        for e in entries:
            ft = e.get("failure_type")
            if ft not in FAILURE_TYPES:
                continue
            for cand in (os.path.join(BASE, e["file"]),
                         os.path.join(BASE, "data", folder, ft, os.path.basename(e["file"]))):
                if os.path.exists(cand):
                    pool[ft].append({"path": cand, "id": e["trace_id"], "src": tag})
                    break
    return pool

POOL = load_pool()
POOL_TOTAL = sum(len(v) for v in POOL.values())
SRC_MIX = Counter(t["src"] for v in POOL.values() for t in v)
print(f"[Server] pool={POOL_TOTAL} " + " ".join(f"{k}:{len(v)}" for k, v in POOL.items()))

# ──────────────────────────────────────────────────────────────
# Verdict parsing
# ──────────────────────────────────────────────────────────────
KW = {
    "prompt_injection": ["prompt injection","injection","injected","ignore all previous",
                         "hijack","exfiltration","untrusted content","external instruction"],
    "memory_overflow":  ["memory overflow","token limit","context_length_exceeded","hard crash",
                         "process killed","step limit","max token","unrecoverable","token budget"],
    "reasoning_loop":   ["reasoning loop","repeated the same","same action","same query",
                         "no progress","identical query","looping","cycling","unrefined input"],
    "context_collapse": ["context collapse","forgot","instruction loss","context warning",
                         "truncat","failed to recall","informal register","constraint loss"],
    "goal_drift":       ["goal drift","drifted","original goal","deviated","scope creep",
                         "constraint ignor","premature","without verif","broadcast"],
    "tool_misuse":      ["tool misuse","wrong tool","incorrect tool","invalid action",
                         "parameter error","format error","public_search","internal_hr"],
    "hallucination":    ["hallucin","fabricat","made up","no records","invented","assumed",
                         "unsupported claim","not returned by"],
}

def parse_type(verdict: str):
    v = (verdict or "").lower()
    if "failure_type:" in v:
        for line in verdict.split("\n"):
            if "failure_type:" in line.lower():
                val = line.split(":", 1)[-1].strip().lower()
                for ft in FAILURE_TYPES:
                    if ft in val or ft.replace("_", " ") in val:
                        return ft, "label"
    for ft, words in KW.items():
        if any(w in v for w in words):
            return ft, "keyword"
    return "unknown", "none"

def extract_fix(verdict: str):
    for m in ("fix:", "repair:", "recommendation:", "clause:"):
        low = (verdict or "").lower()
        if m in low:
            i = low.index(m) + len(m)
            return verdict[i:i + 420].strip()
    return (verdict or "")[-280:].strip()

def field(verdict: str, key: str):
    for line in (verdict or "").split("\n"):
        if line.lower().startswith(key.lower() + ":"):
            return line.split(":", 1)[-1].strip()
    return ""

# ──────────────────────────────────────────────────────────────
# Session telemetry (in-memory, this process only)
# ──────────────────────────────────────────────────────────────
RUNS = []          # every council invocation this session
LOG  = []          # activity feed

def log(kind, msg):
    LOG.insert(0, {"t": round(time.time() - SESSION_START, 1), "kind": kind, "msg": msg})
    del LOG[400:]

log("boot", f"Pool loaded — {POOL_TOTAL} traces across {len([k for k,v in POOL.items() if v])} types")
log("boot", f"Ledger online — {ledger.stats()['total_repairs']} repairs, embedder {ledger.mode}")

# ──────────────────────────────────────────────────────────────
# Routes — static
# ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE, "dashboard.html")

# ──────────────────────────────────────────────────────────────
# Routes — system
# ──────────────────────────────────────────────────────────────
@app.route("/api/system")
def api_system():
    ok  = sum(1 for r in RUNS if r["correct"])
    n   = len(RUNS)
    lat = [r["total_time"] for r in RUNS]
    return jsonify({
        "pool_counts":  {k: len(v) for k, v in POOL.items()},
        "pool_total":   POOL_TOTAL,
        "source_mix":   dict(SRC_MIX),
        "types_active": len([k for k, v in POOL.items() if v]),
        "ledger_size":  ledger.stats()["total_repairs"],
        "embedder":     ledger.mode,
        "model":        getattr(coroner, "MODEL", None) or os.getenv("AASE_MODEL", "groq"),
        "uptime":       round(time.time() - SESSION_START, 1),
        "session_runs": n,
        "session_ok":   ok,
        "session_acc":  round(ok / n * 100, 1) if n else None,
        "median_lat":   round(sorted(lat)[len(lat)//2], 1) if lat else None,
    })

@app.route("/api/log")
def api_log():
    return jsonify(LOG[:80])

# ──────────────────────────────────────────────────────────────
# Routes — traces
# ──────────────────────────────────────────────────────────────
@app.route("/api/trace/<ftype>")
def api_trace(ftype):
    if ftype not in POOL or not POOL[ftype]:
        return jsonify({"error": f"no traces for {ftype}"}), 404
    pick = random.choice(POOL[ftype])
    try:
        d = json.load(open(pick["path"]))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    tr = d.get("trace", [])
    return jsonify({
        "trace_id":     pick["id"],
        "source":       pick["src"],
        "ground_truth": ftype,
        "task":         d.get("task") or d.get("question", ""),
        "trace":        tr,
        "steps":        len(tr),
        "errors":       sum(1 for s in tr if str(s.get("status","")).lower() == "error"),
        "warnings":     sum(1 for s in tr if str(s.get("status","")).lower() == "warning"),
        "annotated":    {
            "mistake_agent": d.get("mistake_agent"),
            "mistake_step":  d.get("mistake_step"),
            "mistake_reason": (d.get("mistake_reason") or "")[:400],
        } if d.get("mistake_reason") else None,
    })

# ──────────────────────────────────────────────────────────────
# Routes — council
# ──────────────────────────────────────────────────────────────
@app.route("/api/council", methods=["POST"])
def api_council():
    body  = request.get_json(force=True)
    trace = body.get("trace", [])
    task  = body.get("task", "")
    gt    = body.get("ground_truth", "")
    tid   = body.get("trace_id", "—")
    trace_str = json.dumps(trace, indent=2)[:3000]
    unknown   = "unknown — determine from trace"

    summary = f"failure_type:{gt}|task:{task[:80]}"
    hit = ledger.find_similar_repair(summary)

    out = {"trace_id": tid, "ledger_hit": bool(hit)}
    if hit:
        out["ledger_fix"]  = hit["fix"][:400]
        out["ledger_dist"] = round(hit["similarity_distance"], 4)

    t0 = time.time()
    out["prosecutor"]   = prosecutor.analyze(trace_str, unknown, task)
    out["t_prosecutor"] = round(time.time() - t0, 1)

    t1 = time.time()
    out["defender"]   = defender.analyze(trace_str, unknown, task)
    out["t_defender"] = round(time.time() - t1, 1)

    t2 = time.time()
    verdict = coroner.decide(trace_str, out["prosecutor"], out["defender"], unknown)
    out["coroner"]   = verdict
    out["t_coroner"] = round(time.time() - t2, 1)

    diagnosed, how = parse_type(verdict)
    out["diagnosed"]     = diagnosed
    out["parse_mode"]    = how
    out["ground_truth"]  = gt
    out["correct"]       = (diagnosed == gt)
    out["evidence"]      = field(verdict, "EVIDENCE")
    out["root_cause"]    = field(verdict, "ROOT_CAUSE")

    fix = extract_fix(verdict)
    out["fix"] = fix

    t3 = time.time()
    after = surgeon.apply_fix(BASE_PROMPT, diagnosed, fix)
    out["t_surgeon"]     = round(time.time() - t3, 1)
    out["prompt_before"] = BASE_PROMPT
    out["prompt_after"]  = after

    ledger.store_repair(summary, diagnosed if diagnosed != "unknown" else gt,
                        fix, "success" if out["correct"] else "partial")
    out["ledger_total"] = ledger.stats()["total_repairs"]
    out["total_time"]   = round(time.time() - t0, 1)

    RUNS.append({"trace_id": tid, "gt": gt, "dx": diagnosed,
                 "correct": out["correct"], "total_time": out["total_time"],
                 "ts": round(time.time() - SESSION_START, 1),
                 "ledger_hit": bool(hit), "parse_mode": how})
    log("council", f"{tid} · {gt} → {diagnosed} · {'match' if out['correct'] else 'diverge'} · {out['total_time']}s")
    return jsonify(out)

# ──────────────────────────────────────────────────────────────
# Routes — ledger
# ──────────────────────────────────────────────────────────────
@app.route("/api/ledger")
def api_ledger():
    q     = (request.args.get("q") or "").lower()
    ftype = request.args.get("type") or ""
    limit = int(request.args.get("limit") or 120)
    try:
        raw = ledger.collection.get(include=["metadatas", "documents"])
    except Exception as e:
        return jsonify({"error": str(e), "rows": [], "total": 0}), 200

    rows = []
    for i, (meta, doc) in enumerate(zip(raw.get("metadatas", []), raw.get("documents", []))):
        meta = meta or {}
        row = {
            "id":      raw["ids"][i],
            "type":    meta.get("failure_type", "—"),
            "outcome": meta.get("outcome", "—"),
            "summary": (meta.get("trace_summary") or "")[:200],
            "fix":     (doc or "")[:500],
        }
        if ftype and row["type"] != ftype:
            continue
        if q and q not in (row["fix"] + row["summary"] + row["type"]).lower():
            continue
        rows.append(row)
    rows.reverse()
    by_type = Counter(r["type"] for r in rows)
    by_out  = Counter(r["outcome"] for r in rows)
    return jsonify({
        "rows":    rows[:limit],
        "total":   len(rows),
        "by_type": dict(by_type),
        "by_outcome": dict(by_out),
        "ledger_size": ledger.stats()["total_repairs"],
    })

@app.route("/api/ledger/probe", methods=["POST"])
def api_probe():
    """Immunity probe — does the ledger already know this failure signature?"""
    body = request.get_json(force=True)
    text = body.get("text", "")
    t0 = time.time()
    hit = ledger.find_similar_repair(text)
    ms  = round((time.time() - t0) * 1000, 1)
    return jsonify({
        "hit": bool(hit),
        "latency_ms": ms,
        "fix": hit["fix"][:400] if hit else None,
        "type": hit["failure_type"] if hit else None,
        "distance": round(hit["similarity_distance"], 4) if hit else None,
    })

# ──────────────────────────────────────────────────────────────
# Routes — analytics
# ──────────────────────────────────────────────────────────────
@app.route("/api/analytics")
def api_analytics():
    out = {"session": {}, "saved": None}

    per = defaultdict(lambda: {"ok": 0, "n": 0})
    conf = defaultdict(lambda: defaultdict(int))
    for r in RUNS:
        per[r["gt"]]["n"] += 1
        if r["correct"]:
            per[r["gt"]]["ok"] += 1
        conf[r["gt"]][r["dx"]] += 1

    lat = sorted(r["total_time"] for r in RUNS)
    out["session"] = {
        "runs":     len(RUNS),
        "correct":  sum(1 for r in RUNS if r["correct"]),
        "per_type": {k: dict(v) for k, v in per.items()},
        "confusion": {k: dict(v) for k, v in conf.items()},
        "latency": {
            "min": lat[0] if lat else None,
            "p50": lat[len(lat)//2] if lat else None,
            "p90": lat[int(len(lat)*0.9)] if lat else None,
            "max": lat[-1] if lat else None,
        },
        "timeline": [{"ts": r["ts"], "ok": r["correct"], "t": r["total_time"],
                      "gt": r["gt"], "dx": r["dx"], "id": r["trace_id"]} for r in RUNS],
        "parse_modes": dict(Counter(r["parse_mode"] for r in RUNS)),
    }

    for fname in ("accuracy_results_real.json", "accuracy_results.json"):
        fp = os.path.join(BASE, "data", fname)
        if os.path.exists(fp):
            try:
                out["saved"] = json.load(open(fp))
                out["saved"]["_file"] = fname
                break
            except Exception:
                pass
    return jsonify(out)

# ──────────────────────────────────────────────────────────────
# Routes — Failure DNA
# ──────────────────────────────────────────────────────────────
@app.route("/api/dna")
def api_dna():
    """
    Failure DNA computed from the actual repair ledger.
    Each 'strain' is a failure type; the vector is its real distribution
    of stored outcomes. Nothing here is invented.
    """
    try:
        raw = ledger.collection.get(include=["metadatas"])
    except Exception as e:
        return jsonify({"error": str(e), "strains": []}), 200

    agg = defaultdict(lambda: {"n": 0, "success": 0, "partial": 0})
    for meta in raw.get("metadatas", []):
        meta = meta or {}
        ft = meta.get("failure_type", "unknown")
        agg[ft]["n"] += 1
        o = meta.get("outcome", "")
        if o == "success":
            agg[ft]["success"] += 1
        elif o == "partial":
            agg[ft]["partial"] += 1

    total = sum(v["n"] for v in agg.values()) or 1
    strains = []
    for ft in FAILURE_TYPES + ["unknown"]:
        v = agg.get(ft)
        if not v or v["n"] == 0:
            continue
        immunity = round(v["success"] / v["n"] * 100)
        strains.append({
            "type":      ft,
            "count":     v["n"],
            "share":     round(v["n"] / total * 100, 1),
            "success":   v["success"],
            "partial":   v["partial"],
            "immunity":  immunity,
            "pool":      len(POOL.get(ft, [])),
        })
    strains.sort(key=lambda s: -s["count"])
    return jsonify({"strains": strains, "total": total,
                    "ledger_size": ledger.stats()["total_repairs"]})

# ──────────────────────────────────────────────────────────────
# Routes — batch evaluation
# ──────────────────────────────────────────────────────────────
BATCH = {"id": None, "running": False, "done": 0, "total": 0,
         "results": [], "started": None, "finished": None, "cancel": False}
BATCH_LOCK = threading.Lock()

def batch_worker(per_type, types, source):
    global BATCH
    picks = []
    for ft in types:
        avail = [t for t in POOL.get(ft, []) if source == "any" or t["src"] == source]
        random.shuffle(avail)
        picks.extend([(ft, p) for p in avail[:per_type]])
    random.shuffle(picks)

    with BATCH_LOCK:
        BATCH["total"] = len(picks)
    log("batch", f"Batch started — {len(picks)} traces")

    for ft, p in picks:
        with BATCH_LOCK:
            if BATCH["cancel"]:
                break
        try:
            d = json.load(open(p["path"]))
        except Exception:
            continue
        trace_str = json.dumps(d.get("trace", []), indent=2)[:3000]
        task = d.get("task") or d.get("question", "")
        unknown = "unknown — determine from trace"
        t0 = time.time()
        try:
            pa = prosecutor.analyze(trace_str, unknown, task)
            da = defender.analyze(trace_str, unknown, task)
            vd = coroner.decide(trace_str, pa, da, unknown)
            dx, how = parse_type(vd)
            err = None
        except Exception as e:
            dx, how, err = "error", "none", str(e)[:200]
        el = round(time.time() - t0, 1)
        ok = (dx == ft)
        row = {"id": p["id"], "gt": ft, "dx": dx, "ok": ok,
               "t": el, "src": p["src"], "err": err, "parse": how}
        with BATCH_LOCK:
            BATCH["results"].append(row)
            BATCH["done"] += 1
        if not err:
            try:
                ledger.store_repair(f"failure_type:{ft}|task:{task[:80]}",
                                    dx if dx != "unknown" else ft,
                                    extract_fix(vd), "success" if ok else "partial")
            except Exception:
                pass
            RUNS.append({"trace_id": p["id"], "gt": ft, "dx": dx, "correct": ok,
                         "total_time": el, "ts": round(time.time()-SESSION_START,1),
                         "ledger_hit": False, "parse_mode": how})

    with BATCH_LOCK:
        BATCH["running"]  = False
        BATCH["finished"] = time.time()
    ok = sum(1 for r in BATCH["results"] if r["ok"])
    n  = len(BATCH["results"])
    log("batch", f"Batch complete — {ok}/{n} correct" + (f" ({ok/n*100:.1f}%)" if n else ""))

    try:
        per = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in BATCH["results"]:
            per[r["gt"]]["total"] += 1
            if r["ok"]:
                per[r["gt"]]["correct"] += 1
        payload = {
            "dataset": source,
            "total_traces": n,
            "correct": ok,
            "accuracy_pct": round(ok / n * 100, 1) if n else 0,
            "avg_time_sec": round(sum(r["t"] for r in BATCH["results"]) / n, 1) if n else 0,
            "per_type": {k: dict(v) for k, v in per.items()},
            "all_results": BATCH["results"],
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        json.dump(payload, open(os.path.join(BASE, "data", "batch_results.json"), "w"), indent=2)
    except Exception:
        pass

@app.route("/api/batch/start", methods=["POST"])
def api_batch_start():
    global BATCH
    with BATCH_LOCK:
        if BATCH["running"]:
            return jsonify({"error": "batch already running"}), 409
        body = request.get_json(force=True)
        per_type = max(1, min(int(body.get("per_type", 3)), 50))
        types    = body.get("types") or [k for k, v in POOL.items() if v]
        source   = body.get("source", "any")
        BATCH = {"id": uuid.uuid4().hex[:8], "running": True, "done": 0, "total": 0,
                 "results": [], "started": time.time(), "finished": None, "cancel": False}
    threading.Thread(target=batch_worker, args=(per_type, types, source), daemon=True).start()
    return jsonify({"started": True, "batch_id": BATCH["id"]})

@app.route("/api/batch/status")
def api_batch_status():
    with BATCH_LOCK:
        res = list(BATCH["results"])
        b = {k: BATCH[k] for k in ("id", "running", "done", "total", "started", "finished")}
    ok = sum(1 for r in res if r["ok"])
    b["results"]  = res[-400:]
    b["correct"]  = ok
    b["accuracy"] = round(ok / len(res) * 100, 1) if res else None
    b["elapsed"]  = round((b["finished"] or time.time()) - b["started"], 1) if b["started"] else 0
    per = defaultdict(lambda: {"ok": 0, "n": 0})
    for r in res:
        per[r["gt"]]["n"] += 1
        if r["ok"]:
            per[r["gt"]]["ok"] += 1
    b["per_type"] = {k: dict(v) for k, v in per.items()}
    return jsonify(b)

@app.route("/api/batch/cancel", methods=["POST"])
def api_batch_cancel():
    with BATCH_LOCK:
        BATCH["cancel"] = True
    log("batch", "Cancel requested")
    return jsonify({"cancelled": True})

@app.route("/api/export")
def api_export():
    """Download a full session report as JSON."""
    payload = {
        "generated":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "pool_total":  POOL_TOTAL,
        "pool_counts": {k: len(v) for k, v in POOL.items()},
        "source_mix":  dict(SRC_MIX),
        "ledger_size": ledger.stats()["total_repairs"],
        "embedder":    ledger.mode,
        "session_runs": RUNS,
        "batch":       {"results": BATCH["results"], "id": BATCH["id"]},
    }
    return Response(json.dumps(payload, indent=2),
                    mimetype="application/json",
                    headers={"Content-Disposition": "attachment; filename=aase_session_report.json"})


if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  AASE CONSOLE   →   http://127.0.0.1:8000")
    print(f"  pool {POOL_TOTAL} traces · ledger {ledger.stats()['total_repairs']} · {ledger.mode}")
    print("=" * 58 + "\n")
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)