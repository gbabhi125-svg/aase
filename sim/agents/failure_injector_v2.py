import json, os, random, time

BASE = "/home/claude/aase"
POOL_FILE = f"{BASE}/data/benchmarks/task_pool_full.json"
OUT_DIR = f"{BASE}/data/injected"

random.seed(42)

FAILURE_TYPES = [
    "hallucination",
    "tool_misuse",
    "reasoning_loop",
    "context_collapse",
    "format_violation",
    "stale_memory",
    "goal_drift",
]

def log_step(trace, step, action, inp, out, status, error=None):
    trace.append({
        "step": step, "timestamp": time.time(), "action": action,
        "input": str(inp)[:300], "output": str(out)[:300],
        "status": status, "error": error
    })

def inject_hallucination(task):
    trace = []
    log_step(trace, 1, "ingest", task, task, "success")
    log_step(trace, 2, "tool_call", task, "TOOL_RESULT: empty", "success")
    log_step(trace, 3, "llm_call", "empty tool result",
              "Fabricated a specific figure/fact not present in tool output", "success")
    log_step(trace, 4, "validate", None, {"valid": False, "reason": "unsupported_claim"}, "error",
              "Agent stated a fact with no supporting tool evidence")
    return trace

def inject_tool_misuse(task):
    trace = []
    log_step(trace, 1, "tool_selection", task, "selected: wrong_tool", "success")
    log_step(trace, 2, "wrong_tool", task, "irrelevant/empty result", "success")
    log_step(trace, 3, "llm_call", "irrelevant result", "produced an answer anyway", "success")
    log_step(trace, 4, "validate", None, {"valid": False, "reason": "wrong_tool_used"}, "error",
              "Correct tool existed but was not selected")
    return trace

def inject_reasoning_loop(task):
    trace = []
    for i in range(1, 6):
        log_step(trace, i, "search", task, "no results", "success")
    log_step(trace, 6, "loop_detected", task, None, "error", "identical action repeated 5x, no progress")
    return trace

def inject_context_collapse(task):
    trace = []
    log_step(trace, 1, "ingest", task, task, "success")
    log_step(trace, 30, "context_warning", "70% full", "core instructions may be truncated", "warning")
    log_step(trace, 60, "llm_call", "late step", "tone/format drifted from system prompt", "success")
    log_step(trace, 61, "validate", None, {"valid": False, "reason": "instruction_forgotten"}, "error")
    return trace

def inject_format_violation(task):
    trace = []
    log_step(trace, 1, "ingest", task, task, "success")
    log_step(trace, 2, "llm_call", task, "free-text answer instead of required JSON schema", "success")
    log_step(trace, 3, "validate", None, {"valid": False, "reason": "schema_mismatch"}, "error",
              "Downstream parser expected structured output")
    return trace

def inject_stale_memory(task):
    trace = []
    log_step(trace, 1, "memory_fetch", task, "cached fact from 3 sessions ago", "success")
    log_step(trace, 2, "llm_call", "stale cached fact", "answered using outdated cached value", "success")
    log_step(trace, 3, "validate", None, {"valid": False, "reason": "stale_data_used"}, "error",
              "Underlying source had since changed; cache was never invalidated")
    return trace

def inject_goal_drift(task):
    trace = []
    log_step(trace, 1, "ingest", task, task, "success")
    log_step(trace, 2, "subtask_1", task, "completed", "success")
    log_step(trace, 3, "subtask_2", "unrelated tangent introduced by agent", "pursued tangent", "success")
    log_step(trace, 4, "validate", None, {"valid": False, "reason": "original_goal_abandoned"}, "error",
              "Agent optimized a sub-step and lost the original task objective")
    return trace

INJECTORS = {
    "hallucination": inject_hallucination,
    "tool_misuse": inject_tool_misuse,
    "reasoning_loop": inject_reasoning_loop,
    "context_collapse": inject_context_collapse,
    "format_violation": inject_format_violation,
    "stale_memory": inject_stale_memory,
    "goal_drift": inject_goal_drift,
}

def generate(n_per_type=1500):
    """
    Balanced sampling: use ALL of the small domains (HumanEval, ToolBench)
    every single failure type, then top up with MultiWOZ to reach n_per_type.
    This maximizes total volume (near the pool's real ceiling) while stopping
    MultiWOZ from drowning out the other two domains.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    pool = json.load(open(POOL_FILE))

    by_source = {"multiwoz_2.1": [], "humaneval": [], "toolbench": []}
    for rec in pool:
        by_source[rec["source"]].append(rec)

    total = 0
    manifest = []
    for ftype, fn in INJECTORS.items():
        sample = list(by_source["humaneval"]) + list(by_source["toolbench"])
        remaining = max(0, n_per_type - len(sample))
        mwoz_sample = random.sample(by_source["multiwoz_2.1"], min(remaining, len(by_source["multiwoz_2.1"])))
        sample += mwoz_sample
        for i, task_rec in enumerate(sample):
            trace = fn(task_rec["task"])
            result = {
                "id": f"{ftype}_{i+1:04d}",
                "injected_failure": ftype,
                "task_id": task_rec["task_id"],
                "task": task_rec["task"],
                "domain": task_rec["domain"],
                "source": task_rec["source"],
                "trace": trace,
                "success": False,
            }
            fname = f"{OUT_DIR}/{ftype}_{i+1:04d}.json"
            json.dump(result, open(fname, "w"), indent=2)
            manifest.append({"file": fname, "failure_type": ftype, "source": task_rec["source"]})
            total += 1
    json.dump(manifest, open(f"{BASE}/data/injected_manifest.json", "w"), indent=2)
    print(f"Generated {total} failure traces across {len(INJECTORS)} failure types")
    print(f"(target {n_per_type} per type, full HumanEval+ToolBench included every type, MultiWOZ tops up)")
    return total

if __name__ == "__main__":
    generate(n_per_type=1500)
