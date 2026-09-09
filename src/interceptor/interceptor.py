"""
AASE Interceptor — catches failures from the Subject Agent in real time.
This is the entry point of AASE. It wraps any agent, watches its trace,
classifies the failure type, and hands off to the Diagnosis Council.
"""

import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.council.prosecutor import ProsecutorAgent
from src.council.defender import DefenderAgent
from src.council.coroner import CoronerAgent
from src.surgeon.surgeon import SurgeonAgent
from src.ledger.ledger import RepairLedger


FAILURE_SIGNATURES = {
    "hallucination": [
        lambda trace: any(s.get("error","") == "hallucination_detected" for s in trace),
        lambda trace: any("mismatch" in s.get("output","").lower() for s in trace),
        lambda trace: any("fabricat" in s.get("output","").lower() for s in trace),
    ],
    "tool_misuse": [
        lambda trace: any(s.get("error","") == "tool_misuse" for s in trace),
        lambda trace: any("wrong tool" in s.get("output","").lower() for s in trace),
        lambda trace: any("wrong_tool" in s.get("error","").lower() for s in trace),
    ],
    "reasoning_loop": [
        lambda trace: any(s.get("error","") == "reasoning_loop" for s in trace),
        lambda trace: len([s for s in trace if s.get("action") == "search"]) >= 3,
        lambda trace: any("loop" in s.get("error","").lower() for s in trace),
    ],
    "context_collapse": [
        lambda trace: any(s.get("error","") == "context_collapse" for s in trace),
        lambda trace: any("informal" in s.get("output","").lower() for s in trace),
        lambda trace: any("truncat" in s.get("output","").lower() for s in trace),
    ],
    "goal_drift": [
        lambda trace: any(s.get("error","") == "goal_drift" for s in trace),
        lambda trace: any("drift" in s.get("output","").lower() for s in trace),
    ],
    "prompt_injection": [
        lambda trace: any(s.get("error","") == "prompt_injection" for s in trace),
        lambda trace: any("injection" in s.get("output","").lower() for s in trace),
        lambda trace: any("ignore" in s.get("input","").lower() and "instruction" in s.get("input","").lower() for s in trace),
    ],
    "memory_overflow": [
        lambda trace: any("context_length_exceeded" in s.get("error","") for s in trace),
        lambda trace: any("memory_overflow" in s.get("error","") for s in trace),
        lambda trace: any("token" in s.get("error","").lower() and "limit" in s.get("error","").lower() for s in trace),
    ],
}


class AASEInterceptor:
    """
    Main AASE controller.
    1. Receives execution trace from Subject Agent
    2. Classifies failure type
    3. Checks Repair Ledger for known fix
    4. If known — applies instantly (Fast Path)
    5. If unknown — runs Diagnosis Council (Slow Path)
    6. Calls Surgeon to apply fix
    7. Records outcome in Repair Ledger
    """

    def __init__(self):
        self.ledger = RepairLedger()
        self.prosecutor = ProsecutorAgent()
        self.defender = DefenderAgent()
        self.coroner = CoronerAgent()
        self.surgeon = SurgeonAgent()
        self.stats = {
            "total_failures": 0,
            "ledger_hits": 0,
            "council_runs": 0,
            "repairs_successful": 0,
            "repairs_failed": 0,
        }

    def classify_failure(self, trace: list) -> str:
        """Classify failure type from execution trace."""
        for failure_type, checks in FAILURE_SIGNATURES.items():
            if any(check(trace) for check in checks):
                return failure_type
        # Default: look for any error step
        error_steps = [s for s in trace if s.get("status") == "error"]
        if error_steps:
            error_msg = error_steps[-1].get("error", "")
            if any(w in error_msg.lower() for w in ["hallucin", "fabricat", "mismatch"]):
                return "hallucination"
            if any(w in error_msg.lower() for w in ["tool", "api", "wrong"]):
                return "tool_misuse"
            if any(w in error_msg.lower() for w in ["loop", "repeat"]):
                return "reasoning_loop"
            if any(w in error_msg.lower() for w in ["context", "truncat", "memory"]):
                return "context_collapse"
        return "unknown"

    def handle_failure(self, agent, trace: list, task: str = "") -> dict:
        """
        Main AASE pipeline for handling a detected failure.
        Returns repair result with method, fix applied, and updated agent.
        """
        self.stats["total_failures"] += 1

        # Step 1: Classify
        failure_type = self.classify_failure(trace)
        print(f"\n[AASE] Failure detected: {failure_type.upper()}")

        # Step 2: Check Repair Ledger (Fast Path)
        trace_summary = self._summarize_trace(trace, failure_type, task)
        ledger_result = self.ledger.find_similar_repair(trace_summary)

        if ledger_result:
            print(f"[AASE] Ledger HIT — applying known fix instantly")
            self.stats["ledger_hits"] += 1
            fix = ledger_result["fix"]
            method = "ledger_hit"
        else:
            # Step 3: Run Diagnosis Council (Slow Path)
            print(f"[AASE] No ledger match — convening Diagnosis Council...")
            self.stats["council_runs"] += 1
            fix, council_log = self._run_council(trace, failure_type, task)
            method = "council_debate"

        # Step 4: Apply fix via Surgeon
        if fix:
            updated_prompt = self.surgeon.apply_fix(
                agent.system_prompt, failure_type, fix
            )
            agent.system_prompt = updated_prompt
            repair_success = True
            self.stats["repairs_successful"] += 1
            print(f"[AASE] Repair applied via Surgeon")
        else:
            repair_success = False
            self.stats["repairs_failed"] += 1
            print(f"[AASE] Repair could not be determined — escalating to human")

        # Step 5: Record in Repair Ledger
        self.ledger.store_repair(
            trace_summary=trace_summary,
            failure_type=failure_type,
            fix_applied=fix or "no_fix",
            outcome="success" if repair_success else "failed"
        )

        return {
            "repaired": repair_success,
            "failure_type": failure_type,
            "fix_applied": fix,
            "method": method,
            "updated_system_prompt": agent.system_prompt if repair_success else None,
        }

    def _summarize_trace(self, trace: list, failure_type: str, task: str) -> str:
        error_steps = [s for s in trace if s.get("status") in ("error","warning")]
        return (
            f"failure_type: {failure_type} | "
            f"task: {task[:100]} | "
            f"steps: {len(trace)} | "
            f"error_at: {error_steps[-1].get('step','?') if error_steps else 'N/A'} | "
            f"error: {error_steps[-1].get('error','none') if error_steps else 'none'}"
        )

    def _run_council(self, trace: list, failure_type: str, task: str):
        trace_str = json.dumps(trace, indent=2)[:3000]

        # Prosecutor argues
        prosecutor_arg = self.prosecutor.analyze(trace_str, failure_type, task)
        print(f"[Prosecutor] {prosecutor_arg[:120]}...")

        # Defender argues
        defender_arg = self.defender.analyze(trace_str, failure_type, task)
        print(f"[Defender]   {defender_arg[:120]}...")

        # Coroner gives verdict
        verdict = self.coroner.decide(trace_str, prosecutor_arg, defender_arg, failure_type)
        print(f"[Coroner]    {verdict[:120]}...")

        # Extract fix from verdict
        fix = self._extract_fix(verdict)
        council_log = {
            "prosecutor": prosecutor_arg,
            "defender": defender_arg,
            "coroner_verdict": verdict,
            "extracted_fix": fix
        }
        return fix, council_log

    def _extract_fix(self, verdict: str) -> str:
        """Pull the recommended fix clause from the Coroner's verdict."""
        verdict_lower = verdict.lower()
        for marker in ["fix:", "repair:", "add to system prompt:", "recommendation:", "clause:"]:
            if marker in verdict_lower:
                idx = verdict_lower.index(marker) + len(marker)
                return verdict[idx:idx+400].strip()
        # Return last 300 chars of verdict as the fix if no marker found
        return verdict[-300:].strip()

    def print_stats(self):
        print("\n===== AASE STATS =====")
        for k, v in self.stats.items():
            print(f"  {k}: {v}")
        if self.stats["total_failures"] > 0:
            hit_rate = self.stats["ledger_hits"] / self.stats["total_failures"] * 100
            print(f"  ledger_hit_rate: {hit_rate:.1f}%")
        print("======================")
