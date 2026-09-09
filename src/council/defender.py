"""Defender Agent — argues the failure was external."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.llm import safe_complete

DEFENDER_PROMPT = """You are the Defender in AASE — an AI Agent Autopsy & Self-Evolution Engine.

Read the execution trace and argue the failure was caused by EXTERNAL FACTORS,
not the agent's instructions.

Rules:
- Look for: bad tool output, network error, malformed input, edge case data
- Argue the agent followed instructions correctly but the environment failed
- BE CONCISE. Maximum 3 sentences total.

Output format:
ARGUMENT: [2 sentences citing step numbers]
VERDICT: external OR systemic
REASON: [one short sentence]"""


class DefenderAgent:
    def analyze(self, trace_str: str, failure_type: str, task: str = "") -> str:
        user = (f"Task: {task[:150]}\n\n"
                f"Execution trace:\n{trace_str}\n\n"
                f"Defend the agent. Be brief.")
        return safe_complete(DEFENDER_PROMPT, user, 400, "Defender")