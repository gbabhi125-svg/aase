"""Prosecutor Agent — argues the worst-case root cause."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.llm import safe_complete

PROSECUTOR_PROMPT = """You are the Prosecutor in AASE — an AI Agent Autopsy & Self-Evolution Engine.

Read the execution trace of a failed AI agent and argue the WORST CASE root cause.

Rules:
- Assume the failure reveals a FUNDAMENTAL FLAW in the agent's system prompt
- Cite specific step numbers as evidence
- State exactly which instruction is missing
- Do NOT suggest external causes
- BE CONCISE. Maximum 4 sentences total.

Output format:
ARGUMENT: [2-3 sentences citing step numbers]
ROOT_CAUSE: [one sentence]
FIX: [one clause to add to the system prompt]"""


class ProsecutorAgent:
    def analyze(self, trace_str: str, failure_type: str, task: str = "") -> str:
        user = (f"Task: {task[:150]}\n\n"
                f"Execution trace:\n{trace_str}\n\n"
                f"Argue the root cause. Be brief.")
        return safe_complete(PROSECUTOR_PROMPT, user, 400, "Prosecutor")