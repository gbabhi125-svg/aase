"""Subject Agent — the AI agent AASE monitors and repairs."""

import time, uuid, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.llm import complete, LLMError


class SubjectAgent:

    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self.trace = []
        self.system_prompt = """You are an enterprise pipeline agent.
Process tasks step by step using the available tools.
Always verify tool outputs before using them in your response.
If a tool returns an error or empty result, report it clearly — never fabricate data.
Maintain formal professional tone at all times."""

    def log(self, step, action, input_data, output, status, error=None):
        entry = {"step": step, "timestamp": round(time.time(), 2),
                 "action": action, "input": str(input_data)[:300],
                 "output": str(output)[:300], "status": status}
        if error:
            entry["error"] = error
        self.trace.append(entry)
        return entry

    def run_task(self, task: str) -> dict:
        self.trace = []
        step = 1
        self.log(step, "ingest", task, "Task received", "success")
        step += 1

        try:
            output = complete(self.system_prompt, task, 800)
            self.log(step, "llm_response", task[:100], output[:200], "success")
            step += 1
        except LLMError as e:
            self.log(step, "llm_response", task[:100], None, "error", f"{e.tag}: {e.message[:200]}")
            return {"agent_id": self.agent_id, "task": task, "output": None,
                    "trace": self.trace, "success": False, "error": e.tag}

        validation = self._validate(output)
        self.log(step, "validate", output[:100], validation,
                 "success" if validation["valid"] else "error",
                 None if validation["valid"] else validation["reason"])

        return {"agent_id": self.agent_id, "task": task, "output": output,
                "trace": self.trace, "success": validation["valid"]}

    def _validate(self, output) -> dict:
        if output is None:
            return {"valid": False, "reason": "null_output"}
        if len(str(output).strip()) < 10:
            return {"valid": False, "reason": "output_too_short"}
        if any(p in str(output).lower() for p in ["error:", "exception:", "traceback"]):
            return {"valid": False, "reason": "error_in_output"}
        return {"valid": True, "reason": "passed"}