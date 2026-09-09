"""Return types for AASE runs."""

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict


@dataclass
class RepairRecord:
    """What AASE did about one failure."""
    failure_type: str
    method: str
    diagnosis_seconds: float
    prosecutor: str = ""
    defender: str = ""
    coroner_verdict: str = ""
    clause_added: str = ""
    prompt_before: str = ""
    prompt_after: str = ""
    retried: bool = False
    retry_succeeded: Optional[bool] = None

    def summary(self) -> str:
        head = f"{self.failure_type} · via {self.method} · {self.diagnosis_seconds:.1f}s"
        if self.retried:
            head += f" · retry {'succeeded' if self.retry_succeeded else 'failed'}"
        return head


@dataclass
class RunResult:
    """
    Result of a wrapped agent run.

    Truthy when the run ultimately succeeded, so `if result:` works.
    str() gives the output, so it drops into code expecting the raw return.
    """
    output: Any
    success: bool
    task: str = ""
    trace: List[Dict] = field(default_factory=list)
    repairs: List[RepairRecord] = field(default_factory=list)
    failure_reason: str = ""
    repaired: bool = False
    total_seconds: float = 0.0

    def __bool__(self):
        return self.success

    def __str__(self):
        return str(self.output) if self.output is not None else ""

    def report(self) -> str:
        lines = [
            f"task     : {self.task[:80]}",
            f"success  : {self.success}",
            f"repaired : {self.repaired}",
            f"elapsed  : {self.total_seconds:.1f}s",
        ]
        if self.failure_reason:
            lines.append(f"failure  : {self.failure_reason}")
        for i, r in enumerate(self.repairs, 1):
            lines.append(f"repair {i}  : {r.summary()}")
            if r.clause_added:
                lines.append(f"  clause : {r.clause_added[:120]}")
        return "\n".join(lines)