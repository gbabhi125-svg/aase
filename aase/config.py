"""Configuration for the AASE wrapper."""

import os
from dataclasses import dataclass, field
from typing import Callable, Optional, List


@dataclass
class AASEConfig:
    """
    Controls how AASE observes, diagnoses and repairs a wrapped agent.

    validator
        Callable (output, task) -> bool | (bool, str). Decides whether a run
        failed. If omitted, a default heuristic is used: empty output, very
        short output, or output containing an error string counts as failure.

    retry_after_repair
        Re-run the task once after the surgeon edits the prompt.

    max_repairs_per_task
        Repair attempts per task before giving up.

    use_ledger
        Check the repair ledger before convening the council.

    prompt_attr
        Attribute on the wrapped agent holding its system prompt. AASE
        auto-detects common names when this is None.

    on_failure / on_repair
        Callbacks for logging or telemetry.

    verbose
        Print progress to stdout.
    """

    validator: Optional[Callable] = None
    retry_after_repair: bool = True
    max_repairs_per_task: int = 1
    use_ledger: bool = True
    prompt_attr: Optional[str] = None
    on_failure: Optional[Callable] = None
    on_repair: Optional[Callable] = None
    verbose: bool = True
    trace_char_limit: int = 3000

    prompt_candidates: List[str] = field(default_factory=lambda: [
        "system_prompt", "system_message", "instructions", "prompt",
        "system", "role_prompt", "backstory",
    ])

    @classmethod
    def from_env(cls) -> "AASEConfig":
        return cls(
            retry_after_repair=os.getenv("AASE_RETRY", "1") not in ("0", "false", "False"),
            use_ledger=os.getenv("AASE_LEDGER", "1") not in ("0", "false", "False"),
            verbose=os.getenv("AASE_VERBOSE", "1") not in ("0", "false", "False"),
        )