"""
The wrapper. Observes any agent, detects failure, diagnoses and repairs.
"""

import os
import sys
import time
import json
import inspect
from typing import Any, Optional, List, Dict

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.parser import parse_failure_type, extract_fix, is_api_failure
from src.council.prosecutor import ProsecutorAgent
from src.council.defender import DefenderAgent
from src.council.coroner import CoronerAgent
from src.surgeon.surgeon import SurgeonAgent

from .config import AASEConfig
from .result import RunResult, RepairRecord

_ERROR_MARKERS = ("error:", "exception:", "traceback", "failed to",
                  "i cannot access", "something went wrong")


def _default_validator(output, task):
    """Conservative default: only obvious breakage counts as failure."""
    if output is None:
        return False, "agent returned nothing"
    text = str(output).strip()
    if not text:
        return False, "agent returned an empty string"
    if len(text) < 10:
        return False, "output too short ({} chars)".format(len(text))
    low = text.lower()
    for m in _ERROR_MARKERS:
        if low.startswith(m) or ("\n" + m) in low:
            return False, "output contains an error marker: '{}'".format(m)
    return True, ""


def _find_callable(agent):
    """Locate the method that runs a task."""
    if callable(agent) and not inspect.isclass(agent):
        for name in ("run", "invoke", "execute"):
            fn = getattr(agent, name, None)
            if callable(fn):
                return name, fn
        return "__call__", agent
    for name in ("run", "invoke", "execute", "call", "__call__"):
        fn = getattr(agent, name, None)
        if callable(fn):
            return name, fn
    raise TypeError(
        "wrap_agent() needs an object with .run(), .invoke(), .execute() "
        "or .__call__(), or a plain function. Got: {}".format(type(agent).__name__)
    )


def _find_prompt_attr(agent, config):
    if config.prompt_attr:
        return config.prompt_attr if hasattr(agent, config.prompt_attr) else None
    for name in config.prompt_candidates:
        val = getattr(agent, name, None)
        if isinstance(val, str) and len(val) > 10:
            return name
    return None


class AASEAgent:
    """
    A wrapped agent. Same interface as the original, plus failure diagnosis
    and repair.
    """

    def __init__(self, agent: Any, config: Optional[AASEConfig] = None):
        self.agent = agent
        self.config = config or AASEConfig()
        self._method_name, self._call = _find_callable(agent)
        self._prompt_attr = _find_prompt_attr(agent, self.config)

        self._prosecutor = ProsecutorAgent()
        self._defender = DefenderAgent()
        self._coroner = CoronerAgent()
        self._surgeon = SurgeonAgent()

        self._ledger = None
        if self.config.use_ledger:
            try:
                from src.ledger.ledger import RepairLedger
                self._ledger = RepairLedger()
            except Exception as e:
                self._log("ledger unavailable ({}) — continuing without it".format(
                    type(e).__name__))

        self.history: List[RunResult] = []

        if self.config.verbose:
            self._log("wrapped {} · calls .{}()".format(
                type(agent).__name__, self._method_name))
            if self._prompt_attr:
                self._log("system prompt found at .{} — repairs will be applied".format(
                    self._prompt_attr))
            else:
                self._log("no system prompt attribute found — "
                          "AASE will diagnose but cannot repair")

    # ── public ───────────────────────────────────────────────────────

    def run(self, task: str, **kwargs) -> RunResult:
        started = time.time()
        output = self._invoke(task, **kwargs)
        ok, reason = self._validate(output, task)

        result = RunResult(output=output, success=ok, task=str(task),
                           failure_reason="" if ok else reason)

        if ok:
            result.total_seconds = time.time() - started
            self.history.append(result)
            return result

        self._log("failure detected — {}".format(reason))
        if self.config.on_failure:
            try:
                self.config.on_failure(task, output, reason)
            except Exception:
                pass

        trace = self._build_trace(task, output, reason)
        result.trace = trace

        attempts = 0
        while attempts < self.config.max_repairs_per_task:
            attempts += 1
            record = self._diagnose_and_repair(task, trace)
            if record is None:
                break
            result.repairs.append(record)

            if not self.config.retry_after_repair or not self._prompt_attr:
                break

            self._log("retrying task with repaired prompt")
            output2 = self._invoke(task, **kwargs)
            ok2, reason2 = self._validate(output2, task)
            record.retried = True
            record.retry_succeeded = ok2

            if ok2:
                self._log("retry succeeded")
                result.output = output2
                result.success = True
                result.repaired = True
                result.failure_reason = ""
                break

            self._log("retry still failing — {}".format(reason2))
            result.output = output2
            result.failure_reason = reason2
            trace = self._build_trace(task, output2, reason2)

        result.total_seconds = time.time() - started
        self.history.append(result)
        return result

    def __call__(self, task: str, **kwargs):
        return self.run(task, **kwargs)

    def stats(self) -> Dict:
        total = len(self.history)
        ok = sum(1 for r in self.history if r.success)
        repaired = sum(1 for r in self.history if r.repaired)
        attempted = sum(len(r.repairs) for r in self.history)
        by_type: Dict[str, int] = {}
        for r in self.history:
            for rep in r.repairs:
                by_type[rep.failure_type] = by_type.get(rep.failure_type, 0) + 1
        return {
            "runs": total,
            "succeeded": ok,
            "failed": total - ok,
            "success_rate_pct": round(ok / total * 100, 1) if total else None,
            "repairs_attempted": attempted,
            "recovered_by_repair": repaired,
            "failure_types_seen": by_type,
            "ledger_size": self._ledger.stats()["total_repairs"] if self._ledger else None,
        }

    @property
    def system_prompt(self) -> Optional[str]:
        if self._prompt_attr:
            return getattr(self.agent, self._prompt_attr)
        return None

    def __getattr__(self, name):
        """Pass through anything AASE doesn't define to the wrapped agent."""
        try:
            agent = self.__dict__["agent"]
        except KeyError:
            raise AttributeError(name)
        return getattr(agent, name)

    # ── internals ────────────────────────────────────────────────────

    def _log(self, msg):
        if self.config.verbose:
            print("[AASE] {}".format(msg))

    def _invoke(self, task, **kwargs):
        try:
            return self._call(task, **kwargs)
        except Exception as e:
            return "ERROR: {}: {}".format(type(e).__name__, e)

    def _validate(self, output, task):
        fn = self.config.validator or _default_validator
        try:
            res = fn(output, task)
        except TypeError:
            res = fn(output)
        if isinstance(res, tuple):
            return bool(res[0]), (res[1] if len(res) > 1 else "")
        return bool(res), ("" if res else "validator returned False")

    def _build_trace(self, task, output, reason):
        return [
            {"step": 1, "action": "ingest", "input": str(task)[:200],
             "output": "Task received.", "status": "success"},
            {"step": 2, "action": "agent.{}".format(self._method_name),
             "input": str(task)[:200], "output": str(output)[:500],
             "status": "success"},
            {"step": 3, "action": "validate", "input": "check output",
             "output": "FAILED: {}".format(reason), "status": "error",
             "error": "validation_failed: {}".format(reason)},
        ]

    def _diagnose_and_repair(self, task, trace) -> Optional[RepairRecord]:
        t0 = time.time()
        signature = "agent:{}|task:{}".format(type(self.agent).__name__, str(task)[:80])

        # fast path — ledger
        if self._ledger:
            try:
                hit = self._ledger.find_similar_repair(signature)
            except Exception:
                hit = None
            if hit:
                self._log("ledger hit — applying stored fix, skipping council")
                rec = RepairRecord(
                    failure_type=hit.get("failure_type", "unknown"),
                    method="ledger",
                    diagnosis_seconds=round(time.time() - t0, 2),
                    clause_added=hit["fix"][:400],
                )
                self._apply(rec, hit.get("failure_type", "unknown"), hit["fix"])
                return rec

        # slow path — council
        self._log("convening diagnosis council")
        tstr = json.dumps(trace, indent=2)[:self.config.trace_char_limit]
        unk = "unknown — determine from trace"

        pa = self._prosecutor.analyze(tstr, unk, str(task))
        da = self._defender.analyze(tstr, unk, str(task))
        vd = self._coroner.decide(tstr, pa, da, unk)

        tag = is_api_failure(vd) or is_api_failure(pa) or is_api_failure(da)
        if tag:
            self._log("council unavailable — {}".format(tag))
            return None

        dx, _ = parse_failure_type(vd)
        fix = extract_fix(vd)
        self._log("diagnosis: {}".format(dx))

        rec = RepairRecord(
            failure_type=dx,
            method="council",
            diagnosis_seconds=round(time.time() - t0, 2),
            prosecutor=pa[:600],
            defender=da[:600],
            coroner_verdict=vd[:800],
        )
        self._apply(rec, dx, fix)

        if self._ledger:
            try:
                self._ledger.store_repair(signature, dx, fix, "applied")
            except Exception:
                pass
        return rec

    def _apply(self, record: RepairRecord, failure_type: str, fix: str):
        if not self._prompt_attr:
            record.clause_added = fix[:400]
            self._log("no writable prompt attribute — "
                      "diagnosis recorded, no repair applied")
            return
        before = getattr(self.agent, self._prompt_attr)
        after = self._surgeon.apply_fix(before, failure_type, fix)
        setattr(self.agent, self._prompt_attr, after)
        record.prompt_before = before
        record.prompt_after = after
        record.clause_added = after.replace(before, "").strip()[:400]
        self._log("prompt repaired (+{} chars)".format(len(after) - len(before)))
        if self.config.on_repair:
            try:
                self.config.on_repair(record)
            except Exception:
                pass


def wrap_agent(agent: Any, config: Optional[AASEConfig] = None, **kwargs) -> AASEAgent:
    """
    Wrap an agent so failures are diagnosed and repaired automatically.

        from aase import wrap_agent
        agent = wrap_agent(my_agent)
        result = agent.run("some task")

    Any AASEConfig field can be passed as a keyword:

        agent = wrap_agent(my_agent, validator=my_check, verbose=False)
    """
    if config is None:
        config = AASEConfig(**kwargs) if kwargs else AASEConfig()
    elif kwargs:
        for k, v in kwargs.items():
            setattr(config, k, v)
    return AASEAgent(agent, config)