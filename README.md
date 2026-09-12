# AASE — Agent Autopsy &amp; Self-Evolution Engine

Wrap any LLM agent. When it fails, three specialised agents argue the root
cause, a fourth rewrites the agent's own instructions, and the repair is
stored so the same failure is never diagnosed twice.

```python
from aase import wrap_agent

agent = wrap_agent(my_agent)
result = agent.run("Retrieve the Q3 revenue for Division B")

print(result.output)
print(result.report())   # what failed, what was diagnosed, what changed
```

That is the whole integration. No subclassing, no framework adoption, no
retraining.

---

## Contents

- [The problem](#the-problem)
- [How it works](#how-it-works)
- [Install](#install)
- [The Operator Console](#the-operator-console)
- [Failure taxonomy](#failure-taxonomy)
- [Checks](#checks)
- [Datasets](#datasets)
- [Results](#results)
- [Findings](#findings)
- [Known limitations](#known-limitations)
- [Scripts](#scripts)
- [Repository layout](#repository-layout)

---

## The problem

An LLM agent fails in production. It states a figure its database never
returned, repeats a failing search indefinitely, or obeys an instruction
hidden inside a PDF it was asked to summarise. An engineer reads the logs,
guesses the cause, edits the prompt by hand, and redeploys. No record
survives of what was tried. The next engineer repeats the work.

Existing research detects or attributes agent failures. The Who&amp;When
benchmark formalised failure attribution — which agent failed, and at which
step. That is diagnosis. Nothing in that line of work then repairs the agent.

AASE closes the loop: it takes the diagnosis and edits the failing agent's
instructions so the cause is removed, then verifies by re-running the
identical task.

---

## How it works

agent.run(task)
│
▼
validator ──── passes ───▶ return output
│
fails
▼
ledger probe ── hit ───▶ apply stored fix ──┐
│ │
miss │
▼ │
┌────────────────────────────────────┐ │
│ Prosecutor argues systemic fault │ │
│ Defender argues external cause │ │
│ Coroner rules on evidence │ │
└────────────────────────────────────┘ │
│ │
▼ │
Surgeon rewrites the system prompt ◀────────┘
│
▼
retry the same task ──▶ re-validate ──▶ store the signature


**The Defender exists to prevent over-repair.** Without an argument for
external causes, every failure attracts a prompt edit — including failures
the agent did not cause. The prompt accumulates defensive clauses against
problems that were never its fault, and degrades.

**No model decides whether the agent succeeded.** Pass and fail are decided
by deterministic Python checkers.

---

## Install

```bash
git clone https://github.com/gbabhi125-svg/aase.git
cd aase
pip install -e ".[all]"
cp .env.example .env        # add an API key
```

`.env`:

LLM_PROVIDER=groq # groq | gemini
GROQ_API_KEY=...
GROQ_MODEL=qwen/qwen3.8-27b
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.6-flash

SIMILARITY_THRESHOLD=0.05 # measured, not guessed — see below
LLM_MIN_GAP=1 # seconds between calls, for free tiers


Verify the provider before anything else:

```bash
python -m src.llm
# provider = groq
# model    = qwen/qwen3.8-27b
# result   = OK (1.2s)
```

### Try it

```bash
python examples/quickstart.py       # a real agent fails and is repaired, live
python examples/frameworks.py       # four agent shapes, one line each, offline
python live.py                      # the Operator Console
```

---

## The Operator Console

```bash
pip install fastapi uvicorn
python live.py
# open http://127.0.0.1:8000
```

A web console with four ways in and the controls that make a cycle
inspectable rather than merely watchable.

### Input modes

| Mode | Calls | What it does |
|---|---|---|
| **Library** | 3 | Browse all 1,234 indexed traces — 1,050 synthetic, 184 real — filterable by type and source. Selecting one shows its ground-truth label and, for real traces, the human annotation. The council diagnoses it **without being told the label**, then the ruling panel reports whether it matched. |
| **Compose** | 4 | Supply your own agent prompt, task, tool output and check. The agent is executed live, validated, diagnosed, repaired and re-run. This is the "here is my failure, show me" path. |
| **Paste** | 3 | Drop in a raw trace array, or a whole trace file from `data/` — the `.trace` array is extracted automatically, the task auto-filled, and the file's own `failure_type` used as ground truth. |
| **Ledger** | 0 | Browse every stored repair. Probe the vector memory with any text and see the nearest matches with real cosine distances and latency. |

### Controls

- **Operator override** — reject the Surgeon's clause, write your own, re-run.
  One call. This is the direct test of whether the *diagnosis* or merely the
  *edit* produced the improvement.
- **Freeze** — stop a running cycle before the next model call.
- **Export** — download the whole session as JSON.
- **Focus mode** — collapse the side rails and expand the pipeline for recording.
- **Light / dark** — both themes built separately, not inverted.

### What is on screen

The **pipeline graph** shows seven nodes wired together, each lighting as it
works, with traffic lights on every edge — grey idle, blue in flight, green
passed, red failed. The return path from Surgeon to Agent turns green or red
depending on whether the retry held.

The **council chamber** holds three fixed seats that fill with the real
arguments as each agent speaks, with latency.

The **surgery deck** shows prompt versions v1 → v2 → v3 with a word-level
diff, additions in green and deletions struck through.

The **fleet panel** tracks each agent's health, prompt version and run history
across the session.

---

## Failure taxonomy

Seven types. The Coroner applies an ordered decision procedure so the most
distinctive signal is checked first.

| Type | Definition | Distinguishing signal |
|---|---|---|
| `hallucination` | States facts or values no tool returned | Tool errored or returned empty; agent gave a specific value anyway |
| `tool_misuse` | Wrong tool, wrong parameters, invalid action format | A different available tool was the correct one |
| `reasoning_loop` | Same action repeated with no progress | Identical input 3+ times, no strategy change |
| `context_collapse` | Gradual loss of an instruction set at the start | Behaviour changes after a context warning; agent keeps running |
| `goal_drift` | Final output violates the original task's constraints | Each step looks fine; the result differs in scope, length or recipient |
| `prompt_injection` | Obeys an instruction found in retrieved content | External data contains an imperative; the next action follows it |
| `memory_overflow` | A hard limit terminated execution | Token or step limit exceeded, process killed. Not gradual |

`context_collapse` and `memory_overflow` are deliberately separate even though
both concern context. One degrades while running; the other stops dead. They
require opposite repairs — re-anchoring instructions versus bounding the
working set — so merging them would produce the wrong fix.

---

## Checks

Eight, four of them parameterised. The check decides pass or fail, and a
model is never asked.

| Check | Fails when |
|---|---|
| No invented figures | Output states a number absent from both task and tool output |
| No repeated action | Output proposes retrying something that already failed |
| No instruction-following | Output acts on a command found inside retrieved content |
| Bounded memory | Output proposes no batching or state clearing for bulk work |
| Must contain | Output is missing a string you supply |
| Must not contain | Output contains a string you supply |
| Line limit | Output exceeds a line count you set |
| Regex | Your pattern does not match, or matches when it should not |

Supply your own to `wrap_agent`:

```python
def validator(output, task):
    if "$" in output and "no data" in tool_result:
        return False, "stated a figure the tool never returned"
    return True, ""

agent = wrap_agent(my_agent, validator=validator)
```

Return `bool` or `(bool, reason)`.

---

## Datasets

### Source corpora

| Source | Used | One record contains | Why |
|---|---|---|---|
| **MultiWOZ 2.1** | 10,438 dialogues | Human–system conversation log, domain databases, slot ontology | Standard benchmark for multi-turn task-oriented agents; goals are machine-checkable |
| **HumanEval** | 164 problems | Function signature, docstring, canonical solution, unit tests | One domain where success is binary |
| **ToolBench** | 10 sample queries | Natural-language request plus callable API specs | Standard tool-selection benchmark. Only the repository sample was reachable; the full 200k corpus is on Google Drive |
| **Who&amp;When** | 184 traces | A real execution trace from a failing multi-agent system with expert annotation naming the failing agent, step and reason | Real failures not constructed by the author. 126 CaptainAgent teams, 58 hand-crafted Magentic-One, on GAIA and AssistantBench |

Combined real task pool: **10,612 tasks.**

### Generated evaluation set

**No model is trained anywhere in this project.** The generated data is a
labelled failure benchmark, not training data. Each real task passes through
one of seven deterministic injectors that builds a trace exhibiting that
failure together with a consistent task description.

**1,050 traces — 150 per type × 7 types.** Each file holds `trace_id`,
`failure_type`, `task`, `domain`, `source_dataset`, the step array,
`num_steps`, `failure_detected_at_step` and `ground_truth_fix`.

### Trace format

```json
{"step": 2, "action": "read_pdf", "input": "...", "output": "...",
 "status": "error", "error": "prompt_injection_detected"}
```

`step` sequence number · `action` what was done · `output` what came out —
this is what the council reads most closely · `status` success, warning or
error; mark the failing step `error` · `error` a short machine-readable
reason · `input` optional.

---

## Results

All figures are computed from raw counts in saved output files. None is
estimated or reported as a mean of percentages.

### Diagnosis on the controlled benchmark

**70 of 70 correct** across all seven types, ten traces each, zero API
exclusions. The confusion matrix is diagonal. Reported as 10/10 per category
rather than as a percentage — the confidence interval on ten samples is wide.

### Diagnosis on real-world traces

Five independent runs over the same 37-trace Who&amp;When sample.

| Run | Scored | Correct | Accuracy | Note |
|---|---|---|---|---|
| 1 | 24 | 8 | 33.3% | truncated by rate limit |
| 2 | 36 | 7 | 19.4% | 1 API exclusion |
| 3 | 37 | 16 | 43.2% | complete |
| 4 | 12 | 6 | 50.0% | truncated by rate limit |
| 5 | 37 | 13 | 35.1% | complete |
| **Pooled** | **146** | **50** | **34.2%** | range 19.4–50.0% |

The pooled figure aggregates raw counts. It is not the mean of the five
percentages, which would weight a 12-trace run equally with a 37-trace run.

For reference, the Who&amp;When authors report their best automated attribution
method at **53.5%** on identifying the failure-responsible agent and 14.2% on
pinpointing the decisive step, with OpenAI o1 and DeepSeek R1 both failing to
reach practical usability. They report annotator uncertainty between 15% and
30% across three human experts.

### Repair — does the agent get better?

An under-specified agent, 34 tasks across seven failure surfaces, outcomes
decided by deterministic checkers. Three conditions over identical tasks:
**A** no repair, **B** one fixed generic clause with no diagnosis, **C** full
AASE.

Run three times. Two completed; one was truncated by rate limits.

| | Run 1 | Run 3 | Pooled |
|---|---|---|---|
| Tasks scored | 33 | 34 | 67 |
| Failed without AASE | 4 | 2 | 6 (9.0%) |
| Repaired by AASE on retry | 3 | 0 | **3 (50%)** |
| Repaired by generic control | 1 | 0 | **1 (17%)** |
| Failure rate before → after | 12.1% → 3.0% | 5.9% → 5.9% | — |

**The two runs disagree sharply.** Run 1 produced four failures across four
surfaces, three repaired. Run 3 produced two failures, both on the
repeated-action surface, none repaired, control also none. Which tasks fail
is not stable across runs on identical inputs, and at these sample sizes a
repair rate cannot be estimated with precision.

Condition B is the informative comparison. Both B and C modify the prompt;
only C determines what is wrong first. For a repeated-query failure in run 1,
the Surgeon wrote:

> *"If a tool call returns the same or ambiguous result as the previous
> attempt, do not execute a third identical action. Instead, stop immediately
> and…"*

A named condition, a threshold and a required action. "Be careful" cannot
encode that, and did not fix the same case.

**Task T16 — an agent facing a repeatedly rate-limited fetch — failed in every
run and was never repaired under any condition, including an operator-written
clause.** Reported as an unresolved case rather than excluded.

### Confidence calibration — inconclusive

A ranked output format was tested over the same decision procedure, asking the
Coroner for a primary label, a runner-up, a confidence level and an ambiguity
flag.

| | top-1 | top-2 |
|---|---|---|
| Overall (N=37) | 24.3% | — |
| High confidence (N=13) | 31% | — |
| Medium confidence (N=23) | 22% | — |
| Flagged ambiguous | 18% | — |
| Not flagged | 27% | — |

High-confidence verdicts scored barely above medium. The ambiguity flag
separated hard from easy cases by nine points — the right direction, not a
conclusive one. **Confidence calibration was not demonstrated in this test.**

Top-1 of 24.3% sits inside the 19.4–50.0% band observed across the five
single-label runs, so the difference is not attributed to the output format.

`memory_overflow` scored **0/6 on both top-1 and top-2.** Since top-2 admits
two of seven labels, chance alone would recover roughly two of six. Recovering
none indicates the Coroner is not uncertain on these traces but consistently
confident in a different reading — `tool_misuse` in every case.

---

## Findings

### Keyword mapping of human annotations is unreliable

Who&amp;When supplies free-text explanations, not labels in this taxonomy.
Relabelling all 184 traces with an LLM applied to the annotation text alone —
the trace was never shown, so it cannot act as a second diagnosis — changed
**105 of 184 labels (57.1%)**. Agreement between methods: 42.9%.

| Failure type | Keyword | LLM-derived | Change |
|---|---|---|---|
| hallucination | 171 | 81 | −90 |
| tool_misuse | 7 | 69 | +62 |
| goal_drift | 3 | 27 | +24 |
| memory_overflow | 0 | 6 | +6 |
| reasoning_loop | 2 | 1 | −1 |
| context_collapse | 1 | 0 | −1 |

Annotations such as *"The code provided by the BingAPI_Expert is incorrect"*
contain the word "incorrect" and were captured as hallucination by keyword
matching, when they describe tool misuse. An earlier 39.1% figure obtained
under keyword labels was therefore measuring label quality, not diagnosis
quality. Every relabelling decision is recorded in `data/relabel_audit.json`.

### The label captures the symptom; the trace reveals the cause

Six traces were relabelled `memory_overflow`. On all six, in every run, the
Coroner assigned a different type — usually `tool_misuse`. Five of the six
annotations share one construction: a mistake, then *"leading to"* or
*"exhausting"*, then a terminal condition.

> *"The expert wrote code with bugs multiple times, leading to the exhaustion
> of the step limits."*

The annotation-only labeller anchored on the consequence. The Coroner, reading
the trace, anchored on the cause. Both readings are defensible. For repair
purposes the Coroner's is more actionable — a system cannot be repaired
against "ran out of steps", only against "wrote code that failed repeatedly".

We do not claim the Coroner is more accurate than the human annotators. A
seven-category taxonomy forces a choice the annotators did not have to make.

### Stability tracks label determinacy

Of 12 traces judged in more than one complete run, **7 received a different
diagnosis**. On the synthetic set verdicts were stable. Neither code, traces
nor labels changed between runs; the variation is the model resolving close
calls differently.

**One case does not fit this explanation and is reported rather than smoothed
over.** `REAL_ALGO_10` carries the annotation *"The agent fabricated the
population figures for Seattle and Colville"* — an unambiguous description of
hallucination — yet the Coroner assigned `tool_misuse` in one run and
`goal_drift` in another. Not all instability is attributable to genuine label
ambiguity.

### A silent embedder switch disabled the Repair Ledger

The ledger held 323 repairs written by `sentence-transformers`, but was being
queried with hash-fallback vectors after the model download failed in a later
session. Cosine distance from a signature to its own stored vector was
**1.0102** — orthogonal. Every probe reported "no match", and no threshold
could have fixed it.

Diagnosed with `scripts/probe_ledger.py`, confirmed with
`scripts/ledger_root_cause.py`, repaired with `scripts/rebuild_ledger.py`.
After rebuilding:

| Probe | Median distance |
|---|---|
| Signature vs itself | **0.0000** |
| Unrelated text | **1.5389** |

A separation of 1.54. `SIMILARITY_THRESHOLD` was then set from measurement
rather than guesswork. The collection now records which embedder wrote it and
the ledger refuses to query across a mismatch rather than returning noise.

---

## Known limitations

- **Sample sizes are small throughout.** Six failures is not a basis for a
  repair-rate estimate; ten traces per category cannot distinguish 100% from
  90%; twelve traces judged twice is thin for a stability claim.
- **Real-world ground truth is derived, not given.** Who&amp;When supplies
  free-text explanations. Every real-world figure is an agreement measure
  between two labelling procedures, not accuracy against verified truth.
- **The synthetic set is author-constructed and unambiguous by design.** The
  70/70 result shows the taxonomy and decision procedure are internally
  consistent; it does not transfer to unseen data, as the real-world section
  shows directly.
- **Diagnosis is not deterministic on ambiguous traces**, and at least one
  clearly-labelled case also varied between runs.
- **Confidence calibration was not demonstrated.**
- **`memory_overflow` is not reliably distinguished** from `tool_misuse` on
  real annotated data, even with a runner-up allowed.
- **Task T16 has never been repaired** under any condition.
- **API rate limits truncated several runs.** Excluded traces are reported
  separately, but truncation reduces effective sample size.
- **Repair is measured by re-running the identical task once.** Whether a
  rewritten prompt prevents recurrence over sustained production use is not
  established.
- **AASE cannot wrap Claude, Copilot or Grok.** Those are closed products with
  no interception point, for anyone. AASE wraps agents built on an API — which
  is what every LangChain, AutoGen and CrewAI deployment is.

---

## Scripts

| Script | Purpose | API calls |
|---|---|---|
| `run_repair_experiment.py` | Does the agent get better? Three conditions, deterministic checkers | ~90 |
| `run_accuracy_test.py` | Diagnosis accuracy against ground truth | 3 per trace |
| `run_ranked_test.py` | Ranked verdict, confidence calibration, top-2 | 3 per trace |
| `run_baseline.py` | Three-condition comparison | ~140 |
| `generate_data.py` | Regenerate the synthetic set with task/trace consistency | 0 |
| `relabel_real_data.py` | Relabel Who&amp;When from annotations, with audit trail | 1 per trace |
| `inspect_failures.py` | Raw Coroner output for failing categories | 3 per trace |
| `inspect_disagreements.py` | Cross-run instability and confusion analysis | 0 |
| `probe_ledger.py` | Is retrieval working? Threshold sweep | 0 |
| `ledger_root_cause.py` | Why a signature does not match itself | 0 |
| `rebuild_ledger.py` | Re-embed the ledger with the active embedder | 0 |

---

## Repository layout

aase/ the wrapper — wrap_agent(), config, result types
src/
council/ prosecutor, defender, coroner
surgeon/ prompt surgery with per-type clause specifications
ledger/ ChromaDB repair memory, embedder-compatibility guard
llm.py provider abstraction (groq | gemini), typed API errors
parser.py single shared verdict parser, single and ranked
live.py Operator Console backend — FastAPI + SSE
live.html Operator Console frontend — single file, no build step
examples/
quickstart.py a real agent fails and is repaired, live
frameworks.py four agent shapes, one line each, offline
scripts/ experiments and diagnostics
data/ traces, manifests, results, audit files


---

## Citation

The real-world evaluation set is the Who&amp;When benchmark:

```bibtex
@inproceedings{zhang2025whowhen,
  title  = {Which Agent Causes Task Failures and When? On Automated
            Failure Attribution of LLM Multi-Agent Systems},
  author = {Zhang, Shaokun and others},
  booktitle = {ICML},
  year   = {2025}
}
```

---

MIT