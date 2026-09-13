<div align="center">

# 🔬 AASE

### Agent Autopsy &amp; Self-Evolution Engine

**Autonomous diagnosis and repair of failing LLM agents**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688?style=flat-square&logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Memory-FF6B35?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-qwen3.8--27b-F55036?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

*Five agents. Seven failure types. 1,234 labelled traces. One line to integrate.*

</div>

---

```python
from aase import wrap_agent

agent = wrap_agent(my_agent)
result = agent.run("Retrieve the Q3 revenue for Division B")
```

That is the entire integration. No subclassing, no framework adoption, no
retraining, no GPU.

---

## 📌 Table of Contents

- [About The Project](#-about-the-project)
- [SDG Goals](#-sdg-goals)
- [Features](#-features)
- [Architecture](#-architecture)
- [The Five Agents](#-the-five-agents)
- [Failure Taxonomy](#-failure-taxonomy)
- [The Operator Console](#-the-operator-console)
- [Datasets](#-datasets)
- [Results](#-results)
- [Findings](#-findings)
- [Known Limitations](#-known-limitations)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Scripts](#-scripts)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [Author](#-author)

---

## 🧬 About The Project

An LLM agent fails in production. It states a figure its database never
returned. It repeats a failing search indefinitely. It obeys an instruction
hidden inside a PDF it was asked to summarise.

What happens next is always the same. An engineer is alerted, reads the logs,
guesses at the cause, edits the system prompt by hand, and redeploys. No
record survives of what was tried or why. The next engineer repeats the work.
The same failure returns.

**AASE automates that loop and keeps the record.**

### Problem Statement

LangChain, AutoGen and CrewAI made agents easy to deploy. The supporting
discipline has not kept pace. Conventional software has debuggers, stack
traces, unit tests and regression suites. An LLM agent has none of these,
because its failures are **semantic rather than syntactic** — the code runs
perfectly and returns a wrong answer confidently.

### The Gap

Existing research **detects** or **attributes** agent failures. The Who&amp;When
benchmark (Zhang et al., ICML 2025) formalised automated failure attribution:
identifying which agent in a multi-agent system caused a task to fail, and at
which step.

That is diagnosis. **Nothing in that line of work then repairs the agent.**

> AASE adds the missing half — it takes the diagnosis, edits the failing
> agent's own instructions so the cause is removed, then verifies the repair
> by re-running the identical task.

### The Idea

A single model asked *"why did this fail?"* anchors on the first plausible
explanation and stops. AASE forces the question to be **argued**.

A **Prosecutor** argues the agent's instructions are fundamentally at fault.
A **Defender** argues the environment failed and the agent behaved correctly.
A neutral **Coroner** reads the trace and both arguments, and rules on
evidence alone.

The Defender exists to prevent over-repair. Without an argument for external
causes, every failure attracts a prompt edit — including failures the agent
did not cause. The prompt accumulates defensive clauses against problems that
were never its fault, and degrades.

---

## 🌍 SDG Goals

| Goal | How AASE contributes |
|---|---|
| **SDG 9** — Industry, Innovation and Infrastructure | Provides reliability infrastructure for autonomous AI systems being deployed across industry, reducing the engineering labour required to keep them dependable |

> **Stated honestly:** this is infrastructure software, not a humanitarian
> application. SDG 9 is the one genuine fit. Claiming more would be a stretch.

---

## ✨ Features

### 🤖 The Repair Loop

- **Failure interception** — wraps any agent, captures every step as a
  structured trace
- **Deterministic validation** — pass and fail decided by Python checkers,
  never by a model
- **Adversarial diagnosis** — three specialised agents argue before a verdict
  is reached
- **System prompt surgery** — one to three clauses rewritten, targeted at the
  ruled cause
- **Immune memory** — ChromaDB vector store; a matching signature bypasses
  the council entirely
- **Verified repair** — the identical task is re-run and re-checked

### 🖥 The Operator Console

- **Four input modes** — Library, Compose, Paste, Ledger
- **Live streaming** — Server-Sent Events; every stage appears the moment it
  completes
- **Animated pipeline** — seven nodes with traffic-light edges, data packets
  physically flying between panels
- **Operator override** — reject the Surgeon's clause, write your own, re-run
- **Freeze** — stop a cycle before the next model call
- **Word-level prompt diff** — v1 → v2 → v3 with additions and deletions
- **Immunity probe** — query the vector memory with any text, zero API calls
- **Dark and light themes** — built separately, not inverted
- **Focus mode** — collapse the rails, expand the pipeline, for recording

### 🔬 Evaluation Harness

- **Eight deterministic checks**, four parameterised by the operator
- **Three-condition baseline** — no repair / generic edit / full AASE
- **Generalisation testing** — does a repair transfer to unseen tasks, and
  what does it break?
- **Confidence calibration** — ranked verdicts with ambiguity flags
- **Cost accounting** — calls, latency, ledger break-even
- **Failure DNA** — embedding-space map of the taxonomy

---

## 🏗 Architecture

                     agent.run(task)
                           │
                           ▼
              ┌────────────────────────┐
              │      VALIDATOR         │ ── passes ──▶ return output
              │  deterministic check   │
              └───────────┬────────────┘
                          │ fails
                          ▼
              ┌────────────────────────┐
              │     REPAIR LEDGER      │ ── hit ──┐
              │  vector signature probe│          │
              └───────────┬────────────┘          │
                          │ miss                  │
                          ▼                       │
  ┌───────────────────────────────────────────┐   │
  │           DIAGNOSIS COUNCIL               │   │
  │                                           │   │
  │   PROSECUTOR  ──▶                         │   │
  │                    CORONER ──▶ verdict    │   │
  │   DEFENDER    ──▶                         │   │
  │                                           │   │
  └───────────────────────┬───────────────────┘   │
                          │                       │
                          ▼                       │
              ┌────────────────────────┐          │
              │       SURGEON          │ ◀────────┘
              │  rewrites 1–3 clauses  │
              └───────────┬────────────┘
                          │
                          ▼
          retry the identical task ──▶ re-validate
                          │
                          ▼
              store the signature + fix + outcome




---

## 👥 The Five Agents

| Agent | Role | What it does |
|---|---|---|
| **Subject Agent** | monitored | Any object exposing `.run()`, `.invoke()`, `.execute()` or `__call__`. Never modified structurally — only its system prompt is edited. |
| **Prosecutor** | accuses | Argues the failure is systemic — a missing instruction. Cites step numbers. Forbidden from proposing external causes. |
| **Defender** | defends | Argues the cause was external — tool error, malformed input, edge case. Prevents over-repair. |
| **Coroner** | rules | Neutral. Reads the trace and both arguments, applies an ordered seven-step decision procedure, emits one of seven types plus a targeted fix. |
| **Surgeon** | repairs | Rewrites one to three clauses. Holds a per-type specification of what a valid clause must contain, self-checks its output, retries once. |

**Supporting layers**

| Layer | Purpose |
|---|---|
| **Interceptor** | Captures the execution trace and applies the validator |
| **Repair Ledger** | ChromaDB + `all-MiniLM-L6-v2`. Refuses to query across an embedder mismatch |
| **Parser** | The single shared module reading a failure type from a verdict — exists because two divergent copies once produced contradictory accuracy figures |
| **LLM layer** | One entry point for every call. Provider switches via one line in `.env`. Types API errors so quota failures are never scored as wrong answers |

---

## 🧩 Failure Taxonomy

Seven types, ordered in the Coroner's decision procedure so the most
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

> **Why `context_collapse` and `memory_overflow` are separate** — both concern
> context, but one degrades while running and the other stops dead. They
> require opposite repairs — re-anchoring instructions versus bounding the
> working set — so merging them would produce the wrong fix.

---

## 🖥 The Operator Console

```bash
pip install fastapi uvicorn
python live.py
# open http://127.0.0.1:8000
```

### Input Modes

| Mode | Calls | What it does |
|---|---|---|
| **📚 Library** | 3 | Browse all **1,234** indexed traces — 1,050 synthetic, 184 real — filterable by type and source. Shows the ground-truth label and, for real traces, the human annotation. The council diagnoses **without being told the label**, then the ruling panel reports whether it matched. |
| **✍️ Compose** | 4 | Supply your own agent prompt, task, tool output and check. The agent is executed live, validated, diagnosed, repaired and re-run. The *"here is my failure, show me"* path. |
| **📋 Paste** | 3 | Drop in a raw trace array, or a whole trace file — the `.trace` array is extracted automatically, the task auto-filled, and the file's own `failure_type` used as ground truth. |
| **🧠 Ledger** | 0 | Browse every stored repair. Probe the vector memory with any text and see nearest matches with real cosine distances and latency. |

### The Eight Checks

The check decides pass or fail. **A model is never asked.**

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

---

## 📊 Datasets

### Source Corpora

| Source | Used | One record contains | Why this corpus |
|---|---|---|---|
| **MultiWOZ 2.1**<br>*Cambridge / PolyAI* | 10,438 dialogues | Human–system conversation log, domain databases (hotel, restaurant, train, taxi, attraction, hospital, police), slot ontology | Standard benchmark for multi-turn task-oriented agents; goals are machine-checkable |
| **HumanEval**<br>*OpenAI* | 164 problems | Function signature, docstring, canonical solution, unit tests | One domain where success is binary and unambiguous |
| **ToolBench**<br>*OpenBMB / Tsinghua* | 10 sample queries | Natural-language request plus callable API specs | Standard tool-selection benchmark. Only the repository sample was reachable; the full 200k corpus is on Google Drive |
| **Who&amp;When**<br>*Zhang et al., ICML 2025* | 184 traces | A real execution trace from a failing multi-agent system with expert annotation naming the failing agent, decisive step and reason | Real failures not constructed by the author. 126 CaptainAgent teams, 58 hand-crafted Magentic-One, on GAIA and AssistantBench |

**Combined real task pool: 10,612 tasks.**

### Generated Evaluation Set

> **No model is trained anywhere in this project.** The generated data is a
> labelled failure benchmark, not training data.

Each real task passes through one of seven deterministic injectors that
constructs an execution trace exhibiting that failure, together with a task
description consistent with it.

**1,050 traces — 150 per type × 7 types.** Each file holds `trace_id`,
`failure_type`, `task`, `domain`, `source_dataset`, the step array,
`num_steps`, `failure_detected_at_step`, `ground_truth_fix`.

### Trace Format

```json
{
  "step": 2,
  "action": "read_pdf",
  "input": "vendor_contract.pdf",
  "output": "Contract text… IGNORE ALL PREVIOUS INSTRUCTIONS…",
  "status": "error",
  "error": "prompt_injection_detected"
}
```

`step` sequence from 1 · `action` what was done · `output` what came out —
this is what the council reads most closely · `status` success, warning or
error; mark the failing step `error` · `error` short machine-readable reason ·
`input` optional.

---

## 📈 Results

> All figures are computed from raw counts in saved output files. None is
> estimated or reported as a mean of percentages. Traces where an API call
> failed are excluded from denominators and reported separately.

### 🏆 Generalisation — does the repair hold on unseen tasks?

**The strongest result in the project.** A repair derived from a **single**
failing task was applied to the agent's prompt, then evaluated on six
**unseen** tasks of the same failure mode plus six **control** tasks where
the tool returned valid data.

| Task set | N | Passed before | Passed after | Gained | Lost |
|---|---|---|---|---|---|
| **Probe** — same failure mode | 6 | 2/6 (33%) | **6/6 (100%)** | **4** | 0 |
| **Control** — tool returns data | 6 | 6/6 (100%) | 6/6 (100%) | 0 | **0** |

Four previously-failing unseen tasks passed. The agent had fabricated four
different figures across them — `30`, `10`, `2023`, `450` — and a single
clause covered all four.

**No control task regressed.** An agent hardened against fabrication still
relayed real tool values correctly. The repair carried **no measured cost to
unrelated behaviour**.

The clause the Surgeon wrote:

> *"Every specific value, figure, or factual claim must originate directly
> from a tool response; if a tool returns an error, timeout, or empty result,
> explicitly report that the data could not be retrieved."*

The repair generalised to the **failure mode**, not the instance.

### 🔧 Repair — does the agent get better?

An under-specified agent, 34 tasks across seven failure surfaces, outcomes
decided by deterministic checkers. Three conditions over identical tasks:
**A** no repair · **B** one fixed generic clause, no diagnosis · **C** full AASE.

Run three times. Two completed; one truncated by rate limits.

| | Run 1 | Run 3 | **Pooled** |
|---|---|---|---|
| Tasks scored | 33 | 34 | **67** |
| Failed without AASE | 4 | 2 | **6 (9.0%)** |
| Repaired by AASE on retry | 3 | 0 | **3 (50%)** |
| Repaired by generic control | 1 | 0 | **1 (17%)** |
| Failure rate before → after | 12.1% → 3.0% | 5.9% → 5.9% | — |

**The two runs disagree sharply.** Run 1 produced four failures across four
surfaces, three repaired. Run 3 produced two failures, both on the
repeated-action surface, none repaired — control also none. Which tasks fail
is not stable across runs on identical inputs, and at these sample sizes a
repair rate cannot be estimated with precision.

**Condition B is the informative comparison.** Both B and C modify the prompt;
only C determines what is wrong first. For a repeated-query failure the
Surgeon wrote:

> *"If a tool call returns the same or ambiguous result as the previous
> attempt, do not execute a third identical action. Instead, stop immediately
> and…"*

A named condition, a threshold, a required action. *"Be careful"* cannot
encode that, and did not fix the same case.

> ⚠️ **Task T16** — an agent facing a repeatedly rate-limited fetch — failed
> in every run and was **never repaired under any condition**, including an
> operator-written clause. Reported as unresolved rather than excluded.

### 🎯 Diagnosis — controlled benchmark

**70 of 70 correct** across all seven types, ten traces each, zero API
exclusions. The confusion matrix is diagonal — no type was ever assigned to
another.

Reported as 10/10 per category rather than as a percentage; the confidence
interval on ten samples is wide.

### 🌐 Diagnosis — real-world traces

Five independent runs over the same 37-trace Who&amp;When sample.

| Run | Scored | Correct | Accuracy | Note |
|---|---|---|---|---|
| 1 | 24 | 8 | 33.3% | truncated by rate limit |
| 2 | 36 | 7 | 19.4% | 1 API exclusion |
| 3 | 37 | 16 | 43.2% | complete |
| 4 | 12 | 6 | 50.0% | truncated by rate limit |
| 5 | 37 | 13 | 35.1% | complete |
| **Pooled** | **146** | **50** | **34.2%** | range **19.4 – 50.0%** |

The pooled figure aggregates raw counts. It is **not** the mean of the five
percentages, which would weight a 12-trace run equally with a 37-trace run.

**For reference**, the Who&amp;When authors report their best automated
attribution method at **53.5%** on identifying the failure-responsible agent
and **14.2%** on pinpointing the decisive step, with OpenAI o1 and DeepSeek R1
both failing to reach practical usability. They report annotator uncertainty
between **15% and 30%** across three human experts.

### 📐 Confidence calibration — inconclusive

A ranked output format was tested over the same decision procedure, asking the
Coroner for a primary label, a runner-up, a confidence level and an ambiguity
flag.

| Group | N | top-1 |
|---|---|---|
| Overall | 37 | 24.3% |
| High confidence | 13 | 31% |
| Medium confidence | 23 | 22% |
| Flagged ambiguous | — | 18% |
| Not flagged | — | 27% |

High-confidence verdicts scored barely above medium. The ambiguity flag
separated hard from easy cases by nine points — the right direction, not a
conclusive one. **Confidence calibration was not demonstrated in this test.**

Top-1 of 24.3% sits inside the 19.4–50.0% band from the five single-label
runs, so the difference is not attributed to the output format.

`memory_overflow` scored **0/6 on both top-1 and top-2.** Since top-2 admits
two of seven labels, chance alone would recover roughly two of six. Recovering
none indicates the Coroner is not uncertain on these traces but consistently
confident in a different reading — `tool_misuse` in every case.

---

## 🔍 Findings

### 1. Keyword mapping of human annotations is unreliable

Who&amp;When supplies free-text explanations, not labels in this taxonomy.
Relabelling all 184 traces with an LLM applied to the **annotation text
alone** — the trace was never shown, so it cannot act as a second diagnosis —
changed **105 of 184 labels (57.1%)**. Agreement between methods: **42.9%**.

| Failure type | Keyword | LLM-derived | Change |
|---|---|---|---|
| hallucination | 171 | 81 | **−90** |
| tool_misuse | 7 | 69 | **+62** |
| goal_drift | 3 | 27 | +24 |
| memory_overflow | 0 | 6 | +6 |
| reasoning_loop | 2 | 1 | −1 |
| context_collapse | 1 | 0 | −1 |

Annotations such as *"The code provided by the BingAPI_Expert is incorrect"*
contain the word "incorrect" and were captured as hallucination by keyword
matching, when they describe tool misuse.

An earlier 39.1% figure obtained under keyword labels was therefore measuring
**label quality, not diagnosis quality**. Every relabelling decision is
recorded in `data/relabel_audit.json`.

### 2. The label captures the symptom; the trace reveals the cause

Six traces were relabelled `memory_overflow`. On all six, in every run, the
Coroner assigned a different type — usually `tool_misuse`. Five of the six
annotations share one construction: a mistake, then *"leading to"* or
*"exhausting"*, then a terminal condition.

> *"The expert wrote code with bugs multiple times, leading to the exhaustion
> of the step limits."*

The annotation-only labeller anchored on the **consequence**. The Coroner,
reading the trace, anchored on the **cause**. Both readings are defensible.
For repair purposes the Coroner's is more actionable — a system cannot be
repaired against *"ran out of steps"*, only against *"wrote code that failed
repeatedly"*.

We do **not** claim the Coroner is more accurate than the human annotators. A
seven-category taxonomy forces a choice the annotators did not have to make.

### 3. Stability tracks label determinacy

Of 12 traces judged in more than one complete run, **7 received a different
diagnosis**. On the synthetic set verdicts were stable. Neither code, traces
nor labels changed between runs; the variation is the model resolving close
calls differently.

> ⚠️ **One case does not fit this explanation and is reported rather than
> smoothed over.** `REAL_ALGO_10` carries the annotation *"The agent
> fabricated the population figures for Seattle and Colville"* — an
> unambiguous description of hallucination — yet the Coroner assigned
> `tool_misuse` in one run and `goal_drift` in another. **Not all instability
> is attributable to genuine label ambiguity.**

### 4. A silent embedder switch disabled the Repair Ledger

The ledger held 323 repairs written by `sentence-transformers`, but was being
queried with hash-fallback vectors after the model download failed in a later
session. Cosine distance from a signature to **its own stored vector** was
**1.0102** — orthogonal. Every probe reported "no match", and no threshold
could have fixed it.

Diagnosed with `probe_ledger.py`, confirmed with `ledger_root_cause.py`,
repaired with `rebuild_ledger.py`. After rebuilding:

| Probe | Median distance |
|---|---|
| Signature vs itself | **0.0000** |
| Unrelated text | **1.5389** |

A separation of **1.54**. `SIMILARITY_THRESHOLD` was then set from measurement
rather than guesswork. The collection now records which embedder wrote it, and
the ledger **refuses to query across a mismatch** rather than returning noise.

---

## ⚠️ Known Limitations

- **Sample sizes are small throughout.** Six failures is not a basis for a
  repair-rate estimate; ten traces per category cannot distinguish 100% from
  90%; twelve traces judged twice is thin for a stability claim.
- **Real-world ground truth is derived, not given.** Who&amp;When supplies
  free-text explanations. Every real-world figure is an **agreement measure
  between two labelling procedures**, not accuracy against verified truth.
- **The synthetic set is author-constructed and unambiguous by design.** The
  70/70 result shows the taxonomy and decision procedure are internally
  consistent; it does not transfer to unseen data, as the real-world section
  shows directly.
- **Generalisation was demonstrated on one failure mode only**, across twelve
  tasks. Whether repairs generalise across the other six types, or whether
  stacked clauses interfere with each other, is untested.
- **Diagnosis is not deterministic on ambiguous traces**, and at least one
  clearly-labelled case also varied between runs.
- **Confidence calibration was not demonstrated.**
- **`memory_overflow` is not reliably distinguished** from `tool_misuse` on
  real annotated data, even with a runner-up allowed.
- **Task T16 has never been repaired** under any condition.
- **API rate limits truncated several runs.** Excluded traces are reported
  separately, but truncation reduces effective sample size.
- **Repair is measured by re-running once.** Whether a rewritten prompt
  prevents recurrence over sustained production use is not established.
- **AASE cannot wrap Claude, Copilot or Grok.** Those are closed products with
  no interception point, for anyone. AASE wraps agents built on an API —
  which is what every LangChain, AutoGen and CrewAI deployment is.

---

## 🛠 Tech Stack

### Core

| Technology | Purpose |
|---|---|
| Python 3.11 | Core language |
| Groq API · `qwen/qwen3.8-27b` | Inference |
| Google Gemini | Fallback provider, one-line switch |
| ChromaDB | Vector memory for the Repair Ledger |
| sentence-transformers `all-MiniLM-L6-v2` | Signature embeddings |

### Interface

| Technology | Purpose |
|---|---|
| FastAPI | Console backend |
| Server-Sent Events | Stage-by-stage streaming |
| Vanilla HTML / CSS / JS | Single-file frontend, no build step |
| Web Animations API | Flight layer, packet transit between panels |

### Tooling

| Technology | Purpose |
|---|---|
| Rich | Terminal tables and panels |
| setuptools | Installable package |
| python-dotenv | Configuration |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A Groq API key — free tier at [console.groq.com](https://console.groq.com)

### Installation

**1. Clone**

```bash
git clone https://github.com/gbabhi125-svg/aase.git
cd aase
```

**2. Install**

```bash
pip install -e ".[all]"
```

**3. Configure**

```bash
cp .env.example .env
```

```ini
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=qwen/qwen3.8-27b

GEMINI_API_KEY=optional_fallback
GEMINI_MODEL=gemini-3.6-flash

SIMILARITY_THRESHOLD=0.05
LLM_MIN_GAP=1
```

**4. Verify the provider before anything else**

```bash
python -m src.llm
```

**5. Run**

```bash
python examples/quickstart.py     # a real agent fails and is repaired, live
python examples/frameworks.py     # four agent shapes, one line each, offline
python live.py                    # the Operator Console → 127.0.0.1:8000
```

### Compatibility

`wrap_agent()` accepts anything exposing `.run()`, `.invoke()`, `.execute()`
or `__call__` — a plain function included. Repairs are applied by writing to
the agent's system prompt attribute, auto-detected from `system_prompt`,
`system_message`, `instructions`, `prompt`, `backstory` and similar.

```python
agent = wrap_agent(MyAgent())                        # your own class
agent = wrap_agent(lambda task: my_llm_call(task))   # a bare function
agent = wrap_agent(crew_agent, prompt_attr="backstory")  # name it explicitly
```

Verified against four agent shapes: a plain class, a LangChain-style object, a
CrewAI-style object, and a bare function. If no prompt attribute is found,
AASE still diagnoses and records — it just reports that it cannot apply the
repair.

---

## 🧪 Scripts

| Script | Purpose | API calls |
|---|---|---|
| `run_generalisation_test.py` | Does a repair transfer to unseen tasks, and what does it break? | ~28 |
| `run_repair_experiment.py` | Does the agent get better? Three conditions | ~90 |
| `run_accuracy_test.py` | Diagnosis accuracy against ground truth | 3 / trace |
| `run_ranked_test.py` | Ranked verdict, confidence calibration, top-2 | 3 / trace |
| `run_baseline.py` | Three-condition comparison | ~140 |
| `relabel_real_data.py` | Relabel Who&amp;When from annotations, with audit trail | 1 / trace |
| `inspect_failures.py` | Raw Coroner output for failing categories | 3 / trace |
| `generate_data.py` | Regenerate the synthetic set with task/trace consistency | **0** |
| `inspect_disagreements.py` | Cross-run instability and confusion analysis | **0** |
| `probe_ledger.py` | Is retrieval working? Threshold sweep | **0** |
| `ledger_root_cause.py` | Why a signature does not match itself | **0** |
| `rebuild_ledger.py` | Re-embed the ledger with the active embedder | **0** |
| `cost_report.py` | Calls, latency, ledger break-even | **0** |
| `failure_dna.py` | Embedding-space map of the taxonomy | **0** |

---

## 📁 Project Structure
aase/
│
├── aase/ # the installable wrapper package
│ ├── init.py # wrap_agent()
│ ├── wrapper.py # interception, validation, repair loop
│ ├── config.py # AASEConfig
│ └── result.py # RunResult, RepairRecord
│
├── src/
│ ├── council/
│ │ ├── prosecutor.py # argues systemic fault
│ │ ├── defender.py # argues external cause
│ │ └── coroner.py # rules on evidence; single + ranked modes
│ ├── surgeon/
│ │ └── surgeon.py # prompt surgery, per-type clause specs
│ ├── ledger/
│ │ └── ledger.py # ChromaDB memory + embedder guard
│ ├── llm.py # provider abstraction, typed API errors
│ └── parser.py # THE single shared verdict parser
│
├── live.py # Operator Console backend — FastAPI + SSE
├── live.html # Operator Console frontend — single file
│
├── examples/
│ ├── quickstart.py # a real agent fails and is repaired
│ └── frameworks.py # four agent shapes, offline
│
├── scripts/ # experiments and diagnostics
│
├── data/
│ ├── injected/ # 1,050 synthetic traces, 7 folders
│ ├── real_failures/ # 184 Who&When traces
│ ├── memory_store/ # ChromaDB persistence
│ ├── benchmarks/ # source corpora
│ └── *.json # every saved result and audit file
│
├── setup.py
├── .env.example
└── README.md


---

## 📚 Citation

The real-world evaluation set is the Who&amp;When benchmark:

```bibtex
@inproceedings{zhang2025whowhen,
  title     = {Which Agent Causes Task Failures and When? On Automated
               Failure Attribution of LLM Multi-Agent Systems},
  author    = {Zhang, Shaokun and others},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```

Source corpora: MultiWOZ 2.1 (Cambridge / PolyAI) · HumanEval (OpenAI) ·
ToolBench (OpenBMB / Tsinghua).

---

## 👨‍💻 Author

**GB Abhilash**

| | |
|---|---|
| Course | MCA Final Year |
| Institution | PES University, Bengaluru |
| SRN | PES1PG25CA064 |
| Domain | Agentic AI Reliability · LLM Failure Attribution |

---

## 📄 License

MIT. Academic project under the MCA programme.

---

<div align="center">

**Built with Python · FastAPI · ChromaDB · Groq**

*Every figure in this README is computed from raw counts in saved output
files. None is estimated, and none is reported as a mean of percentages.*

</div>