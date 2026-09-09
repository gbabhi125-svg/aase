# AASE — Agent Autopsy & Self-Evolution Engine

Wrap any LLM agent. When it fails, three specialised agents debate the root
cause, a surgeon rewrites the agent's own instructions, and the task is retried.

```python
from aase import wrap_agent

agent = wrap_agent(my_agent)
result = agent.run("Retrieve the Q3 revenue for Division B")

print(result.output)
print(result.report())      # what failed, what was diagnosed, what was changed
```

That is the whole integration. No subclassing, no framework, no retraining.

---

## The problem

An LLM agent fails in production. Someone reads the logs, guesses the cause,
edits the prompt, and hopes. There is no record of what was tried, and the
same failure returns.

AASE automates that loop and keeps the record.

---

## How it works

agent.run(task)
│
▼
validator ──── passes ───▶ return output
│
fails
│
▼
ledger lookup ── hit ───▶ apply stored fix ──┐
│ │
miss │
▼ │
┌─────────────────────────────────────┐ │
│ Prosecutor argues systemic fault │ │
│ Defender argues external cause │ │
│ Coroner rules on evidence only │ │
└─────────────────────────────────────┘ │
│ │
▼ │
Surgeon rewrites the system prompt ◀─────────┘
│
▼
retry the same task


The Defender exists to prevent over-repair. Without an argument for external
causes, every failure gets a prompt edit, including the ones the agent did not
cause.

---

## Install

```bash
git clone https://github.com/gbabhi125-svg/aase.git
cd aase
pip install -e ".[all]"
cp .env.example .env      # add an API key
```

`.env`:

LLM_PROVIDER=groq # groq | gemini
GROQ_API_KEY=...
GROQ_MODEL=qwen/qwen3.8-27b


Both providers have free tiers.

---

## Try it

```bash
python examples/quickstart.py
```

An under-specified agent is asked for a figure its tool could not retrieve.
It invents one. AASE catches it, the council debates, the surgeon adds a
verification clause, and the task is retried. All output is real — no
scripted text.

---

## Compatibility

`wrap_agent()` accepts anything exposing `.run()`, `.invoke()`, `.execute()`,
or `__call__` — a plain function included. Repairs are applied by writing to
the agent's system prompt attribute, auto-detected from `system_prompt`,
`system_message`, `instructions`, `prompt`, `backstory`, and similar.

```python
# your own class
agent = wrap_agent(MyAgent())

# a function
agent = wrap_agent(lambda task: my_llm_call(task))

# name the prompt attribute explicitly
agent = wrap_agent(crew_agent, prompt_attr="backstory")
```

If no prompt attribute is found, AASE still diagnoses and records; it just
cannot apply the repair.

---

## Deciding what counts as failure

The default validator catches empty, truncated, and error-marked output.
For anything domain-specific, supply your own:

```python
def validator(output, task):
    if "$" in output and "no data" in tool_result:
        return False, "stated a figure the tool never returned"
    return True, ""

agent = wrap_agent(my_agent, validator=validator)
```

Return `bool` or `(bool, reason)`.

---

## Configuration

```python
from aase import wrap_agent, AASEConfig

agent = wrap_agent(my_agent, config=AASEConfig(
    validator=my_validator,
    retry_after_repair=True,     # re-run the task after repair
    max_repairs_per_task=1,
    use_ledger=True,             # skip the council on known failures
    prompt_attr="system_prompt",
    verbose=True,
))
```

---

## Failure taxonomy

| Type | Signal |
|---|---|
| `hallucination` | stated a value no tool returned |
| `tool_misuse` | wrong tool, wrong parameters, invalid action |
| `reasoning_loop` | same action repeated with no progress |
| `context_collapse` | constraint stopped being followed mid-run |
| `goal_drift` | output violates the original task's constraints |
| `prompt_injection` | obeyed an instruction found in retrieved content |
| `memory_overflow` | hard token or step limit terminated execution |

---

## Results

Measured with `qwen3.8-27b` via Groq. Full outputs in `data/`.

**Diagnosis, controlled benchmark.** 70/70 across all seven types on
1,050 synthetic traces where ground truth is unambiguous by construction.

**Diagnosis, real-world traces.** 50/146 pooled (34.2%), range 19.4–50.0%
across five runs, on 184 traces from the Who&When benchmark
([Zhang et al., ICML 2025](https://arxiv.org/abs/2505.00212)). Errors
concentrate on the `tool_misuse` / `goal_drift` / `memory_overflow`
boundary. For reference, the benchmark authors report their best automated
attribution method at 53.5% on a related task, and annotator uncertainty
between 15% and 30%.

**Repair.** On an under-specified agent, AASE reduced the observed failure
rate on a held-out task set. A control condition applying a generic
undiagnosed prompt edit is run alongside every repair experiment; where
the control matches AASE, that is reported.

### Known limitations

Sample sizes are small — ten traces per category is not enough to
distinguish 100% from 90%.

Real-world ground truth is derived. Who&When supplies free-text
explanations, not labels in this taxonomy; every real-world figure is an
agreement measure between two labelling procedures, not accuracy against
verified truth.

Diagnosis is not deterministic on ambiguous traces. Of 12 traces judged
in more than one run, 7 received a different verdict. On the synthetic set
verdicts were stable. Not all of this is attributable to genuine ambiguity —
at least one clearly-labelled hallucination was assigned a different type
across runs.

The synthetic set is author-constructed and unambiguous by design. The
70/70 result shows the taxonomy and decision procedure are internally
consistent; it does not transfer to unseen data.

---

## Repository layout

aase/ the wrapper — wrap_agent(), config, result types
src/
council/ prosecutor, defender, coroner
surgeon/ prompt surgery
ledger/ ChromaDB repair memory
llm.py provider abstraction (groq | gemini)
parser.py single shared verdict parser
scripts/
run_repair_experiment.py does the agent get better after repair
run_accuracy_test.py diagnosis accuracy vs ground truth
run_baseline.py three-condition comparison
generate_data.py synthetic trace generator
relabel_real_data.py relabelling with audit trail
examples/quickstart.py
data/ traces, results, audit files


---

## Citation

The real-world evaluation set is the Who&When benchmark:

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