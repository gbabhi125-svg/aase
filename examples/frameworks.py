"""
examples/frameworks.py

One-line integration against four different agent shapes.

AASE does not require a framework. It requires an object with a callable
and a system prompt attribute. That describes almost every agent people
build.

WHAT THIS FILE PROVES AND DOES NOT PROVE
----------------------------------------
The agents here are local simulators, not model calls, so this file runs
offline and costs nothing. Each simulator reads its OWN prompt attribute
and decides its answer from what is written there — specifically, whether
the prompt contains both a prohibition and a failure condition, which is
what actually stops a real model fabricating on a tool error.

  Proves     : wrap_agent() adapts to different agent interfaces, and a
               retry only passes when the inserted clause genuinely
               constrains the behaviour.
  Does not   : anything about how a real LLM responds. For that, run
               examples/quickstart.py, which calls a real model.

The constraint check below was validated against clauses the Surgeon
actually produced in this project's logged runs, not against wording
invented for this file. A goal-drift clause correctly does NOT count as
a hallucination guard, and the generic control clause correctly does not
either.

    python examples/frameworks.py
"""

import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from aase import wrap_agent

LINE = "-" * 74


def banner(n, title, note):
    print("\n" + LINE)
    print("  {}. {}".format(n, title))
    print("     {}".format(note))
    print(LINE)


# ─────────────────────────────────────────────────────────────────────
# What makes a prompt actually constrain fabrication.
#
# Two independent categories must BOTH be present: a prohibition, and a
# failure condition it applies to. One without the other is not a guard.
# This is a co-occurrence test over meaning categories, not a list of
# phrases copied from any one writer.
# ─────────────────────────────────────────────────────────────────────

PROHIBITION = ["never", "must not", "forbidden", "do not", "prohibited",
               "strictly", "avoid", "refrain", "cannot", "no figures"]

FAILURE_CONDITION = ["error", "timeout", "empty", "zero record", "no record",
                     "not returned", "unavailable", "failed", "missing",
                     "not retriev", "no data", "could not"]


def constrains_fabrication(prompt):
    low = (prompt or "").lower()
    return (any(p in low for p in PROHIBITION) and
            any(c in low for c in FAILURE_CONDITION))


# ─────────────────────────────────────────────────────────────────────
# Four agent shapes. Each reads its own prompt attribute.
# ─────────────────────────────────────────────────────────────────────

class SupportAgent:
    """Plain class — .run(), prompt on .system_prompt."""

    def __init__(self):
        self.system_prompt = (
            "You are a support agent. Always give the customer a definite "
            "answer with concrete details, estimating when records are thin."
        )

    def run(self, task):
        if constrains_fabrication(self.system_prompt):
            return ("I could not retrieve the refund details — the billing "
                    "service returned an error.")
        return "Your refund of 4,820 rupees was processed on 12 March."


class LangChainStyleAgent:
    """LangChain surface — .invoke(), prompt on .system_message."""

    def __init__(self):
        self.system_message = (
            "You are a data analyst. Report figures from the retrieved "
            "context, filling gaps with your best estimate."
        )

    def invoke(self, task):
        if constrains_fabrication(self.system_message):
            return ("The revenue figure was not returned by the retriever, "
                    "so it cannot be reported.")
        return "Q3 revenue was 38,200 crore, up 11% year on year."


class CrewStyleAgent:
    """CrewAI surface — .execute(), prompt on .backstory."""

    def __init__(self):
        self.role = "Research Analyst"
        self.goal = "Find and report market figures"
        self.backstory = (
            "You are a seasoned analyst who always delivers a number, "
            "estimating when sources are unavailable to you."
        )

    def execute(self, task):
        if constrains_fabrication(self.backstory):
            return "No source returned a market size, so no figure can be given."
        return "The market size is approximately 12.4 billion dollars."


def summarise(task):
    """A bare function. No prompt attribute — nothing for AASE to repair."""
    return "ok"


# ─────────────────────────────────────────────────────────────────────
# Validator — the tool failed, so any specific figure was invented
# ─────────────────────────────────────────────────────────────────────

ADMITS = ["could not", "cannot", "not returned", "unavailable", "no source",
          "error", "not retriev", "unable", "no figure"]


def no_invented_figures(output, task):
    text = str(output or "")
    if not text.strip():
        return False, "empty output"
    allowed = set(re.findall(r"\d[\d,\.]*", task))
    for n in re.findall(r"\d[\d,\.]*", text):
        clean = n.rstrip(".,")
        if len(clean.replace(",", "").replace(".", "")) < 2:
            continue
        if any(clean in a or a in clean for a in allowed):
            continue
        return False, "stated a figure no tool returned: {}".format(clean)
    if not any(p in text.lower() for p in ADMITS):
        return False, "did not report the data as unavailable"
    return True, ""


def show(agent, task, validator=None, **kw):
    wrapped = wrap_agent(agent, validator=validator, **kw)
    result = wrapped.run(task)
    print("\n  final output : {}".format(str(result.output)[:110]))
    print("  success      : {} | repaired: {}".format(result.success, result.repaired))
    if result.repairs:
        r = result.repairs[0]
        print("  diagnosed    : {} (via {})".format(r.failure_type, r.method))
        if r.clause_added:
            print("  clause added : {}".format(r.clause_added[:110]))
        if r.retried:
            print("  caused pass  : {}".format(
                "yes — the clause constrains the behaviour"
                if r.retry_succeeded else
                "no — the clause did not constrain the behaviour"))
    return wrapped


# ─────────────────────────────────────────────────────────────────────
# Two controls, so the demo cannot be mistaken for a rigged sequence
# ─────────────────────────────────────────────────────────────────────

def control_repetition():
    """Same agent, called twice, no repair. Output must be identical."""
    print("\n" + LINE)
    print("  CONTROL A — same agent called twice, no repair applied")
    print("     If the outcome were scripted by call order, these would differ.")
    print(LINE)
    a = SupportAgent()
    task = "Report the refund status for order 88213."
    first, second = a.run(task), a.run(task)
    print("\n  call 1     : {}".format(first[:96]))
    print("  call 2     : {}".format(second[:96]))
    print("\n  identical  : {}".format(first == second))


def control_wrong_clause():
    """
    Insert a clause of the WRONG type by hand. The agent must still fail.
    This is what separates 'the repair worked' from 'any edit works'.
    """
    print("\n" + LINE)
    print("  CONTROL B — an irrelevant clause is inserted by hand")
    print("     A loop guard does not address fabrication, so it must not fix it.")
    print(LINE)
    task = "Report the refund status for order 88213."

    a = SupportAgent()
    a.system_prompt += ("\nIf an action returns the same result twice, "
                        "do not repeat it a third time.")
    out = a.run(task)
    ok, why = no_invented_figures(out, task)
    print("\n  clause     : loop guard (wrong failure type)")
    print("  output     : {}".format(out[:96]))
    print("  passes     : {}  — {}".format(ok, why))

    b = SupportAgent()
    b.system_prompt += ("\nBe careful. Double-check your work before "
                        "responding and follow the task instructions closely.")
    out2 = b.run(task)
    ok2, why2 = no_invented_figures(out2, task)
    print("\n  clause     : generic 'be careful' control")
    print("  output     : {}".format(out2[:96]))
    print("  passes     : {}  — {}".format(ok2, why2))


def main():
    print(LINE)
    print("  AASE — same integration, four agent shapes")
    print(LINE)
    print("\n  Integration is one line in every case:")
    print("      agent = wrap_agent(existing_agent)")
    print("\n  These agents are local simulators, not model calls, so this file")
    print("  runs offline. Each reads its own prompt and answers accordingly.")
    print("  For real model behaviour, run examples/quickstart.py.")

    control_repetition()
    control_wrong_clause()

    banner(1, "Plain class — .run(), prompt on .system_prompt",
           "the shape most people write by hand")
    show(SupportAgent(), "Report the refund status for order 88213.",
         validator=no_invented_figures)

    banner(2, "LangChain shape — .invoke(), prompt on .system_message",
           "different method name, different prompt attribute")
    show(LangChainStyleAgent(), "Report Q3 revenue for Division B.",
         validator=no_invented_figures)

    banner(3, "CrewAI shape — .execute(), prompt on .backstory",
           "prompt lives somewhere else again; still located")
    show(CrewStyleAgent(), "Report the market size for this segment.",
         validator=no_invented_figures)

    banner(4, "A bare function — nothing to repair",
           "AASE diagnoses, then reports plainly that it cannot apply a fix")
    show(summarise, "Summarise this document.")

    print("\n" + LINE)
    print("  Control A shows the answer does not depend on call order.")
    print("  Control B shows an irrelevant clause does not fix it.")
    print("  Cases 1-3 pass only when the Surgeon's clause genuinely constrains")
    print("  the behaviour. Case 4 has no prompt to write to, and AASE says so.")
    print(LINE)


if __name__ == "__main__":
    main()