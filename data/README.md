# AASE Real-World Failure Dataset

## Source
Who&When benchmark — ICML 2025 Spotlight
Paper: "Which Agent Causes Task Failures and When? On Automated Failure 
        Attribution of LLM Multi-Agent Systems"
Authors: Zhang et al., 2025
Repo: https://github.com/ag2ai/Agents_Failure_Attribution

## What This Is
184 REAL execution traces from actual failing LLM multi-agent systems.
NOT synthetic. NOT generated. Real agents, real tasks, real failures.

Each trace includes:
- The actual task the agent was given
- Full step-by-step execution history (avg 22 steps, max 130)
- Human-annotated ground truth: which agent made the mistake, at which step, and why

## Systems These Came From
- CaptainAgent-generated multi-agent teams (on GAIA + AssistantBench tasks)
- Hand-crafted Magentic-One systems (Microsoft)

## Failure Type Distribution (mapped to AASE taxonomy)
  hallucination: 171
  tool_misuse: 7
  goal_drift: 3
  reasoning_loop: 2
  context_collapse: 1
  TOTAL: 184

## Key Finding
Real-world agent failures cluster heavily in hallucination-type errors
(171/184 = 93%).
This validates AASE's design priority on hallucination detection.

## Agents That Failed Most
WebSurfer (33), Verification_Expert (18), Orchestrator (18)

## Usage in AASE
These traces are the REAL evaluation set. AASE's Diagnosis Council is tested
on these to measure diagnosis accuracy against human-annotated ground truth.
