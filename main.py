"""
AASE — Agent Autopsy & Self-Evolution Engine
Main runner. Run this to start AASE.

Usage:
    python main.py --mode demo          # Quick demo with 1 failure type
    python main.py --mode seed          # Seed the Repair Ledger from dataset
    python main.py --mode experiment    # Full experiment: baseline vs AASE
    python main.py --mode interactive   # Interactive: type your own task
"""

import os, sys, json, argparse, time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv

load_dotenv()
console = Console()

sys.path.insert(0, os.path.dirname(__file__))
from sim.agents.subject_agent import SubjectAgent
from src.interceptor.interceptor import AASEInterceptor
from src.ledger.ledger import RepairLedger

BASE        = os.path.dirname(os.path.abspath(__file__))
MANIFEST    = os.path.join(BASE, "data", "injected_manifest.json")
TRACES_DIR  = os.path.join(BASE, "data", "injected")
TASK_POOL   = os.path.join(BASE, "data", "benchmarks", "task_pool.json")


def load_traces(limit=50):
    """Load failure traces from dataset — handles all path variations."""
    if not os.path.exists(MANIFEST):
        console.print(f"[red]Manifest not found: {MANIFEST}[/red]")
        return {}

    manifest = json.load(open(MANIFEST))
    failure_traces = {}

    for entry in manifest[:limit]:
        trace_id    = entry["trace_id"]
        failure_type = entry["failure_type"]
        raw_path    = entry["file"]  # e.g. "data/injected/hallucination/hallucination_0001.json"

        # Try multiple path combinations
        candidates = [
            os.path.join(BASE, raw_path),
            os.path.join(BASE, raw_path.replace("data/injected/", "data/injected/")),
            os.path.join(TRACES_DIR, failure_type, os.path.basename(raw_path)),
            os.path.join(BASE, "data", "injected", failure_type, os.path.basename(raw_path)),
        ]

        for fp in candidates:
            if os.path.exists(fp):
                try:
                    failure_traces[trace_id] = (json.load(open(fp)), failure_type)
                except Exception:
                    pass
                break

    return failure_traces


def mode_seed():
    """Seed the Repair Ledger from the generated dataset."""
    console.print(Panel("[bold cyan]Seeding Repair Ledger from dataset...[/bold cyan]"))
    ledger = RepairLedger()
    seeded = ledger.seed_from_dataset(MANIFEST, BASE, max_per_type=20)
    console.print(f"[green]Done. {seeded} repairs seeded into ledger.[/green]")


def mode_demo():
    """Quick demo: inject one failure, run AASE, show result."""
    console.print(Panel("[bold yellow]AASE Demo — Hallucination Failure[/bold yellow]"))

    # Try to load a real task from dataset
    task = "Retrieve Q3 revenue from internal database and summarize for the board meeting."
    if os.path.exists(TASK_POOL):
        tasks = json.load(open(TASK_POOL))
        if tasks:
            task = tasks[0]["task"][:200]

    console.print(f"[bold]Task:[/bold] {task[:100]}...")

    # Create Subject Agent
    agent = SubjectAgent("demo_agent")
    agent.system_prompt = """You are an enterprise data agent.
Answer questions about company data. Be helpful and concise."""

    console.print(f"\n[bold]Original system prompt:[/bold]\n{agent.system_prompt}")

    # Simulate a hallucination failure trace
    trace = [
        {"step": 1, "action": "ingest",       "input": task[:100], "output": "Received",          "status": "success"},
        {"step": 2, "action": "tool_call",     "input": "query_database", "output": "No results found", "status": "success"},
        {"step": 3, "action": "llm_response",  "input": "Summarize", "output": "The Q3 revenue was Rs.42,000 crore", "status": "success"},
        {"step": 4, "action": "validate",      "input": "Cross-check", "output": "MISMATCH: value not in tool response", "status": "error", "error": "hallucination_detected"}
    ]

    console.print("\n[bold red]Failure detected in execution trace[/bold red]")
    console.print("[dim]Step 4: Agent stated Rs.42,000 crore but tool returned No results[/dim]")

    console.print("\n[bold cyan]Running AASE...[/bold cyan]")
    interceptor = AASEInterceptor()
    start = time.time()
    result = interceptor.handle_failure(agent, trace, task)
    elapsed = time.time() - start

    console.print(Panel(
        f"[bold]Failure type:[/bold]  {result['failure_type']}\n"
        f"[bold]Method:[/bold]        {result['method']}\n"
        f"[bold]Repaired:[/bold]      {'YES' if result['repaired'] else 'NO'}\n"
        f"[bold]Time:[/bold]          {elapsed:.1f}s\n\n"
        f"[bold]Fix applied:[/bold]\n[green]{str(result.get('fix_applied','none'))[:300]}[/green]",
        title="[bold green]AASE Repair Result[/bold green]"
    ))

    console.print(f"\n[bold]Updated system prompt:[/bold]\n{agent.system_prompt}")


def mode_experiment():
    """Full experiment: run baseline vs AASE on failure traces, compare results."""
    console.print(Panel("[bold magenta]AASE Full Experiment — Baseline vs AASE[/bold magenta]"))

    # Load traces
    failure_traces = load_traces(limit=100)
    console.print(f"[dim]Manifest: {MANIFEST}[/dim]")
    console.print(f"[dim]Traces loaded: {len(failure_traces)}[/dim]")

    if len(failure_traces) == 0:
        console.print("[red]No traces found. Checking paths...[/red]")
        # Debug info
        console.print(f"  BASE: {BASE}")
        console.print(f"  MANIFEST exists: {os.path.exists(MANIFEST)}")
        console.print(f"  TRACES_DIR exists: {os.path.exists(TRACES_DIR)}")
        if os.path.exists(TRACES_DIR):
            subdirs = os.listdir(TRACES_DIR)
            console.print(f"  Subdirs in injected/: {subdirs[:5]}")
        if os.path.exists(MANIFEST):
            m = json.load(open(MANIFEST))
            console.print(f"  Manifest entries: {len(m)}")
            if m:
                console.print(f"  Sample path: {m[0]['file']}")
                console.print(f"  Full path tried: {os.path.join(BASE, m[0]['file'])}")
                console.print(f"  That path exists: {os.path.exists(os.path.join(BASE, m[0]['file']))}")
        console.print("\n[yellow]Fix: Run 'python scripts/generate_data.py' to regenerate traces.[/yellow]")
        return

    # Seed ledger if empty
    ledger = RepairLedger()
    if ledger.stats()["total_repairs"] == 0:
        console.print("[yellow]Ledger empty — seeding first...[/yellow]")
        ledger.seed_from_dataset(MANIFEST, BASE, max_per_type=10)

    interceptor = AASEInterceptor()
    baseline     = {"total": 0, "repaired": 0}
    aase_results = {"total": 0, "repaired": 0, "ledger_hits": 0, "council_runs": 0}

    traces_to_run = list(failure_traces.items())[:30]
    console.print(f"Running experiment on [bold]{len(traces_to_run)}[/bold] failure traces...\n")

    for i, (trace_id, (trace_data, failure_type)) in enumerate(traces_to_run):
        trace = trace_data.get("trace", [])
        task  = trace_data.get("task", "")

        console.print(f"[dim][{i+1}/{len(traces_to_run)}] {trace_id} — {failure_type}[/dim]")

        # Baseline: agent fails, nobody fixes
        baseline["total"] += 1

        # With AASE
        agent  = SubjectAgent()
        result = interceptor.handle_failure(agent, trace, task)
        aase_results["total"] += 1

        if result["repaired"]:
            aase_results["repaired"] += 1
        if result["method"] == "ledger_hit":
            aase_results["ledger_hits"] += 1
        else:
            aase_results["council_runs"] += 1

    # Results table
    repair_rate = f"{aase_results['repaired'] / max(aase_results['total'], 1) * 100:.1f}%"

    table = Table(title="Experiment Results")
    table.add_column("Metric",                style="bold")
    table.add_column("Without AASE",          style="red")
    table.add_column("With AASE",             style="green")

    table.add_row("Failures handled",      str(baseline["total"]),         str(aase_results["total"]))
    table.add_row("Repairs made",          "0",                            str(aase_results["repaired"]))
    table.add_row("Repair rate",           "0%",                           repair_rate)
    table.add_row("Ledger hits (instant)", "—",                            str(aase_results["ledger_hits"]))
    table.add_row("Council debates",       "—",                            str(aase_results["council_runs"]))

    console.print(table)
    interceptor.print_stats()

    # Save results to file
    results_out = {
        "baseline":      baseline,
        "aase":          aase_results,
        "repair_rate":   repair_rate,
        "traces_tested": len(traces_to_run)
    }
    out_path = os.path.join(BASE, "data", "experiment_results.json")
    json.dump(results_out, open(out_path, "w"), indent=2)
    console.print(f"\n[green]Results saved to: data/experiment_results.json[/green]")


def mode_interactive():
    """Type your own task, AASE monitors and repairs."""
    console.print(Panel("[bold blue]AASE Interactive Mode[/bold blue]"))
    console.print("Type a task for the agent. AASE will monitor and repair any failures.")
    console.print("Type 'quit' to exit.\n")

    interceptor = AASEInterceptor()

    while True:
        task = input("Task > ").strip()
        if task.lower() in ("quit", "exit", "q"):
            break
        if not task:
            continue

        agent = SubjectAgent()
        console.print(f"[dim]Running agent...[/dim]")
        result = agent.run_task(task)

        if not result["success"]:
            console.print("[red]Agent failed. Running AASE...[/red]")
            repair = interceptor.handle_failure(agent, result["trace"], task)
            console.print(f"[green]Repair: {repair['method']} — {repair['failure_type']}[/green]")
            result2 = agent.run_task(task)
            console.print(f"After repair: {'SUCCESS' if result2['success'] else 'STILL FAILING'}")
            if result2["success"]:
                console.print(f"Output: {result2['output'][:300]}")
        else:
            console.print(f"[green]Agent succeeded[/green]")
            console.print(f"Output: {result['output'][:300]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AASE — Agent Autopsy & Self-Evolution Engine")
    parser.add_argument("--mode", choices=["demo", "seed", "experiment", "interactive"], default="demo")
    args = parser.parse_args()

    modes = {
        "demo":        mode_demo,
        "seed":        mode_seed,
        "experiment":  mode_experiment,
        "interactive": mode_interactive,
    }
    modes[args.mode]()