"""
AASE — Agent Autopsy & Self-Evolution Engine.

Wrap any agent. When it fails, three specialised agents debate the root cause,
a surgeon rewrites the agent's instructions, and the task is retried.

    from aase import wrap_agent

    agent = wrap_agent(my_agent)
    result = agent.run("Retrieve the Q3 revenue for Division B")

Works with anything exposing .run(task), .invoke(task), .__call__(task),
or a plain function. LangChain, AutoGen, CrewAI, or your own class.
"""

from .wrapper import wrap_agent, AASEAgent
from .config import AASEConfig
from .result import RunResult, RepairRecord

__version__ = "0.1.0"
__all__ = ["wrap_agent", "AASEAgent", "AASEConfig", "RunResult", "RepairRecord"]