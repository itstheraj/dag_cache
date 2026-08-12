"""dagcache: VCR cassettes for agent trajectories.

Record agent runs as DAGs of tool/LLM calls, replay the canonical path on
repeat tasks, and only pay for the LLM when the world diverges.

    import dagcache

    @dagcache.tool(pure=True)
    def search_kb(query: str) -> str: ...

    @dagcache.tool(pure=False)
    def send_reply(ticket_id: str, body: str) -> str: ...

    @dagcache.llm            # user-facing synthesis: re-run on replay
    def draft(prompt: str) -> str: ...

    @dagcache.agent
    def resolve_ticket(ticket: dict) -> str:
        ...  # ordinary agentic code

First call runs live and records; subsequent calls with the same input
*shape* replay the cached tool chain with fresh bindings. Divergence falls
back to the live agent automatically.
"""

from .agent import agent
from .graph import DAG, Binding, Node
from .policy import configure, get_config, reset_config
from .replay import Divergence, Executor
from .tracer import llm, tool

__version__ = "0.1.0"

__all__ = [
    "agent",
    "tool",
    "llm",
    "configure",
    "get_config",
    "reset_config",
    "DAG",
    "Node",
    "Binding",
    "Divergence",
    "Executor",
    "__version__",
]
