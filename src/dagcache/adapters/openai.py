"""OpenAI SDK integration.

    from openai import OpenAI
    from dagcache.adapters.openai import wrap_client

    client = wrap_client(OpenAI())

    @dagcache.agent
    def my_agent(task: dict):
        resp = client.chat.completions.create(model="gpt-5", messages=[...], tools=[...])
        ...

Recording: every ``chat.completions.create`` call becomes a node. Calls
whose response contains tool_calls are planning nodes (never re-executed --
the DAG edges encode their decisions); plain-text responses are output
nodes (re-executed at verified replay with prompts patched to fresh data).

No OpenAI import happens here -- the wrapper is duck-typed, so it also works
with any client exposing ``.chat.completions.create`` (Azure, OpenRouter,
LiteLLM proxies, fakes in tests).
"""

from __future__ import annotations

from typing import Any

from .. import graph
from ..bindings import _jsonable
from ..tracer import LLM_REGISTRY, current_recorder

PLANNING = "planning"
OUTPUT = "output"
AUTO = "auto"


def _content_of(resp: Any) -> Any:
    return resp.choices[0].message.content


def _mirror(resp: Any) -> Any:
    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        return {
            "content": getattr(msg, "content", None),
            "tool_calls": [
                {
                    "id": getattr(tc, "id", None),
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in tool_calls
            ],
        }
    return getattr(msg, "content", None)


def _has_tool_calls(resp: Any) -> bool:
    try:
        return bool(resp.choices[0].message.tool_calls)
    except (AttributeError, IndexError, TypeError):
        return False


class _CompletionsProxy:
    def __init__(self, real: Any, role: str):
        self._real = real
        self._role = role

    def create(self, **kwargs):
        resp = self._real.create(**kwargs)
        rec = current_recorder()
        if rec is not None:
            model = kwargs.get("model", "unknown")
            name = f"openai:{model}"
            if self._role == PLANNING or (self._role == AUTO and _has_tool_calls(resp)):
                kind = graph.LLM_PLAN
            else:
                kind = graph.LLM_OUTPUT
                # Latest wrapped client wins: re-wrapping a new client for
                # the same model must not leave a stale closure behind.
                LLM_REGISTRY[name] = self._make_rerun()
            rec.add_node(kind, name, graph.LLM, {"kwargs": _jsonable(kwargs)}, _mirror(resp))
        return resp

    def _make_rerun(self):
        real = self._real

        def rerun(kwargs):
            return _content_of(real.create(**kwargs))

        return rerun

    def __getattr__(self, item):
        return getattr(self._real, item)


class _ChatProxy:
    def __init__(self, real_chat: Any, role: str):
        self._real = real_chat
        self._role = role

    @property
    def completions(self):
        return _CompletionsProxy(self._real.completions, self._role)

    def __getattr__(self, item):
        return getattr(self._real, item)


class WrappedClient:
    def __init__(self, real_client: Any, role: str):
        self._real = real_client
        self._role = role

    @property
    def chat(self):
        return _ChatProxy(self._real.chat, self._role)

    def __getattr__(self, item):
        return getattr(self._real, item)


def wrap_client(client: Any, *, role: str = AUTO) -> WrappedClient:
    """Wrap an OpenAI-style client. ``role``: "auto" | "planning" | "output".

    Use ``role="output"`` for a client dedicated to final-answer synthesis
    (always re-run on replay) or ``role="planning"`` for a client dedicated
    to tool-selection (never re-run). "auto" classifies per response.
    """
    if role not in (PLANNING, OUTPUT, AUTO):
        raise ValueError("role must be 'auto', 'planning' or 'output'")
    return WrappedClient(client, role)
