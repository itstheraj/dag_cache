"""LangChain integration.

    from dagcache.adapters.langchain import wrap_tools, wrap_llm

    tools = wrap_tools([search_tool, order_tool])   # BaseTool-ish objects
    llm = wrap_llm(ChatOpenAI(...))                  # chat model-ish object

    @dagcache.agent
    def my_agent(task: dict):
        chain = build_my_chain(llm=llm, tools=tools)
        return chain.invoke(...)

Duck-typed: works with any tool object exposing ``name`` + ``func``/``_run``
and any LLM exposing ``invoke``. The real ``langchain`` package is never
imported here.
"""

from __future__ import annotations

import inspect
from typing import Any

from .. import graph
from ..bindings import _jsonable
from ..tracer import LLM_REGISTRY, current_recorder, tool as dagcache_tool


def _wrap_single_tool(t: Any, pure: bool) -> Any:
    name = getattr(t, "name", None) or t.__class__.__name__
    real_fn = getattr(t, "func", None) or getattr(t, "_run", None)
    if real_fn is None:
        if callable(t):
            real_fn = t
        else:
            raise TypeError(f"cannot wrap {t!r}: no func/_run/call found")
    try:
        sig = inspect.signature(real_fn)
    except (TypeError, ValueError):
        sig = None

    def call(*args, **kwargs):
        return real_fn(*args, **kwargs)

    if sig is not None:
        call.__signature__ = sig  # lets the recorder bind named args

    wrapped = dagcache_tool(call, pure=pure, name=name)
    original = t

    class ToolProxy:
        def invoke(self, input, *args, **kwargs):
            if isinstance(input, dict):
                return wrapped(**input)
            return wrapped(input)

        def run(self, input, *args, **kwargs):
            return self.invoke(input, *args, **kwargs)

        def _run(self, *args, **kwargs):
            return wrapped(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            return wrapped(*args, **kwargs)

        def __getattr__(self, item):
            return getattr(original, item)

    proxy = ToolProxy()
    proxy.name = name
    proxy.description = getattr(original, "description", "")
    return proxy


def wrap_tools(tools: list, *, pure: bool = True) -> list:
    """Wrap LangChain-style tools so calls are recorded as tool nodes."""
    return [_wrap_single_tool(t, pure) for t in tools]


def _make_lc_rerun(llm: Any):
    def rerun(messages):
        resp = llm.invoke(messages)
        return getattr(resp, "content", resp)

    return rerun


class LLMProxy:
    def __init__(self, real_llm: Any, planning: bool, name: str | None):
        self._real = real_llm
        self._planning = planning
        self._name = name or f"langchain:{real_llm.__class__.__name__}"

    def invoke(self, input, *args, **kwargs):
        resp = self._real.invoke(input, *args, **kwargs)
        rec = current_recorder()
        if rec is not None:
            content = getattr(resp, "content", resp)
            if self._planning:
                kind = graph.LLM_PLAN
            else:
                kind = graph.LLM_OUTPUT
                LLM_REGISTRY[self._name] = _make_lc_rerun(self._real)
            rec.add_node(kind, self._name, graph.LLM,
                         {"messages": _jsonable(input)}, _jsonable(content))
        return resp

    def __getattr__(self, item):
        return getattr(self._real, item)


def wrap_llm(llm: Any, *, planning: bool = False, name: str | None = None) -> LLMProxy:
    """Wrap a LangChain-style chat model.

    ``planning=True`` for models that pick tools/steps (never re-executed);
    default False for models producing final answers (re-run on replay).
    """
    return LLMProxy(llm, planning, name)
