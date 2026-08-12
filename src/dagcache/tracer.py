"""Recording machinery and the public ``@tool`` / ``@llm`` decorators.

Recording is context-local (``contextvars``), so concurrent asyncio tasks or
threads each get their own recording. The agent function's code runs
completely unmodified while recording -- wrappers observe, they don't
interfere.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import time
from typing import Any, Callable

from . import graph
from .bindings import infer_binding, _jsonable
from .graph import DAG, Node
from .keys import structure

_current: contextvars.ContextVar["Recorder | None"] = contextvars.ContextVar(
    "dagcache_recorder", default=None
)

# Registries map node names back to live callables so the replay executor
# can re-execute them. They hold the *unwrapped* functions.
TOOL_REGISTRY: dict[str, Callable] = {}
TOOL_PURITY: dict[str, str] = {}
LLM_REGISTRY: dict[str, Callable] = {}


def current_recorder() -> "Recorder | None":
    return _current.get()


def bind_call(fn: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Bind a call to its parameter names (positional args included)."""
    try:
        return dict(inspect.signature(fn).bind(*args, **kwargs).arguments)
    except (TypeError, ValueError):
        return {**{f"arg{i}": a for i, a in enumerate(args)}, **kwargs}


class Recorder:
    """Accumulates nodes for one agent run. Used as a context manager."""

    def __init__(self, task_kind: str, fingerprint: str, inputs: dict[str, Any]):
        self.task_kind = task_kind
        self.fingerprint = fingerprint
        self.inputs = inputs
        self.nodes: list[Node] = []
        self._live_outputs: dict[str, Any] = {}
        self._token: contextvars.Token | None = None

    def __enter__(self) -> "Recorder":
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            _current.reset(self._token)
            self._token = None

    def add_node(
        self,
        kind: str,
        name: str,
        purity: str,
        args_live: dict[str, Any],
        result_live: Any,
        duration_ms: float = 0.0,
    ) -> None:
        nid = f"n{len(self.nodes)}"
        recorded = _jsonable(result_live)
        prior = [(n.id, self._live_outputs[n.id]) for n in self.nodes]
        bindings = {
            k: infer_binding(v, self.inputs, prior) for k, v in args_live.items()
        }
        self.nodes.append(
            Node(
                id=nid,
                kind=kind,
                name=name,
                purity=purity,
                args=bindings,
                recorded_output=recorded,
                output_shape=structure(recorded),
                duration_ms=duration_ms,
            )
        )
        self._live_outputs[nid] = result_live

    def finalize(self) -> DAG:
        """Compute Luigi-style ``requires`` edges and return the DAG.

        Data dependencies come from node-output bindings. Effectful and LLM
        nodes also depend on the previous node, preserving side-effect order
        (pure nodes may float freely for future parallel execution).
        """
        for i, node in enumerate(self.nodes):
            deps = {b.path[0] for b in node.args.values() if b.source == "node"}
            if i > 0 and node.purity != graph.PURE:
                deps.add(self.nodes[i - 1].id)
            node.requires = sorted(deps, key=lambda x: int(x[1:]))
        return DAG(
            task_kind=self.task_kind, fingerprint=self.fingerprint, nodes=self.nodes
        )


def tool(fn: Callable | None = None, *, pure: bool = True, name: str | None = None):
    """Register a function as a cacheable tool node.

    ``pure=True``  -- no side effects; skippable in frozen replay.
    ``pure=False`` -- side effects; re-executed even in verified replay and
                      pinned to chain order.
    """

    def deco(f: Callable) -> Callable:
        tname = name or f.__name__
        TOOL_REGISTRY[tname] = f
        TOOL_PURITY[tname] = graph.PURE if pure else graph.EFFECTFUL

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            rec = _current.get()
            if rec is None:
                return f(*args, **kwargs)
            bound = bind_call(f, args, kwargs)
            t0 = time.perf_counter()
            result = f(*args, **kwargs)
            rec.add_node(
                graph.TOOL,
                tname,
                TOOL_PURITY[tname],
                bound,
                result,
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
            return result

        wrapper._dagcache_unwrapped = f  # type: ignore[attr-defined]
        return wrapper

    return deco(fn) if fn is not None else deco


def llm(
    fn: Callable | None = None, *, name: str | None = None, planning: bool = False
):
    """Register a function that calls an LLM.

    ``planning=True``  -- the call decides what to do next. Its decision is
                          baked into the DAG; it is never re-executed.
    ``planning=False`` -- the call produces user-facing output. Re-executed
                          during verified replay so text reflects fresh data.
    """

    def deco(f: Callable) -> Callable:
        lname = name or f.__name__
        LLM_REGISTRY[lname] = f
        kind = graph.LLM_PLAN if planning else graph.LLM_OUTPUT

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            rec = _current.get()
            if rec is None:
                return f(*args, **kwargs)
            bound = bind_call(f, args, kwargs)
            t0 = time.perf_counter()
            result = f(*args, **kwargs)
            rec.add_node(
                kind, lname, graph.LLM, bound, result,
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
            return result

        wrapper._dagcache_unwrapped = f  # type: ignore[attr-defined]
        return wrapper

    return deco(fn) if fn is not None else deco
