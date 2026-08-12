"""The Luigi-style worker: execute a cached DAG against new inputs.

Two modes:

- ``verified`` (default) -- re-execute every tool with freshly resolved
  arguments (real side effects, fresh data) and re-run output LLM calls.
  Planning LLM calls are never re-run: the DAG *is* the plan. Any drift
  (unresolvable binding, changed output shape, missing callable, tool
  error) raises :class:`Divergence` and the caller falls back to the live
  agent from scratch.

- ``frozen`` -- VCR mode. Nothing executes; recorded outputs are returned.
  Luigi's ``output().exists()`` applied to the whole graph. For tests/CI.
"""

from __future__ import annotations

import json
import re
from typing import Any

from . import graph
from .bindings import BindingError, _jsonable, resolve
from .graph import DAG
from .keys import structure
from .tracer import LLM_REGISTRY, TOOL_REGISTRY


class Divergence(Exception):
    """The world no longer matches the cached path. Fall back to the agent."""


def _replacement_pairs(recorded: Any, fresh: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if isinstance(recorded, str) and isinstance(fresh, str):
        pairs.append((recorded, fresh))
    try:
        pairs.append(
            (json.dumps(recorded, sort_keys=True), json.dumps(fresh, sort_keys=True))
        )
    except (TypeError, ValueError):
        pass
    pairs.append((str(recorded), str(fresh)))
    # Field-level pairs: a prompt typically embeds *parts* of a tool result
    # ("status=shipped", "It is 21 degrees"), not the whole JSON blob.
    _leaf_pairs(recorded, fresh, pairs)
    # Replacement is word-boundary-bounded, so short values are safe.
    return [(o, n) for o, n in pairs if len(o) >= 2 and o != n]


def _leaf_pairs(recorded: Any, fresh: Any, out: list[tuple[str, str]]) -> None:
    """Pair up leaves at matching paths that changed between runs --
    strings, and numbers like prices/temps/quantities."""
    if isinstance(recorded, dict) and isinstance(fresh, dict):
        for k in recorded:
            if k in fresh:
                _leaf_pairs(recorded[k], fresh[k], out)
    elif isinstance(recorded, list) and isinstance(fresh, list):
        for a, b in zip(recorded, fresh):
            _leaf_pairs(a, b, out)
    elif isinstance(recorded, str) and isinstance(fresh, str) and recorded != fresh:
        out.append((recorded, fresh))
    elif (
        isinstance(recorded, (int, float))
        and not isinstance(recorded, bool)
        and type(recorded) is type(fresh)
        and recorded != fresh
    ):
        out.append((str(recorded), str(fresh)))


def _patch_literals(value: Any, dag: DAG, outputs: dict[str, Any]) -> Any:
    """Substitute fresh upstream outputs into recorded literal arguments.

    An ``llm_output`` node records its prompt as a literal; the prompt embeds
    whatever the tools returned at record time. Before re-running the call we
    swap in today's values so the synthesized text reflects fresh data.
    """
    if isinstance(value, str):
        out = value
        for node in dag.nodes:
            if node.id not in outputs or node.recorded_output is None:
                continue
            for old, new in _replacement_pairs(node.recorded_output, outputs[node.id]):
                # Word-boundary-bounded: "21" must not rewrite "2100".
                out = re.sub(r"(?<!\w)" + re.escape(old) + r"(?!\w)", lambda _: new, out)
        return out
    if isinstance(value, list):
        return [_patch_literals(v, dag, outputs) for v in value]
    if isinstance(value, dict):
        return {k: _patch_literals(v, dag, outputs) for k, v in value.items()}
    return value


class Executor:
    def __init__(self, mode: str = "verified"):
        if mode not in ("verified", "frozen"):
            raise ValueError("mode must be 'verified' or 'frozen'")
        self.mode = mode

    def _callable_for(self, node) -> Any:
        if node.kind == graph.TOOL:
            fn = TOOL_REGISTRY.get(node.name)
        elif node.kind == graph.LLM_OUTPUT:
            fn = LLM_REGISTRY.get(node.name)
        else:
            return None
        if fn is None:
            raise Divergence(f"no live callable registered for {node.kind}:{node.name}")
        return getattr(fn, "_dagcache_unwrapped", fn)

    def run(self, dag: DAG, inputs: dict[str, Any]) -> Any:
        outputs: dict[str, Any] = {}
        for node in dag.topo_order():
            if node.kind == graph.LLM_PLAN or self.mode == "frozen":
                # The plan is what's cached; frozen mode caches outputs too.
                outputs[node.id] = node.recorded_output
                continue
            fn = self._callable_for(node)
            args = {}
            for arg_name, binding in node.args.items():
                try:
                    args[arg_name] = resolve(binding, inputs, outputs)
                except BindingError as e:
                    raise Divergence(f"{node.name}.{arg_name}: {e}") from None
            args = _patch_literals(args, dag, outputs)
            try:
                fresh = fn(**args)
            except Divergence:
                raise
            except Exception as e:
                raise Divergence(f"{node.name} raised {type(e).__name__}: {e}") from None
            if structure(_jsonable(fresh)) != node.output_shape:
                raise Divergence(
                    f"{node.name} output shape drifted:"
                    f" recorded {node.output_shape!r}, got {structure(_jsonable(fresh))!r}"
                )
            outputs[node.id] = fresh
        return outputs[dag.terminal().id]
