"""Turn recorded literal arguments into re-bindable slots, and back.

At record time we infer the *provenance* of every tool argument: did this
value come straight from the agent's input (``bind:input.ticket.title``)?
From an upstream tool's output (``bind:node.n0.order_id``)? Or did the LLM
make it up (``literal``)? At replay time we resolve bindings against the
*new* invocation's inputs and the *fresh* upstream outputs -- which is how a
cached path runs on data it has never seen.
"""

from __future__ import annotations

from typing import Any

from .graph import Binding


class BindingError(Exception):
    pass


# Values too generic to safely provenance-match. Binding `True` or `1` to
# whatever `True` or `1` we find first in the input is a footgun, so these
# stay literal.
def _worth_matching(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)) and abs(value) < 100:
        return False
    if isinstance(value, str) and len(value) < 2:
        return False
    return True


def _walk(root: Any, path: list) -> Any:
    cur = root
    for seg in path:
        if isinstance(cur, dict):
            if seg not in cur:
                raise BindingError(f"missing key {seg!r} in {cur!r}")
            cur = cur[seg]
        elif isinstance(cur, (list, tuple)):
            try:
                cur = cur[int(seg)]
            except (ValueError, IndexError):
                raise BindingError(f"missing index {seg!r} in {cur!r}") from None
        else:
            if not hasattr(cur, seg):
                raise BindingError(f"missing attribute {seg!r} on {type(cur).__name__}")
            cur = getattr(cur, seg)
    return cur


def _find(value: Any, root: Any, base: list, out: list, depth: int = 0) -> None:
    """Collect paths under ``root`` whose value deep-equals ``value``."""
    if depth > 12 or len(out) > 8:
        return
    try:
        if type(root) is type(value) and root == value:
            out.append(base)
            return
    except Exception:
        pass
    if isinstance(root, dict):
        for k, v in root.items():
            _find(value, v, base + [k], out, depth + 1)
    elif isinstance(root, (list, tuple)):
        for i, v in enumerate(root):
            _find(value, v, base + [i], out, depth + 1)
    elif hasattr(root, "__dict__"):
        for k, v in vars(root).items():
            if not k.startswith("_"):
                _find(value, v, base + [k], out, depth + 1)


def infer_binding(
    value: Any, inputs: dict[str, Any], prior: list[tuple[str, Any]]
) -> Binding:
    """Infer where a recorded argument value came from.

    ``prior`` is ``[(node_id, live_output), ...]`` in execution order.
    Inputs win over node outputs (caller-provided data is the most stable
    reference); earliest nodes win over later ones (deterministic).
    """
    if not _worth_matching(value):
        return Binding(source="literal", value=_jsonable(value))

    found: list = []
    _find(value, inputs, [], found)
    if found:
        return Binding(source="input", path=min(found, key=len))

    for node_id, output in prior:
        found = []
        _find(value, output, [], found)
        if found:
            return Binding(source="node", path=[node_id] + min(found, key=len))

    return Binding(source="literal", value=_jsonable(value))


def resolve(binding: Binding, inputs: dict[str, Any], outputs: dict[str, Any]) -> Any:
    """Resolve a binding against this invocation's inputs and fresh outputs."""
    if binding.source == "literal":
        return binding.value
    if binding.source == "input":
        try:
            return _walk(inputs, binding.path)
        except BindingError as e:
            raise BindingError(f"input binding {binding.path}: {e}") from None
    if binding.source == "node":
        if not binding.path or binding.path[0] not in outputs:
            raise BindingError(f"node binding references unavailable node {binding.path!r}")
        try:
            return _walk(outputs[binding.path[0]], binding.path[1:])
        except BindingError as e:
            raise BindingError(f"node binding {binding.path}: {e}") from None
    raise BindingError(f"unknown binding source {binding.source!r}")


def _jsonable(value: Any) -> Any:
    """Best-effort conversion to a JSON-serializable value (for storage)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(v) for v in value), key=repr)
    if hasattr(value, "__dict__"):
        return {
            "__class__": type(value).__qualname__,
            **{
                k: _jsonable(v)
                for k, v in vars(value).items()
                if not k.startswith("_")
            },
        }
    return repr(value)
