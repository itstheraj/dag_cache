"""Task identity: structural fingerprints and path keys.

Cache matching is deliberately *not* about argument values. Two invocations
are "the same task" when they hit the same @agent entrypoint with inputs of
the same *shape* (types and keys, never values). A cached solution is
identified by the chain of tool names it used -- the path -- not by the
arguments the LLM happened to invent last time.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def structure(value: Any) -> Any:
    """Recursive structural fingerprint of a value: types + dict keys only.

    ``{"a": 1, "b": "x"}`` and ``{"a": 99, "b": "y"}`` have the same
    structure; ``{"a": 1}`` and ``[1]`` do not.
    """
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return {
            "dict": {
                str(k): structure(v)
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            }
        }
    if isinstance(value, (list, tuple)):
        if not value:
            return {"list": ["empty"]}
        elems = {json.dumps(structure(v), sort_keys=True) for v in value}
        return {"list": sorted(elems)}
    if hasattr(value, "__dict__"):
        attrs = {
            k: structure(v)
            for k, v in sorted(vars(value).items())
            if not k.startswith("_")
        }
        return {"obj": type(value).__qualname__, "attrs": attrs}
    return f"other:{type(value).__qualname__}"


def fingerprint(inputs: dict[str, Any], extra: Any = None) -> str:
    """Stable hash of the *shape* of an @agent call's bound arguments.

    ``extra`` (from ``@agent(key=...)``) is mixed in by *value*, letting a
    discriminator like a ticket category separate same-shaped tasks into
    different path caches.
    """
    payload = json.dumps(
        {
            "shapes": {k: structure(v) for k, v in sorted(inputs.items())},
            "key": extra,
        },
        sort_keys=True,
        default=repr,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def path_key(tool_names: list[str]) -> str:
    """Stable hash of a tool chain, e.g. ``search_kb > fetch_order``."""
    return hashlib.sha256(">".join(tool_names).encode()).hexdigest()[:16]
