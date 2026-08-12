"""The cached artifact: a DAG of tool/LLM nodes with Luigi-flavored semantics.

Every node has ``requires`` (upstream node ids), a recorded output, and a
purity classification. The replay executor is the Luigi "worker": it walks
the DAG in dependency order, skipping nodes whose recorded output is
acceptable (frozen mode -- Luigi's ``output().exists()``) and re-running
everything else with freshly resolved arguments (verified mode).

Node kinds:

- ``tool``       -- a call to a ``@dagcache.tool`` function
- ``llm_plan``   -- an LLM planning call (its *decision* is cached in the
                    DAG edges; never re-executed at replay)
- ``llm_output`` -- an LLM synthesis call (re-executed at verified replay so
                    user-facing text reflects fresh data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PURE = "pure"
EFFECTFUL = "effectful"
LLM = "llm"

TOOL = "tool"
LLM_PLAN = "llm_plan"
LLM_OUTPUT = "llm_output"


@dataclass
class Binding:
    """How to obtain one argument value at replay time.

    source ``input``:   walk the agent's own call arguments along ``path``
    source ``node``:    walk a prior node's (fresh) output; ``path[0]`` is
                        the upstream node id
    source ``literal``: the LLM synthesized this value; reuse the recorded
                        ``value`` as-is
    """

    source: str
    path: list = field(default_factory=list)
    value: Any = None

    def to_dict(self) -> dict:
        return {"source": self.source, "path": self.path, "value": self.value}

    @classmethod
    def from_dict(cls, d: dict) -> "Binding":
        return cls(source=d["source"], path=list(d.get("path", [])), value=d.get("value"))


@dataclass
class Node:
    id: str
    kind: str  # TOOL | LLM_PLAN | LLM_OUTPUT
    name: str
    purity: str  # PURE | EFFECTFUL | LLM
    args: dict[str, Binding] = field(default_factory=dict)
    recorded_output: Any = None
    output_shape: Any = None
    requires: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "purity": self.purity,
            "args": {k: b.to_dict() for k, b in self.args.items()},
            "recorded_output": self.recorded_output,
            "output_shape": self.output_shape,
            "requires": list(self.requires),
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(
            id=d["id"],
            kind=d["kind"],
            name=d["name"],
            purity=d["purity"],
            args={k: Binding.from_dict(v) for k, v in d.get("args", {}).items()},
            recorded_output=d.get("recorded_output"),
            output_shape=d.get("output_shape"),
            requires=list(d.get("requires", [])),
            duration_ms=d.get("duration_ms", 0.0),
        )


@dataclass
class DAG:
    task_kind: str
    fingerprint: str
    nodes: list[Node]
    # Store-managed metadata (not part of the cached artifact itself)
    id: int | None = None
    status: str = "staging"  # staging | approved | dead
    recordings: int = 1
    hits: int = 0
    fallbacks: int = 0

    def path(self) -> list[str]:
        """The canonical tool chain -- this is what we cache *on*."""
        return [n.name for n in self.nodes if n.kind == TOOL]

    def path_key(self) -> str:
        from .keys import path_key

        return path_key(self.path())

    def terminal(self) -> Node:
        if not self.nodes:
            raise ValueError("DAG has no nodes")
        return self.nodes[-1]

    def topo_order(self) -> list[Node]:
        """Kahn's algorithm with original index as tiebreak (stable order)."""
        index = {n.id: i for i, n in enumerate(self.nodes)}
        indegree = {n.id: 0 for n in self.nodes}
        downstream: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for n in self.nodes:
            for dep in n.requires:
                if dep not in indegree:
                    raise ValueError(f"node {n.id} requires unknown node {dep}")
                indegree[n.id] += 1
                downstream[dep].append(n.id)
        ready = sorted((nid for nid, d in indegree.items() if d == 0), key=index.get)
        order: list[Node] = []
        by_id = {n.id: n for n in self.nodes}
        while ready:
            nid = ready.pop(0)
            order.append(by_id[nid])
            for nxt in downstream[nid]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)
            ready.sort(key=index.get)
        if len(order) != len(self.nodes):
            raise ValueError("DAG contains a cycle")
        return order

    def to_dict(self) -> dict:
        return {
            "task_kind": self.task_kind,
            "fingerprint": self.fingerprint,
            "path": self.path(),
            "nodes": [n.to_dict() for n in self.nodes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DAG":
        return cls(
            task_kind=d["task_kind"],
            fingerprint=d["fingerprint"],
            nodes=[Node.from_dict(n) for n in d["nodes"]],
        )
