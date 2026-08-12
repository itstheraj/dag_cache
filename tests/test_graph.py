import pytest

from dagcache.graph import DAG, TOOL, LLM_OUTPUT, Binding, Node


def make_node(nid, name, kind=TOOL, purity="pure", requires=(), args=None):
    return Node(
        id=nid,
        kind=kind,
        name=name,
        purity=purity,
        args=args or {},
        recorded_output=f"out-{nid}",
        output_shape="str",
        requires=list(requires),
    )


def test_serde_roundtrip():
    n0 = make_node("n0", "search")
    n1 = make_node(
        "n1", "fetch", requires=["n0"],
        args={"order_id": Binding(source="node", path=["n0", "order_id"])},
    )
    dag = DAG(task_kind="k", fingerprint="fp", nodes=[n0, n1])
    back = DAG.from_dict(dag.to_dict())
    assert back.task_kind == "k"
    assert back.nodes[1].args["order_id"].path == ["n0", "order_id"]
    assert back.nodes[1].requires == ["n0"]


def test_path_excludes_llm_nodes():
    nodes = [
        make_node("n0", "search"),
        make_node("n1", "draft", kind=LLM_OUTPUT, purity="llm"),
        make_node("n2", "send"),
    ]
    dag = DAG(task_kind="k", fingerprint="fp", nodes=nodes)
    assert dag.path() == ["search", "send"]
    assert dag.terminal().id == "n2"


def test_topo_order_respects_dependencies():
    nodes = [
        make_node("n0", "a"),
        make_node("n1", "b", requires=["n2"]),  # declared before its dep
        make_node("n2", "c"),
    ]
    dag = DAG(task_kind="k", fingerprint="fp", nodes=nodes)
    order = [n.id for n in dag.topo_order()]
    assert order.index("n2") < order.index("n1")


def test_topo_order_detects_cycles():
    nodes = [
        make_node("n0", "a", requires=["n1"]),
        make_node("n1", "b", requires=["n0"]),
    ]
    dag = DAG(task_kind="k", fingerprint="fp", nodes=nodes)
    with pytest.raises(ValueError, match="cycle"):
        dag.topo_order()
