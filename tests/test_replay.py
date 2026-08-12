"""Focused Executor tests: numeric-leaf patching, missing callables, frozen."""

import pytest

import dagcache
from dagcache.graph import DAG, TOOL, LLM_OUTPUT, Binding, Node
from dagcache.replay import Divergence, Executor


@dagcache.tool(name="re_thermometer", pure=True)
def thermometer(city: str) -> dict:
    return {"city": city, "temp": thermometer.current}


thermometer.current = 21


@dagcache.llm(name="re_phrase")
def phrase(prompt: str) -> str:
    return f"text: {prompt}"


def recorded_dag():
    """DAG as recorded from: temp = thermometer(city); phrase(f'It is {temp} degrees')."""
    n0 = Node(
        id="n0", kind=TOOL, name="re_thermometer", purity="pure",
        args={"city": Binding(source="input", path=["city"])},
        recorded_output={"city": "Berlin", "temp": 21},
        output_shape={"dict": {"city": "str", "temp": "int"}},
    )
    n1 = Node(
        id="n1", kind=LLM_OUTPUT, name="re_phrase", purity="llm",
        args={"prompt": Binding(source="literal", value="It is 21 degrees in Berlin")},
        recorded_output="text: It is 21 degrees in Berlin",
        output_shape="str", requires=["n0"],
    )
    return DAG(task_kind="weather", fingerprint="fp", nodes=[n0, n1])


def test_numeric_leaf_patching():
    thermometer.current = 30
    out = Executor("verified").run(recorded_dag(), {"city": "Paris"})
    assert out == "text: It is 30 degrees in Paris"


def test_bounded_replacement_protects_substrings():
    thermometer.current = 2100  # recorded "21" must not rewrite inside this...
    dag = recorded_dag()
    # ...but the fresh 2100 still lands in the prompt correctly
    out = Executor("verified").run(dag, {"city": "Paris"})
    assert out == "text: It is 2100 degrees in Paris"


def test_missing_callable_diverges():
    dag = recorded_dag()
    dag.nodes[0].name = "no_such_tool"
    with pytest.raises(Divergence, match="no live callable"):
        Executor("verified").run(dag, {"city": "Paris"})


def test_shape_drift_diverges():
    @dagcache.tool(name="re_thermometer2", pure=True)
    def broken(city: str):
        return ["not", "a", "dict"]

    dag = recorded_dag()
    dag.nodes[0].name = "re_thermometer2"
    with pytest.raises(Divergence, match="shape drifted"):
        Executor("verified").run(dag, {"city": "Paris"})


def test_unresolvable_binding_diverges():
    dag = recorded_dag()
    dag.nodes[0].args["city"] = Binding(source="input", path=["nonexistent"])
    with pytest.raises(Divergence):
        Executor("verified").run(dag, {"city": "Paris"})


def test_frozen_returns_recorded():
    thermometer.current = 99
    out = Executor("frozen").run(recorded_dag(), {"city": "Paris"})
    assert out == "text: It is 21 degrees in Berlin"
