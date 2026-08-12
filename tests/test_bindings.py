import pytest

from dagcache.bindings import BindingError, infer_binding, resolve
from dagcache.graph import Binding


def test_infer_input_binding_nested():
    inputs = {"ticket": {"id": "T-1", "title": "broken widget"}}
    b = infer_binding("broken widget", inputs, [])
    assert b.source == "input"
    assert b.path == ["ticket", "title"]


def test_infer_node_binding():
    prior = [("n0", {"order_id": "O-9", "status": "shipped"})]
    b = infer_binding("O-9", {"ticket": {"x": "unrelated"}}, prior)
    assert b.source == "node"
    assert b.path == ["n0", "order_id"]


def test_generic_values_stay_literal():
    assert infer_binding(True, {"flag": True}, []).source == "literal"
    assert infer_binding(5, {"count": 5}, []).source == "literal"
    assert infer_binding("x", {"s": "x"}, []).source == "literal"
    assert infer_binding("never seen before", {"s": "else"}, []).source == "literal"


def test_input_wins_over_node_output():
    prior = [("n0", {"v": "shared-value"})]
    inputs = {"a": {"v": "shared-value"}}
    assert infer_binding("shared-value", inputs, prior).source == "input"


def test_resolve_from_object_attributes():
    class Ticket:
        pass

    t = Ticket()
    t.title = "hello world"
    b = infer_binding("hello world", {"ticket": t}, [])
    assert b.source == "input"
    assert b.path == ["ticket", "title"]

    fresh = Ticket()
    fresh.title = "fresh greeting"
    assert resolve(b, {"ticket": fresh}, {}) == "fresh greeting"


def test_resolve_node_binding_uses_fresh_output():
    b = Binding(source="node", path=["n0", "order_id"])
    assert resolve(b, {}, {"n0": {"order_id": "O-new"}}) == "O-new"


def test_resolve_literal():
    b = Binding(source="literal", value="canned")
    assert resolve(b, {}, {}) == "canned"


def test_resolve_failures_raise():
    with pytest.raises(BindingError):
        resolve(Binding(source="input", path=["missing"]), {}, {})
    with pytest.raises(BindingError):
        resolve(Binding(source="node", path=["n9", "x"]), {}, {})
    with pytest.raises(BindingError):
        resolve(Binding(source="input", path=["a", "deep"]), {"a": {"b": 1}}, {})
