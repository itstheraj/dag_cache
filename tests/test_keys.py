from dagcache.keys import fingerprint, path_key, structure


def test_structure_ignores_values():
    a = {"id": "T-1", "title": "broken", "tags": ["a", "b"]}
    b = {"id": "T-99", "title": "totally different", "tags": ["x"]}
    assert structure(a) == structure(b)


def test_structure_distinguishes_shapes():
    assert structure({"a": 1}) != structure({"a": "1"})
    assert structure({"a": 1}) != structure({"a": 1, "b": 2})
    assert structure([1]) != structure({"0": 1})
    assert structure(None) != structure(False)


def test_structure_handles_objects():
    class Ticket:
        def __init__(self, title):
            self.title = title

    assert structure(Ticket("a")) == structure(Ticket("b"))
    assert "Ticket" in repr(structure(Ticket("a")))


def test_fingerprint_stable_and_value_independent():
    i1 = {"ticket": {"id": "T-1", "n": 3}}
    i2 = {"ticket": {"id": "T-2", "n": 400}}
    assert fingerprint(i1) == fingerprint(i2)
    assert fingerprint(i1) != fingerprint({"ticket": {"id": "T-2"}})


def test_path_key():
    assert path_key(["a", "b"]) == path_key(["a", "b"])
    assert path_key(["a", "b"]) != path_key(["b", "a"])
    assert path_key(["a", "b"]) != path_key(["a"])
