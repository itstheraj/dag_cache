from dagcache.cli import main
from dagcache.graph import DAG, TOOL, Node
from dagcache.policy import get_config
from dagcache.store import get_store


def make_dag(tool_names=("a", "b")):
    nodes = [
        Node(id=f"n{i}", kind=TOOL, name=n, purity="pure", output_shape="str")
        for i, n in enumerate(tool_names)
    ]
    return DAG(task_kind="task", fingerprint="fp1", nodes=nodes)


def db():
    return get_config().db_path


def test_ls_approve_show(capsys):
    store = get_store(db())
    rowid = store.save_dag(make_dag())

    assert main(["--db", db(), "ls"]) == 0
    out = capsys.readouterr().out
    assert "a > b" in out and "staging" in out

    assert main(["--db", db(), "approve", str(rowid)]) == 0
    assert store.get(rowid).status == "approved"

    assert main(["--db", db(), "show", str(rowid)]) == 0
    assert '"task_kind": "task"' in capsys.readouterr().out


def test_export_import_prune(capsys, tmp_path):
    store = get_store(db())
    rowid = store.save_dag(make_dag())
    cassette = tmp_path / "c.json"

    assert main(["--db", db(), "export", str(rowid), "-o", str(cassette)]) == 0
    assert main(["--db", db(), "prune", "--status", "staging"]) == 0
    assert store.list_dags() == []
    assert main(["--db", db(), "import", str(cassette)]) == 0
    assert len(store.list_dags()) == 1


def test_diff(capsys):
    store = get_store(db())
    a = store.save_dag(make_dag(("a",)))
    b = store.save_dag(make_dag(("b",)))
    assert main(["--db", db(), "diff", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert '"a"' in out and '"b"' in out


def test_missing_id_errors(capsys):
    assert main(["--db", db(), "show", "999"]) == 1
    assert "no DAG" in capsys.readouterr().err
