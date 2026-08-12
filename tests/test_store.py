from dagcache.graph import DAG, TOOL, Node
from dagcache.policy import configure, get_config
from dagcache.store import get_store


def make_dag(kind="task", fp="fp1", tool_names=("a", "b")):
    nodes = [
        Node(id=f"n{i}", kind=TOOL, name=n, purity="pure", output_shape="str")
        for i, n in enumerate(tool_names)
    ]
    return DAG(task_kind=kind, fingerprint=fp, nodes=nodes)


def store():
    return get_store(get_config().db_path)


def test_save_dedupes_identical_paths():
    s = store()
    s.save_dag(make_dag())
    s.save_dag(make_dag())
    rows = s.list_dags()
    assert len(rows) == 1
    assert rows[0]["recordings"] == 2


def test_lookup_finds_staging_with_auto_replay():
    s = store()
    s.save_dag(make_dag())
    dag = s.lookup("task", "fp1", get_config())
    assert dag is not None
    assert dag.path() == ["a", "b"]
    assert dag.status == "staging"


def test_lookup_requires_approval_when_auto_replay_off():
    configure(auto_replay=False)
    s = store()
    rowid = s.save_dag(make_dag())
    assert s.lookup("task", "fp1", get_config()) is None
    s.approve(rowid)
    assert s.lookup("task", "fp1", get_config()) is not None


def test_approved_beats_staging_and_stats_rank_competing_paths():
    s = store()
    id_a = s.save_dag(make_dag(tool_names=("a",)))
    id_b = s.save_dag(make_dag(tool_names=("b",)))
    # more observed success wins among staging
    for _ in range(3):
        s.record_hit(id_b)
    assert s.lookup("task", "fp1", get_config()).path() == ["b"]
    # approval wins regardless of stats
    s.approve(id_a)
    assert s.lookup("task", "fp1", get_config()).path() == ["a"]


def test_fallbacks_kill_flaky_staging_dags():
    s = store()
    rowid = s.save_dag(make_dag())
    for _ in range(get_config().fallback_demote_threshold):
        s.record_fallback(rowid, get_config())
    assert s.list_dags()[0]["status"] == "dead"
    assert s.lookup("task", "fp1", get_config()) is None


def test_approved_dags_survive_fallbacks():
    s = store()
    rowid = s.save_dag(make_dag())
    s.approve(rowid)
    for _ in range(10):
        s.record_fallback(rowid, get_config())
    assert s.lookup("task", "fp1", get_config()) is not None


def test_ttl_expiry():
    configure(default_ttl_seconds=0)  # everything is instantly stale
    s = store()
    s.save_dag(make_dag(), get_config())
    assert s.lookup("task", "fp1", get_config()) is None


def test_prune_and_demote():
    s = store()
    id_a = s.save_dag(make_dag(tool_names=("a",)))
    s.save_dag(make_dag(tool_names=("b",)))
    s.demote(id_a)  # staging -> staging, still fine
    assert s.prune(status="staging") == 2
    assert s.list_dags() == []


def test_cassette_roundtrip(tmp_path):
    s = store()
    rowid = s.save_dag(make_dag())
    path = tmp_path / "cassette.json"
    s.export_cassette(rowid, str(path))
    s.prune(status="staging")
    new_id = s.import_cassette(str(path))
    assert s.get(new_id).path() == ["a", "b"]
