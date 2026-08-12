"""Persistence: SQLite by default, JSON cassettes for VCS review.

Rows are keyed by (task_kind, fingerprint, path_key): recording the same
tool chain for the same task shape again just bumps ``recordings`` -- that
repetition is how a path earns "canonical" status. Competing paths for the
same task shape coexist and are ranked by stats at lookup time.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

from .graph import DAG
from .policy import Config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_kind TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    path_key TEXT NOT NULL,
    path_json TEXT NOT NULL,
    dag_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'staging',
    recordings INTEGER NOT NULL DEFAULT 1,
    hits INTEGER NOT NULL DEFAULT 0,
    fallbacks INTEGER NOT NULL DEFAULT 0,
    ttl_seconds INTEGER,
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    UNIQUE (task_kind, fingerprint, path_key)
);
"""

_STORES: dict[str, "Store"] = {}


def get_store(db_path: str) -> "Store":
    if db_path not in _STORES:
        _STORES[db_path] = Store(db_path)
    return _STORES[db_path]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, db_path: str):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    # -- writes -----------------------------------------------------------

    def save_dag(self, dag: DAG, cfg: Config | None = None) -> int:
        dag_json = json.dumps(dag.to_dict())
        path_json = json.dumps(dag.path())
        key = dag.path_key()
        row = self._conn.execute(
            "SELECT id, recordings FROM dags WHERE task_kind=? AND fingerprint=? AND path_key=?",
            (dag.task_kind, dag.fingerprint, key),
        ).fetchone()
        if row:
            self._conn.execute(
                "UPDATE dags SET recordings=recordings+1, dag_json=?, last_used_at=? WHERE id=?",
                (dag_json, _now(), row["id"]),
            )
            self._conn.commit()
            return row["id"]
        ttl = cfg.default_ttl_seconds if cfg else None
        cur = self._conn.execute(
            "INSERT INTO dags (task_kind, fingerprint, path_key, path_json, dag_json,"
            " status, recordings, ttl_seconds, created_at, last_used_at)"
            " VALUES (?,?,?,?,?,'staging',1,?,?,?)",
            (dag.task_kind, dag.fingerprint, key, path_json, dag_json, ttl, _now(), _now()),
        )
        self._conn.commit()
        return cur.lastrowid

    # -- reads ------------------------------------------------------------

    def _row_to_dag(self, row: sqlite3.Row) -> DAG:
        dag = DAG.from_dict(json.loads(row["dag_json"]))
        dag.id = row["id"]
        dag.status = row["status"]
        dag.recordings = row["recordings"]
        dag.hits = row["hits"]
        dag.fallbacks = row["fallbacks"]
        return dag

    def _expired(self, row: sqlite3.Row) -> bool:
        if row["ttl_seconds"] is None:
            return False
        created = datetime.fromisoformat(row["created_at"])
        return datetime.now(timezone.utc) > created + timedelta(seconds=row["ttl_seconds"])

    def lookup(self, task_kind: str, fingerprint: str, cfg: Config) -> DAG | None:
        """Best cached path for this task shape: approved beats staging,
        then by observed success (hits + recordings), oldest first on ties."""
        statuses = ("approved", "staging") if cfg.auto_replay else ("approved",)
        rows = self._conn.execute(
            f"SELECT * FROM dags WHERE task_kind=? AND fingerprint=?"
            f" AND status IN ({','.join('?' * len(statuses))})",
            (task_kind, fingerprint, *statuses),
        ).fetchall()
        rows = [r for r in rows if not self._expired(r)]
        if not rows:
            return None
        rows.sort(
            key=lambda r: (
                0 if r["status"] == "approved" else 1,
                -(r["hits"] + r["recordings"]),
                r["id"],
            )
        )
        return self._row_to_dag(rows[0])

    def get(self, rowid: int) -> DAG | None:
        row = self._conn.execute("SELECT * FROM dags WHERE id=?", (rowid,)).fetchone()
        return self._row_to_dag(row) if row else None

    def list_dags(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, task_kind, fingerprint, path_json, status, recordings,"
            " hits, fallbacks, created_at, last_used_at FROM dags ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    # -- lifecycle --------------------------------------------------------

    def record_hit(self, rowid: int) -> None:
        self._conn.execute(
            "UPDATE dags SET hits=hits+1, last_used_at=? WHERE id=?", (_now(), rowid)
        )
        self._conn.commit()

    def record_fallback(self, rowid: int, cfg: Config) -> None:
        self._conn.execute(
            "UPDATE dags SET fallbacks=fallbacks+1, last_used_at=? WHERE id=?",
            (_now(), rowid),
        )
        if cfg.fallback_demote_threshold > 0:
            self._conn.execute(
                "UPDATE dags SET status='dead' WHERE id=? AND status='staging'"
                " AND fallbacks>=?",
                (rowid, cfg.fallback_demote_threshold),
            )
        self._conn.commit()

    def approve(self, rowid: int) -> bool:
        cur = self._conn.execute(
            "UPDATE dags SET status='approved' WHERE id=?", (rowid,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def demote(self, rowid: int) -> bool:
        cur = self._conn.execute(
            "UPDATE dags SET status='staging' WHERE id=?", (rowid,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def prune(self, *, status: str | None = None, older_than_days: int | None = None) -> int:
        sql, params = "DELETE FROM dags WHERE 1=1", []
        if status:
            sql += " AND status=?"
            params.append(status)
        if older_than_days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
            sql += " AND created_at < ?"
            params.append(cutoff)
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return cur.rowcount

    # -- cassettes --------------------------------------------------------

    def export_cassette(self, rowid: int, path: str) -> None:
        row = self._conn.execute("SELECT * FROM dags WHERE id=?", (rowid,)).fetchone()
        if not row:
            raise KeyError(f"no DAG with id {rowid}")
        payload = {"meta": {k: row[k] for k in row.keys() if k != "dag_json"},
                   "dag": json.loads(row["dag_json"])}
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)

    def import_cassette(self, path: str) -> int:
        with open(path) as fh:
            payload = json.load(fh)
        dag = DAG.from_dict(payload["dag"])
        return self.save_dag(dag)
