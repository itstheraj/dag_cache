"""``dagcache`` command line: inspect, approve, diff, prune cached paths."""

from __future__ import annotations

import argparse
import difflib
import json
import sys

from .policy import get_config
from .store import get_store


def _store(args):
    cfg = get_config()
    return get_store(args.db or cfg.db_path)


def cmd_ls(args) -> int:
    rows = _store(args).list_dags()
    if not rows:
        print("no cached DAGs")
        return 0
    print(f"{'id':>4}  {'status':<9} {'rec':>4} {'hits':>5} {'fb':>3}  task_kind :: path")
    for r in rows:
        path = " > ".join(json.loads(r["path_json"]))
        print(
            f"{r['id']:>4}  {r['status']:<9} {r['recordings']:>4} {r['hits']:>5}"
            f" {r['fallbacks']:>3}  {r['task_kind']} :: {path}"
        )
    return 0


def cmd_show(args) -> int:
    dag = _store(args).get(args.id)
    if dag is None:
        print(f"no DAG with id {args.id}", file=sys.stderr)
        return 1
    print(json.dumps(dag.to_dict(), indent=2))
    return 0


def cmd_approve(args) -> int:
    ok = _store(args).approve(args.id)
    print(f"approved {args.id}" if ok else f"no DAG with id {args.id}")
    return 0 if ok else 1


def cmd_demote(args) -> int:
    ok = _store(args).demote(args.id)
    print(f"demoted {args.id} to staging" if ok else f"no DAG with id {args.id}")
    return 0 if ok else 1


def cmd_prune(args) -> int:
    n = _store(args).prune(status=args.status, older_than_days=args.older_than)
    print(f"pruned {n} DAG(s)")
    return 0


def cmd_export(args) -> int:
    try:
        _store(args).export_cassette(args.id, args.output)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"exported DAG {args.id} -> {args.output}")
    return 0


def cmd_import(args) -> int:
    rowid = _store(args).import_cassette(args.path)
    print(f"imported {args.path} -> DAG {rowid}")
    return 0


def cmd_diff(args) -> int:
    store = _store(args)
    a, b = store.get(args.id1), store.get(args.id2)
    if a is None or b is None:
        print("both ids must exist", file=sys.stderr)
        return 1
    left = json.dumps(a.to_dict(), indent=2, sort_keys=True).splitlines()
    right = json.dumps(b.to_dict(), indent=2, sort_keys=True).splitlines()
    print("\n".join(difflib.unified_diff(left, right, f"dag:{args.id1}", f"dag:{args.id2}")))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dagcache", description=__doc__)
    parser.add_argument("--db", help="path to store.db (default: config/env)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ls", help="list cached DAGs").set_defaults(fn=cmd_ls)

    p = sub.add_parser("show", help="dump a DAG as JSON")
    p.add_argument("id", type=int)
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("approve", help="mark a DAG canonical")
    p.add_argument("id", type=int)
    p.set_defaults(fn=cmd_approve)

    p = sub.add_parser("demote", help="return a DAG to staging")
    p.add_argument("id", type=int)
    p.set_defaults(fn=cmd_demote)

    p = sub.add_parser("prune", help="delete cached DAGs")
    p.add_argument("--status", choices=["staging", "approved", "dead"])
    p.add_argument("--older-than", type=int, metavar="DAYS")
    p.set_defaults(fn=cmd_prune)

    p = sub.add_parser("export", help="export a DAG as a JSON cassette")
    p.add_argument("id", type=int)
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("import", help="import a JSON cassette")
    p.add_argument("path")
    p.set_defaults(fn=cmd_import)

    p = sub.add_parser("diff", help="diff two DAGs")
    p.add_argument("id1", type=int)
    p.add_argument("id2", type=int)
    p.set_defaults(fn=cmd_diff)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
