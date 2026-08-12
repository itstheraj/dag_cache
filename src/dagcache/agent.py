"""The ``@agent`` entrypoint decorator: watch, record, replay, fall back."""

from __future__ import annotations

import functools
from typing import Callable

from .keys import fingerprint
from .policy import get_config
from .replay import Divergence, Executor
from .store import get_store
from .tracer import Recorder, bind_call


def agent(fn: Callable | None = None, *, kind: str | None = None, key: Callable | None = None):
    """Mark an agent entrypoint.

    On each call: look up an approved/staging DAG for (task kind, input
    shape). If found, replay it (zero LLM planning). If replay diverges or
    no DAG exists, run the function live while recording, and store the
    observed path as a candidate.

    ``key`` is an optional discriminator invoked with the *same arguments*
    as the wrapped function; its return *value* is mixed into the cache
    fingerprint. Use it to keep same-shaped but semantically different
    tasks apart::

        @dagcache.agent(key=lambda ticket: ticket["category"])
        def resolve_ticket(ticket: dict): ...

    Convention: the function's return value should be the result of its
    final LLM/tool call -- replay returns the terminal node's output.
    """

    def deco(f: Callable) -> Callable:
        task_kind = kind or f.__qualname__

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            cfg = get_config()
            if not cfg.enabled:
                return f(*args, **kwargs)
            inputs = bind_call(f, args, kwargs)
            # A failing key must raise: silently merging distinct tasks
            # into one path cache is the worst possible outcome.
            extra = key(*args, **kwargs) if key else None
            fp = fingerprint(inputs, extra=extra)
            store = get_store(cfg.db_path)

            if not cfg.force_record:
                dag = store.lookup(task_kind, fp, cfg)
                if dag is not None:
                    try:
                        result = Executor(cfg.replay_mode).run(dag, inputs)
                    except Divergence:
                        store.record_fallback(dag.id, cfg)
                    else:
                        store.record_hit(dag.id)
                        return result

            recorder = Recorder(task_kind, fp, inputs)
            with recorder:
                ret = f(*args, **kwargs)
            if recorder.nodes:
                store.save_dag(recorder.finalize(), cfg)
            return ret

        wrapper._dagcache_unwrapped = f  # type: ignore[attr-defined]
        return wrapper

    return deco(fn) if fn is not None else deco
