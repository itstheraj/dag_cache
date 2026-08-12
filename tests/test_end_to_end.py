"""End-to-end: record an agent run, replay it on new inputs with zero LLM
planning calls, patch fresh data into literals, and fall back on drift."""

import pytest

import dagcache
from dagcache.policy import configure, get_config
from dagcache.store import get_store

CALLS = {"plan": 0, "draft": 0, "search": 0, "fetch": 0, "send": 0}
SEEN = {"search": [], "fetch": [], "send": []}
WORLD = {"status": "shipped", "fetch_shape": "dict"}


@dagcache.tool(name="e2e_search", pure=True)
def search_kb(query: str) -> str:
    CALLS["search"] += 1
    SEEN["search"].append(query)
    return f"kb-article about {query}"


@dagcache.tool(name="e2e_fetch", pure=True)
def fetch_order(order_id: str):
    CALLS["fetch"] += 1
    SEEN["fetch"].append(order_id)
    if WORLD["fetch_shape"] == "list":
        return ["unexpected"]
    return {"order_id": order_id, "status": WORLD["status"]}


@dagcache.tool(name="e2e_send", pure=False)
def send_reply(ticket_id: str, body: str) -> str:
    CALLS["send"] += 1
    SEEN["send"].append((ticket_id, body))
    return f"sent:{ticket_id}"


@dagcache.llm(name="e2e_plan", planning=True)
def plan(prompt: str) -> str:
    CALLS["plan"] += 1
    return "search -> fetch -> send -> draft"


@dagcache.llm(name="e2e_draft")
def draft(prompt: str) -> str:
    CALLS["draft"] += 1
    return f"reply: {prompt}"


@dagcache.agent
def resolve_ticket(ticket: dict) -> str:
    plan(f"handle {ticket['title']}")
    article = search_kb(ticket["title"])
    order = fetch_order(ticket["order_id"])
    send_reply(ticket["id"], f"see {article}")
    status = order["status"] if isinstance(order, dict) else "unknown"
    return draft(f"article={article}; status={status}")


@pytest.fixture(autouse=True)
def reset_state():
    for k in CALLS:
        CALLS[k] = 0
    for k in SEEN:
        SEEN[k].clear()
    WORLD["status"] = "shipped"
    WORLD["fetch_shape"] = "dict"


T1 = {"id": "T-100", "title": "where is my order", "order_id": "O-100"}
T2 = {"id": "T-200", "title": "package never arrived", "order_id": "O-200"}


def test_first_run_records_second_run_replays_verified():
    r1 = resolve_ticket(T1)
    assert r1 == "reply: article=kb-article about where is my order; status=shipped"
    assert CALLS == {"plan": 1, "draft": 1, "search": 1, "fetch": 1, "send": 1}

    WORLD["status"] = "delivered"  # the world moved on
    before = dict(CALLS)
    r2 = resolve_ticket(T2)

    # planning LLM never ran; everything else re-executed fresh
    assert CALLS["plan"] == before["plan"]
    assert CALLS["search"] == before["search"] + 1
    assert CALLS["fetch"] == before["fetch"] + 1
    assert CALLS["send"] == before["send"] + 1
    assert CALLS["draft"] == before["draft"] + 1

    # bindings resolved against the NEW ticket
    assert SEEN["search"][-1] == "package never arrived"
    assert SEEN["fetch"][-1] == "O-200"
    assert SEEN["send"][-1][0] == "T-200"
    # synthesized literal had the stale article patched to the fresh one
    assert "package never arrived" in SEEN["send"][-1][1]

    # fresh data flowed into the final answer
    assert r2 == "reply: article=kb-article about package never arrived; status=delivered"

    rows = get_store(get_config().db_path).list_dags()
    assert rows[0]["hits"] == 1
    assert rows[0]["fallbacks"] == 0


def test_frozen_mode_executes_nothing():
    r1 = resolve_ticket(T1)
    configure(replay_mode="frozen")
    before = dict(CALLS)
    r2 = resolve_ticket(T2)
    assert CALLS == before  # VCR mode: no tool ran, no LLM ran
    assert r2 == r1  # recorded answer returned verbatim


def test_drift_falls_back_to_live_agent():
    resolve_ticket(T1)
    WORLD["fetch_shape"] = "list"  # API changed under us
    before = dict(CALLS)
    r3 = resolve_ticket(T2)

    # replay diverged at fetch_order -> live agent took over (planning ran)
    assert CALLS["plan"] == before["plan"] + 1
    rows = get_store(get_config().db_path).list_dags()
    assert rows[0]["fallbacks"] == 1
    assert r3.startswith("reply: ")


def test_disabled_config_passes_through():
    configure(enabled=False)
    resolve_ticket(T1)
    resolve_ticket(T2)
    assert CALLS["plan"] == 2  # every run live
    assert get_store(get_config().db_path).list_dags() == []


def test_force_record_keeps_recording():
    configure(force_record=True)
    resolve_ticket(T1)
    resolve_ticket(T2)
    assert CALLS["plan"] == 2
    rows = get_store(get_config().db_path).list_dags()
    assert rows[0]["recordings"] == 2
    assert rows[0]["hits"] == 0
