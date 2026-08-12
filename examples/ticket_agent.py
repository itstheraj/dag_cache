"""Runnable demo: a support-ticket agent with a fake LLM.

    python examples/ticket_agent.py

Run 1 records. Run 2 (new ticket, same shape) replays the cached tool chain
with zero LLM planning calls. Run 3 simulates world drift (a tool starts
returning a different shape) and falls back to the live agent.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dagcache

CALLS = {"plan": 0, "draft": 0, "search": 0, "fetch": 0, "send": 0}
WORLD = {"status": "shipped", "fetch_shape": "dict"}


@dagcache.tool(pure=True)
def search_kb(query: str) -> str:
    CALLS["search"] += 1
    return f"kb-article about {query}"


@dagcache.tool(pure=True)
def fetch_order(order_id: str):
    CALLS["fetch"] += 1
    if WORLD["fetch_shape"] == "list":
        return ["unexpected", "list"]  # the API changed under us
    return {"order_id": order_id, "status": WORLD["status"]}


@dagcache.tool(pure=False)
def send_reply(ticket_id: str, body: str) -> str:
    CALLS["send"] += 1
    print(f"  [side effect] emailed customer for {ticket_id}")
    return f"sent:{ticket_id}"


@dagcache.llm(planning=True)
def plan(prompt: str) -> str:
    CALLS["plan"] += 1
    return "search_kb -> fetch_order -> send_reply -> draft"


@dagcache.llm
def draft(prompt: str) -> str:
    CALLS["draft"] += 1
    return f"Dear customer, {prompt}"


@dagcache.agent
def resolve_ticket(ticket: dict) -> str:
    plan(f"handle ticket {ticket['title']}")
    article = search_kb(ticket["title"])
    order = fetch_order(ticket["order_id"])
    send_reply(ticket["id"], f"see {article}")
    status = order["status"] if isinstance(order, dict) else "unknown"
    return draft(f"article={article}; status={status}")


def show(label, result, before):
    deltas = {k: CALLS[k] - before[k] for k in CALLS}
    print(f"{label}\n  -> {result}\n  calls this run: {deltas}\n")


def main():
    dagcache.configure(db_path=os.path.join(os.path.dirname(__file__), ".demo-store.db"))

    t1 = {"id": "T-100", "title": "where is my order", "order_id": "O-100"}
    t2 = {"id": "T-200", "title": "package never arrived", "order_id": "O-200"}

    before = dict(CALLS)
    show("RUN 1 (live, recorded):", resolve_ticket(t1), before)

    WORLD["status"] = "delivered"  # world moved on; cached path should still work
    before = dict(CALLS)
    show("RUN 2 (verified replay: tools re-run, planning LLM skipped):",
         resolve_ticket(t2), before)

    WORLD["fetch_shape"] = "list"  # world broke; replay must fall back
    before = dict(CALLS)
    show("RUN 3 (divergence -> live fallback):", resolve_ticket(t2), before)

    os.remove(os.path.join(os.path.dirname(__file__), ".demo-store.db"))


if __name__ == "__main__":
    main()
