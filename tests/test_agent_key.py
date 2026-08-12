import dagcache
from dagcache.store import get_store
from dagcache.policy import get_config

RAN = []


@dagcache.tool(name="key_refund", pure=False)
def refund(order_id: str) -> str:
    RAN.append(("refund", order_id))
    return f"refunded:{order_id}"


@dagcache.tool(name="key_weather", pure=True)
def weather(city: str) -> str:
    RAN.append(("weather", city))
    return f"sunny in {city}"


@dagcache.agent(key=lambda task: task["category"])
def handle(task: dict) -> str:
    if task["category"] == "refund":
        return refund(task["target"])
    return weather(task["target"])


def test_key_separates_same_shaped_tasks():
    r1 = handle({"category": "refund", "target": "O-1"})
    r2 = handle({"category": "weather", "target": "Berlin"})
    assert r1 == "refunded:O-1"
    assert r2 == "sunny in Berlin"

    # two distinct fingerprints were cached despite identical input shapes
    rows = get_store(get_config().db_path).list_dags()
    assert len(rows) == 2
    paths = {r["path_json"] for r in rows}
    assert paths == {'["key_refund"]', '["key_weather"]'}

    # replays stay in their lane: refund replay never calls the weather tool
    RAN.clear()
    assert handle({"category": "refund", "target": "O-2"}) == "refunded:O-2"
    assert RAN == [("refund", "O-2")]
