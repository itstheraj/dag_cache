"""Adapters are duck-typed: fakes stand in for openai/langchain objects."""

import json
from types import SimpleNamespace

import pytest

import dagcache
from dagcache.adapters.langchain import wrap_llm, wrap_tools
from dagcache.adapters.openai import wrap_client

# ---------------------------------------------------------------- openai fake


def make_tool_resp(name, arguments: dict):
    tc = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tc]))]
    )


def make_text_resp(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))]
    )


class FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeOpenAIClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


LOOKUP_CALLS = []


@dagcache.tool(name="oa_lookup", pure=True)
def oa_lookup(query: str) -> str:
    LOOKUP_CALLS.append(query)
    return f"data for {query}"


def build_openai_agent(responses):
    raw = FakeOpenAIClient(responses)
    client = wrap_client(raw)

    @dagcache.agent
    def oa_agent(task: dict):
        plan_resp = client.chat.completions.create(
            model="gpt-fake",
            messages=[{"role": "user", "content": f"plan for {task['title']}"}],
        )
        tc = plan_resp.choices[0].message.tool_calls[0]
        result = oa_lookup(**json.loads(tc.function.arguments))
        final = client.chat.completions.create(
            model="gpt-fake",
            messages=[{"role": "user", "content": f"summarize {result}"}],
        )
        return final.choices[0].message.content

    return oa_agent, raw


@pytest.fixture(autouse=True)
def reset():
    LOOKUP_CALLS.clear()


def test_openai_adapter_records_plan_and_output():
    agent_fn, raw = build_openai_agent([
        make_tool_resp("oa_lookup", {"query": "refund policy"}),
        make_text_resp("old summary"),
    ])
    out = agent_fn({"title": "refund policy"})
    assert out == "old summary"
    assert len(raw.chat.completions.calls) == 2

    from dagcache.store import get_store
    from dagcache.policy import get_config

    rows = get_store(get_config().db_path).list_dags()
    assert len(rows) == 1
    assert json.loads(rows[0]["path_json"]) == ["oa_lookup"]


def test_openai_adapter_replay_skips_planning_call():
    agent_fn, raw = build_openai_agent([
        make_tool_resp("oa_lookup", {"query": "refund policy"}),
        make_text_resp("old summary"),
    ])
    agent_fn({"title": "refund policy"})

    # Replay needs only ONE more scripted response: the output call.
    raw.chat.completions._responses.append(make_text_resp("fresh summary"))
    out = agent_fn({"title": "warranty policy"})

    assert out == "fresh summary"
    # the tool arg was inferred as bind:input.task.title -> resolved fresh
    assert LOOKUP_CALLS[-1] == "warranty policy"
    # two calls at record + exactly one at replay (the planning call was skipped)
    assert len(raw.chat.completions.calls) == 3
    # the rerun output call carried the FRESH tool data in its prompt
    assert "summarize data for warranty policy" in str(
        raw.chat.completions.calls[-1]["messages"]
    )


# ------------------------------------------------------------ langchain fakes


class FakeLCTool:
    def __init__(self, name, func, description="fake tool"):
        self.name = name
        self.func = func
        self.description = description


class FakeLCLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, input, **kwargs):
        self.calls.append(input)
        return SimpleNamespace(content=f"lc-answer: {input}")


LC_TOOL_CALLS = []


def lc_search(query: str) -> str:
    LC_TOOL_CALLS.append(query)
    return f"lc-data for {query}"


@pytest.fixture(autouse=True)
def reset_lc():
    LC_TOOL_CALLS.clear()


def test_langchain_tools_record_and_replay():
    tool = wrap_tools([FakeLCTool("lc_search", lc_search)])[0]

    @dagcache.agent
    def lc_agent(task: dict):
        return tool.invoke({"query": task["q"]})

    assert lc_agent({"q": "first question"}) == "lc-data for first question"
    assert lc_agent({"q": "second question"}) == "lc-data for second question"
    assert LC_TOOL_CALLS == ["first question", "second question"]


def test_langchain_planning_llm_skipped_on_replay():
    tool = wrap_tools([FakeLCTool("lc_search", lc_search)])[0]
    planner = wrap_llm(FakeLCLLM(), planning=True, name="lc_planner")
    drafter = wrap_llm(FakeLCLLM(), name="lc_drafter")

    @dagcache.agent
    def lc_agent(task: dict):
        planner.invoke(f"plan: {task['q']}")
        data = tool.invoke({"query": task["q"]})
        return drafter.invoke(f"write about {data}") .content

    r1 = lc_agent({"q": "alpha topic"})
    assert r1 == "lc-answer: write about lc-data for alpha topic"

    r2 = lc_agent({"q": "beta topic"})
    assert r2 == "lc-answer: write about lc-data for beta topic"
    # planner ran once (record only); drafter ran twice (re-run for freshness)
    assert len(planner._real.calls) == 1
    assert len(drafter._real.calls) == 2
    assert "beta topic" in drafter._real.calls[-1]
