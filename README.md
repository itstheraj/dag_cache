# dagcache

**VCR cassettes for agent trajectories.** Watch an agent run, infer the DAG
of tool/LLM calls it made, freeze the good ones, and replay them on repeat
tasks — the LLM only fires for genuinely new paths.

Luigi inverts here: Luigi makes you *declare* the DAG up front. dagcache
*learns* the DAG from observed agent behavior and solidifies it once proven.

```
pip install dagcache        # zero runtime dependencies
```

## Quickstart

```python
import dagcache

@dagcache.tool(pure=True)            # read-only: skippable in frozen replay
def search_kb(query: str) -> str: ...

@dagcache.tool(pure=False)           # side effects: re-executed on replay
def send_reply(ticket_id: str, body: str) -> str: ...

@dagcache.llm(planning=True)         # decides what to do: never re-executed
def plan(prompt: str) -> str: ...

@dagcache.llm                        # writes user-facing text: re-run fresh
def draft(prompt: str) -> str: ...

@dagcache.agent
def resolve_ticket(ticket: dict) -> str:
    plan(f"handle {ticket['title']}")
    article = search_kb(ticket["title"])
    order = fetch_order(ticket["order_id"])
    send_reply(ticket["id"], f"see {article}")
    return draft(f"article={article}; status={order['status']}")
```

- **Run 1** executes live and records the path `search_kb > fetch_order >
  send_reply` (+ the LLM nodes) into `.dagcache/store.db`.
- **Run 2** with a different ticket of the same *shape* replays the cached
  path: tools re-execute with bindings resolved against the new ticket
  (`order_id` comes from the new input, not the recording), the planning
  LLM never runs, and the final draft is re-generated with fresh data
  patched into its prompt.
- **Run 3**, if the world drifted (a tool returns a different shape, a
  binding can't resolve, a tool raises), the replay aborts and the live
  agent takes over automatically. You can never do worse than the status
  quo.

Try it: `python examples/ticket_agent.py`

## What gets cached, exactly

The cache key is **the chain, not the args**: task kind (the `@agent`
function) + structural fingerprint of the inputs (types and keys, never
values). The cached artifact is a **DAG skeleton** whose nodes are tool/LLM
calls with args stored as *bindings*:

| Binding | Meaning at replay |
|---|---|
| `input` | walked out of the new call's arguments (`ticket.title`) |
| `node`  | walked out of an upstream node's *fresh* output (`n1.order_id`) |
| `literal` | the LLM made this value up; reused as-is (with recorded→fresh substrings patched) |

If two different chains solve the same task shape, both are stored and
ranked by observed success (hits + recordings); `dagcache approve` pins the
canonical one. A staging DAG that keeps diverging dies after 3 fallbacks.

## Replay modes

- **verified** (default): re-execute tools with fresh args (real side
  effects, fresh data), skip planning LLM calls, re-run output LLM calls.
  Luigi's worker, where the DAG is the plan.
- **frozen**: execute nothing, return recorded outputs. Luigi's
  `output().exists()` across the whole graph — VCR mode for tests/CI.

```python
dagcache.configure(db_path=".dagcache/store.db", replay_mode="frozen")
# or env: DAGCACHE_MODE=record|off, DAGCACHE_REPLAY=verified|frozen, DAGCACHE_DB=...
```

Same-shaped but semantically different tasks need a discriminator — its
*value* joins the fingerprint:

```python
@dagcache.agent(key=lambda ticket: ticket["category"])
```

## CLI

```
dagcache ls                    # id, status, recordings, hits, fallbacks, path
dagcache show 3                # full DAG JSON
dagcache approve 3             # mark canonical (survives fallback storms)
dagcache demote 3              # back to staging
dagcache diff 3 7              # unified diff of two candidate paths
dagcache export 3 -o c.json    # cassette for code review
dagcache import c.json
dagcache prune --status dead --older-than 30
```

## Framework integrations

**OpenAI** (duck-typed — works with any `.chat.completions.create` client):

```python
from dagcache.adapters.openai import wrap_client
client = wrap_client(OpenAI())   # role="auto"|"planning"|"output"
```

Responses containing `tool_calls` are recorded as planning nodes (never
re-executed); text responses as output nodes (re-executed at verified
replay with prompts patched to fresh data).

**LangChain** (duck-typed — no langchain import required):

```python
from dagcache.adapters.langchain import wrap_tools, wrap_llm
agent = build_agent(llm=wrap_llm(ChatOpenAI(...)), tools=wrap_tools([...]))
```

## Honest limitations

- **Fuzzy matching is off by default.** Matching is exact on (task kind,
  input shape, key). Semantic/embedding matching is deliberately not in
  v0.1 — a fuzzy hit replaying an *effectful* path is how the cache refunds
  the wrong customer.
- **Side effects are real.** Verified replay re-executes effectful tools.
  That's automation, not caching. Mark them `pure=False` and keep fuzzy
  matching off.
- **Synthesized literals can go stale** if they embed values that came
  neither from inputs nor tool outputs (e.g. a date the LLM invented).
  They replay verbatim; divergence checks only cover output *shape*.
- **Prompt patching is heuristic**: recorded→fresh value substitution with
  word-boundary matching. Drift beyond shape (e.g. a price that changed but
  still looks like a price) is not detected.
- **The agent function must be replayable-shaped**: its return value should
  be its final LLM/tool call's result. Custom post-processing after the
  last call won't run at replay.
- Cache poisoning is a thing: scope your store per principal if agent
  inputs are attacker-controlled, and `approve` paths before production.

## Ruby

A RubyLLM plugin on the same mental model lives in the companion repo
**`ruby_llm-dagcache`** — YAML cassettes, `DagCache.watch(agent)`,
automatic `RubyLLM::Tool` instrumentation.

## Prior art

- [Agentic Plan Caching (arXiv 2506.14852)](https://arxiv.org/abs/2506.14852)
  — research prototype; still spends an LLM call adapting templates.
- [Agent Workflow Memory (arXiv 2409.07429)](https://arxiv.org/abs/2409.07429)
  — injects past workflows into prompts; doesn't skip the agent.
- GPTCache / LangChain caches — single LLM-call semantic caching.
- LangGraph checkpointing / Temporal — resumption, not cross-task reuse.
- vcrpy / VCR — the right mental model, at the wrong layer.
