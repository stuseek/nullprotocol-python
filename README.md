# NullProtocol for Python

Run focused AI tasks with saved instructions, context and Knowledge. Create an agent in the [dashboard](https://app.nullprotocol.ai), then use it from your Python backend.

Python 3.10+. Both synchronous and asynchronous clients are included. This SDK calls NullProtocol-hosted agents; its model operations and History are handled by the existing HTTP API.

The package is being prepared for its first release. From this checkout:

```sh
python -m pip install .
```

## First request

1. Create an agent in the dashboard. Choose a model and write its instructions.
2. An Owner creates a Space SDK key under **Connect**. Keep it on your backend.
3. Set `NULLPROTOCOL_KEY` and replace `support` below with your agent ID.

```python
from nullprotocol import NullProtocol

with NullProtocol() as client:
    agent = client.load("support")
    result = agent.extract(
        "Order 42: two blue mugs.",
        {"orderId": "number", "quantity": "number", "item": "string"},
    )
    if result.success:
        print(result.data)
        print(result.usage)
        print(result.operation_id)
    else:
        print(result.error_code, result.error)
```

The same Space key loads the agent and authorizes its hosted model. The next call picks up saved instruction/model/context changes. The operation appears in History. A team needs enabled model access and available credit.

You can also pass `key=...` and `endpoint=...` explicitly. Environment defaults are `NULLPROTOCOL_KEY` and `NULLPROTOCOL_API_URL`. The production endpoint is `https://api.nullprotocol.ai`. `timeout=30.0` is in **seconds**. Use the context manager or call `close()` to release the HTTP pool.

## Four operations

```python
fields = agent.extract(text, {"orderId": "number"})
summary = agent.summarize(text, max_length=300, focus="next step")
review = agent.validate("Every claim must follow the policy.", draft, reference=policy)
decision = agent.decide(
    text,
    [
        {"action": "request_photo", "description": "Ask for a photo of the damage."},
        {"action": "human_review", "description": "Ask a person to review the case."},
    ],
)
```

These snippets use an agent loaded as above and your own `text`, `draft` and `policy` variables. A successful extraction exposes `data`; summarization exposes `summary` and `key_points`; validation exposes `recommendation`, `score`, `reasoning`; a decision exposes `action`, `parameters`, `reasoning`.

Always check `result.success`. A completed validation with recommendation `fail` still has `success=True`: the model operation succeeded, the claim did not pass. Keep business rules in code. The schema checks output structure, not truth.

The Python summary limit defaults to 200 characters to match the JavaScript SDK; it is sent explicitly to the HTTP API. Invalid output gets the server's normal repair attempt. This client does not add its own generation or repair loop.

## Async applications

```python
import asyncio
from nullprotocol import AsyncNullProtocol


async def main():
    async with AsyncNullProtocol() as client:
        agent = await client.load("support")
        result = await agent.summarize("The buyer received a broken mug.")
        print(result.summary if result.success else result.error)


asyncio.run(main())
```

Methods that perform network I/O are awaited. `register_action()` and `with_scope()` are local and do not need `await`. Async handlers and guards are supported by `AsyncNullProtocol`. A sync client requires sync handlers. Use the async client in an async web server; sync callbacks in an async action must not block its event loop.

## Context, notes and scopes

```python
order = agent.with_scope("order:42")
order.context.set("delivery", {"delivered": True}, inclusion="selected")
entry = order.context.get("delivery")
result = order.extract(
    "Read delivery status.",
    {"delivered": "boolean"},
    context_keys=["delivery"],
)
note = order.memory.add("The buyer supplied a packaging photo.")
notes = order.memory.list()
order.memory.delete(note["id"])
order.context.delete("delivery")
```

`scope` is mapped to the API's `conversation` field. It can identify an order, user, project or job. IDs are flat: `user:42:session:a` does not inherit `user:42`. A scope selects data, not permissions: authorize the end user in your backend. Without a scope, notes are agent-wide; with one, notes belong to that ID and include agent-wide facts. Nothing is remembered automatically.

`agent.space.set/get/list/delete` manages shared Space context; `agent.context` selects the current agent/scope. `always` is the default inclusion. A narrower value replaces the whole value under the same key; fields are not merged. `context.set(..., if_version=None)` creates only, `if_version=entry["version"]` guards replacement, and omitting the parameter overwrites.

State-management results such as notes, settings and context entries are dictionaries preserving the server's JSON field names (`agentId`, `disabledActions`, etc.). Model-operation result attributes use Python `snake_case`; keys inside extracted `data` and action `parameters` remain exactly as specified by your application.

## Knowledge

```python
result = agent.extract(
    "How many days is the return window?",
    {"days": "number"},
    knowledge=True,
)
result = agent.validate(
    "The statement must agree with the policy.",
    "Personalised goods can always be returned.",
    knowledge={"query": "Return exceptions for personalised goods"},
)
```

Add/prepare documents and connect Space sources in the dashboard first. All four operations accept `knowledge=True` or `{"query": "..."}`. Omitted/False means no search. Scope IDs do not narrow Knowledge connections. `result.knowledge` contains the actual source selection, and `result.knowledge_usage` tracks search embeddings separately from generation `result.usage`.

Reference texts under Context are included automatically and in full. Python uses the HTTP API's default 6000-character reference-text budget; the run endpoint does not accept the JavaScript SDK's `documentBudget` option. Longer sources belong in Knowledge. The complete request must still fit the model's window.

## Local actions

```python
from nullprotocol import ActionContext


def send_message(parameters, context: ActionContext):
    print(parameters["text"])  # Replace with your delivery API.
    return {"sent": True}


agent.register_action(
    "send_message",
    send_message,
    description="Send a helpful reply to the customer.",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string", "minLength": 1}},
        "required": ["text"],
        "additionalProperties": False,
    },
    guard=lambda parameters, decision: len(parameters["text"]) <= 1000,
)
decision = agent.decide("The customer says: Good morning!", ["send_message"])
if decision.success:
    execution = agent.execute(decision)
    print(execution.success, execution.outcome, execution.history_error)
```

Handlers receive `(parameters, ActionContext)`. `context.input` is the original task; `context.decision` is a copy of its decision snapshot. The SDK supplies registered parameter schemas as structured data in the decision's `context` because the HTTP endpoint only accepts action names and descriptions. The original task and schema contract are visible in History. Automatic Knowledge queries still use the original task, not that schema wrapper.

Before a handler runs, `execute` rereads pause/disabled settings, validates its parameters against the registered JSON Schema and checks the optional guard. A guard must return exactly `True`. Changing the returned result or values in a guard does not change the checked execution snapshot. Only an original successful decision from that agent object can be executed; a JSON/HTTP response loaded elsewhere is not executable. There is no implicit `chat` action or automatic tool loop.

Decisions are consumed once by this object, including failed/refused executions. Handlers are never automatically retried. Cancellation can leave an external effect’s outcome unknown; it does not make the consumed decision safe to execute again. HTTP model-call idempotency is **not** durable idempotency for your external action: use your own stable business-operation ID/transaction before sending messages, issuing refunds or making other writes, including across processes/restarts. `execute` records its outcome on the decision's History operation. Recording failure is exposed as `execution.history_error` and does not undo an action that already ran. A local handler makes no second model call; this first Python release does not include the JavaScript handler's `reply()` helper.

Action schemas use JSON Schema 2020-12, local references and the formats available to the installed `jsonschema` FormatChecker. Unsupported formats/remote references are rejected at registration. `decide` without a registered handler is supported when you only want a payload; validate that payload before using it yourself.

## Errors, usage and retries

- `result.success=False`: the operation failed or was refused. Read `error_code` and `error`.
- `APIError`: the HTTP API refused the request. Read `status_code`, `code`, optional `retry_after`, and `request_id` for a run.
- `TransportError`: no response was received; the operation may already have completed. Keep `request_id`.
- `ProtocolError`: an unexpected response shape, including malformed usage. Do not treat it as a successful task.
- `ValueError`: invalid local configuration or input detected before dispatch.
- `result.history_error`: a usable result may exist even if its History record failed. Preserve the result.

`usage` is `None` when generation was not reported; it is not fabricated as zero. Known failure usage is retained. `usage.uncounted_requests` indicates partial reporting. Search usage stays separate in `knowledge_usage`.

The client generates one idempotency key per operation and preserves the exact body/key across safe transport retries. `max_retries=2` is the default. You can provide your own stable event ID:

```python
result = agent.extract(text, schema, idempotency_key="event-42-extract-v1")
```

Run timeouts, `429 rate_limited`, `409 in_progress` and `503 busy` are retried within the configured limit. Numeric Retry-After delays up to 30 seconds are honored; a longer or unsupported delay is returned to your application rather than ignored. Stored configuration failures, 500 errors, and unknown outcomes are not retried automatically. GETs may retry safely. Agent creation, context writes, note creation, settings writes and action recording are not automatically replayed after an uncertain write.

A completed HTTP response is cached **even if it is a failure**. After fixing a definite refusal (for example paused agent), deliberately choose a new attempt ID. After an unknown outcome, inspect History first. Do not generate a fresh ID automatically after a timeout.

## Create or edit an agent in Python

```python
agent = client.create(
    agent_id="invoice-reader",
    name="Invoice reader",
    model="your-enabled-model-alias",
    instructions="Read only supplied facts. Do not invent missing values.",
)
settings = agent.settings()
agent.update(instructions="Use null for missing values.", if_revision=settings["revision"])
agent.update(paused=True)
```

Use a model alias enabled for the team. `with_scope()` creates a new binding with copied action registrations, not the previous object's executable decisions. The bound objects share their client's HTTP pool; closing the client closes SDK access for all of them. When you supply your own `httpx.Client`/`AsyncClient` as `http_client`, you own and close that HTTP client yourself.

Standalone Ollama/OpenAI/Anthropic providers are available in the JavaScript SDK; this Python release is for saved hosted agents. It has no browser key storage, automatic memory extraction, payment flow or hosted workflow engine.

## Examples and development

Examples in `examples/` use your backend environment key and an existing `support` agent. They make real hosted requests when run. The scoped-data example explicitly removes only its own note/key; it uses a fixed demonstration scope, so use a disposable test agent when trying it.

```sh
python -m pip install -e '.[dev]'
pytest
ruff check .
ruff format --check .
mypy src
```

Tests use a scripted HTTP fixture, not paid providers. The workspace integration proof also runs this client against the actual API/gateway and an isolated PostgreSQL database with scripted generation/embeddings. These checks verify integration, not model accuracy.
