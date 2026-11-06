from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError

import httpx
import pytest

from nullprotocol import (
    APIError,
    AsyncNullProtocol,
    NullProtocol,
    ProtocolError,
    TransportError,
)

SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


def client(api, **options):
    return NullProtocol(
        key="test-space-key",
        http_client=httpx.Client(transport=httpx.MockTransport(api)),
        **options,
    )


def test_sync_four_operations_and_usage(api):
    with client(api) as c:
        agent = c.load("support")
        extracted = agent.extract("30 days", {"days": "number"})
        assert extracted.success and extracted.data == {"days": 30}
        assert extracted.usage.input_tokens == 100
        assert extracted.usage.output_tokens == 20
        assert extracted.operation_id and extracted.request_id
        assert agent.summarize("A mug is broken.").summary
        validated = agent.validate("Follow policy.", "Refund gift cards.", reference=None)
        assert validated.success and validated.recommendation == "fail"
        assert agent.decide("A greeting", ["send_message"]).action == "send_message"
    bodies = [json.loads(r.content) for r in api.requests if r.url.path.endswith("/run")]
    assert bodies[1]["maxLength"] == 200
    assert bodies[2]["reference"] is None
    assert (
        len({r.headers["Idempotency-Key"] for r in api.requests if r.url.path.endswith("/run")})
        == 4
    )


def test_explicit_scope_context_memory_shared_and_versions(api):
    with client(api) as c:
        agent = c.load("support", scope="order:42")
        entry = agent.context.set("order", {"id": 42}, inclusion="selected", if_version=None)
        assert agent.context.get("order")["value"] == {"id": 42}
        assert len(agent.context.list()) == 1
        agent.context.set("order", {"id": 42}, if_version=entry["version"])
        agent.space.set("shop", {"currency": "EUR"})
        note = agent.memory.add("Packaging photo supplied.")
        assert agent.memory.list()[0]["text"] == "Packaging photo supplied."
        agent.memory.delete(note["id"])
        agent.extract("Read order", {"id": "number"}, context_keys=["order"])
        agent.context.delete("order")
    for req in api.requests:
        if "/agents/support/context" in req.url.path:
            assert req.url.params["conversation"] == "order:42"
        if req.url.path == "/v1/context/shop":
            assert "conversation" not in req.url.params
    put = [json.loads(r.content) for r in api.requests if r.method == "PUT"]
    assert put[0]["ifVersion"] is None
    assert put[1]["ifVersion"] == entry["version"]
    assert "ifVersion" not in put[2]


def test_create_update_and_flat_scope(api):
    with client(api) as c:
        agent = c.create(model="ministral-14b", name="Support", instructions="Read facts.")
        assert agent.agent_id == "issued-id"
        agent.update(instructions="", paused=False, disabled_actions=[], if_revision=0)
        changed = api.settings
        assert changed["instructions"] == "" and changed["paused"] is False
        assert changed["disabledActions"] == [] and changed["ifRevision"] == 0
        scoped = agent.with_scope("project:alpha")
        assert scoped.scope == "project:alpha" and agent.scope is None
        assert scoped.with_scope(None).scope is None
        with pytest.raises(ValueError):
            agent.update(if_revision=1)


def test_operation_failures_preserve_partial_usage_and_unknown(api):
    with client(api) as c:
        agent = c.load("support")
        api.result_override = {
            "success": False,
            "errorCode": "invalid_reply",
            "error": "Bad output",
            "usage": {"inputTokens": 101, "outputTokens": 8, "uncountedRequests": 1},
            "historyError": "storage unavailable",
        }
        result = agent.extract("x", {})
        assert not result.success and result.error_code == "invalid_reply"
        assert result.usage.uncounted_requests == 1
        assert result.history_error == "storage unavailable"
        api.result_override = {
            "success": False,
            "errorCode": "provider_error",
            "error": "Unknown outcome",
        }
        assert agent.extract("x", {}).usage is None
        api.result_override = {
            "success": False,
            "errorCode": "documents_too_large",
            "error": "Too large",
        }
        refused = agent.extract("x", {})
        assert refused.usage is None and refused.attempts is None


def test_knowledge_metadata_and_forward_fields(api):
    with client(api) as c:
        agent = c.load("support")
        api.result_override = {
            "success": True,
            "data": {"days": 30},
            "knowledgeUsage": {"inputTokens": 12},
            "knowledge": {"id": "selection", "sources": []},
            "futureField": {"retained": True},
        }
        result = agent.extract(
            "Return window", {"days": "number"}, knowledge={"query": " returns "}
        )
        assert result.knowledge_usage.input_tokens == 12
        assert result.usage is None
        assert result.knowledge["id"] == "selection"
        assert result.raw["futureField"] == {"retained": True}
        result.raw["futureField"]["retained"] = False
        assert result.raw["futureField"]["retained"] is True
        assert json.loads(api.requests[-1].content)["knowledge"] == {"query": "returns"}


def test_retry_run_timeout_reuses_id_and_body_without_duplicate_model_work(api, monkeypatch):
    monkeypatch.setattr("nullprotocol._transport.time.sleep", lambda _: None)
    seen = []

    def fault(req):
        if not req.url.path.endswith("/run"):
            return None
        seen.append((req.headers["Idempotency-Key"], req.content))
        if len(seen) == 1:
            api.override = None
            api(req)  # First request completed, but its response is lost.
            api.override = fault
            raise httpx.ReadTimeout("response lost")
        return None

    api.override = fault
    with client(api) as c:
        result = c.load("support").extract("x", {}, idempotency_key="event-42-read")
    assert result.success and seen[0] == seen[1]
    assert result.request_id == "event-42-read" and api.model_calls == 1


@pytest.mark.parametrize(
    "status,code",
    [
        (503, "inference_unavailable"),
        (409, "request_outcome_unknown"),
        (409, "idempotency_mismatch"),
        (500, "internal_error"),
        (401, "unauthorized"),
    ],
)
def test_stored_failures_and_unknown_outcomes_not_retried(api, status, code):
    attempts = []

    def fault(req):
        if req.url.path.endswith("/run"):
            attempts.append(req)
            return httpx.Response(status, json={"error": code})

    api.override = fault
    with client(api) as c, pytest.raises(APIError) as err:
        c.load("support").extract("x", {}, idempotency_key="event-42")
    assert err.value.code == code and err.value.request_id == "event-42"
    assert len(attempts) == 1


@pytest.mark.parametrize(
    "status,code", [(429, "rate_limited"), (503, "busy"), (409, "in_progress")]
)
def test_transient_retry_respects_stable_id(api, monkeypatch, status, code):
    delays = []
    monkeypatch.setattr("nullprotocol._transport.time.sleep", delays.append)
    attempts = []

    def fault(req):
        if req.url.path.endswith("/run"):
            attempts.append(req)
            if len(attempts) == 1:
                return httpx.Response(status, json={"error": code}, headers={"Retry-After": "0"})

    api.override = fault
    with client(api) as c:
        assert c.load("support").extract("x", {}).success
    assert delays == [0] and attempts[0].content == attempts[1].content
    assert attempts[0].headers["Idempotency-Key"] == attempts[1].headers["Idempotency-Key"]


@pytest.mark.parametrize("retry_after", ["31", "NaN", "Wed, 21 Oct 2026 07:28:00 GMT"])
def test_does_not_ignore_long_or_unsupported_retry_after(api, retry_after):
    attempts = []

    def fault(req):
        if req.url.path.endswith("/run"):
            attempts.append(req)
            return httpx.Response(
                429, json={"error": "rate_limited"}, headers={"Retry-After": retry_after}
            )

    api.override = fault
    with client(api) as c, pytest.raises(APIError) as err:
        c.load("support").extract("x", {})
    assert err.value.retry_after == retry_after
    assert len(attempts) == 1


def test_timeout_error_exposes_id_but_never_key(api, monkeypatch):
    monkeypatch.setattr("nullprotocol._transport.time.sleep", lambda _: None)

    def fault(req):
        if req.url.path.endswith("/run"):
            raise httpx.ConnectError("unreachable")

    api.override = fault
    with client(api) as c, pytest.raises(TransportError) as err:
        c.load("support").summarize("x", idempotency_key="retry-this-id")
    assert err.value.request_id == "retry-this-id"
    assert "test-space-key" not in str(err.value)


def test_writes_without_idempotency_are_not_retried(api):
    writes = []

    def fault(req):
        if req.method == "PUT":
            writes.append(req)
            raise httpx.ReadTimeout("unknown write")

    api.override = fault
    with client(api) as c, pytest.raises(TransportError):
        c.load("support").context.set("policy", "30 days")
    assert len(writes) == 1


def test_no_following_redirect_and_non_json_error_status(api):
    def fault(req):
        if req.url.path.endswith("/run"):
            return httpx.Response(302, headers={"Location": "https://other.example/secret"})

    api.override = fault
    with client(api) as c, pytest.raises(APIError) as err:
        c.load("support").extract("x", {})
    assert err.value.status_code == 302
    assert all(r.url.host == "api.nullprotocol.ai" for r in api.requests)
    api.override = lambda req: httpx.Response(502, text="<html>bad gateway</html>")
    with client(api) as c, pytest.raises(APIError) as err:
        c.load("support")
    assert err.value.status_code == 502


@pytest.mark.parametrize(
    "result",
    [
        None,
        [],
        {"success": True, "usage": {"inputTokens": 1}},
        {"success": False},
        {"success": 1},
        {"success": True, "summary": 42},
    ],
)
def test_malformed_results_are_not_success(api, result):
    api.result_override = result
    if result is None:
        api.override = lambda req: (
            httpx.Response(200, json={"result": None}) if req.url.path.endswith("/run") else None
        )
    with client(api) as c, pytest.raises(ProtocolError) as err:
        c.load("support").summarize("x")
    assert err.value.request_id


@pytest.mark.parametrize("scope", ["", "x\n", "x" * 129, "😀" * 65])
def test_invalid_scope_rejected_before_network(api, scope):
    with client(api) as c, pytest.raises(ValueError):
        c.load("support", scope=scope)
    assert api.requests == []


@pytest.mark.parametrize("agent_id", ["../other", "/support", "SUPPORT", "x?z=1"])
def test_invalid_agent_id_rejected_before_network(api, agent_id):
    with client(api) as c, pytest.raises(ValueError):
        c.load(agent_id)
    assert api.requests == []


@pytest.mark.parametrize("key", ["has spaces", "", "x\n", "x" * 129])
def test_invalid_request_id_does_not_call_model(api, key):
    with client(api) as c, pytest.raises(ValueError):
        c.load("support").extract("x", {}, idempotency_key=key)
    assert api.model_calls == 0


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), b"bytes", {"bad": object()}, "\ud800"]
)
def test_non_json_data_rejected_locally(api, value):
    with client(api) as c, pytest.raises(ValueError):
        c.load("support").extract(value, {})
    assert api.model_calls == 0


def test_body_limit_and_unicode_scope(api):
    with client(api) as c:
        agent = c.load("support", scope="😀" * 64)
        with pytest.raises(ValueError):
            agent.extract("x" * 131072, {})
        assert agent.extract("x", {}).success
        assert json.loads(api.requests[-1].content)["conversation"] == "😀" * 64


def test_closed_and_external_client_ownership(api):
    external = httpx.Client(transport=httpx.MockTransport(api))
    c = NullProtocol(key="test", http_client=external)
    agent = c.load("support")
    c.close()
    assert not external.is_closed
    with pytest.raises(TransportError):
        agent.extract("x", {})
    external.close()


def test_environment_and_endpoint_normalization(api, monkeypatch):
    monkeypatch.setenv("NULLPROTOCOL_KEY", "env-test-key")
    external = httpx.Client(transport=httpx.MockTransport(api))
    with NullProtocol(endpoint="http://localhost:4010/v1/", http_client=external) as c:
        c.load("support")
    assert api.requests[0].url.path == "/v1/agents/support/settings"
    assert api.requests[0].headers["Authorization"] == "Bearer env-test-key"


@pytest.mark.parametrize(
    "options",
    [
        dict(key=""),
        dict(key="x\n"),
        dict(endpoint="https://user:pass@example.com"),
        dict(endpoint="ftp://example.com"),
        dict(endpoint="https://example.com/other"),
        dict(timeout=0),
        dict(timeout=float("nan")),
        dict(max_retries=-1),
        dict(max_retries=True),
    ],
)
def test_bad_setup(options):
    with pytest.raises(ValueError):
        NullProtocol(**({"key": "test"} | options))


def test_decide_execute_schema_guard_snapshot_and_recording(api):
    delivered = []
    with client(api) as c:
        agent = c.load("support")
        agent.register_action(
            "send_message",
            lambda p, ctx: delivered.append((p["text"], ctx.input)) or {"sent": True},
            input_schema=SCHEMA,
            guard=lambda p, d: True,
        )
        decision = agent.decide("Good morning", ["send_message"], knowledge=True)
        posted = json.loads(api.requests[-1].content)
        assert posted["knowledge"] == {"query": "Good morning"}
        assert posted["context"]["actionParameterSchemas"]["send_message"] == SCHEMA
        decision.parameters["text"] = "tampered"
        outcome = agent.execute(decision)
        assert outcome.success and outcome.result == {"sent": True}
        assert delivered == [("Good morning!", "Good morning")]
        assert api.actions[0]["input"] == {"text": "Good morning!"}
        assert api.actions[0]["modelCalls"] == 0
        assert agent.execute(decision).error_code == "already_executed"
        replay = agent.decide(
            "Good morning", ["send_message"], knowledge=True, idempotency_key=decision.request_id
        )
        assert agent.execute(replay).error_code == "already_executed"
        assert len(delivered) == 1
        with pytest.raises(FrozenInstanceError):
            decision.success = False


@pytest.mark.parametrize(
    "mode,code",
    [
        ("schema", "invalid_parameters"),
        ("guard", "guard_rejected"),
        ("guard_raise", "guard_error"),
        ("disabled", "action_disabled"),
        ("paused", "agent_paused"),
        ("unregistered", "action_not_registered"),
        ("handler_raise", "handler_failed"),
    ],
)
def test_execution_refusals_and_failures(api, mode, code):
    delivered = []

    def guard(p, d):
        if mode == "guard_raise":
            raise RuntimeError("guard crashed")
        return mode != "guard"

    def handler(p, ctx):
        if mode == "handler_raise":
            raise RuntimeError("handler crashed")
        delivered.append(p)

    with client(api) as c:
        agent = c.load("support")
        if mode != "unregistered":
            agent.register_action("send_message", handler, input_schema=SCHEMA, guard=guard)
        if mode == "schema":
            api.result_override = {
                "success": True,
                "action": "send_message",
                "parameters": {"wrong": 1},
                "reasoning": "x",
            }
        result = agent.decide("Hello", ["send_message"])
        if mode == "paused":
            api.settings["paused"] = True
        if mode == "disabled":
            api.settings["disabledActions"] = ["send_message"]
        outcome = agent.execute(result)
        assert not outcome.success and outcome.error_code == code
        assert delivered == []
        assert api.actions[-1]["outcome"] == ("failed" if mode == "handler_raise" else "refused")
        assert agent.execute(result).error_code == "already_executed"


def test_guard_cannot_mutate_the_checked_execution_snapshot(api):
    sent = []

    def guard(params, decision):
        params["text"] = "changed"
        decision.parameters["text"] = "also changed"
        return True

    with client(api) as c:
        agent = c.load("support")
        agent.register_action(
            "send_message", lambda p, ctx: sent.append(p["text"]), input_schema=SCHEMA, guard=guard
        )
        assert agent.execute(agent.decide("Hello", ["send_message"])).success
    assert sent == ["Good morning!"]


def test_foreign_or_serialized_decision_and_history_error(api):
    with client(api) as c:
        one, other = c.load("support"), c.load("support")
        one.register_action("send_message", lambda p, ctx: {"sent": True})
        decision = one.decide("Hello", ["send_message"])
        assert other.execute(decision).error_code == "not_a_decision"
        assert one.execute(decision.raw).error_code == "not_a_decision"
        api.override = lambda req: (
            httpx.Response(503, json={"error": "busy"})
            if req.url.path.endswith("/actions")
            else None
        )
        outcome = one.execute(decision)
        assert outcome.success and outcome.history_error
        assert one.execute(decision).error_code == "already_executed"


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "format": "made-up"},
        {"$ref": "https://example.com/schema"},
        {"$dynamicRef": "https://example.com/schema"},
    ],
)
def test_unsupported_action_schema_rejected_without_network(api, schema):
    with client(api) as c:
        agent = c.load("support")
        with pytest.raises(ValueError):
            agent.register_action("send_message", lambda p, ctx: None, input_schema=schema)
    assert api.model_calls == 0


def test_async_operations_scopes_and_handlers(api):
    async def run():
        external = httpx.AsyncClient(transport=httpx.MockTransport(api))
        async with AsyncNullProtocol(key="test", http_client=external) as c:
            agent = await c.load("support", scope="order:42")
            assert (await agent.extract("x", {})).success
            assert (await agent.summarize("x")).summary
            assert (await agent.validate("x", "y")).recommendation == "fail"
            note = await agent.memory.add("Fact")
            assert await agent.memory.list()
            await agent.memory.delete(note["id"])
            entry = await agent.context.set("order", {"id": 42})
            assert (await agent.context.get("order"))["version"] == entry["version"]
            await agent.context.delete("order")
            sent = []

            async def handler(p, ctx):
                await asyncio.sleep(0)
                sent.append(p["text"])
                return {"sent": True}

            async def guard(p, d):
                return True

            agent.register_action("send_message", handler, guard=guard, input_schema=SCHEMA)
            decision = await agent.decide("Hello", ["send_message"])
            executions = await asyncio.gather(agent.execute(decision), agent.execute(decision))
            assert sum(e.success for e in executions) == 1
            assert len(sent) == 1
            results = await asyncio.gather(*(agent.extract("x", {}) for _ in range(20)))
            assert len({r.request_id for r in results}) == 20
        assert not external.is_closed
        await external.aclose()

    asyncio.run(run())


def test_async_transport_timeout_and_cancellation(api, monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr("nullprotocol._transport.asyncio.sleep", no_sleep)
    calls = []

    def fault(req):
        if req.url.path.endswith("/run"):
            calls.append(req)
            raise httpx.ReadTimeout("lost response")

    api.override = fault

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as external:
            async with AsyncNullProtocol(key="test", http_client=external) as c:
                agent = await c.load("support")
                with pytest.raises(TransportError) as err:
                    await agent.extract("x", {}, idempotency_key="async-event")
                assert err.value.request_id == "async-event"
        assert len(calls) == 3
        assert len({r.content for r in calls}) == 1

    asyncio.run(run())


def test_action_schema_property_named_format_is_data(api):
    with client(api) as c:
        agent = c.load("support")
        agent.register_action(
            "send_message",
            lambda p, ctx: None,
            input_schema={
                "type": "object",
                "properties": {"format": {"type": "string"}, "$ref": {"type": "string"}},
                "default": {"format": "not-a-schema-format"},
            },
        )


@pytest.mark.parametrize(
    "metadata",
    [
        dict(attempts=True),
        dict(repaired="yes"),
        dict(historyError=42),
        dict(knowledge=[]),
        dict(usage={"inputTokens": True, "outputTokens": 0}),
    ],
)
def test_invalid_metadata_is_not_coerced(api, metadata):
    api.result_override = {"success": True, "data": {"days": 30}, **metadata}
    with client(api) as c, pytest.raises(ProtocolError):
        c.load("support").extract("x", {})


def test_action_format_validation_and_local_references(api):
    schema = {
        "type": "object",
        "$defs": {"date": {"type": "string", "format": "date"}},
        "properties": {"date": {"$ref": "#/$defs/date"}},
        "required": ["date"],
    }
    with client(api) as c:
        agent = c.load("support")
        agent.register_action("send_message", lambda p, ctx: None, input_schema=schema)
        api.result_override = {
            "success": True,
            "action": "send_message",
            "parameters": {"date": "2026-02-29"},
            "reasoning": "x",
        }
        assert agent.execute(agent.decide("x", ["send_message"])).error_code == "invalid_parameters"
        api.result_override = {
            "success": True,
            "action": "send_message",
            "parameters": {"date": "2026-02-28"},
            "reasoning": "x",
        }
        assert agent.execute(agent.decide("x", ["send_message"])).success


def test_sync_client_rejects_async_guard_or_handler(api):
    async def guard(p, d):
        return True

    async def handler(p, ctx):
        return {"sent": True}

    with client(api) as c:
        agent = c.load("support")
        agent.register_action("send_message", lambda p, ctx: None, guard=guard)
        assert agent.execute(agent.decide("x", ["send_message"])).error_code == "guard_error"
        agent.register_action("send_message", handler)
        assert agent.execute(agent.decide("x", ["send_message"])).error_code == "handler_failed"


def test_async_cancellation_is_not_converted_to_retry(api):
    async def run():
        started = asyncio.Event()
        calls = []

        async def http(request):
            if request.url.path.endswith("/run"):
                calls.append(request)
                started.set()
                await asyncio.Event().wait()
            return api(request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as external:
            async with AsyncNullProtocol(key="test", http_client=external) as c:
                agent = await c.load("support")
                pending = asyncio.create_task(
                    agent.extract("x", {}, idempotency_key="cancelled-event")
                )
                await started.wait()
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                assert len(calls) == 1
                assert calls[0].headers["Idempotency-Key"] == "cancelled-event"

    asyncio.run(run())


def test_unserializable_handler_result_does_not_repeat_side_effect(api):
    writes = []
    with client(api) as c:
        agent = c.load("support")
        agent.register_action("send_message", lambda p, ctx: writes.append(True) or object())
        decision = agent.decide("x", ["send_message"])
        result = agent.execute(decision)
        assert result.success and result.history_error
        assert agent.execute(decision).error_code == "already_executed"
    assert writes == [True]


def test_nonfinite_or_missing_response_metadata_is_protocol_error(api):
    with client(api) as c:
        agent = c.load("support")
        api.override = lambda req: (
            httpx.Response(200, text='{"result":{"success":true,"data":NaN},"operationId":null}')
            if req.url.path.endswith("/run")
            else None
        )
        with pytest.raises(ProtocolError):
            agent.extract("x", {})
        api.override = lambda req: (
            httpx.Response(200, json={"result": {"success": True, "data": {}}})
            if req.url.path.endswith("/run")
            else None
        )
        with pytest.raises(ProtocolError):
            agent.extract("x", {})
