from __future__ import annotations

import inspect
import time
from copy import deepcopy
from dataclasses import replace
from typing import Any

from ._common import (
    ActionChoice,
    AgentState,
    Knowledge,
    MadeDecision,
    RegisteredAction,
    object_field,
    operation_payload,
    summary_options,
    validation_options,
)
from ._errors import APIError, ProtocolError
from ._knowledge import AgentKnowledge, AsyncAgentKnowledge
from ._stores import (
    AsyncContextStore,
    AsyncMemoryStore,
    AsyncSpaceContextStore,
    ContextStore,
    MemoryStore,
    SpaceContextStore,
)
from ._transport import _UNSET, AsyncTransport, Transport, json_body
from ._types import ActionContext, Execution, Result


def _result(response: dict[str, Any], task_id: str, operation: str) -> Result[Any]:
    result = Result.from_response(response, task_id)
    if not result.success:
        return result
    raw = result.raw
    valid = (
        operation == "extract"
        and "data" in raw
        or operation == "summarize"
        and isinstance(result.summary, str)
        and isinstance(result.key_points, list)
        and all(isinstance(x, str) for x in result.key_points)
        or operation == "validate"
        and result.recommendation in ("pass", "fail", "conditional")
        and isinstance(result.score, (int, float))
        and not isinstance(result.score, bool)
        and 0 <= result.score <= 1
        and isinstance(result.reasoning, str)
        or operation == "decide"
        and isinstance(result.action, str)
        and isinstance(result.parameters, dict)
        and isinstance(result.reasoning, str)
    )
    if not valid:
        raise ProtocolError(
            f"Invalid {operation} result in the server response.", request_id=task_id
        )
    return result


def _knowledge_for_decision(
    context: Any,
    values: dict[str, Any],
    knowledge: Knowledge | None,
) -> Knowledge | None:
    if (
        knowledge is True
        and isinstance(values["context"], dict)
        and "actionParameterSchemas" in values["context"]
    ):
        return {"query": context if isinstance(context, str) else json_body(context).decode()}
    return knowledge


def _changes(values: dict[str, Any]) -> dict[str, Any]:
    mapping = {"disabled_actions": "disabledActions", "if_revision": "ifRevision"}
    body = {mapping.get(k, k): v for k, v in values.items() if v is not None}
    if not any(k != "ifRevision" for k in body):
        raise ValueError("Supply at least one setting to update.")
    return body


def _refused(made: MadeDecision | None, code: str, message: str) -> Execution:
    return Execution(
        False, "refused", made.snapshot.action if made else None, error_code=code, error=message
    )


def _view(result: Result[Any]) -> Result[Any]:
    return Result.from_response(
        {"result": result.raw, "operationId": result.operation_id}, result.request_id
    )


def _check_action(
    made: MadeDecision,
    action: RegisteredAction | None,
    settings: dict[str, Any],
) -> Execution | None:
    name = made.snapshot.action
    if settings.get("paused"):
        return _refused(made, "agent_paused", "This agent is paused. The handler did not run.")
    if name in settings.get("disabledActions", []):
        return _refused(
            made, "action_disabled", "This action is disabled. The handler did not run."
        )
    if action is None:
        return _refused(made, "action_not_registered", "No handler is registered for this action.")
    if action.validator is not None:
        problems = list(action.validator.iter_errors(made.snapshot.parameters))
        if problems:
            message = "; ".join(problem.message for problem in problems[:5])
            return _refused(made, "invalid_parameters", message)
    return None


def _record_body(made: MadeDecision, outcome: Execution, started: float) -> dict[str, Any]:
    body = {
        "name": made.snapshot.action,
        "input": made.snapshot.parameters,
        "outcome": outcome.outcome,
        "modelCalls": 0,
        "requestBytes": 0,
        "durationMs": min(86400000, max(0, round((time.monotonic() - started) * 1000))),
    }
    if outcome.success:
        body["result"] = outcome.result
    if outcome.error_code is not None:
        body["errorCode"] = outcome.error_code
    return body


class Agent(AgentState):
    """A saved hosted agent. Call methods synchronously; use AsyncAgent in async apps."""

    def __init__(self, transport: Transport, agent_id: str, scope: str | None = None) -> None:
        super().__init__(agent_id, scope)
        self._transport = transport
        self.context = ContextStore(transport, self._path + "/context", self.scope)
        self.space = SpaceContextStore(transport)
        self.knowledge = AgentKnowledge(transport, self.agent_id)
        self.memory = MemoryStore(transport, self._path + "/memory", self.scope)

    def with_scope(self, scope: str | None) -> Agent:
        """Bind another scope, copying action registrations but not decisions."""
        bound = Agent(self._transport, self.agent_id, scope)
        bound._actions = self._actions.copy()
        return bound

    def settings(self) -> dict[str, Any]:
        return object_field(self._transport.request("GET", self._path + "/settings"), "agent")

    def update(
        self,
        *,
        name: str | None = None,
        model: str | None = None,
        instructions: str | None = None,
        paused: bool | None = None,
        disabled_actions: list[str] | None = None,
        if_revision: int | None = None,
    ) -> dict[str, Any]:
        body = _changes(
            {
                "name": name,
                "model": model,
                "instructions": instructions,
                "paused": paused,
                "disabled_actions": disabled_actions,
                "if_revision": if_revision,
            }
        )
        return object_field(
            self._transport.request("PATCH", self._path + "/settings", body=body), "agent"
        )

    def _run(
        self,
        operation: str,
        values: dict[str, Any],
        knowledge: Knowledge | None,
        context_keys: list[str] | None,
        idempotency_key: str | None,
        images: list[str] | None = None,
    ) -> Result[Any]:
        task_id = self._task_id(idempotency_key)
        body = operation_payload(operation, values, self.scope, knowledge, context_keys, images)
        return _result(
            self._transport.request(
                "POST",
                self._path + "/run",
                body=body,
                task_id=task_id,
            ),
            task_id,
            operation,
        )

    def extract(
        self,
        data: Any,
        schema: dict[str, Any],
        *,
        images: list[str] | None = None,
        knowledge: Knowledge | None = None,
        context_keys: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> Result[Any]:
        return self._run(
            "extract",
            {"data": data, "schema": schema},
            knowledge,
            context_keys,
            idempotency_key,
            images,
        )

    def summarize(
        self,
        content: Any,
        *,
        max_length: int = 200,
        focus: str | None = None,
        images: list[str] | None = None,
        knowledge: Knowledge | None = None,
        context_keys: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> Result[Any]:
        return self._run(
            "summarize",
            summary_options(content, max_length, focus),
            knowledge,
            context_keys,
            idempotency_key,
            images,
        )

    def validate(
        self,
        criteria: str,
        subject: Any,
        *,
        reference: Any = _UNSET,
        images: list[str] | None = None,
        knowledge: Knowledge | None = None,
        context_keys: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> Result[Any]:
        return self._run(
            "validate",
            validation_options(criteria, subject, reference, _UNSET),
            knowledge,
            context_keys,
            idempotency_key,
            images,
        )

    def decide(
        self,
        context: Any,
        actions: list[ActionChoice],
        *,
        images: list[str] | None = None,
        knowledge: Knowledge | None = None,
        context_keys: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> Result[Any]:
        original = deepcopy(context)
        values = self._choices(original, actions)
        result = self._run(
            "decide",
            values,
            _knowledge_for_decision(original, values, knowledge),
            context_keys,
            idempotency_key,
            images,
        )
        return self._remember(result, original)

    def execute(self, decision: Result[Any]) -> Execution:
        made, problem = self._consume(decision)
        if problem:
            return _refused(
                made, problem, "Use an unconsumed successful decision from this agent object."
            )
        assert made is not None
        started = time.monotonic()
        action = self._actions.get(made.snapshot.action or "")
        try:
            outcome = _check_action(made, action, self.settings())
            if outcome is None and action is not None and action.guard is not None:
                allowed = action.guard(deepcopy(made.snapshot.parameters), _view(made.snapshot))
                if inspect.isawaitable(allowed):
                    if inspect.iscoroutine(allowed):
                        allowed.close()
                    outcome = _refused(
                        made, "guard_error", "Use AsyncNullProtocol for async guards."
                    )
                elif allowed is not True:
                    outcome = _refused(
                        made, "guard_rejected", "The action was rejected by its guard."
                    )
        except Exception as exc:
            outcome = _refused(
                made, exc.code if isinstance(exc, APIError) else "guard_error", str(exc)
            )
        if outcome is None:
            assert action is not None
            try:
                value = action.handler(
                    deepcopy(made.snapshot.parameters),
                    ActionContext(_view(made.snapshot), deepcopy(made.input)),
                )
                if inspect.isawaitable(value):
                    if inspect.iscoroutine(value):
                        value.close()
                    raise TypeError("Use AsyncNullProtocol for async handlers.")
                outcome = Execution(True, "completed", made.snapshot.action, result=value)
            except Exception as exc:
                outcome = Execution(
                    False,
                    "failed",
                    made.snapshot.action,
                    error_code="handler_failed",
                    error=str(exc),
                )
        try:
            if made.snapshot.operation_id is None:
                raise ProtocolError(
                    "The decision has no History operation ID; execution was not recorded."
                )
            self._transport.request(
                "POST",
                self._path + f"/operations/{made.snapshot.operation_id}/actions",
                body=_record_body(made, outcome, started),
            )
        except Exception as exc:
            outcome = replace(outcome, history_error=str(exc))
        return outcome


class AsyncAgent(AgentState):
    """Async counterpart of Agent. Network I/O uses httpx.AsyncClient."""

    def __init__(self, transport: AsyncTransport, agent_id: str, scope: str | None = None) -> None:
        super().__init__(agent_id, scope)
        self._transport = transport
        self.context = AsyncContextStore(transport, self._path + "/context", self.scope)
        self.space = AsyncSpaceContextStore(transport)
        self.knowledge = AsyncAgentKnowledge(transport, self.agent_id)
        self.memory = AsyncMemoryStore(transport, self._path + "/memory", self.scope)

    def with_scope(self, scope: str | None) -> AsyncAgent:
        bound = AsyncAgent(self._transport, self.agent_id, scope)
        bound._actions = self._actions.copy()
        return bound

    async def settings(self) -> dict[str, Any]:
        return object_field(await self._transport.request("GET", self._path + "/settings"), "agent")

    async def update(
        self,
        *,
        name: str | None = None,
        model: str | None = None,
        instructions: str | None = None,
        paused: bool | None = None,
        disabled_actions: list[str] | None = None,
        if_revision: int | None = None,
    ) -> dict[str, Any]:
        body = _changes(
            {
                "name": name,
                "model": model,
                "instructions": instructions,
                "paused": paused,
                "disabled_actions": disabled_actions,
                "if_revision": if_revision,
            }
        )
        return object_field(
            await self._transport.request("PATCH", self._path + "/settings", body=body), "agent"
        )

    async def _run(
        self,
        operation: str,
        values: dict[str, Any],
        knowledge: Knowledge | None,
        context_keys: list[str] | None,
        idempotency_key: str | None,
        images: list[str] | None = None,
    ) -> Result[Any]:
        task_id = self._task_id(idempotency_key)
        body = operation_payload(operation, values, self.scope, knowledge, context_keys, images)
        return _result(
            await self._transport.request(
                "POST",
                self._path + "/run",
                body=body,
                task_id=task_id,
            ),
            task_id,
            operation,
        )

    async def extract(
        self,
        data: Any,
        schema: dict[str, Any],
        *,
        images: list[str] | None = None,
        knowledge: Knowledge | None = None,
        context_keys: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> Result[Any]:
        return await self._run(
            "extract",
            {"data": data, "schema": schema},
            knowledge,
            context_keys,
            idempotency_key,
            images,
        )

    async def summarize(
        self,
        content: Any,
        *,
        max_length: int = 200,
        focus: str | None = None,
        images: list[str] | None = None,
        knowledge: Knowledge | None = None,
        context_keys: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> Result[Any]:
        return await self._run(
            "summarize",
            summary_options(content, max_length, focus),
            knowledge,
            context_keys,
            idempotency_key,
            images,
        )

    async def validate(
        self,
        criteria: str,
        subject: Any,
        *,
        reference: Any = _UNSET,
        images: list[str] | None = None,
        knowledge: Knowledge | None = None,
        context_keys: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> Result[Any]:
        return await self._run(
            "validate",
            validation_options(criteria, subject, reference, _UNSET),
            knowledge,
            context_keys,
            idempotency_key,
            images,
        )

    async def decide(
        self,
        context: Any,
        actions: list[ActionChoice],
        *,
        images: list[str] | None = None,
        knowledge: Knowledge | None = None,
        context_keys: list[str] | None = None,
        idempotency_key: str | None = None,
    ) -> Result[Any]:
        original = deepcopy(context)
        values = self._choices(original, actions)
        result = await self._run(
            "decide",
            values,
            _knowledge_for_decision(original, values, knowledge),
            context_keys,
            idempotency_key,
            images,
        )
        return self._remember(result, original)

    async def execute(self, decision: Result[Any]) -> Execution:
        made, problem = self._consume(decision)
        if problem:
            return _refused(
                made, problem, "Use an unconsumed successful decision from this agent object."
            )
        assert made is not None
        started = time.monotonic()
        action = self._actions.get(made.snapshot.action or "")
        try:
            outcome = _check_action(made, action, await self.settings())
            if outcome is None and action is not None and action.guard is not None:
                allowed = action.guard(deepcopy(made.snapshot.parameters), _view(made.snapshot))
                if inspect.isawaitable(allowed):
                    allowed = await allowed
                if allowed is not True:
                    outcome = _refused(
                        made, "guard_rejected", "The action was rejected by its guard."
                    )
        except Exception as exc:
            outcome = _refused(
                made, exc.code if isinstance(exc, APIError) else "guard_error", str(exc)
            )
        if outcome is None:
            assert action is not None
            try:
                value = action.handler(
                    deepcopy(made.snapshot.parameters),
                    ActionContext(_view(made.snapshot), deepcopy(made.input)),
                )
                if inspect.isawaitable(value):
                    value = await value
                outcome = Execution(True, "completed", made.snapshot.action, result=value)
            except Exception as exc:
                outcome = Execution(
                    False,
                    "failed",
                    made.snapshot.action,
                    error_code="handler_failed",
                    error=str(exc),
                )
        try:
            if made.snapshot.operation_id is None:
                raise ProtocolError(
                    "The decision has no History operation ID; execution was not recorded."
                )
            await self._transport.request(
                "POST",
                self._path + f"/operations/{made.snapshot.operation_id}/actions",
                body=_record_body(made, outcome, started),
            )
        except Exception as exc:
            outcome = replace(outcome, history_error=str(exc))
        return outcome
