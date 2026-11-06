from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar
from uuid import UUID

from ._errors import ProtocolError

T = TypeVar("T")


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ProtocolError(f"Invalid {name} in the server response.")
    return value


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    uncounted_requests: int | None = None

    @classmethod
    def from_dict(cls, value: Any) -> Usage | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ProtocolError("Invalid generation usage in the server response.")
        uncounted = value.get("uncountedRequests")
        if uncounted is not None:
            uncounted = _count(uncounted, "uncountedRequests")
            if uncounted == 0:
                raise ProtocolError("uncountedRequests must be positive when reported.")
        return cls(
            _count(value.get("inputTokens"), "inputTokens"),
            _count(value.get("outputTokens"), "outputTokens"),
            uncounted,
        )


@dataclass(frozen=True)
class KnowledgeUsage:
    input_tokens: int | None = None
    uncounted_requests: int | None = None

    @classmethod
    def from_dict(cls, value: Any) -> KnowledgeUsage | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ProtocolError("Invalid Knowledge usage in the server response.")
        counted, uncounted = value.get("inputTokens"), value.get("uncountedRequests")
        if counted is not None:
            counted = _count(counted, "Knowledge inputTokens")
        if uncounted is not None:
            uncounted = _count(uncounted, "Knowledge uncountedRequests")
            if uncounted == 0:
                raise ProtocolError("Knowledge uncountedRequests must be positive.")
        return cls(counted, uncounted)


@dataclass(frozen=True, eq=False)
class Result(Generic[T]):
    """A model-operation result. raw preserves all server fields without renaming JSON data."""

    success: bool
    operation_id: str | None
    request_id: str
    data: T | None = None
    summary: str | None = None
    key_points: list[str] | None = None
    recommendation: str | None = None
    score: float | None = None
    reasoning: str | None = None
    action: str | None = None
    parameters: dict[str, Any] | None = None
    error_code: str | None = None
    error: str | None = None
    attempts: int | None = None
    repaired: bool | None = None
    usage: Usage | None = None
    knowledge_usage: KnowledgeUsage | None = None
    knowledge: dict[str, Any] | None = None
    history_error: str | None = None
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def raw(self) -> dict[str, Any]:
        return deepcopy(self._raw)

    @classmethod
    def from_response(cls, response: dict[str, Any], request_id: str) -> Result[Any]:
        value = response.get("result")
        if not isinstance(value, dict) or type(value.get("success")) is not bool:
            raise ProtocolError(
                "Missing operation result in the server response.", request_id=request_id
            )
        if "operationId" not in response:
            raise ProtocolError(
                "Missing operationId in the server response.", request_id=request_id
            )
        operation_id = response.get("operationId")
        if operation_id is not None and not isinstance(operation_id, str):
            raise ProtocolError(
                "Invalid operationId in the server response.", request_id=request_id
            )
        if operation_id is not None:
            try:
                UUID(operation_id)
            except ValueError as exc:
                raise ProtocolError("Invalid operationId UUID.", request_id=request_id) from exc
        for name in ("attempts",):
            if name in value:
                try:
                    _count(value[name], name)
                except ProtocolError as exc:
                    exc.request_id = request_id
                    raise
        if "repaired" in value and type(value["repaired"]) is not bool:
            raise ProtocolError("Invalid repaired flag.", request_id=request_id)
        if "historyError" in value and not isinstance(value["historyError"], str):
            raise ProtocolError("Invalid historyError.", request_id=request_id)
        if "knowledge" in value and not isinstance(value["knowledge"], dict):
            raise ProtocolError("Invalid Knowledge evidence.", request_id=request_id)
        if not value["success"] and not all(
            isinstance(value.get(key), str) for key in ("errorCode", "error")
        ):
            raise ProtocolError("Missing operation error details.", request_id=request_id)
        try:
            usage = Usage.from_dict(value.get("usage"))
            knowledge_usage = KnowledgeUsage.from_dict(value.get("knowledgeUsage"))
        except ProtocolError as exc:
            exc.request_id = request_id
            raise
        return cls(
            success=value["success"],
            operation_id=operation_id,
            request_id=request_id,
            data=deepcopy(value.get("data")),
            summary=value.get("summary"),
            key_points=deepcopy(value.get("keyPoints")),
            recommendation=value.get("recommendation"),
            score=value.get("score"),
            reasoning=value.get("reasoning"),
            action=value.get("action"),
            parameters=deepcopy(value.get("parameters")),
            error_code=value.get("errorCode"),
            error=value.get("error"),
            attempts=value.get("attempts"),
            repaired=value.get("repaired"),
            usage=usage,
            knowledge_usage=knowledge_usage,
            knowledge=deepcopy(value.get("knowledge")),
            history_error=value.get("historyError"),
            _raw=deepcopy(value),
        )


@dataclass(frozen=True)
class Execution:
    success: bool
    outcome: str
    action: str | None = None
    result: Any = None
    error_code: str | None = None
    error: str | None = None
    history_error: str | None = None


@dataclass(frozen=True)
class ActionContext:
    decision: Result[Any]
    input: Any
