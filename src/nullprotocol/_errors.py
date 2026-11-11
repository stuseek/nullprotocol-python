from __future__ import annotations

from typing import Any


class NullProtocolError(Exception):
    """Base SDK error. Failed model operations are returned as Result, not raised."""


class KnowledgeError(NullProtocolError):
    """Preparation waiting failed locally; no model operation was run."""

    def __init__(
        self, code: str, message: str, *, source_id: str, version: str, reason: str | None = None
    ) -> None:
        self.code, self.source_id, self.version, self.reason = code, source_id, version, reason
        super().__init__(message)


class APIError(NullProtocolError):
    """An HTTP failure. Keep request_id when retrying an uncertain run."""

    def __init__(
        self,
        status_code: int,
        code: str,
        *,
        request_id: str | None = None,
        details: dict[str, Any] | None = None,
        retry_after: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.details = details or {}
        self.retry_after = retry_after
        messages = {
            "invalid_context_key": (
                "A Context key must have 1–64 lowercase letters, digits, dots, underscores or "
                "hyphens, starting with a letter; for example order_id."
            ),
            "invalid_context_value": (
                "Context value must be JSON-serializable. JSON null is allowed; "
                "use delete to remove an entry."
            ),
        }
        super().__init__(f"NullProtocol HTTP {status_code}: {messages.get(code, code)}")


class TransportError(NullProtocolError):
    """The response was not received. A run may have completed: retain request_id."""

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        self.request_id = request_id
        super().__init__(message)


class ProtocolError(NullProtocolError):
    """The server response does not match the expected response contract."""

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        self.request_id = request_id
        super().__init__(message)
