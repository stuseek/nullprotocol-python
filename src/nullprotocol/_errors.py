from __future__ import annotations

from typing import Any


class NullProtocolError(Exception):
    """Base SDK error. Failed model operations are returned as Result, not raised."""


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
        super().__init__(f"NullProtocol HTTP {status_code}: {code}")


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
