from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from typing import Any, NoReturn
from urllib.parse import urlsplit

import httpx

from ._errors import APIError, ProtocolError, TransportError

_UNSET = object()


def json_body(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("Request data must be valid UTF-8 JSON (no NaN or Infinity).") from exc


def utf16_length(value: str) -> int:
    try:
        return len(value.encode("utf-16-le")) // 2
    except UnicodeError as exc:
        raise ValueError("Text must not contain unpaired surrogates.") from exc


def identifier(value: str, kind: str) -> str:
    pattern = r"[a-z0-9][a-z0-9._-]{0,63}" if kind == "agent ID" else r"[a-z][a-z0-9._-]{0,63}"
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        if kind == "context key":
            raise ValueError(
                "Invalid context key. Use 1–64 lowercase letters, digits, dots, underscores or "
                "hyphens, starting with a letter; for example order_id."
            )
        raise ValueError(f"Invalid {kind}.")
    return value


def scope_id(value: str | None) -> str | None:
    if value is not None and (
        not isinstance(value, str)
        or not 1 <= utf16_length(value) <= 128
        or re.search(r"[\x00-\x1f\x7f-\x9f]", value)
    ):
        raise ValueError("scope must be 1–128 UTF-16 characters without control characters.")
    return value


def request_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[\x21-\x7e]{1,128}", value):
        raise ValueError("idempotency_key must be 1–128 visible ASCII characters without spaces.")
    return value


class _Config:
    def __init__(
        self, key: str | None, endpoint: str | None, timeout: float, max_retries: int
    ) -> None:
        actual = key if key is not None else os.environ.get("NULLPROTOCOL_KEY")
        if (
            not isinstance(actual, str)
            or not actual.strip()
            or re.search(r"[\s\x00-\x1f\x7f]", actual)
        ):
            raise ValueError("Set key or NULLPROTOCOL_KEY to an active Space SDK key.")
        address = (
            endpoint or os.environ.get("NULLPROTOCOL_API_URL") or "https://api.nullprotocol.ai"
        )
        parsed = urlsplit(address)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/") not in ("", "/v1")
        ):
            raise ValueError("endpoint must be an HTTP(S) origin, optionally ending in /v1.")
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive number of seconds.")
        if type(max_retries) is not int or not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be an integer from 0 to 5.")
        self.base = address.rstrip("/").removesuffix("/v1") + "/v1"
        self.key = actual
        self.timeout = timeout
        self.max_retries = max_retries


def _invalid_constant(_value: str) -> NoReturn:
    raise ValueError("Non-finite numbers are not valid JSON.")


def _decode(response: httpx.Response, task_id: str | None) -> dict[str, Any]:
    try:
        body = response.json(parse_constant=_invalid_constant)
    except ValueError as exc:
        if not response.is_success:
            raise APIError(
                response.status_code,
                "non_json_response",
                request_id=task_id,
                retry_after=response.headers.get("retry-after"),
            ) from exc
        raise ProtocolError("Expected a JSON response.", request_id=task_id) from exc
    if not response.is_success:
        detail = body if isinstance(body, dict) else {}
        error = detail.get("error", "http_error")
        raise APIError(
            response.status_code,
            error if isinstance(error, str) else "http_error",
            request_id=task_id,
            details=detail,
            retry_after=response.headers.get("retry-after"),
        )
    if not isinstance(body, dict):
        raise ProtocolError("Expected a JSON object response.", request_id=task_id)
    return body


def _delay(response: httpx.Response | None, attempt: int) -> float | None:
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after is not None:
            try:
                value = float(retry_after)
            except ValueError:
                return None  # Do not guess when a server supplies an unsupported date format.
            if not math.isfinite(value) or value < 0 or value > 30:
                return None  # Return the response instead of ignoring a long delay.
            return value
    return float(min(0.25 * (2**attempt), 2.0))


def _retryable(response: httpx.Response) -> bool:
    if response.status_code == 429:
        return True
    if response.status_code == 409:
        try:
            return bool(response.json().get("error") == "in_progress")
        except (ValueError, AttributeError):
            return False
    if response.status_code == 503:
        try:
            return bool(response.json().get("error") == "busy")
        except (ValueError, AttributeError):
            return False
    return False


def _budget(timeout: float, deadline: float | None) -> float:
    if deadline is None:
        return timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Knowledge preparation wait timed out.")
    return min(timeout, remaining)


class Transport:
    def __init__(self, config: _Config, client: httpx.Client | None = None) -> None:
        self.config = config
        self.client = client if client is not None else httpx.Client(follow_redirects=False)
        self.owned = client is None
        self.closed = False

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            if self.owned:
                self.client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Any = _UNSET,
        params: dict[str, str] | None = None,
        task_id: str | None = None,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        if self.closed:
            raise TransportError("This NullProtocol client is closed.", request_id=task_id)
        content = None if body is _UNSET else json_body(body)
        headers = {"Authorization": f"Bearer {self.config.key}", "Accept": "application/json"}
        if content is not None:
            headers["Content-Type"] = "application/json"
        if task_id is not None:
            headers["Idempotency-Key"] = task_id
        retries = self.config.max_retries if method == "GET" or task_id is not None else 0
        for attempt in range(retries + 1):
            timeout = _budget(self.config.timeout, deadline)
            response = None
            try:
                response = self.client.request(
                    method,
                    self.config.base + path,
                    headers=headers,
                    content=content,
                    params=params,
                    timeout=timeout,
                    follow_redirects=False,
                )
            except httpx.TransportError as exc:
                _budget(self.config.timeout, deadline)
                if attempt == retries:
                    raise TransportError(
                        "NullProtocol could not be reached. "
                        "Preserve request_id before retrying a run.",
                        request_id=task_id,
                    ) from exc
            _budget(self.config.timeout, deadline)
            if response is not None and (attempt == retries or not _retryable(response)):
                return _decode(response, task_id)
            delay = _delay(response, attempt)
            if delay is None:
                assert response is not None
                return _decode(response, task_id)
            time.sleep(_budget(delay, deadline))
        raise AssertionError("Unreachable transport state")


class AsyncTransport:
    def __init__(self, config: _Config, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self.client = client if client is not None else httpx.AsyncClient(follow_redirects=False)
        self.owned = client is None
        self.closed = False

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            if self.owned:
                await self.client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        body: Any = _UNSET,
        params: dict[str, str] | None = None,
        task_id: str | None = None,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        if self.closed:
            raise TransportError("This NullProtocol client is closed.", request_id=task_id)
        content = None if body is _UNSET else json_body(body)
        headers = {"Authorization": f"Bearer {self.config.key}", "Accept": "application/json"}
        if content is not None:
            headers["Content-Type"] = "application/json"
        if task_id is not None:
            headers["Idempotency-Key"] = task_id
        retries = self.config.max_retries if method == "GET" or task_id is not None else 0
        for attempt in range(retries + 1):
            timeout = _budget(self.config.timeout, deadline)
            response = None
            try:
                response = await self.client.request(
                    method,
                    self.config.base + path,
                    headers=headers,
                    content=content,
                    params=params,
                    timeout=timeout,
                    follow_redirects=False,
                )
            except httpx.TransportError as exc:
                _budget(self.config.timeout, deadline)
                if attempt == retries:
                    raise TransportError(
                        "NullProtocol could not be reached. "
                        "Preserve request_id before retrying a run.",
                        request_id=task_id,
                    ) from exc
            _budget(self.config.timeout, deadline)
            if response is not None and (attempt == retries or not _retryable(response)):
                return _decode(response, task_id)
            delay = _delay(response, attempt)
            if delay is None:
                assert response is not None
                return _decode(response, task_id)
            await asyncio.sleep(_budget(delay, deadline))
        raise AssertionError("Unreachable transport state")
