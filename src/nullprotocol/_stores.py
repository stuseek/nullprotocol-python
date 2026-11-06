from __future__ import annotations

from typing import Any

from ._common import list_field, object_field, version
from ._knowledge import AsyncKnowledgeLibrary, KnowledgeLibrary
from ._transport import _UNSET, AsyncTransport, Transport, identifier


def _context_body(value: Any, inclusion: str | None, if_version: Any) -> dict[str, Any]:
    body = {"value": value}
    if inclusion is not None:
        if inclusion not in ("always", "selected"):
            raise ValueError("inclusion must be 'always' or 'selected'.")
        body["inclusion"] = inclusion
    if if_version is not _UNSET:
        body["ifVersion"] = None if if_version is None else version(if_version)
    return body


class ContextStore:
    def __init__(self, transport: Transport, path: str, scope: str | None = None) -> None:
        self._transport, self._path = transport, path
        self._params = {"conversation": scope} if scope is not None else None

    def list(self) -> list[dict[str, Any]]:
        return list_field(
            self._transport.request("GET", self._path, params=self._params), "entries"
        )

    def get(self, key: str) -> dict[str, Any]:
        return object_field(
            self._transport.request(
                "GET",
                self._path + "/" + identifier(key, "context key"),
                params=self._params,
            ),
            "entry",
        )

    def set(
        self,
        key: str,
        value: Any,
        *,
        inclusion: str | None = None,
        if_version: Any = _UNSET,
    ) -> dict[str, Any]:
        return object_field(
            self._transport.request(
                "PUT",
                self._path + "/" + identifier(key, "context key"),
                params=self._params,
                body=_context_body(value, inclusion, if_version),
            ),
            "entry",
        )

    def delete(self, key: str) -> None:
        self._transport.request(
            "DELETE",
            self._path + "/" + identifier(key, "context key"),
            params=self._params,
        )


class AsyncContextStore:
    def __init__(self, transport: AsyncTransport, path: str, scope: str | None = None) -> None:
        self._transport, self._path = transport, path
        self._params = {"conversation": scope} if scope is not None else None

    async def list(self) -> list[dict[str, Any]]:
        return list_field(
            await self._transport.request(
                "GET",
                self._path,
                params=self._params,
            ),
            "entries",
        )

    async def get(self, key: str) -> dict[str, Any]:
        return object_field(
            await self._transport.request(
                "GET",
                self._path + "/" + identifier(key, "context key"),
                params=self._params,
            ),
            "entry",
        )

    async def set(
        self,
        key: str,
        value: Any,
        *,
        inclusion: str | None = None,
        if_version: Any = _UNSET,
    ) -> dict[str, Any]:
        return object_field(
            await self._transport.request(
                "PUT",
                self._path + "/" + identifier(key, "context key"),
                params=self._params,
                body=_context_body(value, inclusion, if_version),
            ),
            "entry",
        )

    async def delete(self, key: str) -> None:
        await self._transport.request(
            "DELETE",
            self._path + "/" + identifier(key, "context key"),
            params=self._params,
        )


def _note(text: str, scope: str | None) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip() or len(text.strip().encode()) > 4096:
        raise ValueError("A note must be nonempty text of at most 4096 UTF-8 bytes.")
    return {"text": text, **({"conversation": scope} if scope is not None else {})}


class MemoryStore:
    def __init__(self, transport: Transport, path: str, scope: str | None) -> None:
        self._transport, self._path, self._scope = transport, path, scope

    def list(self) -> list[dict[str, Any]]:
        return list_field(
            self._transport.request(
                "GET",
                self._path,
                params={"conversation": self._scope} if self._scope else None,
            ),
            "notes",
        )

    def add(self, text: str) -> dict[str, Any]:
        return object_field(
            self._transport.request(
                "POST",
                self._path,
                body=_note(text, self._scope),
            ),
            "note",
        )

    def delete(self, note_id: str) -> None:
        self._transport.request("DELETE", self._path + "/" + version(note_id))


class AsyncMemoryStore:
    def __init__(self, transport: AsyncTransport, path: str, scope: str | None) -> None:
        self._transport, self._path, self._scope = transport, path, scope

    async def list(self) -> list[dict[str, Any]]:
        return list_field(
            await self._transport.request(
                "GET",
                self._path,
                params={"conversation": self._scope} if self._scope else None,
            ),
            "notes",
        )

    async def add(self, text: str) -> dict[str, Any]:
        return object_field(
            await self._transport.request(
                "POST",
                self._path,
                body=_note(text, self._scope),
            ),
            "note",
        )

    async def delete(self, note_id: str) -> None:
        await self._transport.request("DELETE", self._path + "/" + version(note_id))


class SpaceContextStore(ContextStore):
    def __init__(self, transport: Transport) -> None:
        super().__init__(transport, "/context")
        self.knowledge = KnowledgeLibrary(transport)


class AsyncSpaceContextStore(AsyncContextStore):
    def __init__(self, transport: AsyncTransport) -> None:
        super().__init__(transport, "/context")
        self.knowledge = AsyncKnowledgeLibrary(transport)
