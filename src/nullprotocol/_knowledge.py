from __future__ import annotations

from typing import Any

from ._common import checked_text, list_field, object_field, version
from ._transport import AsyncTransport, Transport


def _fields(
    name: str | None, text: str | None, *, update: bool = False, if_version: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if name is not None or not update:
        if not isinstance(name, str):
            raise ValueError("Knowledge name must be text.")
        result["name"] = checked_text(name.strip(), "Knowledge name", 120)
    if text is not None or not update:
        if not isinstance(text, str) or not text.strip() or "\0" in text:
            raise ValueError("Knowledge text must be nonempty UTF-8 text without NUL characters.")
        if len(text.encode("utf-8")) > 1048576:
            raise ValueError("A Knowledge document holds up to 1 MiB of UTF-8 text.")
        result["text"] = text
    if update:
        if not result:
            raise ValueError("Supply a new name or text to update a Knowledge document.")
        if not isinstance(if_version, str):
            raise ValueError("if_version is required; read the document's current version first.")
        result["ifVersion"] = version(if_version)
    return result


class _Documents:
    def __init__(self, transport: Transport, path: str, scope: str) -> None:
        self._transport, self._path, self._scope = transport, path, scope

    def _all(self) -> list[dict[str, Any]]:
        return list_field(self._transport.request("GET", self._path), "sources")

    def list(self) -> list[dict[str, Any]]:
        """List documents owned by this scope. Connections are listed separately."""
        return [source for source in self._all() if source.get("scope") == self._scope]

    def get(self, source_id: str) -> dict[str, Any]:
        return object_field(
            self._transport.request("GET", self._path + "/" + version(source_id)), "source"
        )

    def add(self, *, name: str, text: str) -> dict[str, Any]:
        """Save text and begin preparation. Inspect get(id)['state'] for readiness."""
        return self._transport.request("POST", self._path, body=_fields(name, text))

    def update(
        self, source_id: str, *, if_version: str, name: str | None = None, text: str | None = None
    ) -> dict[str, Any]:
        return self._transport.request(
            "PUT",
            self._path + "/" + version(source_id),
            body=_fields(name, text, update=True, if_version=if_version),
        )

    def delete(self, source_id: str, *, if_version: str) -> None:
        self._transport.request(
            "DELETE", self._path + "/" + version(source_id), body={"ifVersion": version(if_version)}
        )

    def retry(self, source_id: str) -> dict[str, Any]:
        """Explicitly retry failed preparation; this can make embedding requests."""
        return self._transport.request(
            "POST", self._path + "/" + version(source_id) + "/retry", body={}
        )


class KnowledgeLibrary(_Documents):
    """The Space's shared library. Adding a source does not connect it to an agent."""

    def __init__(self, transport: Transport) -> None:
        super().__init__(transport, "/knowledge", "space")

    def usage(self) -> dict[str, Any]:
        """Embedding usage over 30 days; unknown sums remain None, known sums are strings."""
        return object_field(self._transport.request("GET", self._path), "usage")


class AgentKnowledge(_Documents):
    def __init__(self, transport: Transport, agent_id: str) -> None:
        super().__init__(transport, f"/agents/{agent_id}/knowledge", "agent")
        self._links = f"/agents/{agent_id}/knowledge-links"

    def connections(self) -> list[dict[str, Any]]:
        return [s for s in self._all() if s.get("scope") == "space" and s.get("attached")]

    def connect(self, space_source_id: str) -> dict[str, Any]:
        return self._transport.request("PUT", self._links + "/" + version(space_source_id), body={})

    def disconnect(self, space_source_id: str) -> dict[str, Any]:
        return self._transport.request(
            "DELETE", self._links + "/" + version(space_source_id), body={}
        )


class _AsyncDocuments:
    def __init__(self, transport: AsyncTransport, path: str, scope: str) -> None:
        self._transport, self._path, self._scope = transport, path, scope

    async def _all(self) -> list[dict[str, Any]]:
        return list_field(await self._transport.request("GET", self._path), "sources")

    async def list(self) -> list[dict[str, Any]]:
        return [source for source in await self._all() if source.get("scope") == self._scope]

    async def get(self, source_id: str) -> dict[str, Any]:
        return object_field(
            await self._transport.request("GET", self._path + "/" + version(source_id)), "source"
        )

    async def add(self, *, name: str, text: str) -> dict[str, Any]:
        return await self._transport.request("POST", self._path, body=_fields(name, text))

    async def update(
        self, source_id: str, *, if_version: str, name: str | None = None, text: str | None = None
    ) -> dict[str, Any]:
        return await self._transport.request(
            "PUT",
            self._path + "/" + version(source_id),
            body=_fields(name, text, update=True, if_version=if_version),
        )

    async def delete(self, source_id: str, *, if_version: str) -> None:
        await self._transport.request(
            "DELETE", self._path + "/" + version(source_id), body={"ifVersion": version(if_version)}
        )

    async def retry(self, source_id: str) -> dict[str, Any]:
        return await self._transport.request(
            "POST", self._path + "/" + version(source_id) + "/retry", body={}
        )


class AsyncKnowledgeLibrary(_AsyncDocuments):
    def __init__(self, transport: AsyncTransport) -> None:
        super().__init__(transport, "/knowledge", "space")

    async def usage(self) -> dict[str, Any]:
        return object_field(await self._transport.request("GET", self._path), "usage")


class AsyncAgentKnowledge(_AsyncDocuments):
    def __init__(self, transport: AsyncTransport, agent_id: str) -> None:
        super().__init__(transport, f"/agents/{agent_id}/knowledge", "agent")
        self._links = f"/agents/{agent_id}/knowledge-links"

    async def connections(self) -> list[dict[str, Any]]:
        return [s for s in await self._all() if s.get("scope") == "space" and s.get("attached")]

    async def connect(self, space_source_id: str) -> dict[str, Any]:
        return await self._transport.request(
            "PUT", self._links + "/" + version(space_source_id), body={}
        )

    async def disconnect(self, space_source_id: str) -> dict[str, Any]:
        return await self._transport.request(
            "DELETE", self._links + "/" + version(space_source_id), body={}
        )
