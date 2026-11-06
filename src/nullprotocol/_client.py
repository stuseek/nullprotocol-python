from __future__ import annotations

from typing import Any

import httpx

from ._agent import Agent, AsyncAgent
from ._common import checked_text, object_field
from ._transport import AsyncTransport, Transport, _Config, identifier


def _create_body(
    model: str,
    instructions: str,
    agent_id: str | None,
    name: str | None,
) -> dict[str, Any]:
    checked_text(model, "model", 120)
    if not isinstance(instructions, str) or len(instructions.encode()) > 16384:
        raise ValueError("instructions must be text of at most 16384 UTF-8 bytes.")
    body = {"provider": "nullprotocol", "model": model, "instructions": instructions}
    if agent_id is not None:
        body["agentId"] = identifier(agent_id, "agent ID")
    if name is not None:
        body["name"] = checked_text(name, "name", 80)
    return body


class NullProtocol:
    """Client for a Space's hosted agents. Timeouts are in seconds, not milliseconds."""

    def __init__(
        self,
        *,
        key: str | None = None,
        endpoint: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._transport = Transport(_Config(key, endpoint, timeout, max_retries), http_client)

    def __enter__(self) -> NullProtocol:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def close(self) -> None:
        self._transport.close()

    def load(self, agent_id: str, *, scope: str | None = None) -> Agent:
        agent = Agent(self._transport, agent_id, scope)
        agent.settings()  # Fail at setup if the agent/key is unavailable.
        return agent

    def create(
        self,
        *,
        model: str,
        instructions: str = "",
        agent_id: str | None = None,
        name: str | None = None,
    ) -> Agent:
        body = _create_body(model, instructions, agent_id, name)
        saved = object_field(self._transport.request("POST", "/agents", body=body), "agent")
        return Agent(self._transport, saved["agentId"])


class AsyncNullProtocol:
    """Async client for asyncio applications. Use async with to release its HTTP pool."""

    def __init__(
        self,
        *,
        key: str | None = None,
        endpoint: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._transport = AsyncTransport(_Config(key, endpoint, timeout, max_retries), http_client)

    async def __aenter__(self) -> AsyncNullProtocol:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._transport.close()

    async def load(self, agent_id: str, *, scope: str | None = None) -> AsyncAgent:
        agent = AsyncAgent(self._transport, agent_id, scope)
        await agent.settings()
        return agent

    async def create(
        self,
        *,
        model: str,
        instructions: str = "",
        agent_id: str | None = None,
        name: str | None = None,
    ) -> AsyncAgent:
        body = _create_body(model, instructions, agent_id, name)
        saved = object_field(await self._transport.request("POST", "/agents", body=body), "agent")
        return AsyncAgent(self._transport, saved["agentId"])
