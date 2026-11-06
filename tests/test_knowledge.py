from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import httpx
import pytest

from nullprotocol import APIError, AsyncNullProtocol, NullProtocol, TransportError


class KnowledgeAPI:
    def __init__(self):
        self.requests = []
        self.ids = {scope: str(uuid4()) for scope in ["space", "agent"]}
        self.documents = {
            scope: {
                "id": self.ids[scope],
                "name": "Policy",
                "text": "30 days.",
                "scope": scope,
                "revision": 1,
                "activeRevision": None,
                "version": str(uuid4()),
                "state": "preparing",
            }
            for scope in self.ids
        }
        self.attached = False

    def __call__(self, request):
        self.requests.append(request)
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        if path.endswith("/settings"):
            return httpx.Response(
                200,
                json={"agent": {"agentId": "support", "model": "test", "provider": "nullprotocol"}},
            )
        if "/knowledge-links/" in path:
            self.attached = request.method == "PUT"
            return httpx.Response(200, json={"attached": self.attached})
        if path.endswith("/retry"):
            return httpx.Response(200, json={"retrying": False})
        scope = "space" if path.startswith("/v1/knowledge") else "agent"
        source = self.documents[scope]
        if path.endswith("/knowledge"):
            if request.method == "POST":
                source.update(body)
                return httpx.Response(
                    201, json={k: source[k] for k in ["id", "revision", "version"]}
                )
            rows = [self.documents["space"] | {"attached": self.attached}]
            if scope == "agent":
                rows.append(self.documents["agent"])
            return httpx.Response(
                200,
                json={
                    "sources": rows,
                    "usage": {"preparation": "25", "search": None, "uncounted": 0, "requests": 1},
                },
            )
        if not path.endswith(source["id"]):
            return httpx.Response(404, json={"error": "knowledge_not_found"})
        if request.method == "GET":
            return httpx.Response(200, json={"source": source})
        if body.get("ifVersion") != source["version"]:
            return httpx.Response(409, json={"error": "version_conflict"})
        if request.method == "DELETE":
            return httpx.Response(200, json={"deleted": True})
        if "text" in body and body["text"] != source["text"]:
            source["revision"] += 1
        source.update(body | {"version": str(uuid4())})
        return httpx.Response(200, json={k: source[k] for k in ["id", "revision", "version"]})


def setup(api):
    return NullProtocol(key="fixture", http_client=httpx.Client(transport=httpx.MockTransport(api)))


def test_document_lifecycle_scopes_connections_and_usage():
    api = KnowledgeAPI()
    with setup(api) as client:
        agent = client.load("support", scope="order:42")
        document = agent.knowledge.add(name=" Policy ", text="30 days.\n")
        assert agent.knowledge.get(document["id"])["text"] == "30 days.\n"
        assert [s["scope"] for s in agent.knowledge.list()] == ["agent"]
        changed = agent.knowledge.update(
            document["id"], text="14 days.", if_version=document["version"]
        )
        assert changed["revision"] == 2
        with pytest.raises(APIError) as conflict:
            agent.knowledge.update(document["id"], text="7 days.", if_version=document["version"])
        assert conflict.value.code == "version_conflict"
        with pytest.raises(APIError):
            agent.knowledge.delete(document["id"], if_version=document["version"])
        agent.knowledge.delete(document["id"], if_version=changed["version"])
        shared = agent.space.knowledge.add(name="Shared", text="Rules")
        assert len(agent.space.knowledge.list()) == 1
        assert agent.knowledge.connections() == []
        agent.knowledge.connect(shared["id"])
        assert [s["id"] for s in agent.knowledge.connections()] == [shared["id"]]
        assert [s["id"] for s in agent.knowledge.list()] == [document["id"]]
        with pytest.raises(APIError):
            agent.knowledge.get(shared["id"])
        agent.knowledge.disconnect(shared["id"])
        assert agent.knowledge.connections() == []
        assert agent.space.knowledge.usage()["search"] is None
        assert agent.space.knowledge.usage()["preparation"] == "25"
        assert agent.knowledge.retry(document["id"]) == {"retrying": False}
        common = agent.space.knowledge.get(shared["id"])
        agent.space.knowledge.update(shared["id"], name="Renamed", if_version=common["version"])
        assert agent.space.knowledge.retry(shared["id"]) == {"retrying": False}
    assert all("conversation" not in req.url.params for req in api.requests)
    assert all(
        "conversation" not in (json.loads(req.content) if req.content else {})
        for req in api.requests
    )


@pytest.mark.parametrize(
    "name,text",
    [
        ("", "x"),
        ("bad\nname", "x"),
        ("x", ""),
        ("x", "   "),
        ("x", "bad\0text"),
        ("x", "\ud800"),
        ("x", "é" * 524289),
        ("😀" * 61, "x"),
    ],
)
def test_invalid_knowledge_input_does_not_dispatch(name, text):
    api = KnowledgeAPI()
    with setup(api) as client:
        agent = client.load("support")
        with pytest.raises(ValueError):
            agent.knowledge.add(name=name, text=text)
    assert len(api.requests) == 1


def test_versions_and_ids_are_required_before_writes():
    api = KnowledgeAPI()
    with setup(api) as client:
        agent = client.load("support")
        with pytest.raises(TypeError):
            agent.knowledge.update(api.ids["agent"], text="x")
        with pytest.raises(ValueError):
            agent.knowledge.update(api.ids["agent"], if_version=str(uuid4()))
        with pytest.raises(ValueError):
            agent.knowledge.delete(api.ids["agent"], if_version="bad")
        with pytest.raises(ValueError):
            agent.knowledge.get("../other")
        with pytest.raises(ValueError):
            agent.knowledge.connect("../other")
    assert len(api.requests) == 1


def test_uncertain_knowledge_writes_are_not_retried():
    api = KnowledgeAPI()
    writes = []

    def http(request):
        if request.method == "POST":
            writes.append(request)
            raise httpx.ReadTimeout("response lost")
        return api(request)

    with httpx.Client(transport=httpx.MockTransport(http)) as external:
        with NullProtocol(key="fixture", http_client=external) as client:
            with pytest.raises(TransportError):
                client.load("support").knowledge.add(name="Policy", text="Rules")
    assert len(writes) == 1


def test_async_management_matches_sync_routes():
    async def run():
        api = KnowledgeAPI()
        async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as external:
            async with AsyncNullProtocol(key="fixture", http_client=external) as client:
                agent = await client.load("support", scope="order:42")
                own = await agent.knowledge.add(name="Own", text="30 days.")
                assert (await agent.knowledge.get(own["id"]))["state"] == "preparing"
                assert len(await agent.knowledge.list()) == 1
                changed = await agent.knowledge.update(
                    own["id"], text="14 days.", if_version=own["version"]
                )
                assert changed["revision"] == 2
                await agent.knowledge.delete(own["id"], if_version=changed["version"])
                shared = await agent.space.knowledge.add(name="Shared", text="Policy")
                assert len(await agent.space.knowledge.list()) == 1
                current = await agent.space.knowledge.get(shared["id"])
                await agent.space.knowledge.update(
                    shared["id"], name="Renamed", if_version=current["version"]
                )
                assert (await agent.space.knowledge.usage())["search"] is None
                assert await agent.knowledge.connections() == []
                await agent.knowledge.connect(shared["id"])
                assert len(await agent.knowledge.connections()) == 1
                await agent.knowledge.disconnect(shared["id"])
                assert await agent.knowledge.connections() == []
                assert await agent.knowledge.retry(own["id"]) == {"retrying": False}
                assert await agent.space.knowledge.retry(shared["id"]) == {"retrying": False}

    asyncio.run(run())
