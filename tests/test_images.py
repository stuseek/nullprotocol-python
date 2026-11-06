import asyncio
import json

import httpx
import pytest

from nullprotocol import AsyncNullProtocol, NullProtocol

PHOTO = "https://images.example.test/mug.jpg"


def test_photos_survive_every_operation_and_idempotency(api):
    with NullProtocol(
        key="fixture", http_client=httpx.Client(transport=httpx.MockTransport(api))
    ) as client:
        agent = client.load("support")
        images = [PHOTO]
        assert agent.extract("", {"days": "number"}, images=images).success
        assert agent.summarize("Describe the mug.", images=images).success
        assert agent.validate("There is damage.", "", images=images).success
        assert agent.decide("What next?", ["send_message"], images=images).success
        agent.extract("", {"days": "number"}, images=images, idempotency_key="photo-42")
        agent.extract("", {"days": "number"}, images=images, idempotency_key="photo-42")
        assert api.model_calls == 5
    bodies = [json.loads(r.content) for r in api.requests if r.url.path.endswith("/run")]
    assert all(body["images"] == [PHOTO] for body in bodies)
    assert bodies[0]["data"] == ""


@pytest.mark.parametrize(
    "images",
    [
        "photo.jpg",
        ["http://example.test/photo.jpg"],
        ["data:image/png;base64,AAAA"],
        ["https://user:password@example.test/photo.jpg"],
        [PHOTO] * 5,
    ],
)
def test_invalid_photos_never_reach_api(api, images):
    with NullProtocol(
        key="fixture", http_client=httpx.Client(transport=httpx.MockTransport(api))
    ) as client:
        agent = client.load("support")
        before = len(api.requests)
        with pytest.raises(ValueError):
            agent.extract("", {"days": "number"}, images=images)
        assert len(api.requests) == before


def test_async_photos(api):
    async def run():
        async with AsyncNullProtocol(
            key="fixture", http_client=httpx.AsyncClient(transport=httpx.MockTransport(api))
        ) as client:
            agent = await client.load("support")
            assert (await agent.extract("", {"days": "number"}, images=[PHOTO])).success
            assert (await agent.summarize("Describe.", images=[PHOTO])).success
            assert (await agent.validate("Damaged.", "", images=[PHOTO])).success
            assert (await agent.decide("Next?", ["send_message"], images=[PHOTO])).success

    asyncio.run(run())
    assert all(
        json.loads(r.content)["images"] == [PHOTO]
        for r in api.requests
        if r.url.path.endswith("/run")
    )
