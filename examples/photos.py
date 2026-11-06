"""Run a saved Vision agent. Set NULLPROTOCOL_KEY on your backend."""

import os

from nullprotocol import NullProtocol

with NullProtocol() as client:
    agent = client.load(os.environ.get("AGENT_ID", "photo-reader"))
    result = agent.extract(
        "Identify the landmark in the photo.",
        {"landmark": "string"},
        images=[os.environ.get("PHOTO_URL", "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=800&q=80")],
    )
    if not result.success:
        print(result.error_code, result.error)
        raise SystemExit(1)
    print(result.data)
    print(result.usage)
    if result.history_error:
        print("History was not saved:", result.history_error)
        raise SystemExit(1)
