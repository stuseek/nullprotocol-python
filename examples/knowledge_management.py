"""Set NULLPROTOCOL_KEY and create support first. Uses hosted tokens; cleans up its source."""

import time

from nullprotocol import NullProtocol, NullProtocolError

with NullProtocol() as client:
    agent = client.load("support")
    source = agent.space.knowledge.add(
        name="Temporary Python SDK policy example",
        text="# Returns\nUnused goods may be returned within 30 days.",
    )
    connected = False
    try:
        deadline = time.monotonic() + 30
        while True:
            status = agent.space.knowledge.get(source["id"])
            if status["version"] != source["version"]:
                raise RuntimeError("The source was edited by another client.")
            if status["state"] == "ready":
                break
            if status["state"] == "failed":
                raise RuntimeError(f"Preparation failed: {status['error']}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Knowledge is still preparing. Try again later.")
            time.sleep(0.25)
        agent.knowledge.connect(source["id"])
        connected = True
        result = agent.extract(
            "Within how many days can unused goods be returned?",
            {"days": "number"},
            knowledge={"query": "Return window for unused goods"},
        )
        if not result.success:
            raise RuntimeError(f"{result.error_code}: {result.error}")
        print(result.data)
        print("Evidence:", result.knowledge)
        print("Generation:", result.usage, "Search:", result.knowledge_usage)
    finally:
        if connected:
            try:
                agent.knowledge.disconnect(source["id"])
            except NullProtocolError as error:
                print("Disconnect:", error)
        try:
            agent.space.knowledge.delete(source["id"], if_version=source["version"])
        except NullProtocolError as error:
            print("Source cleanup:", error)
