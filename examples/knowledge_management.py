"""Set NULLPROTOCOL_KEY and create support first. Uses hosted tokens; cleans up its source."""

from nullprotocol import NullProtocol, NullProtocolError

with NullProtocol() as client:
    agent = client.load("support")
    source = agent.space.knowledge.add(
        name="Temporary Python SDK policy example",
        text="# Returns\nUnused goods may be returned within 30 days.",
    )
    connected = False
    try:
        agent.space.knowledge.wait_until_ready(source, timeout=30)
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
