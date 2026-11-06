"""Connect and prepare policy sources in the agent's Knowledge tab first."""

from nullprotocol import NullProtocol

with NullProtocol() as client:
    agent = client.load("support")
    result = agent.extract(
        "Within how many days can unused goods be returned?",
        {"days": "number"},
        knowledge={"query": "Return window for unused goods"},
    )
    if result.success:
        print(result.data)
        print("Sources:", result.knowledge)
    else:
        print(result.error_code, result.error)
    print("Generation:", result.usage)
    print("Search:", result.knowledge_usage)
