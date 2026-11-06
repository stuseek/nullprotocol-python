"""Save facts for one order. Scope IDs are application data, not user authorization."""

from nullprotocol import NullProtocol

with NullProtocol() as client:
    agent = client.load("support", scope="order:42")
    agent.context.set("order", {"id": 42, "delivered": True}, inclusion="selected", if_version=None)
    note = agent.memory.add("The buyer supplied a photo of the packaging.")
    try:
        result = agent.extract(
            "Read the order ID and delivery status from the supplied data.",
            {"orderId": "number", "delivered": "boolean"},
            context_keys=["order"],
        )
        print(result.data if result.success else result.error)
    finally:
        agent.memory.delete(note["id"])
        agent.context.delete("order")
