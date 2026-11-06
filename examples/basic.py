"""Set NULLPROTOCOL_KEY and create a saved agent with ID 'support' first."""

from nullprotocol import NullProtocol

with NullProtocol() as client:
    agent = client.load("support")
    result = agent.extract(
        "Order 42: two blue mugs.",
        {"orderId": "number", "quantity": "number", "item": "string"},
    )
    if result.success:
        print(result.data)
        print(result.usage)
        print("History:", result.operation_id)
    else:
        print(result.error_code, result.error)
    if result.history_error:
        print("History was not saved:", result.history_error)
