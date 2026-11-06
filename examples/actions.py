"""A locally executed handler, no built-in chat or second model call."""

from typing import Any

from nullprotocol import ActionContext, NullProtocol


def send_message(parameters: dict[str, Any], context: ActionContext) -> dict[str, bool]:
    # Replace printing with your application's delivery API. Use application-level
    # idempotency for real messages, payments and other external writes.
    print("Send:", parameters["text"])
    return {"sent": True}


with NullProtocol() as client:
    agent = client.load("support")
    agent.register_action(
        "send_message",
        send_message,
        description="Send a helpful reply to the customer.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string", "minLength": 1}},
            "required": ["text"],
            "additionalProperties": False,
        },
    )
    decision = agent.decide("The customer says: Good morning!", ["send_message"])
    if decision.success:
        execution = agent.execute(decision)
        print(execution)
    else:
        print(decision.error_code, decision.error)
