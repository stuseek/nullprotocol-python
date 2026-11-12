from __future__ import annotations

import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from typing import Any, TypeVar
from urllib.parse import urlsplit
from uuid import UUID, uuid4
from weakref import WeakKeyDictionary, WeakValueDictionary

from jsonschema import Draft202012Validator, FormatChecker

from ._transport import identifier, json_body, request_id, scope_id, utf16_length
from ._types import Result

Knowledge = bool | dict[str, str]
ActionChoice = str | dict[str, Any]


def object_field(body: dict[str, Any], field: str) -> dict[str, Any]:
    from ._errors import ProtocolError

    value = body.get(field)
    if not isinstance(value, dict):
        raise ProtocolError(f"Missing {field} in the server response.")
    return value


def list_field(body: dict[str, Any], field: str) -> list[dict[str, Any]]:
    from ._errors import ProtocolError

    value = body.get(field)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ProtocolError(f"Missing {field} in the server response.")
    return value


def version(value: str) -> str:
    try:
        UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Expected a UUID version or note ID.") from exc
    return value


def checked_text(value: str, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or utf16_length(value) > maximum:
        raise ValueError(
            f"{name} must be a nonempty string of at most {maximum} UTF-16 characters."
        )
    if re.search(r"[\x00-\x1f\x7f-\x9f]", value):
        raise ValueError(f"{name} must not contain control characters.")
    return value


TASK_NAME = re.compile(r"[a-z][a-z0-9._-]{0,63}")


def task_payload(
    task: str,
    input: Any,
    reference: Any,
    unset: Any,
    scope: str | None,
    knowledge: Knowledge | None,
    images: list[str] | None,
) -> dict[str, Any]:
    """What a run of a saved task sends: the input and this run's own options."""
    if not isinstance(task, str) or not TASK_NAME.fullmatch(task):
        raise ValueError(
            "A task name starts with a lowercase letter and has up to 64 lowercase "
            "letters, digits, dots, underscores or hyphens."
        )
    values = {"input": input} if reference is unset else {"input": input, "reference": reference}
    body = operation_payload("", values, scope, knowledge, None, images)
    del body["operation"]
    return body


def operation_payload(
    operation: str,
    values: dict[str, Any],
    scope: str | None,
    knowledge: Knowledge | None,
    context_keys: list[str] | None,
    images: list[str] | None = None,
) -> dict[str, Any]:
    body = {"operation": operation, **values}
    if scope is not None:
        body["conversation"] = scope
    if knowledge is not None:
        if type(knowledge) is bool:
            body["knowledge"] = knowledge
        elif isinstance(knowledge, dict) and set(knowledge) == {"query"}:
            query = knowledge["query"]
            if (
                not isinstance(query, str)
                or not query.strip()
                or utf16_length(query.strip()) > 2000
            ):
                raise ValueError(
                    "Knowledge query must have 1–2000 UTF-16 characters after trimming."
                )
            # The server returns knowledge_query_too_large if the byte limit is exceeded.
            body["knowledge"] = {"query": query.strip()}
        else:
            raise ValueError("knowledge must be a bool or {'query': '...'}.")
    if context_keys is not None:
        if not isinstance(context_keys, list) or len(context_keys) > 16:
            raise ValueError("context_keys must be a list of at most 16 keys.")
        body["contextKeys"] = [identifier(key, "context key") for key in context_keys]
    if images is not None:
        if not isinstance(images, list) or len(images) > 4:
            raise ValueError("images must be a list of at most four HTTPS photo URLs.")
        for address in images:
            if not isinstance(address, str) or utf16_length(address) > 2048:
                raise ValueError("Each photo needs an HTTPS URL of at most 2048 characters.")
            url = urlsplit(address)
            if url.scheme != "https" or not url.hostname or url.username or url.password:
                raise ValueError("Photo URLs must use HTTPS without embedded credentials.")
        if images:
            body["images"] = list(images)
    if len(json_body(body)) > 131072:
        raise ValueError("The HTTP operation body exceeds 131072 UTF-8 bytes.")
    return body


def summary_options(content: Any, max_length: int, focus: str | None) -> dict[str, Any]:
    if type(max_length) is not int or not 1 <= max_length <= 10000:
        raise ValueError("max_length must be an integer from 1 to 10000.")
    body = {"content": content, "maxLength": max_length}
    if focus is not None:
        if not isinstance(focus, str) or utf16_length(focus) > 1000:
            raise ValueError("focus must have at most 1000 UTF-16 characters.")
        body["focus"] = focus
    return body


def validation_options(
    criteria: str, subject: Any, reference: Any, unset: object
) -> dict[str, Any]:
    if not isinstance(criteria, str) or not 1 <= utf16_length(criteria) <= 16384:
        raise ValueError("criteria must have 1–16384 UTF-16 characters.")
    body = {"criteria": criteria, "subject": subject}
    if reference is not unset:
        body["reference"] = reference
    return body


def schema_validator(schema: dict[str, Any]) -> Draft202012Validator:
    if not isinstance(schema, dict):
        raise ValueError("input_schema must be a JSON Schema object.")
    Draft202012Validator.check_schema(schema)
    checker = FormatChecker()

    def check(value: Any) -> None:
        if not isinstance(value, dict):
            return
        for ref in ("$ref", "$dynamicRef", "$recursiveRef"):
            if ref in value and not str(value[ref]).startswith("#"):
                raise ValueError(
                    "Action schemas must use local references; remote refs are unsupported."
                )
        if "format" in value and value["format"] not in checker.checkers:
            raise ValueError(f"Unsupported action schema format: {value['format']}")
        # Descend through schema positions only. A property literally named
        # format/$ref, an enum value or a default is ordinary application data.
        for keyword in (
            "properties",
            "patternProperties",
            "$defs",
            "definitions",
            "dependentSchemas",
        ):
            for child in value.get(keyword, {}).values():
                check(child)
        for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
            for child in value.get(keyword, []):
                check(child)
        for keyword in (
            "items",
            "additionalItems",
            "additionalProperties",
            "unevaluatedProperties",
            "unevaluatedItems",
            "contains",
            "propertyNames",
            "if",
            "then",
            "else",
            "not",
            "contentSchema",
        ):
            child = value.get(keyword)
            if isinstance(child, list):
                for item in child:
                    check(item)
            else:
                check(child)
        for child in value.get("dependencies", {}).values():
            if isinstance(child, dict):
                check(child)

    check(schema)
    return Draft202012Validator(deepcopy(schema), format_checker=checker)


@dataclass
class RegisteredAction:
    handler: Callable[..., Any]
    description: str | None
    schema: dict[str, Any] | None
    validator: Draft202012Validator | None
    guard: Callable[..., Any] | None


@dataclass
class MadeDecision:
    snapshot: Result[Any]
    input: Any
    executed: bool = False


_AgentT = TypeVar("_AgentT", bound="AgentState")


class AgentState:
    def __init__(self, agent_id: str, scope: str | None) -> None:
        self.agent_id = identifier(agent_id, "agent ID")
        self.scope = scope_id(scope)
        self._actions: dict[str, RegisteredAction] = {}
        self._decisions: WeakKeyDictionary[Result[Any], MadeDecision] = WeakKeyDictionary()
        self._decision_ids: WeakValueDictionary[str, MadeDecision] = WeakValueDictionary()
        self._lock = Lock()

    @property
    def _path(self) -> str:
        return f"/agents/{self.agent_id}"

    def register_action(
        self: _AgentT,
        name: str,
        handler: Callable[..., Any],
        *,
        description: str | None = None,
        input_schema: dict[str, Any] | None = None,
        guard: Callable[..., Any] | None = None,
    ) -> _AgentT:
        checked_text(name, "Action name", 64)
        if not callable(handler) or (guard is not None and not callable(guard)):
            raise ValueError("handler and guard must be callable.")
        if description is not None and (
            not isinstance(description, str) or utf16_length(description) > 1000
        ):
            raise ValueError("Action description must have at most 1000 UTF-16 characters.")
        validator = schema_validator(input_schema) if input_schema is not None else None
        self._actions[name] = RegisteredAction(
            handler,
            description,
            deepcopy(input_schema),
            validator,
            guard,
        )
        return self

    def _choices(self, context: Any, actions: list[ActionChoice]) -> dict[str, Any]:
        if not isinstance(actions, list) or not 1 <= len(actions) <= 32:
            raise ValueError("actions must contain 1–32 action choices.")
        choices = []
        for offered in actions:
            if isinstance(offered, str):
                name = offered
                description = None
            elif isinstance(offered, dict) and set(offered) <= {
                "action",
                "description",
                "parameters",
            }:
                name = offered.get("action", "")
                description = offered.get("description")
            else:
                raise ValueError(
                    "Use action names or {'action': name, 'description': text} choices."
                )
            checked_text(name, "Action name", 64)
            registered = self._actions.get(name)
            if description is None and registered is not None:
                description = registered.description
            choice: dict[str, Any] = {"action": name}
            if description is not None:
                if not isinstance(description, str) or utf16_length(description) > 1000:
                    raise ValueError("Action description must have at most 1000 UTF-16 characters.")
                choice["description"] = description
            schema = registered.schema if registered is not None else None
            if schema is None and isinstance(offered, dict):
                if "parameters" in offered:
                    schema = offered["parameters"]
                    schema_validator(schema)
            if schema is not None:
                choice["parameters"] = deepcopy(schema)
            choices.append(choice)
        return {"context": context, "actions": choices}

    def _remember(
        self, result: Result[Any], original: Any, choices: list[dict[str, Any]] | None
    ) -> Result[Any]:
        if result.success:
            from ._errors import ProtocolError

            # A saved task's actions live on the server, which checked the choice.
            chosen = (
                {}
                if choices is None
                else next((choice for choice in choices if choice["action"] == result.action), None)
            )
            if chosen is None:
                raise ProtocolError(
                    "The server returned an action that was not offered.",
                    request_id=result.request_id,
                )
            schema = chosen.get("parameters")
            if schema is not None and not schema_validator(schema).is_valid(result.parameters):
                raise ProtocolError(
                    "The server returned parameters that do not match the action schema.",
                    request_id=result.request_id,
                )
            snapshot = Result.from_response(
                {"result": result.raw, "operationId": result.operation_id},
                result.request_id,
            )
            with self._lock:
                key = result.operation_id or "request:" + result.request_id
                made = self._decision_ids.get(key)
                if made is None:
                    made = MadeDecision(snapshot, deepcopy(original))
                    self._decision_ids[key] = made
                self._decisions[result] = made
        return result

    def _consume(self, result: Result[Any]) -> tuple[MadeDecision | None, str | None]:
        if not isinstance(result, Result):
            return None, "not_a_decision"
        with self._lock:
            made = self._decisions.get(result)
            if made is None:
                return None, "not_a_decision"
            if made.executed:
                return made, "already_executed"
            made.executed = True
            return made, None

    @staticmethod
    def _task_id(value: str | None) -> str:
        return request_id(str(uuid4()) if value is None else value)
