from __future__ import annotations

import json
from copy import deepcopy
from uuid import uuid4

import httpx
import pytest


class FakeAPI:
    def __init__(self):
        self.requests = []
        self.runs = {}
        self.model_calls = 0
        self.actions = []
        self.entries = {}
        self.notes = []
        self.tasks = {"read-days": "extract", "check-reply": "validate", "route": "decide"}
        self.settings = {
            "agentId": "support",
            "provider": "nullprotocol",
            "model": "scripted",
            "instructions": "Read facts.",
        }
        self.override = None
        self.result_override = None

    def __call__(self, request):
        self.requests.append(request)
        if self.override:
            response = self.override(request)
            if response is not None:
                return response
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        if path.endswith("/settings"):
            if request.method == "PATCH":
                self.settings.update(body)
            return httpx.Response(200, json={"agent": self.settings})
        if path == "/v1/agents" and request.method == "POST":
            self.settings.update(body)
            self.settings["agentId"] = body.get("agentId", "issued-id")
            return httpx.Response(201, json={"agent": self.settings})
        if path.endswith("/run"):
            # A saved task holds the operation the caller does not send.
            task = path.split("/tasks/")[1].split("/")[0] if "/tasks/" in path else None
            operation = self.tasks[task] if task else body["operation"]
            key = request.headers["Idempotency-Key"]
            if key in self.runs:
                old_body, response = self.runs[key]
                if old_body != body:
                    return httpx.Response(409, json={"error": "idempotency_mismatch"})
                return httpx.Response(200, json=response)
            self.model_calls += 1
            outputs = {
                "extract": {"data": {"days": 30}},
                "summarize": {
                    "summary": "A broken mug needs replacing.",
                    "keyPoints": ["Replace the mug."],
                },
                "validate": {
                    "recommendation": "fail",
                    "score": 0,
                    "reasoning": "The rule excludes gift cards.",
                },
                "decide": {
                    "action": body.get("actions", [{"action": "send_message"}])[0]["action"],
                    "parameters": {"text": "Good morning!"},
                    "reasoning": "Reply to the greeting.",
                },
            }
            result = (
                self.result_override
                if self.result_override is not None
                else {
                    "success": True,
                    **outputs[operation],
                    "attempts": 1,
                    "repaired": False,
                    "usage": {"inputTokens": 100, "outputTokens": 20},
                }
            )
            response = {"result": deepcopy(result), "operationId": str(uuid4())}
            if task:
                response["task"] = {"name": task, "revision": 2, "operation": operation}
            self.runs[key] = (body, response)
            return httpx.Response(200, json=response)
        if path.endswith("/actions"):
            self.actions.append(body)
            return httpx.Response(201, json={"recorded": True})
        if "/context" in path:
            base, _, key = path.partition("/context")
            slot = (base, request.url.params.get("conversation"), key.lstrip("/"))
            if request.method == "PUT":
                entry = {
                    "key": slot[-1],
                    "value": body["value"],
                    "inclusion": body.get("inclusion", "always"),
                    "version": str(uuid4()),
                }
                self.entries[slot] = entry
                return httpx.Response(200, json={"entry": entry})
            if request.method == "DELETE":
                self.entries.pop(slot, None)
                return httpx.Response(200, json={"deleted": True})
            if slot[-1]:
                return httpx.Response(200, json={"entry": self.entries[slot]})
            return httpx.Response(
                200,
                json={"entries": [entry for k, entry in self.entries.items() if k[:2] == slot[:2]]},
            )
        if path.endswith("/memory"):
            if request.method == "POST":
                note = {"id": str(uuid4()), **body}
                self.notes.append(note)
                return httpx.Response(201, json={"note": note})
            return httpx.Response(200, json={"notes": self.notes})
        if "/memory/" in path and request.method == "DELETE":
            self.notes = [n for n in self.notes if n["id"] != path.rsplit("/", 1)[1]]
            return httpx.Response(200, json={"deleted": True})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")


@pytest.fixture
def api():
    return FakeAPI()
