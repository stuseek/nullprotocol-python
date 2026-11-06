# Changelog

## 0.2.0 — 2026-11-06

- Agent and Space Knowledge management, including source CRUD, state, preparation retry, connections and embedding usage.
- Explicit document version guards on replacement/deletion, with matching sync and async APIs.
- Management stays agent-wide across data scopes and does not enable retrieval automatically.

## 0.1.0 — 2026-11-06

First Python SDK.

- Synchronous and asynchronous clients for saved NullProtocol agents.
- Extract, summarize, validate and decide through the hosted run API.
- Context, explicit saved notes, shared Space values and flat data scopes.
- Per-call Knowledge selection and separate embedding/generation usage.
- Local action registration, parameter schemas, guards, snapshot execution and action History.
- Stable idempotency keys across transport retries; no replay of uncertain data writes or local handlers.
- Typed results, preserved operation IDs, unknown usage, setup/HTTP/transport errors.

This version does not include standalone model providers or the JavaScript action reply helper.
