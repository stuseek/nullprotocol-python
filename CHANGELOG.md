# Changelog

## 0.5.0

- Add sync/async `agent.run(task, input)` for saved tasks. Results carry `task` with its name, revision and operation. Requires API 0.25.0.

## 0.4.0

- Add sync/async Knowledge wait_until_ready for exact revisions, metadata polling and bounded timeouts; async task cancellation cancels reads.
- Send action parameter schemas as first-class choices (API0.24+) so the engine can repair them before success; reject schema-invalid successful server replies.
- Clarify Context key/value errors and JSON null support. Align the exported __version__ with package metadata.

## 0.3.0

- Add images to all four sync/async operations, with HTTPS URL validation, explicit model capability refusals and unchanged text requests.

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
