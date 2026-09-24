# Contracts

Three documents cross the boundary between `kanon` and the rest of the world. Each is a
versioned JSON shape defined once, as serde types in `kanon::contracts`, and published as a
JSON Schema under [`docs/schemas/`](../schemas/) for consumers in other languages. A Rust
consumer uses the types directly; any other consumer validates against the schema.

## The version rule

Every document carries a `version` field, an integer. A missing `version` means 1.

- Within a version, changes are additive: a field may be added, never removed or renamed, and
  a reader ignores fields it does not know.
- A removal, rename or change of meaning is a new version.
- A reader rejects a document whose version is newer than the one it knows, with one line:
  `trail.jsonl: version 2 is newer than the version 1 this kanon reads`.

`kanon` always writes the version it speaks and checks the version before anything else in a
document, so a newer document is reported as such rather than as a parse error.

## Backend

`kanon eval --backend external --backend-url URL` sends one `POST URL/search` per query and
scores what comes back. Both directions are version 1.

Request ([schema](../schemas/backend-request.schema.json)):

```json
{"version": 1, "query": "enable caching for uploads", "k": 10, "module": null}
```

`module` restricts the search to one source when given; `kanon` sends `null` otherwise.

Response ([schema](../schemas/backend-response.schema.json);
[example](../../examples/search-response.json)):

```json
{"version": 1, "hits": [
  {"page_id": "guides::docs/enable-caching.md", "score": 12.4,
   "heading": "Enable", "unit_id": "guides::docs/enable-caching.md#1"}
]}
```

Hits come best first; `kanon` keeps the first `k` and uses only their order. `heading` may be
omitted (it defaults to empty). `unit_id` is optional and names the [unit](#unit) that matched,
so a backend that retrieves sections is graded on the same cuts `kanon` makes; today `kanon`
reads it and scores the page.

## Trail

`trail.jsonl` is what a serving consumer writes, one JSON object per line, and what
`kanon grade` replays. `pinakes usage` reads the same file. Version 1
([schema](../schemas/trail-entry.schema.json); [example](../../examples/trail.jsonl)):

```json
{"at": "2026-09-16T09:16:55Z", "query": "enable caching for uploads",
 "retrieved": ["guides::docs/enable-caching.md", "handbook::docs/concepts/storage.md"],
 "ranks": [1, 2], "cited": [], "outcome": "bad", "session": "b7c0"}
```

`at` (RFC 3339 UTC) and `query` are required. `retrieved` and `cited` hold page ids in the
`<source>::<path>` form; a line with any other id shape is rejected. `ranks` gives the rank
shown for each entry of `retrieved`, `outcome` is `ok`, `bad` or `unknown`, and `session` is
opaque. Everything optional defaults to empty or unknown, so a consumer logs as much as it has.

## Unit

A unit is one section of a page: the thing `embed` embeds and a hit refers to. Its id is
`<page_id>#<ordinal>`, the ordinal being the 0-based position of the section within its page in
document order. `kanon::contracts::units` cuts them exactly as the reference index does.
Version 1 ([schema](../schemas/unit.schema.json)):

```json
{"version": 1, "id": "guides::docs/enable-caching.md#1",
 "page_id": "guides::docs/enable-caching.md", "heading": "Enable", "ordinal": 1,
 "text": "Enable Caching\nEnable\n\n\nSet `cache.enabled = true` and `cache.size` in the configuration.",
 "sha256": "37b7f00304b910b905de1f8625a032d0160b14817c9d386b1e2f8eb34fc8e774"}
```

This is the second unit of the golden fixture's `guides/docs/enable-caching.md` (the page's
title, its `## Enable` heading and that section's body); `#0` is the intro before the first
heading, with an empty `heading`. `text` is the title, heading and body joined as embedded;
`sha256` is its hash in lower-case hex, so two sides can tell whether they cut the same unit
without exchanging the text. Pages a higher-priority source mirrors are not searchable and
yield no units.

## Keeping the schemas honest

`tests/schemas.rs` generates the schemas from the types and fails when the committed files
differ. After an intended change to a contract type, run
`UPDATE_SCHEMAS=1 cargo test --test schemas` (also part of `just update-golden`), review the
diff, and say in the commit body which contract changed and why the version did or did not.
