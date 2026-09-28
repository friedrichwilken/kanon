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

Hits come best first and `kanon` ranks by their order, keeping the first `k` distinct pages: a
page counts once, at its best rank, and a repeat of it is dropped before the `k` are taken. That
holds for every hit, with or without a `unit_id`, and it is what the built-in backends do, so a
backend that retrieves several units of one page is measured on pages like every other. (A
backend that used to return the same page twice used to fill two of the `k` places with it; a
run file recorded before this change can therefore differ from one recorded after it.) The
first hit's `score` is recorded as the result's `top_score` and decides a negative query
against `--negative-threshold`, so it should be comparable across the backend's own answers.
`heading` may be omitted (it defaults to empty).

`unit_id` is optional and names the [unit](#unit) that matched, in the id `pinakes chunks`
gives it: `<page_id>#<ordinal>`. `kanon` scores pages, so the hit's `page_id` is what counts and
the unit is recorded beside it, as `top_units` in the run file (same length as `top`, `null` for
a hit without one; the key is absent when no hit named a unit). A hit without a `unit_id` is
scored on `page_id` alone, as before.

`kanon` reads `unit_id` more strictly than it used to, within version 1: it was parsed and
ignored, and now a `unit_id` must be `<source>::<path>#<ordinal>` for the page the hit names,
the page being everything before the last `#` (a path may contain one) and the ordinal a plain
number. A hit that is kept and says otherwise, an opaque id or the id of another page, fails
the whole `eval` with the id in the message; a backend with ids of its own should leave
`unit_id` out. Only that shape is checked: `kanon` does not look up whether the page or the
unit exists in the artifact. A hit that is not kept (past the first `k` pages, or a repeat of a
page already kept) is not checked either.

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
document order. The cut and the ids are pinakes's: `kanon::contracts::units` is
`pinakes::chunks::chunks` under this document's names (`page_id` where `chunks.jsonl` says
`page`, plus the `version`), and `pinakes chunks` writes the same ids, texts and hashes. A
consumer that indexes its own units can check its cut against either.
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

## The artifact

The artifact directory is pinakes's contract, not kanon's: `kanon` defines no type for
`manifest.json` or a source's `meta.json` and reads them only through pinakes
(`load_pages`, `Manifest::load`). For a consumer in another language, the shape of
`manifest.json` is pinakes's `docs/schemas/manifest.schema.json`.

The contract carries one integer, `artifact_version`, in `manifest.json` and in every
`meta.json`; a missing one means 1. A reader accepts an equal or lower version and rejects a
higher one, and `kanon` passes pinakes's one line on unchanged, with the file it came from:

```text
artifact/manifest.json: artifact version 2 is newer than this pinakes supports (1); upgrade pinakes
```

Every command that loads the artifact's pages (`eval` in all its forms, `grade`, `embed` and
`queries suggest`) goes through `load_pages`, so a newer `meta.json` stops it. Each of them
also reads the artifact's own `manifest.json` through `Manifest::load`, so a newer manifest
stops it too, before any query, embedding or model call is made. `queries add`, `queries check`
and `queries import` read the workspace's committed manifest through the same loader. An
artifact whose `manifest.json` is present but not a manifest at all fails those commands with
the loader's error rather than being measured or hashed anyway; an artifact with no manifest is
measured as before.

One limit is pinakes's, not kanon's. `Manifest::load` parses the typed manifest first and
compares the version afterwards, so a manifest of a newer major that also removed or renamed a
field kanon needs is reported as `invalid manifest: missing field ...` (a `version` other than
1 as `unsupported manifest version`), not with the line above. `meta.json` has no such gap: its
version is read before anything else. `kanon` cannot close this gap without parsing the file
itself, which it does not do.

A run file records the version it measured as `run.artifact_version`, next to
`manifest_sha256`, so a series says which contract each number came from. It is absent when the
artifact has no manifest, and in run files written before the field existed; `kanon history
--json` shows it per row as `artifact_version`, `null` when absent.

## Keeping the schemas honest

`tests/schemas.rs` generates the schemas from the types and fails when the committed files
differ. After an intended change to a contract type, run
`UPDATE_SCHEMAS=1 cargo test --test schemas` (also part of `just update-golden`), review the
diff, and say in the commit body which contract changed and why the version did or did not.
