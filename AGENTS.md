# AGENTS.md

Guidance for coding agents (and people) working in this repository.

## What this is

`kanon` measures a retriever against a corpus, and keeps measuring it as both change. The
corpus is an artifact directory as [pinakes](https://github.com/friedrichwilken/pinakes)
compiles it; the retriever is the built-in reference index, an embeddings endpoint, or any
search service that answers one HTTP request. Every input and output is a plain file that
belongs in git: a query set, a run, a gate, a report.

## Where it sits

- **pinakes** compiles sources into a curated corpus (artifact, manifest, residue). It is a
  library dependency here, for the reference index and the artifact reader. Nothing about
  curating belongs in this repository.
- **kanon** measures. `eval`, `embed`, `grade`, `queries` and `report` live here.
- **A consumer** serves. It implements the backend contract so `kanon` can score what is
  actually running, and writes `trail.jsonl` so `grade` can learn from real queries.

## Layout

**Depend on nothing else in the crate:** `config` (`kanon.yaml`, or the `eval:` block of
`pinakes.yaml`, and source priorities read from `pinakes.yaml`), `workspace` (`Paths`), `env`
(`KANON_*` with a `PINAKES_*` fallback), `num` and `rng`. `llm` (the model endpoint from
`KANON_LLM_*`) sits on `env`.

**Contracts:** `contracts` (the versioned serde types of the backend contract, the trail and
the unit, the readers that check a document's version, and the source of the JSON Schemas
under `docs/schemas/`); it sits on pinakes alone.

**Measuring:** `eval` (the judge, the metrics, the result file); `embed` (the embeddings file
pair and the `Embedder` trait); `backend` (the `Backend` trait and `bm25`, `tantivy`, `dense`,
`hybrid`, `external`); `grade` (replay a trail, ask the model); `queries` (grow and validate
the judge); `report` (the evaluation sections of a Markdown report); `history` (the numbered
run files `eval --out` writes and the rows `history` reads back).

**Commands:** `error` (`CommandError`), `commands` (one file per subcommand: `eval`, `embed`,
`grade`, `queries`, `report`, `history`; options, outcome, implementation, plus `settings`,
what every command reads first). The binary's `src/cli/` is arguments, dispatch and printing
only.

## Rules

- Human output goes to stderr, data to stdout. Exit codes: 0 ok, 1 error, 2 a failed
  `eval --gate`, 4 a failed `queries check`.
- Errors are `thiserror` types in the library and `anyhow` at the CLI edge.
- The golden tests pin behaviour rather than assert it. Refresh only when the change is
  intended, and say why in the commit body.
- Nothing here touches the network in `cargo test`.
- No personal or employer information in code, fixtures, docs, commits or issues.
