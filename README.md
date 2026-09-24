# kanon

`kanon` measures a retriever against a corpus, and keeps measuring it as both change.

> κανών: a measuring rod. The Alexandrian librarians used the word for their lists of the authors
> worth reading, the first canon. This tool is the rod, not the list.

## Why

Every retrieval system has a recall number somebody once computed and nobody can reproduce.
The query set lived in a notebook, the corpus has changed since, and the number was measured
over chunks the serving index never cut. So the system is tuned by feel, and a change that
helps one question and hurts three ships anyway.

`kanon` makes retrieval quality a file in git: a query set you commit, a run you can replay,
a gate that fails a pull request when recall drops, and a history that shows how the numbers
got where they are. It is retriever-agnostic. It scores the built-in reference index, an
embedding endpoint, or any search service that answers one HTTP request.

## Where it sits

```text
sources  ──►  pinakes   ──►  artifact + manifest  ──►  serapeum  ──►  agents
                                    │                       │
                              queries.jsonl             trail.jsonl
                                    │                       │
                                    └──────►  kanon  ◄──────┘
```

- [pinakes](https://github.com/friedrichwilken/pinakes) compiles a curated corpus.
- `kanon` measures any retriever against it.
- serapeum serves it. It implements the backend contract, so `kanon` can score what is actually
  running.

Each one reads and writes plain files. None needs the others at runtime.

## Status

Pre-release. The evaluation commands (`eval`, `embed`, `grade`, `queries`, `report`) have
moved here out of `pinakes`, where they started, with the same flags and file formats; see
the [issues](https://github.com/friedrichwilken/kanon/issues) for what comes next.

## Quick start

```sh
cargo install --git https://github.com/friedrichwilken/kanon --locked
kanon eval --artifact ./artifact --queries queries.jsonl
```

`eval` prints one table per split on stderr and the result as JSON on stdout:

```text
./artifact: 33 pages, 30 searchable, k = 10
| split | kind | n | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|
| tuning | overall | 12 | 0.917 | 0.917 | 0.917 |
| held-out | overall | 2 | 1.000 | 1.000 | 1.000 |
```

A `kanon.yaml` next to the artifact fixes what a bare `kanon eval` measures; a `pinakes.yaml`
with an `eval:` block works as before. `eval --json baseline.json` records a run,
`eval --gate baseline.json` exits 2 when tuning recall@5 drops by more than `max_recall_drop`,
and `eval --compare bm25,bm25-tantivy,dense,hybrid,external` scores several retrievers over
the same query set. Model endpoints come from `KANON_EMBED_URL` and `KANON_LLM_URL` (the
`PINAKES_*` names still work).

`eval --out runs/` also writes the result as `runs/NNN-<label>.json`, numbered after the last
run in the directory and labelled with `--label` or the git short SHA, with the backend, the
manifest and query-set hashes and the time inside. `kanon history` lists those runs as a
Markdown table (`--json OUT` for the rows as JSON), and `report --runs runs/` adds the same
table as a History section. Commit the directory and the numbers have a series.

## Licence

Apache-2.0.
