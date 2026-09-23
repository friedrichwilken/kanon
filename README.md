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

Pre-release. The evaluation commands are being moved here out of `pinakes`, where they started;
see the [issues](https://github.com/friedrichwilken/kanon/issues) for the plan.

## Quick start

Coming with the first release. The intended shape:

```sh
cargo install --git https://github.com/friedrichwilken/kanon --tag v1
kanon eval --artifact ./artifact --queries queries.jsonl
```

## Licence

Apache-2.0.
