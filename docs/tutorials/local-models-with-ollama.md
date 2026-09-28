# Local models with Ollama

`kanon` talks to two kinds of model endpoint, and both speak the OpenAI wire format that
[Ollama](https://ollama.com) serves on its own (llama.cpp's server does too; text-embeddings-inference
serves the embeddings half only):

- **embeddings**, for `embed` and for the `dense` and `hybrid` backends: `POST /v1/embeddings`;
- **chat**, for `grade` and `queries suggest`: `POST /v1/chat/completions`.

So a laptop, no account and no key is enough to try every part of `kanon` that uses a model.
This page does that with the golden fixture that ships in the repository, and says what a small
local model needs that a hosted one does not.

## Set up

```sh
ollama pull nomic-embed-text      # embeddings
ollama pull llama3.2:3b           # a small chat model
cargo install --git https://github.com/friedrichwilken/kanon --locked

git clone https://github.com/friedrichwilken/kanon && KANON_REPO=$PWD/kanon
mkdir /tmp/golden && cp -R "$KANON_REPO/tests/fixtures/golden/." /tmp/golden && cd /tmp/golden   # a corpus, queries, a config

export KANON_EMBED_URL=http://localhost:11434/v1 KANON_EMBED_MODEL=nomic-embed-text
export KANON_LLM_URL=http://localhost:11434/v1   KANON_LLM_MODEL=llama3.2:3b
```

No `KANON_EMBED_KEY` or `KANON_LLM_KEY`: a local server does not ask for one.

## Embed, then compare the retrievers

```sh
kanon embed --batch 16
kanon eval --compare bm25,dense,hybrid
```

`embed` writes `embeddings.bin` and `embeddings.json` and says which prefixes it used (next
section). `--batch 16` keeps each request short; on a machine with no GPU the default of 64
texts can run into the 30-second request timeout. The chat requests of `grade` and
`queries suggest` time out after 60 seconds, so load the chat model once (send it any prompt)
before the first run: a cold start on a CPU can take that long by itself. `eval --compare` prints one table per
backend over the same queries. `dense` embeds every query through the same endpoint and model
`embeddings.json` names, so `KANON_EMBED_URL` has to be set for `eval` too.

Measure, do not assume: a small local embedding model can lose to BM25 on a corpus this small,
and `hybrid` is there for when it does. The numbers are the point, so commit the runs
(`--out runs/`) and watch them as the model or the corpus changes.

## Prefixes

Open embedding models are trained with a marker in front of the text, and a different one for
a document and for a query. Left out, the model still returns vectors, only worse ones, and
`dense` silently loses to BM25. `embed` therefore puts the document prefix in front of every
unit, records **both** prefixes in `embeddings.json`, and `dense` and `hybrid` read the query
prefix from that file, so a query is never embedded by another convention than the file was
built with.

| model | document prefix | query prefix |
|---|---|---|
| `nomic-embed-text` | `search_document: ` | `search_query: ` |
| `e5-small`, `e5-base`, `e5-large`, `multilingual-e5-*` | `passage: ` | `query: ` |
| `bge-small-en`, `bge-base-en`, `bge-large-en`, `mxbai-embed-large` | none | `Represent this sentence for searching relevant passages: ` |

Those are the defaults for a model of that name (an `org/` path and a `:tag` are ignored);
every other model gets none, which is right for `bge-m3`, `all-minilm` and the OpenAI models.
To set them for any model:

```sh
kanon embed --doc-prefix 'passage: ' --query-prefix 'query: '
```

or in `kanon.yaml` (or the `eval:` block of a `pinakes.yaml`), as `doc_prefix` and
`query_prefix`; most prefixes end in a space and contain a colon, so quote them in YAML
(`doc_prefix: 'passage: '`). A flag beats the config, the config beats the model's default, and an empty
value is a choice: `--doc-prefix ''` switches a known model's prefix off. There is no `eval`
flag for the query side: the file decides.

`embeddings.json` records `model`, `dimension`, `unit_ids`, `manifest_sha256`, `doc_prefix`
and `query_prefix`. A file written before prefixes were recorded reads as having none. The
other direction does not degrade loudly: a `kanon` older than this change ignores the two new
fields and embeds queries without a prefix, so upgrade every `kanon` that reads a file built
with prefixes. If the file was built with the wrong `query_prefix` but the right `doc_prefix`,
that one field can be corrected in `embeddings.json` by hand; a wrong `doc_prefix` needs `embed`
run again. A consumer that embeds queries in-process, next to its own index, has to use the same model and
the same `query_prefix` to stay comparable with what `eval` measured.

## Grade and suggest with a small chat model

```sh
kanon grade --trail "$KANON_REPO/examples/trail.jsonl" --backend dense --k 5
kanon queries suggest --n 5
```

A small model is less careful about the reply format than the hosted models these commands
were written against: it wraps the JSON in a markdown fence, or in a sentence, or prefixes it
with its reasoning. `kanon` reads the last JSON value in the reply that has the expected shape,
whatever surrounds it (an empty `[]` in a sentence, or a format example the model repeated
first, is not taken for the answer), and a reply with none is asked for once more with a
stricter instruction. If that also fails, `grade` skips that query and `queries suggest` skips
that page, and each says how many it skipped ("queries skipped", "pages failed"). When the model
failed on every query or page it was asked about, the command stops and prints the model's raw
reply in the error. That means the model is too small for the job: try a larger one before
changing anything else.

## Check that all of this still works

[`examples/local-models/check.sh`](../../examples/local-models/check.sh) runs every step above
against a local server, in a scratch copy of the fixture, and fails on the first step that does
not hold; `just check-local-models` builds `kanon` and runs it. It needs Ollama serving the two
models, so it is not part of CI. `OLLAMA_URL`, `EMBED_MODEL` and `CHAT_MODEL` point it at
another server or model, and `MIN_DENSE_RECALL5=0.5` makes it fail when the dense backend's
tuning recall@5 falls below that.
