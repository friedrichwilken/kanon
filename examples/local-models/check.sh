#!/usr/bin/env bash
# Runs docs/tutorials/local-models-with-ollama.md end to end against a local model server, so
# the claim that kanon works with nothing but Ollama stays true. It needs the server, the two
# models and `kanon`, `curl` and `jq` on PATH; it is not part of CI (no server there), and
# `just check-local-models` runs it.
#
#   examples/local-models/check.sh
#
# Settings, all optional:
#   OLLAMA_URL          server root, default http://localhost:11434
#   EMBED_MODEL         embedding model, default nomic-embed-text
#   CHAT_MODEL          chat model, default llama3.2:3b
#   KANON               the kanon binary, default kanon (e.g. target/release/kanon)
#   BATCH               texts per embeddings request, default 16 (a CPU-only server is slow)
#   MIN_DENSE_RECALL5   fail unless the dense backend's tuning recall@5 reaches this (default: off)
#
# Everything is written under a scratch copy of the golden fixture; nothing is left behind.

set -euo pipefail

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_URL="${OLLAMA_URL%/}" # a trailing slash would make every request path start with //
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
CHAT_MODEL="${CHAT_MODEL:-llama3.2:3b}"
KANON="${KANON:-kanon}"
BATCH="${BATCH:-16}"
MIN_DENSE_RECALL5="${MIN_DENSE_RECALL5:-}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
step() { printf '\n== %s\n' "$*" >&2; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

for tool in curl jq "$KANON"; do
  command -v "$tool" >/dev/null || fail "$tool is not on PATH (KANON=$KANON)"
done

# A relative path to the binary would not survive the cd below.
if [[ "$KANON" == */* ]]; then
  KANON="$(cd "$(dirname "$KANON")" && pwd)/$(basename "$KANON")"
fi

step "the server and its models"
tags="$(curl -fsS "$OLLAMA_URL/api/tags")" || fail "no server answers at $OLLAMA_URL (is 'ollama serve' running?)"
for model in "$EMBED_MODEL" "$CHAT_MODEL"; do
  jq -e --arg m "$model" '[.models[].name] | any(. == $m or . == ($m + ":latest"))' <<<"$tags" >/dev/null \
    || fail "model $model is not pulled: run 'ollama pull $model'"
done

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cp -R "$root/tests/fixtures/golden/." "$work/"
cd "$work"

export KANON_EMBED_URL="$OLLAMA_URL/v1" KANON_EMBED_MODEL="$EMBED_MODEL"
export KANON_LLM_URL="$OLLAMA_URL/v1" KANON_LLM_MODEL="$CHAT_MODEL"
# A key or URL left over from another setup must not be sent to (or replace) the local server.
unset KANON_EMBED_KEY KANON_LLM_KEY PINAKES_EMBED_URL PINAKES_LLM_URL \
  PINAKES_EMBED_KEY PINAKES_LLM_KEY PINAKES_EMBED_MODEL PINAKES_LLM_MODEL

step "kanon embed"
"$KANON" embed --batch "$BATCH"
jq -e --arg m "$EMBED_MODEL" '.model == $m and .dimension > 0 and (.unit_ids | length) > 0
  and has("doc_prefix") and has("query_prefix")' embeddings.json >/dev/null \
  || fail "embeddings.json does not look right: $(cat embeddings.json | head -c 300)"
case "$EMBED_MODEL" in
  nomic-embed-text*)
    jq -e '.doc_prefix == "search_document: " and .query_prefix == "search_query: "' embeddings.json >/dev/null \
      || fail "nomic-embed-text should record the search_document/search_query prefixes" ;;
esac
echo "embedded $(jq '.unit_ids | length' embeddings.json) units, $(jq .dimension embeddings.json) dims," \
  "prefixes: $(jq -c '[.doc_prefix, .query_prefix]' embeddings.json)" >&2

step "kanon eval --compare bm25,dense,hybrid"
"$KANON" eval --compare bm25,dense,hybrid --json compare.json
printf '%-8s %9s %9s\n' backend 'recall@5' MRR >&2
for backend in bm25 dense hybrid; do
  jq -e --arg b "$backend" '.[$b].tuning.overall.n > 0' compare.json >/dev/null \
    || fail "eval measured nothing for $backend"
  printf '%-8s %9s %9s\n' "$backend" \
    "$(jq -r --arg b "$backend" '.[$b].tuning.overall["recall@5"]' compare.json)" \
    "$(jq -r --arg b "$backend" '.[$b].tuning.overall.mrr' compare.json)" >&2
done
# Whatever the model, a dense backend that finds none of the expected pages in ten tries has not
# embedded or searched anything usable: a broken setup, not a weak model.
jq -e '.dense.tuning.overall["recall@10"] > 0' compare.json >/dev/null \
  || fail "the dense backend found no expected page in any top 10: check the embeddings endpoint and model"
if [[ -n "$MIN_DENSE_RECALL5" ]]; then
  jq -e --argjson min "$MIN_DENSE_RECALL5" '.dense.tuning.overall["recall@5"] >= $min' compare.json >/dev/null \
    || fail "dense recall@5 is below $MIN_DENSE_RECALL5"
fi

step "kanon grade (candidates from the dense backend)"
"$KANON" grade --trail "$root/examples/trail.jsonl" --backend dense --k 5 --out graded.jsonl
[[ -s graded.jsonl ]] || fail "grade wrote no rows"
jq -e -s 'all(.[]; .grade >= 0 and .grade <= 3 and (.id | contains("::")))' graded.jsonl >/dev/null \
  || fail "graded.jsonl has a row outside grade 0-3 or without a page id"
echo "graded $(wc -l <graded.jsonl | tr -d ' ') candidates" >&2

step "kanon queries suggest"
"$KANON" queries suggest --n 3 --out suggestions.jsonl
[[ -s suggestions.jsonl ]] || fail "queries suggest wrote no suggestions"
echo "suggested $(wc -l <suggestions.jsonl | tr -d ' ') queries" >&2

printf '\nPASS: embed, eval --compare, grade and queries suggest all ran against %s\n' "$OLLAMA_URL" >&2
