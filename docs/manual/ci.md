# Gate a pull request

The gate only helps if it runs on every change to the corpus, the query set or the retriever.
Two pieces do that: a composite action that puts `kanon` on the runner's PATH, and a reusable
workflow that runs `kanon eval --gate`, writes the before/after report as the job summary and
as one pull request comment, and can commit the run file on the default branch.

## What the workflow does

`friedrichwilken/kanon/.github/workflows/eval.yml` runs, in the caller's checkout:

```sh
kanon --config $config eval --gate $baseline --json eval.json --out $runs_dir \
  [--artifact $artifact] [--queries $queries] [--backend $backend --backend-url $backend_url]
kanon report --eval-before $baseline --eval-after eval.json --runs $runs_dir
```

The report goes to the job summary and, on a pull request, into one comment that is updated on
every push (found again by its first line, `<!-- kanon-eval: NAME -->`). The job's exit code is
`kanon eval`'s: **0** the gate passed, **2** the gated tuning metric (recall@5, or the config's
`gate_metric`) dropped by more than `max_recall_drop`, **1** an error. A failed gate fails the
job after the summary and the
comment are written, so the numbers are visible on the pull request either way.

When the baseline file does not exist yet, the run is recorded but not gated. With
`commit_run: true` on the default branch, the new run file is committed as
`github-actions[bot]`, and the baseline with it when the gate passed, so the baseline is always
the last run on the default branch that passed. That is how a repository bootstraps: the first
push writes `eval.json`, every pull request after that is measured against it.

Inputs, all optional:

| input | default | meaning |
|---|---|---|
| `config` | `kanon.yaml` | `kanon.yaml`, or a `pinakes.yaml` with an `eval:` block (`--config`) |
| `artifact` | | artifact directory; empty is `artifact` next to the config |
| `prepare` | | a shell command run first, e.g. one that builds or downloads the artifact |
| `queries` | | query file; empty is the one named in the config |
| `baseline` | `eval.json` | the result to gate against; missing means record only |
| `backend` | | `bm25`, `bm25-tantivy`, `dense`, `hybrid` or `external` |
| `backend_url` | | the search endpoint for `backend: external` |
| `runs_dir` | `runs` | where `eval --out` writes the numbered run file |
| `commit_run` | `false` | commit the run file (and a passing baseline) on the default branch |
| `kanon_version` | `latest` | a release version, `latest`, or `source` |
| `name` | `kanon` | tells two calls in one workflow apart (comment, summary) |
| `runs_on` | `ubuntu-latest` | the runner label |

Permissions come from the caller: `contents: read` is enough for the gate, the comment needs
`pull-requests: write`, `commit_run` needs `contents: write`. Pull requests from forks get a
read-only token, so the comment is skipped there with a warning; the summary still shows.

## A corpus repository

The repository holds `pinakes.yaml` (or `kanon.yaml`), `queries.jsonl`, the committed
artifact, `eval.json` and `runs/`. Every pull request is gated; every push to `main` records
a run.

```yaml
name: retrieval

on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: write
  pull-requests: write

jobs:
  gate:
    uses: friedrichwilken/kanon/.github/workflows/eval.yml@main
    with:
      config: pinakes.yaml
      baseline: eval.json
      commit_run: true
```

If the artifact is not committed but built, produce it first:

```yaml
    with:
      prepare: pinakes build --config pinakes.yaml
      artifact: artifact
```

## A service repository

The service implements the backend contract. The workflow deploys it to staging, then measures
what is actually running against the corpus's query set: same workflow, `backend: external`
and the staging URL. The corpus (artifact, `queries.jsonl`, `eval.json`) is checked out or
downloaded in `prepare`; here it is a second checkout.

```yaml
name: retrieval

on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read
  pull-requests: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - run: ./scripts/deploy-staging.sh

  gate:
    needs: deploy
    uses: friedrichwilken/kanon/.github/workflows/eval.yml@main
    with:
      prepare: git clone --depth 1 https://github.com/example-org/corpus corpus
      config: corpus/kanon.yaml
      baseline: corpus/eval.json
      backend: external
      backend_url: https://staging.example.org/search
      name: staging
```

A gate that fails here means the deployed service answers worse than the recorded baseline,
whatever the reference index would have said.

## The action on its own

`uses: friedrichwilken/kanon@main` installs `kanon` and nothing else, for any workflow that
wants to run the commands itself:

```yaml
steps:
  - uses: actions/checkout@v7
  - uses: friedrichwilken/kanon@main
    with:
      version: latest # a version such as 0.1.0, latest, or source
      github-token: ${{ github.token }}
  - run: kanon eval --gate eval.json
```

`latest` and a version number download the release tarball for the runner (Linux x86_64,
Linux aarch64, macOS aarch64) and verify it against its `.sha256` sidecar; `source` runs
`cargo install --locked` on the commit the action was referenced at, which takes a few minutes
and needs no release. While there is no release, or for a runner without a release build, the
download falls back to the source build with a notice.

Outputs: `path` (the directory the binary is in) and `version` (the installed version, or
`source`).
