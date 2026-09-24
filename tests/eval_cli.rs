//! `kanon eval` end to end through the binary: JSON on stdout or in `--json`, the table on
//! stderr, exit 2 when `--gate` fails, the nDCG columns, the negative share and
//! `--gate-metric`.

use std::fs;
use std::path::Path;
use std::process::Command;

fn kanon(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_kanon"))
        .current_dir(dir)
        .arg("--config")
        .arg("nonexistent.yaml")
        .args(args)
        .output()
        .expect("kanon runs")
}

fn workspace() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let page = dir.path().join("artifact/handbook/docs/user/README.md");
    fs::create_dir_all(page.parent().unwrap()).unwrap();
    fs::write(&page, "# Storage\n\nEnable upload caching with a label.\n").unwrap();
    fs::write(
        dir.path().join("queries.jsonl"),
        "{\"id\": \"q\", \"kind\": \"howto\", \"query\": \"enable upload caching\", \
         \"expected\": [\"handbook/docs/user\"]}\n",
    )
    .unwrap();
    dir
}

#[test]
fn eval_writes_json_and_exits_2_when_the_gate_fails() {
    let dir = workspace();
    let root = dir.path();

    // No manifest, no meta.json, no config: the artifact is still measurable.
    let out = kanon(root, &["eval", "--queries", "queries.jsonl"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let summary: serde_json::Value = serde_json::from_str(&stdout).expect("JSON on stdout");
    assert_eq!(summary["tuning"]["overall"]["recall@5"], 1.0);
    assert_eq!(
        summary["queries"][0]["top"][0],
        "handbook::docs/user/README.md"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("| tuning | overall | 1 | 1.000 | 1.000 | 1.000 |"),
        "{stderr}"
    );
    assert!(stderr.contains("1 pages, 1 searchable, k = 10"), "{stderr}");

    // --json writes the file and keeps stdout empty; a matching baseline passes the gate.
    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--json",
            "baseline.json",
        ],
    );
    assert!(out.status.success());
    assert!(out.stdout.is_empty());
    assert!(root.join("baseline.json").is_file());
    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--gate",
            "baseline.json",
        ],
    );
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("gate: tuning recall@5 1.000 → 1.000"));

    // A query that misses drops recall@5 from 1.0 to 0.0: beyond the default 0.05 tolerance.
    fs::write(
        root.join("queries.jsonl"),
        "{\"id\": \"q\", \"kind\": \"howto\", \"query\": \"nothing here\", \
         \"expected\": [\"handbook/docs/user\"]}\n",
    )
    .unwrap();
    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--gate",
            "baseline.json",
        ],
    );
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("FAILED"));

    // Errors exit 1: an unknown page for --without, and a missing query file.
    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--without",
            "handbook::nope.md",
        ],
    );
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("not in the corpus"));
    let out = kanon(root, &["eval"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("no query file"));
}

/// The workspace plus a second page, a graded query over both pages (`README.md` at 3,
/// `quotas.md` at 1) and two negative queries, one that nothing matches and one that the
/// quotas page answers weakly.
fn graded_workspace() -> tempfile::TempDir {
    let dir = workspace();
    let other = dir.path().join("artifact/handbook/docs/user/quotas.md");
    fs::write(&other, "# Quotas\n\nUpload quotas and caching limits.\n").unwrap();
    fs::write(
        dir.path().join("queries.jsonl"),
        "{\"id\": \"q\", \"kind\": \"howto\", \"query\": \"enable upload caching\", \
         \"expected\": [\"handbook::docs/user/README.md\"], \
         \"graded\": {\"handbook::docs/user/README.md\": 3, \"handbook::docs/user/quotas.md\": 1}}\n\
         {\"id\": \"none\", \"kind\": \"negative\", \"query\": \"zebra chess brackets\", \
         \"expected\": []}\n\
         {\"id\": \"weak\", \"kind\": \"negative\", \"query\": \"quotas\", \"expected\": []}\n",
    )
    .unwrap();
    dir
}

/// A graded row and two negative rows: the table carries the nDCG columns and the negative
/// line, the JSON carries `ndcg@5`/`ndcg@10`, `rels`, `top_score` and the `negative` block, and
/// the threshold decides what counts as rejected.
#[test]
fn eval_reports_ndcg_columns_and_the_negative_share() {
    let dir = graded_workspace();
    let root = dir.path();

    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--json",
            "baseline.json",
        ],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("| split | kind | n | recall@5 | recall@10 | MRR | nDCG@5 | nDCG@10 |"),
        "{stderr}"
    );
    assert!(
        stderr.contains("| tuning | overall | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |"),
        "{stderr}"
    );
    assert!(
        stderr.contains("tuning: 2 negative queries, 1 rejected (0.500)"),
        "{stderr}"
    );
    let summary: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("baseline.json")).unwrap()).unwrap();
    assert_eq!(summary["tuning"]["overall"]["ndcg@5"], 1.0);
    assert_eq!(summary["tuning"]["overall"]["ndcg@10"], 1.0);
    assert_eq!(summary["tuning"]["overall"]["n"], 1);
    assert!(summary["tuning"]["per_kind"].get("negative").is_none());
    assert_eq!(summary["tuning"]["negative"]["n"], 2);
    assert_eq!(summary["tuning"]["negative"]["rejected"], 1);
    assert_eq!(summary["tuning"]["negative"]["share"], 0.5);
    let q = &summary["queries"][0];
    assert_eq!(q["rels"], serde_json::json!([3, 1]));
    assert_eq!(q["ndcg5"], 1.0);
    assert!(q["top_score"].is_f64(), "{q}");
    let none = &summary["queries"][1];
    assert_eq!(none["top"], serde_json::json!([]));
    assert!(none.get("top_score").is_none(), "{none}");

    // A threshold above every score rejects the weak one too.
    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--negative-threshold",
            "1000",
        ],
    );
    assert!(out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("2 negative queries, 2 rejected (1.000)"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// `--gate-metric` (or `gate_metric` in the config) names the tuning metric the gate compares;
/// the flag wins over the config and an unknown name is a usage error.
#[test]
fn eval_gates_on_the_named_metric() {
    let dir = graded_workspace();
    let root = dir.path();
    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--json",
            "baseline.json",
        ],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // --gate-metric names the metric in the gate line; the baseline matches, so it passes.
    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--gate",
            "baseline.json",
            "--gate-metric",
            "ndcg5",
        ],
    );
    assert!(out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("gate: tuning nDCG@5 1.000 → 1.000"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--gate",
            "baseline.json",
            "--gate-metric",
            "precision5",
        ],
    );
    assert_eq!(out.status.code(), Some(2), "clap usage error");
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown gate metric"));

    // The config's gate_metric applies to a bare eval; the flag still wins over it. Swapping
    // the grades (README.md 1, quotas.md 3, returned in that order) drops nDCG@5 to
    // (1 + 7/log2 3) / (7 + 1/log2 3) = 0.710 while recall@5 stays at 1.0, so the gate fails on
    // ndcg5 and passes on recall5.
    fs::write(
        root.join("kanon.yaml"),
        "queries: queries.jsonl\nmax_recall_drop: 0.1\ngate_metric: ndcg5\n",
    )
    .unwrap();
    fs::write(
        root.join("queries.jsonl"),
        "{\"id\": \"q\", \"kind\": \"howto\", \"query\": \"enable upload caching\", \
         \"expected\": [\"handbook::docs/user/\"], \
         \"graded\": {\"handbook::docs/user/README.md\": 1, \"handbook::docs/user/quotas.md\": 3}}\n",
    )
    .unwrap();
    let run = |extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_kanon"))
            .current_dir(root)
            .args(["--config", "kanon.yaml", "eval", "--gate", "baseline.json"])
            .args(extra)
            .output()
            .expect("kanon runs")
    };
    let out = run(&[]);
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("gate: tuning nDCG@5 1.000 → 0.710, drop +0.290, max 0.100: FAILED"),
        "{stderr}"
    );
    let out = run(&["--gate-metric", "recall5"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("gate: tuning recall@5 1.000 → 1.000"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A config that fixes `eval.backend` makes a bare `eval` measure that backend,
/// and `--backend` on the command line still overrides it.
#[test]
fn eval_takes_the_backend_from_the_config() {
    let dir = workspace();
    let root = dir.path();
    fs::write(
        root.join("pinakes.yaml"),
        "version: 1\nsources:\n  - name: handbook\n    repo: https://github.com/example-org/handbook.git\n    \
         ref: main\n    resolver:\n      type: glob\n      include: ['docs/**/*.md']\n\
         eval:\n  queries: queries.jsonl\n  backend: bm25-tantivy\n",
    )
    .unwrap();
    let run = |extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_kanon"))
            .current_dir(root)
            .args(["--config", "pinakes.yaml", "eval"])
            .args(extra)
            .output()
            .expect("kanon runs")
    };

    let out = run(&[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(json["backend"], "bm25-tantivy");

    let out = run(&["--backend", "bm25"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(json["backend"], "bm25", "the flag wins over the config");
}

/// `eval.compare` in the config turns a bare `eval` into a comparison, one result per backend.
#[test]
fn eval_takes_the_comparison_from_the_config() {
    let dir = workspace();
    let root = dir.path();
    fs::write(
        root.join("pinakes.yaml"),
        "version: 1\nsources:\n  - name: handbook\n    repo: https://github.com/example-org/handbook.git\n    \
         ref: main\n    resolver:\n      type: glob\n      include: ['docs/**/*.md']\n\
         eval:\n  queries: queries.jsonl\n  compare: [bm25, bm25-tantivy]\n",
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_kanon"))
        .current_dir(root)
        .args(["--config", "pinakes.yaml", "eval"])
        .output()
        .expect("kanon runs");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(json["bm25"]["backend"], "bm25");
    assert_eq!(json["bm25-tantivy"]["backend"], "bm25-tantivy");
}

/// `--backend bm25`, given explicitly with no config, goes through the tagged backend path: the
/// JSON result carries `"backend":"bm25"` and the stderr counts line is bracketed `[bm25]` —
/// unlike the plain path, which has neither (see `eval_plain_has_no_backend_key_or_bracket_tag`
/// below and `eval_writes_json_and_exits_2_when_the_gate_fails` above).
#[test]
fn eval_backend_bm25_explicit_flag_tags_the_output() {
    let dir = workspace();
    let root = dir.path();
    let out = kanon(
        root,
        &["eval", "--queries", "queries.jsonl", "--backend", "bm25"],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(json["backend"], "bm25");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("[bm25]: 1 pages, 1 searchable, k = 10"),
        "{stderr}"
    );
}

/// `--backend bm25 --gate BASELINE`: exit 0 and a `"gate [bm25]:"` line when it passes, exit 2
/// and `"gate [bm25]:"` plus `"FAILED"` when it doesn't — the tagged-path mirror of
/// `eval_writes_json_and_exits_2_when_the_gate_fails`'s plain-path gate coverage.
#[test]
fn eval_backend_bm25_gate_passes_and_fails() {
    let dir = workspace();
    let root = dir.path();

    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--backend",
            "bm25",
            "--json",
            "baseline.json",
        ],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--backend",
            "bm25",
            "--gate",
            "baseline.json",
        ],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("gate [bm25]:"));

    fs::write(
        root.join("queries.jsonl"),
        "{\"id\": \"q\", \"kind\": \"howto\", \"query\": \"nothing here\", \
         \"expected\": [\"handbook/docs/user\"]}\n",
    )
    .unwrap();
    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--backend",
            "bm25",
            "--gate",
            "baseline.json",
        ],
    );
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("gate [bm25]:"), "{stderr}");
    assert!(stderr.contains("FAILED"), "{stderr}");
}

/// A bare `--allow-stale` (no `--backend`) still routes through the tagged backend path, at
/// `bm25` — a quirk of the current dispatch predicate (`allow_stale` only means anything for
/// `dense`/`hybrid`), pinned so a refactor doesn't silently change it.
#[test]
fn eval_allow_stale_alone_goes_through_the_backend_path() {
    let dir = workspace();
    let root = dir.path();
    let out = kanon(
        root,
        &["eval", "--queries", "queries.jsonl", "--allow-stale"],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(json["backend"], "bm25");
    assert!(String::from_utf8_lossy(&out.stderr).contains("[bm25]"));
}

/// The plain path (no backend-selecting flag at all) never tags its output — the mirror image
/// of the `--backend`/`--allow-stale` pins above.
#[test]
fn eval_plain_has_no_backend_key_or_bracket_tag() {
    let dir = workspace();
    let root = dir.path();
    let out = kanon(root, &["eval", "--queries", "queries.jsonl"]);
    assert!(out.status.success());
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(json.get("backend").is_none(), "{json}");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains('['), "{stderr}");
}

/// A configured `eval.compare` only turns a *bare* `eval` into a comparison; `--backend` on the
/// command line still measures just that one backend, ignoring the configured comparison.
#[test]
fn eval_config_compare_is_ignored_when_backend_flag_given() {
    let dir = workspace();
    let root = dir.path();
    fs::write(
        root.join("pinakes.yaml"),
        "version: 1\nsources:\n  - name: handbook\n    repo: https://github.com/example-org/handbook.git\n    \
         ref: main\n    resolver:\n      type: glob\n      include: ['docs/**/*.md']\n\
         eval:\n  queries: queries.jsonl\n  compare: [bm25, bm25-tantivy]\n",
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_kanon"))
        .current_dir(root)
        .args(["--config", "pinakes.yaml", "eval", "--backend", "bm25"])
        .output()
        .expect("kanon runs");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(json["backend"], "bm25");
    assert!(json.get("bm25").is_none(), "{json}");
    assert!(json.get("bm25-tantivy").is_none(), "{json}");
}

/// `eval.embeddings`, when relative, resolves against the config file's directory, not the
/// process's cwd. Pinned through the error `dense` raises when the resolved file is missing,
/// since exercising a real read needs no network: the config lives in `sub/`, so a correct
/// resolution reports `sub/custom/embeddings.json`, not `custom/embeddings.json` off the cwd.
#[test]
fn eval_config_relative_embeddings_path_resolves_against_config_dir() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let sub = root.join("sub");
    let page = sub.join("artifact/handbook/docs/user/README.md");
    fs::create_dir_all(page.parent().unwrap()).unwrap();
    fs::write(&page, "# Storage\n\nEnable upload caching with a label.\n").unwrap();
    fs::write(
        sub.join("queries.jsonl"),
        "{\"id\": \"q\", \"kind\": \"howto\", \"query\": \"enable upload caching\", \
         \"expected\": [\"handbook/docs/user\"]}\n",
    )
    .unwrap();
    fs::write(
        sub.join("pinakes.yaml"),
        "version: 1\nsources:\n  - name: handbook\n    repo: https://github.com/example-org/handbook.git\n    \
         ref: main\n    resolver:\n      type: glob\n      include: ['docs/**/*.md']\n\
         eval:\n  queries: queries.jsonl\n  embeddings: custom/embeddings.bin\n",
    )
    .unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_kanon"))
        .current_dir(root)
        .env("KANON_EMBED_URL", "http://127.0.0.1:1")
        .args(["--config", "sub/pinakes.yaml", "eval", "--backend", "dense"])
        .output()
        .expect("kanon runs");
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    // `--config sub/pinakes.yaml` is relative, so the resolved path stays relative too (joined
    // against the config's directory, "sub", not the process's cwd, which would omit "sub/").
    assert!(stderr.contains("sub/custom/embeddings.json"), "{stderr}");
}

/// `--backend dense` needs `KANON_EMBED_URL`; without it, the exact wording must survive byte
/// for byte.
#[test]
fn eval_backend_dense_without_embed_url_errors() {
    let dir = workspace();
    let root = dir.path();
    let out = Command::new(env!("CARGO_BIN_EXE_kanon"))
        .current_dir(root)
        .env_remove("KANON_EMBED_URL")
        .env_remove("PINAKES_EMBED_URL")
        .args([
            "--config",
            "nonexistent.yaml",
            "eval",
            "--queries",
            "queries.jsonl",
            "--backend",
            "dense",
        ])
        .output()
        .expect("kanon runs");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("KANON_EMBED_URL is not set (needed for --backend dense/hybrid)"),
        "{stderr}"
    );
}
