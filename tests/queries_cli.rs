//! `kanon queries add`, `queries check` and `queries suggest` end to end through the binary:
//! appending rows, rejecting unknown expected ids, accepting suggestions, exit 4 on a `check`
//! failure, and the one-line explanation `suggest` prints without a model endpoint (the model
//! itself is scripted in the library's own tests, which a spawned binary cannot do).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;

use pinakes::manifest::{Manifest, ManifestSource, PageEntry, SelectedBy};

fn kanon(dir: &Path, config: &str, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_kanon"))
        .current_dir(dir)
        .arg("--config")
        .arg(config)
        .args(args)
        .output()
        .expect("kanon runs")
}

/// A workspace with a committed `manifest.json` (two pages) but no `pinakes.yaml`.
fn workspace() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let mut manifest = Manifest::new("2026-09-16T12:00:00Z".to_string());
    let mut pages = BTreeMap::new();
    for path in ["docs/a.md", "docs/b.md"] {
        pages.insert(
            path.to_string(),
            PageEntry {
                sha256: "aa".repeat(32),
                title: path.to_string(),
                doc_type: "howto".to_string(),
                section: String::new(),
                selected_by: SelectedBy::Include,
                rendered_from: None,
            },
        );
    }
    manifest.sources.insert(
        "handbook".to_string(),
        ManifestSource {
            repo: "o/handbook".to_string(),
            repo_url: "https://github.com/o/handbook.git".to_string(),
            git_ref: "main".to_string(),
            commit: "a".repeat(40),
            archived: Some(false),
            resolver: "glob".to_string(),
            unrendered: Vec::new(),
            pages,
            residue: vec![],
            unresolved: vec![],
            render: None,
        },
    );
    manifest.save(&dir.path().join("manifest.json")).unwrap();
    dir
}

#[test]
fn add_normalises_expected_ids_and_rejects_unknown_ones() {
    let dir = workspace();
    let root = dir.path();

    let out = kanon(
        root,
        "nonexistent.yaml",
        &[
            "queries",
            "add",
            "--id",
            "a",
            "--query",
            "what is a",
            "--expected",
            "handbook/docs/a.md",
            "--kind",
            "howto",
            "--queries",
            "queries.jsonl",
        ],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = fs::read_to_string(root.join("queries.jsonl")).unwrap();
    assert_eq!(text.lines().count(), 1);
    let row: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert_eq!(row["expected"][0], "handbook::docs/a.md");

    // An unknown expected id exits 1 and nothing is appended.
    let out = kanon(
        root,
        "nonexistent.yaml",
        &[
            "queries",
            "add",
            "--id",
            "bad",
            "--query",
            "x",
            "--expected",
            "handbook::missing.md",
            "--queries",
            "queries.jsonl",
        ],
    );
    assert_eq!(out.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("matches no page"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        fs::read_to_string(root.join("queries.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1,
        "rejected row is not appended"
    );

    // A directory prefix in the legacy form is normalised with a trailing slash.
    let out = kanon(
        root,
        "nonexistent.yaml",
        &[
            "queries",
            "add",
            "--id",
            "b",
            "--query",
            "what is b",
            "--expected",
            "handbook/docs",
            "--holdout",
            "--queries",
            "queries.jsonl",
        ],
    );
    assert!(out.status.success());
    let lines: Vec<String> = fs::read_to_string(root.join("queries.jsonl"))
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(lines.len(), 2);
    let row: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
    assert_eq!(row["expected"][0], "handbook::docs/");
    assert_eq!(row["holdout"], true);
}

#[test]
fn check_exits_4_on_an_unmet_holdout_share_and_0_once_it_is_met() {
    let dir = workspace();
    let root = dir.path();
    fs::write(
        root.join("pinakes.yaml"),
        "version: 1\nsources:\n  - name: handbook\n    repo: https://github.com/o/handbook.git\n    \
         ref: main\n    resolver:\n      type: glob\n      include: ['**/*.md']\n\
         eval:\n  queries: queries.jsonl\n  holdout_min: 0.6\n",
    )
    .unwrap();
    fs::write(
        root.join("queries.jsonl"),
        "{\"id\": \"a\", \"query\": \"a\", \"expected\": [\"handbook::docs/a.md\"]}\n\
         {\"id\": \"b\", \"query\": \"b\", \"expected\": [\"handbook::docs/b.md\"], \"holdout\": true}\n",
    )
    .unwrap();

    // 1 of 2 rows held out (0.5) is below the configured minimum (0.6).
    let out = kanon(root, "pinakes.yaml", &["queries", "check"]);
    assert_eq!(out.status.code(), Some(4));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("held-out share: 0.500 (minimum 0.600)"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // The default minimum (0.2, no config) is met by the same file.
    let out = kanon(
        root,
        "nonexistent.yaml",
        &["queries", "check", "--queries", "queries.jsonl"],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("ok"));

    // An unknown expected id and a duplicate query id both fail the check (exit 4).
    fs::write(
        root.join("queries.jsonl"),
        "{\"id\": \"a\", \"query\": \"a\", \"expected\": [\"handbook::missing.md\"]}\n\
         {\"id\": \"a\", \"query\": \"a again\", \"expected\": [\"handbook::docs/b.md\"], \"holdout\": true}\n",
    )
    .unwrap();
    let out = kanon(
        root,
        "nonexistent.yaml",
        &["queries", "check", "--queries", "queries.jsonl"],
    );
    assert_eq!(out.status.code(), Some(4));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("unknown: a: expected \"handbook::missing.md\""),
        "{stderr}"
    );
    assert!(stderr.contains("duplicate: a"), "{stderr}");

    // No query file at all and no config: the generic "no query file" error, exit 1.
    let out = kanon(root, "nonexistent.yaml", &["queries", "check"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("no query file"));
}

#[test]
fn add_from_accepts_suggestions_and_keeps_their_origin() {
    let dir = workspace();
    let root = dir.path();
    fs::write(
        root.join("suggestions.jsonl"),
        "{\"id\": \"what-is-a-1\", \"query\": \"what is a\", \"expected\": [\"handbook::docs/a.md\"], \
         \"kind\": \"concept\", \"origin\": \"suggested\", \"page_title\": \"A\"}\n\
         {\"id\": \"what-is-b-2\", \"query\": \"what is b\", \"expected\": [\"handbook::docs/b.md\"], \
         \"kind\": \"concept\", \"origin\": \"suggested\", \"page_title\": \"B\"}\n",
    )
    .unwrap();

    let out = kanon(
        root,
        "nonexistent.yaml",
        &[
            "queries",
            "add",
            "--from",
            "suggestions.jsonl",
            "--accept",
            "what-is-b-2",
            "--queries",
            "queries.jsonl",
        ],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("accepted 1 suggestions"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = fs::read_to_string(root.join("queries.jsonl")).unwrap();
    assert_eq!(text.lines().count(), 1);
    let row: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert_eq!(row["id"], "what-is-b-2");
    assert_eq!(row["origin"], "suggested");
    assert!(row.get("page_title").is_none(), "{row}");

    // An id that is not in the file exits 1 and appends nothing.
    let out = kanon(
        root,
        "nonexistent.yaml",
        &[
            "queries",
            "add",
            "--from",
            "suggestions.jsonl",
            "--accept",
            "nope",
            "--queries",
            "queries.jsonl",
        ],
    );
    assert_eq!(out.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("names no row in the suggestions file"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        fs::read_to_string(root.join("queries.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );

    // --from is exclusive with --id/--query/--expected and needs a selection; a plain add
    // still needs all three.
    let out = kanon(
        root,
        "nonexistent.yaml",
        &[
            "queries",
            "add",
            "--from",
            "suggestions.jsonl",
            "--accept-all",
            "--id",
            "x",
            "--queries",
            "queries.jsonl",
        ],
    );
    assert_eq!(out.status.code(), Some(2), "clap usage error");
    let out = kanon(
        root,
        "nonexistent.yaml",
        &["queries", "add", "--from", "suggestions.jsonl"],
    );
    assert_eq!(out.status.code(), Some(2), "clap usage error");
    let out = kanon(root, "nonexistent.yaml", &["queries", "add", "--id", "x"]);
    assert_eq!(out.status.code(), Some(2), "clap usage error");
}

#[test]
fn suggest_explains_the_missing_endpoint_and_exits_1() {
    let dir = workspace();
    let out = Command::new(env!("CARGO_BIN_EXE_kanon"))
        .current_dir(dir.path())
        .env_remove("KANON_LLM_URL")
        .env_remove("PINAKES_LLM_URL")
        .args(["--config", "nonexistent.yaml", "queries", "suggest"])
        .output()
        .expect("kanon runs");
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("KANON_LLM_URL is not set") && stderr.contains("queries suggest"),
        "{stderr}"
    );
    assert!(out.stdout.is_empty());
    assert!(!dir.path().join("suggestions.jsonl").exists());
}
