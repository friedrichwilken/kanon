//! The artifact contract version (pinakes SPEC §2.8) through the binary: a run records the
//! `artifact_version` of the manifest it measured, and an artifact of a newer contract stops
//! `eval`, `eval --compare` and `embed` with pinakes's own line before anything is measured.

use std::fs;
use std::path::Path;
use std::process::Command;

const NEWER: &str = "artifact version 2 is newer than this pinakes supports (1); upgrade pinakes";

fn kanon(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_kanon"))
        .current_dir(dir)
        // An embeddings endpoint nothing listens on: a request would fail with another message.
        .env("KANON_EMBED_URL", "http://127.0.0.1:1")
        .env("KANON_EMBED_MODEL", "test-model")
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

/// The artifact's `manifest.json`, `artifact_version` left out when `None` (a manifest written
/// before the field existed).
fn write_manifest(root: &Path, artifact_version: Option<u32>) {
    let version =
        artifact_version.map_or(String::new(), |v| format!("\"artifact_version\": {v}, "));
    fs::write(
        root.join("artifact/manifest.json"),
        format!("{{\"version\": 1, {version}\"generated_at\": \"2026-09-16T12:00:00Z\", \"sources\": {{}}}}"),
    )
    .unwrap();
}

fn stderr(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn run_file(root: &Path, name: &str) -> serde_json::Value {
    serde_json::from_str(&fs::read_to_string(root.join(name)).unwrap()).unwrap()
}

#[test]
fn a_run_records_the_artifact_version_of_the_manifest_it_measured() {
    let dir = workspace();
    let root = dir.path();
    let eval = |label: &str| {
        let out = kanon(
            root,
            &[
                "eval",
                "--queries",
                "queries.jsonl",
                "--out",
                "runs",
                "--label",
                label,
            ],
        );
        assert!(out.status.success(), "{}", stderr(&out));
    };

    // No manifest: no version, as before.
    eval("bare");
    assert!(
        run_file(root, "runs/001-bare.json")["run"]
            .get("artifact_version")
            .is_none()
    );

    // A manifest with the field, and one from before the field existed (which means 1).
    write_manifest(root, Some(1));
    eval("versioned");
    let versioned = run_file(root, "runs/002-versioned.json");
    assert_eq!(versioned["run"]["artifact_version"], 1);
    // The hash of the same file sits next to it.
    let sha = versioned["run"]["manifest_sha256"].as_str().unwrap();
    assert_eq!(sha.len(), 64, "{sha}");
    write_manifest(root, None);
    eval("legacy");
    assert_eq!(
        run_file(root, "runs/003-legacy.json")["run"]["artifact_version"],
        1
    );

    // history --json carries it row by row.
    let out = kanon(
        root,
        &["history", "--runs", "runs", "--json", "history.json"],
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let rows = run_file(root, "history.json");
    assert_eq!(rows[0]["artifact_version"], serde_json::Value::Null);
    assert_eq!(rows[1]["artifact_version"], 1);
    assert_eq!(rows[2]["artifact_version"], 1);
}

#[test]
fn a_manifest_of_a_newer_contract_stops_every_eval_before_it_measures() {
    let dir = workspace();
    let root = dir.path();
    write_manifest(root, Some(2));

    let base = ["eval", "--queries", "queries.jsonl"];
    // With and without `--out`, every backend path, `--with`, and a dense backend whose
    // embeddings file does not exist (it would fail on that, if the version were not first).
    for extra in [
        vec![],
        vec!["--out", "runs"],
        vec!["--with", "handbook::docs/user/quotas.md"],
        vec!["--backend", "bm25-tantivy"],
        vec!["--backend", "bm25-tantivy", "--out", "runs"],
        vec!["--backend", "dense"],
        vec!["--compare", "bm25,bm25-tantivy", "--out", "runs"],
    ] {
        let args: Vec<&str> = base.iter().chain(&extra).copied().collect();
        let out = kanon(root, &args);
        assert_eq!(out.status.code(), Some(1), "{args:?}");
        let stderr = stderr(&out);
        assert!(stderr.contains(NEWER), "{args:?}: {stderr}");
        assert!(stderr.contains("artifact/manifest.json: "), "{stderr}");
        assert!(!root.join("runs").exists(), "{args:?}: no run file");
    }
}

#[test]
fn embed_refuses_a_manifest_of_a_newer_contract_before_embedding() {
    let dir = workspace();
    write_manifest(dir.path(), Some(2));
    let out = kanon(dir.path(), &["embed"]);
    assert_eq!(out.status.code(), Some(1));
    let stderr = stderr(&out);
    assert!(stderr.contains(NEWER), "{stderr}");
    assert!(!dir.path().join("embeddings.bin").exists());
}

#[test]
fn a_source_of_a_newer_contract_stops_eval_through_the_artifact_reader() {
    let dir = workspace();
    let root = dir.path();
    fs::write(
        root.join("artifact/handbook/meta.json"),
        "{\"artifact_version\": 2, \"repo\": \"example-org/handbook\"}",
    )
    .unwrap();
    let out = kanon(root, &["eval", "--queries", "queries.jsonl"]);
    assert_eq!(out.status.code(), Some(1));
    let stderr = stderr(&out);
    assert!(stderr.contains(NEWER), "{stderr}");
    assert!(stderr.contains("meta.json"), "{stderr}");

    // The same source at version 1, or with no version at all (written before the field
    // existed), is measured as before.
    for meta in [
        "{\"artifact_version\": 1, \"repo\": \"example-org/handbook\"}",
        "{\"repo\": \"example-org/handbook\"}",
    ] {
        fs::write(root.join("artifact/handbook/meta.json"), meta).unwrap();
        let out = kanon(root, &["eval", "--queries", "queries.jsonl"]);
        assert!(out.status.success(), "{}", self::stderr(&out));
    }
}
