//! `kanon eval --out` and `kanon history` end to end through the binary: numbered run files,
//! the table on stdout, JSON with `--json`, and `report --runs`.

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
         \"expected\": [\"handbook/docs/user\"]}\n\
         {\"id\": \"h\", \"kind\": \"howto\", \"query\": \"nothing here\", \
         \"expected\": [\"handbook/docs/user\"], \"holdout\": true}\n",
    )
    .unwrap();
    dir
}

fn ok(out: &std::process::Output) -> String {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn eval_out_numbers_run_files_and_history_lists_them() {
    let dir = workspace();
    let root = dir.path();

    // The first run creates the directory; --json and --out combine.
    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--out",
            "runs",
            "--label",
            "before curation",
            "--json",
            "baseline.json",
        ],
    );
    let stderr = ok(&out);
    assert!(out.stdout.is_empty(), "--json keeps stdout empty");
    assert!(stderr.contains("wrote baseline.json"), "{stderr}");
    assert!(
        stderr.contains("wrote runs/001-before-curation.json"),
        "{stderr}"
    );
    let text = fs::read_to_string(root.join("runs/001-before-curation.json")).unwrap();
    let run: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(run["tuning"]["overall"]["recall@5"], 1.0);
    assert_eq!(run["run"]["label"], "before-curation");
    assert_eq!(run["run"]["backend"], "bm25");
    assert_eq!(run["run"]["manifest_sha256"], "none");
    assert_eq!(run["run"]["k"], 10);
    assert_eq!(run["run"]["queries_sha256"].as_str().unwrap().len(), 64);
    assert!(run["run"]["at"].as_str().unwrap().ends_with('Z'));

    // The second run gets the next number; without --json the result still goes to stdout.
    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--backend",
            "bm25-tantivy",
            "--out",
            "runs",
            "--label",
            "tantivy",
        ],
    );
    let stderr = ok(&out);
    assert!(stderr.contains("wrote runs/002-tantivy.json"), "{stderr}");
    let stdout: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(stdout["backend"], "bm25-tantivy");
    let mut names: Vec<String> = fs::read_dir(root.join("runs"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, ["001-before-curation.json", "002-tantivy.json"]);

    // A run file is still a valid gate baseline.
    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--gate",
            "runs/001-before-curation.json",
        ],
    );
    ok(&out);

    // history: the table on stdout, one row per run, in order.
    let out = kanon(root, &["history", "--runs", "runs"]);
    let stderr = ok(&out);
    assert!(stderr.contains("runs: 2 runs"), "{stderr}");
    let table = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = table.lines().collect();
    assert_eq!(lines.len(), 4, "{table}");
    assert!(
        lines[0].starts_with(
            "| # | label | backend | tuning recall@5 | recall@10 | MRR | nDCG@5 | nDCG@10 | n |"
        ),
        "{table}"
    );
    assert!(
        lines[2].starts_with(
            "| 001 | before-curation | bm25 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1 \
             | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1 | none | "
        ),
        "{table}"
    );
    assert!(
        lines[3].starts_with("| 002 | tantivy | bm25-tantivy | 1.000 |"),
        "{table}"
    );
}

/// Two runs in `runs/` (`001-before-curation.json`, `002-tantivy.json`) plus `baseline.json`.
fn two_runs(root: &Path) {
    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--out",
            "runs",
            "--label",
            "before-curation",
            "--json",
            "baseline.json",
        ],
    );
    ok(&out);
    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--backend",
            "bm25-tantivy",
            "--out",
            "runs",
            "--label",
            "tantivy",
        ],
    );
    ok(&out);
}

#[test]
fn history_json_report_runs_and_odd_files() {
    let dir = workspace();
    let root = dir.path();
    two_runs(root);
    let out = kanon(root, &["history", "--runs", "runs"]);
    ok(&out);
    let table = String::from_utf8_lossy(&out.stdout).into_owned();
    let first_row = table.lines().nth(2).unwrap();

    // history --json: rows in the file, stdout empty.
    let out = kanon(
        root,
        &["history", "--runs", "runs", "--json", "history.json"],
    );
    let stderr = ok(&out);
    assert!(out.stdout.is_empty());
    assert!(stderr.contains("wrote history.json"), "{stderr}");
    let rows: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("history.json")).unwrap()).unwrap();
    let rows = rows.as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["seq"], 1);
    assert_eq!(rows[0]["file"], "001-before-curation.json");
    assert_eq!(rows[0]["label"], "before-curation");
    assert_eq!(rows[0]["tuning"]["recall@5"], 1.0);
    assert_eq!(rows[0]["holdout"]["n"], 1);
    assert_eq!(rows[1]["backend"], "bm25-tantivy");

    // report --runs renders the same rows after the eval section.
    let out = kanon(
        root,
        &["report", "--eval-after", "baseline.json", "--runs", "runs"],
    );
    ok(&out);
    let report = String::from_utf8_lossy(&out.stdout);
    let eval_at = report.find("## Eval before/after").unwrap();
    let history_at = report.find("## History").unwrap();
    assert!(eval_at < history_at, "{report}");
    assert!(report.contains(first_row), "{report}");

    // A plain result in the directory is a row with dashes; an invalid file names itself.
    fs::copy(root.join("baseline.json"), root.join("runs/003-plain.json")).unwrap();
    let out = kanon(root, &["history", "--runs", "runs"]);
    ok(&out);
    let table = String::from_utf8_lossy(&out.stdout);
    assert!(
        table.contains(
            "| 003 | – | – | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1 \
             | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1 | – | – |"
        ),
        "{table}"
    );
    fs::write(root.join("runs/004-broken.json"), "not json").unwrap();
    let out = kanon(root, &["history", "--runs", "runs"]);
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("004-broken.json"), "{stderr}");
    assert!(stderr.contains("invalid run file"), "{stderr}");
    fs::remove_file(root.join("runs/004-broken.json")).unwrap();

    // The default directory is `runs` next to the config; missing is an error naming it.
    let out = kanon(root, &["history"]);
    ok(&out);
    let out = kanon(root, &["history", "--runs", "elsewhere"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("elsewhere"));
}

/// `--compare a,b --out DIR` writes one run file per backend, `<label>-<backend>`.
#[test]
fn eval_compare_out_writes_one_run_file_per_backend() {
    let dir = workspace();
    let root = dir.path();
    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--compare",
            "bm25,bm25-tantivy",
            "--out",
            "runs",
            "--label",
            "v1",
        ],
    );
    let stderr = ok(&out);
    assert!(stderr.contains("wrote runs/001-v1-bm25.json"), "{stderr}");
    assert!(
        stderr.contains("wrote runs/002-v1-bm25-tantivy.json"),
        "{stderr}"
    );
    let combined: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(combined.get("bm25").is_some());
    let second: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(root.join("runs/002-v1-bm25-tantivy.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(second["run"]["label"], "v1-bm25-tantivy");
    assert_eq!(second["run"]["backend"], "bm25-tantivy");

    // --label without --out is rejected by the parser.
    let out = kanon(
        root,
        &["eval", "--queries", "queries.jsonl", "--label", "x"],
    );
    assert_eq!(out.status.code(), Some(2));
}
