//! `kanon report --svg DIR` end to end through the binary: the chart files the inputs allow
//! are written, named on stderr and linked from the Markdown by the directory as given.

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

fn ok(out: &std::process::Output) -> String {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Two pages and three queries; the second page only answers a query after the corpus grows,
/// so `before.json` (one page) and `after.json` (two pages) differ per query.
fn workspace() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let storage = root.join("artifact/handbook/docs/user/README.md");
    fs::create_dir_all(storage.parent().unwrap()).unwrap();
    fs::write(
        &storage,
        "# Storage\n\nEnable upload caching with a label.\n",
    )
    .unwrap();
    fs::write(
        root.join("queries.jsonl"),
        "{\"id\": \"caching\", \"kind\": \"howto\", \"query\": \"enable upload caching\", \
         \"expected\": [\"handbook/docs/user\"]}\n\
         {\"id\": \"quotas\", \"kind\": \"concept\", \"query\": \"quota rate limits\", \
         \"expected\": [\"handbook/docs/admin\"]}\n\
         {\"id\": \"held\", \"kind\": \"howto\", \"query\": \"upload caching label\", \
         \"expected\": [\"handbook/docs/user\"], \"holdout\": true}\n",
    )
    .unwrap();
    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--json",
            "before.json",
            "--out",
            "runs",
            "--label",
            "one-page",
        ],
    );
    ok(&out);
    let quotas = root.join("artifact/handbook/docs/admin/quotas.md");
    fs::create_dir_all(quotas.parent().unwrap()).unwrap();
    fs::write(&quotas, "# Quotas\n\nQuota rate limits in strict mode.\n").unwrap();
    let out = kanon(
        root,
        &[
            "eval",
            "--queries",
            "queries.jsonl",
            "--json",
            "after.json",
            "--out",
            "runs",
            "--label",
            "two-pages",
        ],
    );
    ok(&out);
    dir
}

#[test]
fn report_svg_writes_the_charts_and_links_them() {
    let dir = workspace();
    let root = dir.path();

    let out = kanon(
        root,
        &[
            "report",
            "--svg",
            "out/charts",
            "--eval-before",
            "before.json",
            "--eval-after",
            "after.json",
            "--runs",
            "runs",
        ],
    );
    let stderr = ok(&out);
    let report = String::from_utf8_lossy(&out.stdout);
    for name in ["recall-per-kind", "rank-movement", "recall-over-runs"] {
        let path = root.join(format!("out/charts/{name}.svg"));
        let svg = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""),
            "{svg}"
        );
        assert!(svg.trim_end().ends_with("</svg>"), "{svg}");
        assert!(
            stderr.contains(&format!("wrote out/charts/{name}.svg")),
            "{stderr}"
        );
        assert!(
            report.contains(&format!("](out/charts/{name}.svg)\n")),
            "{report}"
        );
    }
    // Each link sits under its section: the eval charts before History, the runs chart after.
    let history_at = report.find("## History").unwrap();
    assert!(report.find("recall-per-kind.svg").unwrap() < history_at);
    assert!(report.find("rank-movement.svg").unwrap() < history_at);
    assert!(report.find("recall-over-runs.svg").unwrap() > history_at);

    // The rank chart carries the query that went from a miss to a hit.
    let movement = fs::read_to_string(root.join("out/charts/rank-movement.svg")).unwrap();
    assert!(
        movement.contains("<title>quotas: miss → 1</title>"),
        "{movement}"
    );
    let per_kind = fs::read_to_string(root.join("out/charts/recall-per-kind.svg")).unwrap();
    assert!(
        per_kind.contains("<title>howto held-out: recall@5 1.000 (n = 1)</title>"),
        "{per_kind}"
    );

    // Without --runs and --eval-before, only the per-kind chart is written; the others are
    // neither written nor linked (a stale file from an earlier call is left alone).
    let out = kanon(
        root,
        &[
            "report",
            "--svg",
            "out/charts",
            "--eval-after",
            "after.json",
        ],
    );
    let stderr = ok(&out);
    let report = String::from_utf8_lossy(&out.stdout);
    assert_eq!(stderr.trim(), "wrote out/charts/recall-per-kind.svg");
    assert!(
        report.contains("](out/charts/recall-per-kind.svg)"),
        "{report}"
    );
    assert!(!report.contains("rank-movement"), "{report}");
    assert!(!report.contains("recall-over-runs"), "{report}");

    // Without --svg nothing is written and nothing linked.
    let out = kanon(root, &["report", "--eval-after", "after.json"]);
    let stderr = ok(&out);
    assert!(stderr.is_empty(), "{stderr}");
    assert!(!String::from_utf8_lossy(&out.stdout).contains(".svg"));
    assert!(!root.join("charts").exists());

    // A directory that cannot be created is an error naming it.
    fs::write(root.join("blocker"), "").unwrap();
    let out = kanon(
        root,
        &[
            "report",
            "--svg",
            "blocker/charts",
            "--eval-after",
            "after.json",
        ],
    );
    assert_eq!(out.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("blocker/charts"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
