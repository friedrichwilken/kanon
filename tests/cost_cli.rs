//! The cost of a run through the binary: latency and tokens per query in the result, the run
//! file and the history, and the optional budget that fails `eval` with exit code 2.

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
    fs::write(
        &page,
        "# Storage\n\nEnable upload caching with a label.\n\n## Caching\n\nSet the cache size and \
         the label of the bucket.\n",
    )
    .unwrap();
    fs::write(
        dir.path().join("queries.jsonl"),
        "{\"id\": \"q\", \"kind\": \"howto\", \"query\": \"enable upload caching\", \
         \"expected\": [\"handbook/docs/user\"]}\n",
    )
    .unwrap();
    dir
}

fn err_text(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn json(out: &std::process::Output) -> serde_json::Value {
    serde_json::from_slice(&out.stdout).expect("JSON on stdout")
}

const EVAL: [&str; 3] = ["eval", "--queries", "queries.jsonl"];

fn eval(dir: &Path, extra: &[&str]) -> std::process::Output {
    let args: Vec<&str> = EVAL.iter().chain(extra).copied().collect();
    kanon(dir, &args)
}

#[test]
fn a_run_reports_its_latency_and_the_tokens_it_returned() {
    let dir = workspace();
    let out = eval(dir.path(), &[]);
    assert_eq!(out.status.code(), Some(0), "{}", err_text(&out));
    let stderr = err_text(&out);
    let cost_line = stderr
        .lines()
        .find(|l| l.starts_with("cost: "))
        .expect(&stderr);
    assert!(cost_line.contains("latency p50 "), "{cost_line}");
    assert!(cost_line.contains("tokens per query "), "{cost_line}");
    assert!(!cost_line.contains("not counted"), "{cost_line}");

    let cost = &json(&out)["cost"];
    assert!(cost["latency"]["p50_ms"].as_f64().unwrap() >= 0.0);
    assert!(
        cost["latency"]["p95_ms"].as_f64().unwrap() >= cost["latency"]["p50_ms"].as_f64().unwrap()
    );
    let five = cost["tokens"]["mean@5"].as_f64().unwrap();
    let ten = cost["tokens"]["mean@10"].as_f64().unwrap();
    assert!(five > 0.0 && ten >= five, "{cost}");
    assert_eq!(
        cost["tokens"]["unresolved"], 0,
        "every hit's unit was found"
    );
}

#[test]
fn every_built_in_backend_counts_the_units_it_returns() {
    let dir = workspace();
    for backend in ["bm25", "bm25-tantivy"] {
        let out = eval(dir.path(), &["--backend", backend]);
        assert_eq!(out.status.code(), Some(0), "{backend}: {}", err_text(&out));
        let tokens = &json(&out)["cost"]["tokens"];
        assert!(
            tokens["mean@5"].as_f64().unwrap() > 0.0,
            "{backend}: {tokens}"
        );
        assert_eq!(tokens["unresolved"], 0, "{backend}");
    }
}

#[test]
fn a_budget_passes_or_fails_the_run_with_exit_code_2() {
    let dir = workspace();
    // Ceilings nothing reaches: both lines say ok, exit 0.
    let out = eval(
        dir.path(),
        &["--max-tokens", "100000", "--max-p95-ms", "100000"],
    );
    assert_eq!(out.status.code(), Some(0), "{}", err_text(&out));
    let stderr = err_text(&out);
    assert!(stderr.contains("budget: p95 latency (ms) "), "{stderr}");
    assert!(stderr.contains("budget: tokens per query @5 "), "{stderr}");
    assert!(!stderr.contains("FAILED"), "{stderr}");

    // A token ceiling the run is over.
    let out = eval(dir.path(), &["--max-tokens", "0.5"]);
    assert_eq!(out.status.code(), Some(2), "{}", err_text(&out));
    assert!(
        err_text(&out).contains("budget: tokens per query @5 ")
            && err_text(&out).contains(", max 0.5: FAILED"),
        "{}",
        err_text(&out)
    );
    // The result is still written: the budget gates, it does not withhold.
    assert!(json(&out)["cost"]["tokens"]["mean@5"].as_f64().unwrap() > 0.5);

    // A latency ceiling of zero: no search takes no time.
    let out = eval(dir.path(), &["--max-p95-ms", "0"]);
    assert_eq!(out.status.code(), Some(2), "{}", err_text(&out));
    assert!(
        err_text(&out).contains("p95 latency (ms) "),
        "{}",
        err_text(&out)
    );

    // Either limit failing fails the run, and one unset limit is not checked at all.
    let out = eval(
        dir.path(),
        &["--max-tokens", "0.5", "--max-p95-ms", "100000"],
    );
    assert_eq!(out.status.code(), Some(2));
    let out = eval(dir.path(), &["--max-tokens", "100000"]);
    assert!(
        !err_text(&out).contains("p95 latency"),
        "{}",
        err_text(&out)
    );
}

#[test]
fn the_budget_is_checked_for_each_backend_of_a_comparison() {
    let dir = workspace();
    let out = eval(
        dir.path(),
        &["--compare", "bm25,bm25-tantivy", "--max-tokens", "0.5"],
    );
    assert_eq!(out.status.code(), Some(2), "{}", err_text(&out));
    let stderr = err_text(&out);
    assert!(
        stderr.contains("budget [bm25]: tokens per query @5 "),
        "{stderr}"
    );
    assert!(
        stderr.contains("budget [bm25-tantivy]: tokens per query @5 "),
        "{stderr}"
    );
    let combined = json(&out);
    assert!(combined["bm25"]["cost"]["tokens"].is_object());
    assert!(combined["bm25-tantivy"]["cost"]["tokens"].is_object());

    let out = eval(
        dir.path(),
        &["--backend", "bm25-tantivy", "--max-tokens", "0.5"],
    );
    assert_eq!(out.status.code(), Some(2));
    assert!(
        err_text(&out).contains("budget [bm25-tantivy]: "),
        "{}",
        err_text(&out)
    );
}

#[test]
fn the_config_sets_the_budget_and_a_flag_overrides_it() {
    let dir = workspace();
    fs::write(
        dir.path().join("kanon.yaml"),
        "queries: queries.jsonl\nmax_tokens: 0.5\nmax_p95_ms: 100000\n",
    )
    .unwrap();
    let run = |extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_kanon"))
            .current_dir(dir.path())
            .args(["--config", "kanon.yaml", "eval"])
            .args(extra)
            .output()
            .unwrap()
    };
    let out = run(&[]);
    assert_eq!(out.status.code(), Some(2), "{}", err_text(&out));
    assert!(
        err_text(&out).contains("tokens per query @5 ")
            && err_text(&out).contains("max 0.5: FAILED")
    );
    assert!(
        err_text(&out).contains("p95 latency (ms) ") && err_text(&out).contains("max 100000.0: ok")
    );
    let out = run(&["--max-tokens", "100000"]);
    assert_eq!(out.status.code(), Some(0), "{}", err_text(&out));
}

#[test]
fn a_limit_that_is_not_a_finite_number_is_refused() {
    let dir = workspace();
    for bad in ["-1", "nan", "inf", "fast"] {
        let flag = format!("--max-tokens={bad}");
        let out = eval(dir.path(), &[&flag]);
        assert_ne!(out.status.code(), Some(0), "{bad}");
        assert!(out.stdout.is_empty(), "{bad}: nothing was measured");
        let text = err_text(&out);
        assert!(text.contains("--max-tokens"), "{bad}: {text}");
        assert!(
            text.contains(&format!("{bad:?}")),
            "{bad}: the value is named: {text}"
        );
    }
    // The same for the latency ceiling, and a negative value as a separate argument (which the
    // argument parser takes for a flag) is refused too.
    let out = eval(dir.path(), &["--max-p95-ms=-5"]);
    assert_ne!(out.status.code(), Some(0));
    let out = eval(dir.path(), &["--max-tokens", "-1"]);
    assert_ne!(out.status.code(), Some(0));
    assert!(out.stdout.is_empty());
}

#[test]
fn a_run_file_and_the_history_carry_the_cost() {
    let dir = workspace();
    let root = dir.path();
    let out = eval(root, &["--out", "runs", "--label", "one"]);
    assert_eq!(out.status.code(), Some(0), "{}", err_text(&out));
    let run: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("runs/001-one.json")).unwrap()).unwrap();
    assert!(run["cost"]["tokens"]["mean@5"].as_f64().unwrap() > 0.0);
    assert!(run["cost"]["latency"]["p95_ms"].as_f64().is_some());

    let out = kanon(
        root,
        &["history", "--runs", "runs", "--json", "history.json"],
    );
    assert_eq!(out.status.code(), Some(0), "{}", err_text(&out));
    let rows: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("history.json")).unwrap()).unwrap();
    assert!(rows[0]["p95_ms"].as_f64().is_some(), "{rows}");
    assert_eq!(
        rows[0]["tokens5"].as_f64(),
        run["cost"]["tokens"]["mean@5"].as_f64()
    );
}

#[test]
fn a_baseline_written_before_cost_existed_still_gates() {
    let dir = workspace();
    let root = dir.path();
    let out = eval(root, &[]);
    let mut baseline = json(&out);
    assert!(baseline.as_object_mut().unwrap().remove("cost").is_some());
    fs::write(
        root.join("old.json"),
        serde_json::to_string(&baseline).unwrap(),
    )
    .unwrap();
    let out = eval(root, &["--gate", "old.json"]);
    assert_eq!(out.status.code(), Some(0), "{}", err_text(&out));
    assert!(
        err_text(&out).contains("gate: tuning recall@5"),
        "{}",
        err_text(&out)
    );
}
