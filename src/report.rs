//! `kanon report`: the evaluation sections of a Markdown report, rendered from one or two
//! `eval --json` results and, with `--runs`, the history of a run directory. The corpus
//! sections (residue, duplicates, decisions) stay with `pinakes report`; this renders only
//! what `kanon` measures.
//!
//! The tables are the ones `pinakes report` used to render for `--eval-before/--eval-after`,
//! so a PR body built from both tools reads the same as before the move.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::Path;

use crate::eval::{EvalSummary, Metrics, Split};
use crate::history::{self, HistoryRow};
use crate::svg::Charts;

/// Everything the report is rendered from.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReportInput<'a> {
    /// Evaluation on the previous corpus or retriever.
    pub before: Option<&'a EvalSummary>,
    /// Evaluation on the current corpus or retriever.
    pub after: Option<&'a EvalSummary>,
    /// The rows of a run directory (`--runs`); the History section appears only when given.
    pub history: Option<&'a [HistoryRow]>,
    /// The chart files `--svg DIR` wrote; each is linked under the section it belongs to.
    pub charts: Option<&'a Charts>,
}

/// An image link to a chart file, under the section it belongs to. A path with whitespace
/// or parentheses goes in angle brackets, as Markdown needs.
fn chart_link(out: &mut String, alt: &str, path: Option<&Path>) {
    if let Some(path) = path {
        let path = path.display().to_string();
        if path.contains(|c: char| c.is_whitespace() || c == '(' || c == ')') {
            let _ = writeln!(out, "![{alt}](<{path}>)\n");
        } else {
            let _ = writeln!(out, "![{alt}]({path})\n");
        }
    }
}

/// Render the report as Markdown.
pub fn render(input: ReportInput<'_>) -> String {
    let mut out = String::from("# Retrieval report\n\n");
    let backends: BTreeSet<&str> = [input.before, input.after]
        .into_iter()
        .flatten()
        .map(|e| e.backend.as_str())
        .filter(|b| !b.is_empty())
        .collect();
    if !backends.is_empty() {
        let names: Vec<String> = backends.iter().map(|b| format!("`{b}`")).collect();
        let _ = writeln!(out, "- Backend: {}\n", names.join(", "));
    }
    eval_section(&mut out, input.before, input.after);
    let charts = input.charts;
    chart_link(
        &mut out,
        "Recall@5 per kind, tuning next to held-out",
        charts.and_then(|c| c.recall_per_kind.as_deref()),
    );
    chart_link(
        &mut out,
        "Rank of the first expected hit per query, before to after",
        charts.and_then(|c| c.rank_movement.as_deref()),
    );
    if let Some(rows) = input.history {
        out.push_str("## History\n\n");
        out.push_str(&history::render_table(rows));
        out.push('\n');
        chart_link(
            &mut out,
            "Recall, MRR and nDCG@5 over runs",
            charts.and_then(|c| c.recall_over_runs.as_deref()),
        );
    }
    out
}

fn eval_section(out: &mut String, before: Option<&EvalSummary>, after: Option<&EvalSummary>) {
    out.push_str("## Eval before/after\n\n");
    if before.is_none() && after.is_none() {
        out.push_str("_No evaluation results supplied._\n\n");
        return;
    }
    out.push_str("### Tuning queries\n\n");
    eval_table(out, before.map(|e| &e.tuning), after.map(|e| &e.tuning));
    let holdout_before = before.and_then(|e| e.holdout.as_ref());
    let holdout_after = after.and_then(|e| e.holdout.as_ref());
    if holdout_before.is_some() || holdout_after.is_some() {
        out.push_str("### Held-out queries\n\n");
        eval_table(out, holdout_before, holdout_after);
    }
}

fn eval_table(out: &mut String, before: Option<&Split>, after: Option<&Split>) {
    out.push_str(
        "| kind | recall@5 | recall@10 | MRR | nDCG@5 | nDCG@10 | n |\n|---|---|---|---|---|---|---|\n",
    );
    let mut kinds: BTreeSet<&str> = BTreeSet::new();
    for split in [before, after].into_iter().flatten() {
        kinds.extend(split.per_kind.keys().map(String::as_str));
    }
    eval_row(
        out,
        "overall",
        before.map(|s| &s.overall),
        after.map(|s| &s.overall),
    );
    for kind in kinds {
        eval_row(
            out,
            kind,
            before.and_then(|s| s.per_kind.get(kind)),
            after.and_then(|s| s.per_kind.get(kind)),
        );
    }
    out.push('\n');
}

fn eval_row(out: &mut String, label: &str, before: Option<&Metrics>, after: Option<&Metrics>) {
    let cell = |f: fn(&Metrics) -> f64| match (before, after) {
        (Some(b), Some(a)) => format!("{:.3} → {:.3}", f(b), f(a)),
        (Some(b), None) => format!("{:.3} → –", f(b)),
        (None, Some(a)) => format!("– → {:.3}", f(a)),
        (None, None) => "–".to_string(),
    };
    let n = after
        .or(before)
        .map_or_else(|| "–".to_string(), |m| m.n.to_string());
    let _ = writeln!(
        out,
        "| {label} | {} | {} | {} | {} | {} | {n} |",
        cell(|m| m.recall5),
        cell(|m| m.recall10),
        cell(|m| m.mrr),
        cell(|m| m.ndcg5),
        cell(|m| m.ndcg10)
    );
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::Path;

    use super::*;

    /// Metrics with nDCG@5 and nDCG@10 derived from MRR (`mrr - 0.02`, `mrr + 0.02`), so the
    /// snapshots show every column with a distinct number.
    fn metrics(r5: f64, r10: f64, mrr: f64, n: usize) -> Metrics {
        Metrics {
            recall5: r5,
            recall10: r10,
            mrr,
            ndcg5: mrr - 0.02,
            ndcg10: mrr + 0.02,
            n,
        }
    }

    fn evals() -> (EvalSummary, EvalSummary) {
        let before = EvalSummary {
            tuning: Split {
                overall: metrics(0.8, 0.85, 0.66, 40),
                per_kind: BTreeMap::from([("howto".to_string(), metrics(0.9, 0.95, 0.8, 10))]),
                negative: None,
            },
            holdout: Some(Split {
                overall: metrics(0.7, 0.8, 0.6, 10),
                per_kind: BTreeMap::new(),
                negative: None,
            }),
            queries: vec![],
            backend: String::new(),
        };
        let after = EvalSummary {
            tuning: Split {
                overall: metrics(0.85, 0.9, 0.7, 40),
                per_kind: BTreeMap::from([
                    ("concept".to_string(), metrics(0.7, 0.7, 0.5, 5)),
                    ("howto".to_string(), metrics(0.9, 1.0, 0.85, 10)),
                ]),
                negative: None,
            },
            holdout: Some(Split {
                overall: metrics(0.75, 0.8, 0.65, 10),
                per_kind: BTreeMap::new(),
                negative: None,
            }),
            queries: vec![],
            backend: "bm25".to_string(),
        };
        (before, after)
    }

    /// Compare `rendered` with `tests/snapshots/<name>.md`, refreshing it when
    /// `UPDATE_SNAPSHOTS` is set.
    fn snapshot(name: &str, rendered: &str) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/snapshots")
            .join(format!("{name}.md"));
        if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
            std::fs::write(&path, rendered).unwrap();
        }
        let expected = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("{}: {e}; run UPDATE_SNAPSHOTS=1 cargo test", path.display())
        });
        assert_eq!(
            rendered, expected,
            "{name} changed; run `UPDATE_SNAPSHOTS=1 cargo test` if intended"
        );
    }

    #[test]
    fn full_report_matches_snapshot() {
        let (before, after) = evals();
        let text = render(ReportInput {
            before: Some(&before),
            after: Some(&after),
            history: None,
            charts: None,
        });
        assert!(text.contains(
            "| overall | 0.800 → 0.850 | 0.850 → 0.900 | 0.660 → 0.700 | 0.640 → 0.680 \
             | 0.680 → 0.720 | 40 |"
        ));
        assert!(text.contains(
            "| concept | – → 0.700 | – → 0.700 | – → 0.500 | – → 0.480 | – → 0.520 | 5 |"
        ));
        assert!(text.contains("- Backend: `bm25`\n"));
        snapshot("report_full", &text);
    }

    #[test]
    fn report_without_results_says_so() {
        let text = render(ReportInput::default());
        assert!(text.contains("_No evaluation results supplied._"));
        assert!(!text.contains("Backend"));
        snapshot("report_empty", &text);
    }

    #[test]
    fn report_with_only_the_after_side() {
        let (_, after) = evals();
        let text = render(ReportInput {
            before: None,
            after: Some(&after),
            history: None,
            charts: None,
        });
        assert!(text.contains(
            "| overall | – → 0.850 | – → 0.900 | – → 0.700 | – → 0.680 | – → 0.720 | 40 |"
        ));
        assert!(text.contains("### Held-out queries"));
        snapshot("report_after_only", &text);
    }

    #[test]
    fn report_with_a_history_section() {
        let (before, after) = evals();
        let first = history::Run {
            summary: before,
            run: Some(history::RunInfo {
                label: "a1b2c3d".to_string(),
                at: "2026-09-16T12:00:00Z".to_string(),
                backend: "bm25".to_string(),
                manifest_sha256: "0123456789abcdef".repeat(4),
                queries_sha256: "fedcba9876543210".repeat(4),
                k: 10,
            }),
        };
        let second = history::Run {
            summary: after,
            run: None,
        };
        let rows = vec![
            HistoryRow::of(1, Path::new("runs/001-a1b2c3d.json"), &first),
            HistoryRow::of(2, Path::new("runs/002-plain.json"), &second),
        ];
        let text = render(ReportInput {
            before: None,
            after: None,
            history: Some(&rows),
            charts: None,
        });
        assert!(text.contains("_No evaluation results supplied._\n\n## History\n\n"));
        assert!(
            text.contains("| 001 | a1b2c3d | bm25 | 0.800 | 0.850 | 0.660 | 0.640 | 0.680 | 40 |"),
            "{text}"
        );
        assert!(
            text.contains("| 01234567 | 2026-09-16T12:00:00Z |"),
            "{text}"
        );
        assert!(text.contains("| 002 | – | bm25 | 0.850 |"), "{text}");
        snapshot("report_history", &text);
    }

    #[test]
    fn report_links_the_charts_under_their_sections() {
        let (before, after) = evals();
        let rows = vec![HistoryRow::of(
            1,
            Path::new("runs/001-a.json"),
            &history::Run {
                summary: before.clone(),
                run: None,
            },
        )];
        let charts = Charts {
            recall_over_runs: Some("out/recall-over-runs.svg".into()),
            rank_movement: Some("out/rank-movement.svg".into()),
            recall_per_kind: Some("out/recall-per-kind.svg".into()),
        };
        let text = render(ReportInput {
            before: Some(&before),
            after: Some(&after),
            history: Some(&rows),
            charts: Some(&charts),
        });
        let per_kind = text.find("](out/recall-per-kind.svg)\n\n").unwrap();
        let movement = text.find("](out/rank-movement.svg)\n\n").unwrap();
        let over_runs = text.find("](out/recall-over-runs.svg)\n").unwrap();
        let held_out = text.find("### Held-out queries").unwrap();
        let history_at = text.find("## History").unwrap();
        assert!(held_out < per_kind && per_kind < movement, "{text}");
        assert!(movement < history_at && history_at < over_runs, "{text}");
        assert!(text.ends_with("](out/recall-over-runs.svg)\n\n"), "{text}");

        // A chart that was not written leaves no link.
        let text = render(ReportInput {
            before: Some(&before),
            after: Some(&after),
            history: None,
            charts: Some(&Charts {
                recall_over_runs: None,
                ..charts
            }),
        });
        assert!(!text.contains("recall-over-runs"), "{text}");
        assert!(text.contains("](out/rank-movement.svg)"), "{text}");

        // A directory with a space or a parenthesis needs angle brackets.
        let text = render(ReportInput {
            before: Some(&before),
            after: None,
            history: None,
            charts: Some(&Charts {
                recall_per_kind: Some("my (charts)/recall-per-kind.svg".into()),
                ..Charts::default()
            }),
        });
        assert!(
            text.contains("](<my (charts)/recall-per-kind.svg>)\n"),
            "{text}"
        );
    }
}
