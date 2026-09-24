//! `kanon report`: the evaluation sections of a Markdown report, rendered from one or two
//! `eval --json` results. The corpus sections (residue, duplicates, decisions) stay with
//! `pinakes report`; this renders only what `kanon` measures.
//!
//! The tables are the ones `pinakes report` used to render for `--eval-before/--eval-after`,
//! so a PR body built from both tools reads the same as before the move.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::eval::{EvalSummary, Metrics, Split};

/// Everything the report is rendered from.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReportInput<'a> {
    /// Evaluation on the previous corpus or retriever.
    pub before: Option<&'a EvalSummary>,
    /// Evaluation on the current corpus or retriever.
    pub after: Option<&'a EvalSummary>,
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
    out.push_str("| kind | recall@5 | recall@10 | MRR | n |\n|---|---|---|---|---|\n");
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
        "| {label} | {} | {} | {} | {n} |",
        cell(|m| m.recall5),
        cell(|m| m.recall10),
        cell(|m| m.mrr)
    );
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn metrics(r5: f64, r10: f64, mrr: f64, n: usize) -> Metrics {
        Metrics {
            recall5: r5,
            recall10: r10,
            mrr,
            n,
        }
    }

    fn evals() -> (EvalSummary, EvalSummary) {
        let before = EvalSummary {
            tuning: Split {
                overall: metrics(0.8, 0.85, 0.66, 40),
                per_kind: BTreeMap::from([("howto".to_string(), metrics(0.9, 0.95, 0.8, 10))]),
            },
            holdout: Some(Split {
                overall: metrics(0.7, 0.8, 0.6, 10),
                per_kind: BTreeMap::new(),
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
            },
            holdout: Some(Split {
                overall: metrics(0.75, 0.8, 0.65, 10),
                per_kind: BTreeMap::new(),
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
        });
        assert!(text.contains("| overall | 0.800 → 0.850 | 0.850 → 0.900 | 0.660 → 0.700 | 40 |"));
        assert!(text.contains("| concept | – → 0.700 | – → 0.700 | – → 0.500 | 5 |"));
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
        });
        assert!(text.contains("| overall | – → 0.850 | – → 0.900 | – → 0.700 | 40 |"));
        assert!(text.contains("### Held-out queries"));
        snapshot("report_after_only", &text);
    }
}
