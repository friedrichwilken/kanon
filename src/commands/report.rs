use std::path::PathBuf;

use crate::error::CommandError;
use crate::eval::EvalSummary;
use crate::history;
use crate::report::{self, ReportInput};
use crate::svg::{self, Charts};

/// Inputs for `report`, as file paths.
#[derive(Debug, Clone, Default)]
pub struct ReportOptions {
    /// Eval result on the previous corpus or retriever.
    pub eval_before: Option<PathBuf>,
    /// Eval result on the current corpus or retriever.
    pub eval_after: Option<PathBuf>,
    /// A run directory (`eval --out`) to render a `## History` section from.
    pub runs: Option<PathBuf>,
    /// A directory to write the charts into (`--svg DIR`), linked from the report by the
    /// path as given.
    pub svg: Option<PathBuf>,
}

/// What `report` produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReportOutcome {
    /// The Markdown report.
    pub text: String,
    /// The chart files written under `--svg DIR`; all `None` without the flag.
    pub charts: Charts,
}

/// Run `report`: render the evaluation sections of a Markdown report from eval result files
/// and, with `runs`, the history of a run directory. With `svg`, the charts those inputs
/// allow are written there first and linked from the text.
pub fn report(options: &ReportOptions) -> Result<ReportOutcome, CommandError> {
    let before = options
        .eval_before
        .as_deref()
        .map(EvalSummary::load)
        .transpose()?;
    let after = options
        .eval_after
        .as_deref()
        .map(EvalSummary::load)
        .transpose()?;
    let rows = options
        .runs
        .as_deref()
        .map(history::read_history)
        .transpose()?;
    let charts = match &options.svg {
        Some(dir) => svg::write_charts(dir, before.as_ref(), after.as_ref(), rows.as_deref())?,
        None => Charts::default(),
    };
    let text = report::render(ReportInput {
        before: before.as_ref(),
        after: after.as_ref(),
        history: rows.as_deref(),
        charts: Some(&charts),
    });
    Ok(ReportOutcome { text, charts })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::{EvalOptions, eval};
    use crate::testing::eval_workspace;

    #[test]
    fn report_reads_eval_result_files() {
        let (dir, paths) = eval_workspace();
        let result = dir.path().join("eval.json");
        let options = EvalOptions {
            queries: Some(dir.path().join("queries.jsonl")),
            json: Some(result.clone()),
            ..EvalOptions::default()
        };
        eval(&paths, &options).unwrap();
        let outcome = report(&ReportOptions {
            eval_before: None,
            eval_after: Some(result),
            ..ReportOptions::default()
        })
        .unwrap();
        let text = outcome.text;
        assert!(text.starts_with("# Retrieval report\n"));
        assert!(text.contains("| overall | – → 0.500 |"), "{text}");
        assert!(!text.contains("## History"), "{text}");
        assert!(!text.contains(".svg"), "{text}");
        assert_eq!(outcome.charts, Charts::default());
        assert!(matches!(
            report(&ReportOptions {
                eval_before: Some(dir.path().join("missing.json")),
                ..ReportOptions::default()
            })
            .unwrap_err(),
            CommandError::Eval(_)
        ));
    }

    #[test]
    fn report_renders_the_history_of_a_run_directory() {
        let (dir, paths) = eval_workspace();
        let runs = dir.path().join("runs");
        let options = EvalOptions {
            queries: Some(dir.path().join("queries.jsonl")),
            out: Some(runs.clone()),
            label: Some("first".to_string()),
            ..EvalOptions::default()
        };
        eval(&paths, &options).unwrap();
        let text = report(&ReportOptions {
            runs: Some(runs.clone()),
            ..ReportOptions::default()
        })
        .unwrap()
        .text;
        assert!(text.contains("## History\n\n| # | label |"), "{text}");
        assert!(text.contains("| 001 | first | bm25 | 0.500 |"), "{text}");
        assert!(matches!(
            report(&ReportOptions {
                runs: Some(dir.path().join("missing")),
                ..ReportOptions::default()
            })
            .unwrap_err(),
            CommandError::History(_)
        ));
    }

    #[test]
    fn report_svg_writes_the_charts_the_inputs_allow_and_links_them() {
        let (dir, paths) = eval_workspace();
        let runs = dir.path().join("runs");
        let result = dir.path().join("eval.json");
        let options = EvalOptions {
            queries: Some(dir.path().join("queries.jsonl")),
            out: Some(runs.clone()),
            json: Some(result.clone()),
            ..EvalOptions::default()
        };
        eval(&paths, &options).unwrap();
        let charts_dir = dir.path().join("charts");

        // One result and the runs: the per-kind and over-runs charts, no rank movement.
        let outcome = report(&ReportOptions {
            eval_before: None,
            eval_after: Some(result.clone()),
            runs: Some(runs),
            svg: Some(charts_dir.clone()),
        })
        .unwrap();
        assert_eq!(
            outcome.charts,
            Charts {
                recall_over_runs: Some(charts_dir.join(svg::RECALL_OVER_RUNS)),
                rank_movement: None,
                recall_per_kind: Some(charts_dir.join(svg::RECALL_PER_KIND)),
            }
        );
        for path in outcome.charts.paths() {
            assert!(path.is_file(), "{}", path.display());
            let link = format!("]({})\n", path.display());
            assert!(outcome.text.contains(&link), "{}", outcome.text);
        }
        assert!(!outcome.text.contains("rank-movement"), "{}", outcome.text);

        // Both results: the rank-movement chart too.
        let outcome = report(&ReportOptions {
            eval_before: Some(result.clone()),
            eval_after: Some(result.clone()),
            runs: None,
            svg: Some(charts_dir.clone()),
        })
        .unwrap();
        assert_eq!(
            outcome.charts.rank_movement,
            Some(charts_dir.join(svg::RANK_MOVEMENT))
        );
        assert!(
            outcome.text.contains("rank-movement.svg)"),
            "{}",
            outcome.text
        );

        // Nothing to draw: the directory is not created.
        let outcome = report(&ReportOptions {
            svg: Some(dir.path().join("unused")),
            ..ReportOptions::default()
        })
        .unwrap();
        assert_eq!(outcome.charts, Charts::default());
        assert!(!dir.path().join("unused").exists());

        // A directory that cannot be created is an error naming it.
        std::fs::write(dir.path().join("blocker"), "").unwrap();
        assert!(matches!(
            report(&ReportOptions {
                eval_after: Some(result),
                svg: Some(dir.path().join("blocker/charts")),
                ..ReportOptions::default()
            })
            .unwrap_err(),
            CommandError::Svg(_)
        ));
    }
}
