use std::path::PathBuf;

use crate::error::CommandError;
use crate::eval::EvalSummary;
use crate::history;
use crate::report::{self, ReportInput};

/// Inputs for `report`, as file paths.
#[derive(Debug, Clone, Default)]
pub struct ReportOptions {
    /// Eval result on the previous corpus or retriever.
    pub eval_before: Option<PathBuf>,
    /// Eval result on the current corpus or retriever.
    pub eval_after: Option<PathBuf>,
    /// A run directory (`eval --out`) to render a `## History` section from.
    pub runs: Option<PathBuf>,
}

/// Run `report`: render the evaluation sections of a Markdown report from eval result files
/// and, with `runs`, the history of a run directory.
pub fn report(options: &ReportOptions) -> Result<String, CommandError> {
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
    Ok(report::render(ReportInput {
        before: before.as_ref(),
        after: after.as_ref(),
        history: rows.as_deref(),
    }))
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
        let text = report(&ReportOptions {
            eval_before: None,
            eval_after: Some(result),
            runs: None,
        })
        .unwrap();
        assert!(text.starts_with("# Retrieval report\n"));
        assert!(text.contains("| overall | – → 0.500 |"), "{text}");
        assert!(!text.contains("## History"), "{text}");
        assert!(matches!(
            report(&ReportOptions {
                eval_before: Some(dir.path().join("missing.json")),
                eval_after: None,
                runs: None,
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
        .unwrap();
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
}
