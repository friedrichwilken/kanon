use std::path::PathBuf;

use crate::error::CommandError;
use crate::eval::EvalSummary;
use crate::report::{self, ReportInput};

/// Inputs for `report`, as file paths.
#[derive(Debug, Clone, Default)]
pub struct ReportOptions {
    /// Eval result on the previous corpus or retriever.
    pub eval_before: Option<PathBuf>,
    /// Eval result on the current corpus or retriever.
    pub eval_after: Option<PathBuf>,
}

/// Run `report`: render the evaluation sections of a Markdown report from eval result files.
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
    Ok(report::render(ReportInput {
        before: before.as_ref(),
        after: after.as_ref(),
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
        })
        .unwrap();
        assert!(text.starts_with("# Retrieval report\n"));
        assert!(text.contains("| overall | – → 0.500 |"), "{text}");
        assert!(matches!(
            report(&ReportOptions {
                eval_before: Some(dir.path().join("missing.json")),
                eval_after: None,
            })
            .unwrap_err(),
            CommandError::Eval(_)
        ));
    }
}
