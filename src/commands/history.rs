use std::path::PathBuf;

use crate::error::{CommandError, io_err};
use crate::history::{self, HistoryRow};
use crate::workspace::Paths;

/// The run directory `history` reads by default: `runs` next to the config.
pub const DEFAULT_RUNS_DIR: &str = "runs";

/// Options for `history`.
#[derive(Debug, Clone, Default)]
pub struct HistoryOptions {
    /// The run directory (default: [`DEFAULT_RUNS_DIR`] next to the config).
    pub runs: Option<PathBuf>,
    /// Write the rows as JSON here instead of returning them for the caller to print.
    pub json: Option<PathBuf>,
}

/// What `history` produced.
#[derive(Debug)]
pub struct HistoryOutcome {
    /// The directory that was read.
    pub dir: PathBuf,
    /// One row per run file, in sequence order.
    pub rows: Vec<HistoryRow>,
}

/// Run `history`: read every run file in the run directory, in sequence order. `history`
/// only reads; with `json`, the rows are written there as a JSON array.
pub fn history(paths: &Paths, options: &HistoryOptions) -> Result<HistoryOutcome, CommandError> {
    let dir = options
        .runs
        .clone()
        .unwrap_or_else(|| paths.config_dir().join(DEFAULT_RUNS_DIR));
    let rows = history::read_history(&dir)?;
    if let Some(path) = &options.json {
        let mut text = serde_json::to_string_pretty(&rows)?;
        text.push('\n');
        std::fs::write(path, text).map_err(io_err(path))?;
    }
    Ok(HistoryOutcome { dir, rows })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::{EvalOptions, eval};
    use crate::testing::eval_workspace;

    #[test]
    fn history_reads_the_runs_next_to_the_config_and_writes_json() {
        let (dir, paths) = eval_workspace();
        assert!(matches!(
            history(&paths, &HistoryOptions::default()).unwrap_err(),
            CommandError::History(_)
        ));
        for label in ["a", "b"] {
            let options = EvalOptions {
                queries: Some(dir.path().join("queries.jsonl")),
                out: Some(dir.path().join(DEFAULT_RUNS_DIR)),
                label: Some(label.to_string()),
                ..EvalOptions::default()
            };
            eval(&paths, &options).unwrap();
        }
        let json = dir.path().join("history.json");
        let outcome = history(
            &paths,
            &HistoryOptions {
                runs: None,
                json: Some(json.clone()),
            },
        )
        .unwrap();
        assert_eq!(outcome.dir, dir.path().join("runs"));
        assert_eq!(outcome.rows.len(), 2);
        assert_eq!(outcome.rows[1].label.as_deref(), Some("b"));
        let written: Vec<HistoryRow> =
            serde_json::from_str(&std::fs::read_to_string(&json).unwrap()).unwrap();
        assert_eq!(written, outcome.rows);
    }
}
