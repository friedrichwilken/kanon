use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::Args;

use kanon::commands::{self, HistoryOptions, Paths};
use kanon::history;

#[derive(Args)]
pub(crate) struct HistoryArgs {
    /// Run directory written by `eval --out` (default: `runs` next to the config).
    #[arg(long, value_name = "DIR")]
    runs: Option<PathBuf>,
    /// Write the rows as JSON to this file instead of printing the table.
    #[arg(long, value_name = "OUT")]
    json: Option<PathBuf>,
}

pub(crate) fn run_history(paths: &Paths, args: HistoryArgs) -> Result<ExitCode> {
    let options = HistoryOptions {
        runs: args.runs,
        json: args.json,
    };
    let outcome = commands::history(paths, &options)?;
    eprintln!("{}: {} runs", outcome.dir.display(), outcome.rows.len());
    if let Some(path) = &options.json {
        eprintln!("wrote {}", path.display());
    } else {
        let table = history::render_table(&outcome.rows);
        std::io::stdout().lock().write_all(table.as_bytes())?;
    }
    Ok(ExitCode::SUCCESS)
}
