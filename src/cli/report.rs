use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::Args;

use kanon::commands::{self, ReportOptions};

#[derive(Args)]
pub(crate) struct ReportArgs {
    /// Eval JSON for the previous corpus or retriever.
    #[arg(long, value_name = "EVAL")]
    eval_before: Option<PathBuf>,
    /// Eval JSON for the current corpus or retriever.
    #[arg(long, value_name = "EVAL")]
    eval_after: Option<PathBuf>,
    /// Run directory written by `eval --out` (relative to the current directory); adds a
    /// History section, one row per run.
    #[arg(long, value_name = "DIR")]
    runs: Option<PathBuf>,
    /// Write the charts the inputs allow as SVG files into this directory (created if
    /// missing) and link them from the report by the path as given: recall per kind from the
    /// result, rank movement from both results, recall over runs from `--runs`.
    #[arg(long, value_name = "DIR")]
    svg: Option<PathBuf>,
}

pub(crate) fn run_report(args: ReportArgs) -> Result<ExitCode> {
    let options = ReportOptions {
        eval_before: args.eval_before,
        eval_after: args.eval_after,
        runs: args.runs,
        svg: args.svg,
    };
    let outcome = commands::report(&options)?;
    for path in outcome.charts.paths() {
        eprintln!("wrote {}", path.display());
    }
    std::io::stdout()
        .lock()
        .write_all(outcome.text.as_bytes())?;
    Ok(ExitCode::SUCCESS)
}
