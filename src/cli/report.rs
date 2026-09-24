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
    /// Run directory written by `eval --out`; adds a History section, one row per run.
    #[arg(long, value_name = "DIR")]
    runs: Option<PathBuf>,
}

pub(crate) fn run_report(args: ReportArgs) -> Result<ExitCode> {
    let options = ReportOptions {
        eval_before: args.eval_before,
        eval_after: args.eval_after,
        runs: args.runs,
    };
    let text = commands::report(&options)?;
    std::io::stdout().lock().write_all(text.as_bytes())?;
    Ok(ExitCode::SUCCESS)
}
