//! `kanon` command-line interface: clap subcommands only; the logic lives in the library.
//!
//! Human output goes to stderr, data to stdout. Exit codes: 0 ok, 1 error, 2 a failed
//! `eval --gate`, 4 a failed `queries check`.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand};

use kanon::workspace::{KANON_CONFIG, PINAKES_CONFIG, Paths};

mod cli;

use cli::embed::{EmbedArgs, run_embed};
use cli::eval::{EvalArgs, run_eval};
use cli::grade::{GradeArgs, run_grade};
use cli::queries::{QueriesCommand, run_queries};
use cli::report::{ReportArgs, run_report};

/// Measure a retriever against a corpus, and keep measuring it as both change.
#[derive(Parser)]
#[command(name = "kanon", version, about)]
struct Cli {
    /// Path to the configuration: `kanon.yaml`, or a `pinakes.yaml` whose `eval:` block
    /// stands in for it (the fallback when `kanon.yaml` does not exist). Source priorities for
    /// the mirror rule come from that `pinakes.yaml`, or from the one next to `kanon.yaml`.
    #[arg(long, global = true, default_value = KANON_CONFIG)]
    config: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Measure retrieval quality: table on stderr, JSON on stdout, exit 2 when the gate fails.
    Eval(EvalArgs),
    /// Grow and validate the judge, `queries.jsonl`.
    Queries {
        #[command(subcommand)]
        command: QueriesCommand,
    },
    /// Embed every retrieval unit through an OpenAI-compatible endpoint.
    Embed(EmbedArgs),
    /// Replay a served-query trail against a backend and grade each candidate with a model.
    Grade(GradeArgs),
    /// Render the evaluation sections of a Markdown report on stdout.
    Report(ReportArgs),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::from(1)
        }
    }
}

/// The config to read: the one given, except that the default `kanon.yaml` falls back to a
/// `pinakes.yaml` in the same directory when it does not exist and that one does.
fn config_path(given: PathBuf) -> PathBuf {
    if given.is_file() || given.file_name().is_none_or(|n| n != KANON_CONFIG) {
        return given;
    }
    let fallback = given.with_file_name(PINAKES_CONFIG);
    if fallback.is_file() { fallback } else { given }
}

fn run(cli: Cli) -> Result<ExitCode> {
    let paths = Paths::for_config(&config_path(cli.config));
    match cli.command {
        Command::Eval(args) => run_eval(paths, args),
        Command::Queries { command } => run_queries(&paths, command),
        Command::Embed(args) => run_embed(paths, args),
        Command::Grade(args) => run_grade(&paths, args),
        Command::Report(args) => run_report(args),
    }
}
