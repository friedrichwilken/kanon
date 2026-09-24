use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{ArgGroup, Args, Subcommand};

use kanon::commands::{
    self, Paths, QueriesAcceptOptions, QueriesAddOptions, QueriesImportOptions,
    QueriesSuggestOptions,
};
use kanon::queries::suggest::{DEFAULT_N, DEFAULT_OUT};
use pinakes::llm::UreqChatTransport;

use crate::cli::EXIT_POLICY;

#[derive(Subcommand)]
pub(crate) enum QueriesCommand {
    /// Append a row after checking `--expected` against the committed manifest, or accept rows
    /// of a suggestions file with `--from`.
    Add(QueriesAddArgs),
    /// Fail (exit 4) on unknown expected ids, duplicate ids or too small a held-out share.
    Check {
        /// Query file (default: `queries` from the config).
        #[arg(long, value_name = "FILE")]
        queries: Option<PathBuf>,
    },
    /// Turn `kanon grade`'s output into query rows.
    Import(QueriesImportArgs),
    /// Ask a model for the questions a sample of pages answers; writes a suggestions file for
    /// `queries add --from`, never `queries.jsonl`.
    Suggest(QueriesSuggestArgs),
}

#[derive(Args)]
pub(crate) struct QueriesImportArgs {
    /// `graded.jsonl`, as written by `kanon grade`.
    graded: PathBuf,
    /// Minimum grade for a candidate to enter `expected`.
    #[arg(long, default_value_t = kanon::queries::DEFAULT_MIN_GRADE)]
    min_grade: u8,
    /// Target held-out share for the random `holdout` assignment.
    #[arg(long, default_value_t = kanon::queries::DEFAULT_HOLDOUT_MIN)]
    holdout_share: f64,
    /// Seed for the deterministic PRNG that assigns `holdout`.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Query file to append to (default: `queries` from the config).
    #[arg(long, value_name = "FILE")]
    queries: Option<PathBuf>,
}

#[derive(Args)]
#[command(group = ArgGroup::new("selection").args(["accept", "accept_all"]))]
pub(crate) struct QueriesAddArgs {
    /// Query id.
    #[arg(long, required_unless_present = "from", conflicts_with = "from")]
    id: Option<String>,
    /// The query text.
    #[arg(long, required_unless_present = "from", conflicts_with = "from")]
    query: Option<String>,
    /// Page ids, id prefixes, or the legacy `<source>/<path>` form; every one must exist.
    #[arg(
        long,
        value_name = "ID",
        num_args = 1..,
        required_unless_present = "from",
        conflicts_with = "from"
    )]
    expected: Vec<String>,
    /// Query kind, e.g. `howto`.
    #[arg(long, default_value = "", conflicts_with = "from")]
    kind: String,
    /// Hold this row (or every accepted row) out of tuning decisions.
    #[arg(long)]
    holdout: bool,
    /// Suggestions file written by `queries suggest`; choose rows with `--accept` or
    /// `--accept-all`. The rows keep `"origin": "suggested"`.
    #[arg(long, value_name = "FILE", requires = "selection")]
    from: Option<PathBuf>,
    /// Ids of the suggestions to accept.
    #[arg(long, value_name = "ID", num_args = 1.., requires = "from", conflicts_with = "accept_all")]
    accept: Vec<String>,
    /// Accept every row of the suggestions file.
    #[arg(long, requires = "from")]
    accept_all: bool,
    /// Query file (default: `queries` from the config).
    #[arg(long, value_name = "FILE")]
    queries: Option<PathBuf>,
}

#[derive(Args)]
pub(crate) struct QueriesSuggestArgs {
    /// Pages to sample, stratified by source and section; each yields two or three queries.
    #[arg(long, default_value_t = DEFAULT_N, value_name = "N")]
    n: usize,
    /// Most pages any one source contributes.
    #[arg(long, value_name = "N")]
    per_source: Option<usize>,
    /// Suggestions file to write (replaced when it exists).
    #[arg(long, default_value = DEFAULT_OUT, value_name = "FILE")]
    out: PathBuf,
    /// Model name; falls back to `KANON_LLM_MODEL`.
    #[arg(long)]
    model: Option<String>,
    /// Seed for the deterministic page sampling.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

pub(crate) fn run_queries(paths: &Paths, command: QueriesCommand) -> Result<ExitCode> {
    match command {
        QueriesCommand::Add(args) => run_queries_add(paths, args),
        QueriesCommand::Check { queries } => run_queries_check(paths, queries.as_deref()),
        QueriesCommand::Import(args) => run_queries_import(paths, args),
        QueriesCommand::Suggest(args) => run_queries_suggest(paths, args),
    }
}

fn run_queries_import(paths: &Paths, args: QueriesImportArgs) -> Result<ExitCode> {
    let options = QueriesImportOptions {
        queries: args.queries,
        graded: args.graded,
        min_grade: args.min_grade,
        holdout_share: args.holdout_share,
        seed: args.seed,
    };
    let outcome = commands::queries_import(paths, &options)?;
    for query in &outcome.skipped {
        eprintln!("skipped {query:?}: no candidate at or above --min-grade");
    }
    eprintln!(
        "imported {} queries ({} skipped)",
        outcome.imported.len(),
        outcome.skipped.len()
    );
    Ok(ExitCode::SUCCESS)
}

fn run_queries_add(paths: &Paths, args: QueriesAddArgs) -> Result<ExitCode> {
    if let Some(from) = args.from {
        let options = QueriesAcceptOptions {
            queries: args.queries,
            from,
            accept: args.accept,
            accept_all: args.accept_all,
            holdout: args.holdout,
        };
        let added = commands::queries_accept(paths, &options)?;
        for query in &added {
            eprintln!("added {} ({} expected)", query.id, query.expected.len());
        }
        eprintln!("accepted {} suggestions", added.len());
        return Ok(ExitCode::SUCCESS);
    }
    let options = QueriesAddOptions {
        queries: args.queries,
        id: args.id.unwrap_or_default(),
        query: args.query.unwrap_or_default(),
        expected: args.expected,
        kind: args.kind,
        holdout: args.holdout,
    };
    let query = commands::queries_add(paths, &options)?;
    eprintln!("added {} ({} expected)", query.id, query.expected.len());
    Ok(ExitCode::SUCCESS)
}

fn run_queries_suggest(paths: &Paths, args: QueriesSuggestArgs) -> Result<ExitCode> {
    let options = QueriesSuggestOptions {
        n: args.n,
        per_source: args.per_source,
        out: args.out,
        model: args.model,
        seed: args.seed,
    };
    let transport = UreqChatTransport::new();
    let outcome = commands::queries_suggest(paths, &options, &transport)?;
    eprintln!(
        "{} pages sampled, {} suggestions ({} title quotes, {} empty, {} repeats rejected; \
         {} pages failed)",
        outcome.pages,
        outcome.suggestions.len(),
        outcome.title_quotes,
        outcome.empty,
        outcome.repeats,
        outcome.failed
    );
    eprintln!("wrote {}", options.out.display());
    Ok(ExitCode::SUCCESS)
}

fn run_queries_check(paths: &Paths, queries: Option<&std::path::Path>) -> Result<ExitCode> {
    let report = commands::queries_check(paths, queries)?;
    for (id, expected) in &report.unknown {
        eprintln!("unknown: {id}: expected {expected:?} matches no page in the manifest");
    }
    for id in &report.duplicate_ids {
        eprintln!("duplicate: {id}");
    }
    eprintln!(
        "held-out share: {:.3} (minimum {:.3})",
        report.holdout_share, report.holdout_min
    );
    if report.ok() {
        eprintln!("ok");
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::from(EXIT_POLICY))
    }
}
