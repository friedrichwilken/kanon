use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Args;

use kanon::commands::{self, BackendFlags, GradeOptions, Paths};
use pinakes::llm::UreqChatTransport;

#[derive(Args)]
pub(crate) struct GradeArgs {
    /// Trail file to replay.
    #[arg(long, value_name = "FILE")]
    trail: PathBuf,
    /// Backend to fetch candidates from: bm25 (default), bm25-tantivy, dense, hybrid, external
    /// or a name from the config's `backends`; the same names and config defaults as
    /// `eval --backend`.
    #[arg(long, value_name = "NAME")]
    backend: Option<String>,
    /// The consumer's search endpoint base URL (`--backend external`).
    #[arg(long, value_name = "URL")]
    backend_url: Option<String>,
    /// `embeddings.bin` path (`--backend dense`/`hybrid`; default: `embeddings.bin` next to the
    /// config).
    #[arg(long, value_name = "FILE")]
    embeddings: Option<PathBuf>,
    /// Use embeddings even when their recorded manifest hash does not match the artifact.
    #[arg(long)]
    allow_stale: bool,
    /// Candidates fetched per query.
    #[arg(long, value_name = "N")]
    k: Option<usize>,
    /// Model name; falls back to `KANON_LLM_MODEL`.
    #[arg(long)]
    model: Option<String>,
    /// Write graded rows to this file instead of stdout.
    #[arg(long, value_name = "OUT")]
    out: Option<PathBuf>,
}

pub(crate) fn run_grade(paths: &Paths, args: GradeArgs) -> Result<ExitCode> {
    let mut flags = BackendFlags {
        backend: args.backend,
        backend_url: args.backend_url,
        embeddings: args.embeddings,
        allow_stale: args.allow_stale,
        ..BackendFlags::default()
    };
    commands::apply_backend_config_defaults(paths, &mut flags)?;
    let backend = flags.spec()?;
    let embedder = if backend.needs_embedder() {
        Some(commands::eval_embedder_from_env()?)
    } else {
        None
    };
    let options = GradeOptions {
        trail: args.trail,
        backend,
        allow_stale: flags.allow_stale,
        embedder,
        k: args.k,
        model: args.model,
        out: args.out.clone(),
    };
    let transport = UreqChatTransport::new();
    let outcome = commands::grade(paths, &options, &transport)?;
    eprintln!(
        "{} queries, {} graded rows [{}]",
        outcome.queries,
        outcome.rows.len(),
        outcome.backend
    );
    if let Some(path) = &args.out {
        eprintln!("wrote {}", path.display());
    } else {
        let text = kanon::grade::to_jsonl(&outcome.rows).context("serialising graded rows")?;
        std::io::stdout().lock().write_all(text.as_bytes())?;
    }
    Ok(ExitCode::SUCCESS)
}
