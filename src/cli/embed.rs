use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::Args;

use kanon::commands::{self, EmbedOptions, Paths};
use kanon::embed::HttpEmbedder;

#[derive(Args)]
pub(crate) struct EmbedArgs {
    /// Artifact directory (default: `artifact` next to the config).
    #[arg(long)]
    artifact: Option<PathBuf>,
    /// The embedding model, sent to the endpoint and recorded in `embeddings.json` (default:
    /// `KANON_EMBED_MODEL`).
    #[arg(long, value_name = "NAME")]
    model: Option<String>,
    /// `embeddings.bin` output path (default: `embeddings.bin` next to the config);
    /// `embeddings.json` is written next to it.
    #[arg(long, value_name = "FILE")]
    out: Option<PathBuf>,
    /// Text put in front of every unit before it is embedded, recorded in `embeddings.json`
    /// (default: `doc_prefix` from the config, else the model's known one, else none; pass an
    /// empty string for none).
    #[arg(long, value_name = "TEXT")]
    doc_prefix: Option<String>,
    /// Text `dense` and `hybrid` put in front of every query, recorded in `embeddings.json` so
    /// queries are embedded as the file was built (default: `query_prefix` from the config, else
    /// the model's known one, else none). Give a nomic model `--query-prefix "search_query: "`.
    #[arg(long, value_name = "TEXT")]
    query_prefix: Option<String>,
    /// Texts per embeddings request.
    #[arg(long, default_value_t = kanon::embed::DEFAULT_BATCH)]
    batch: usize,
}

pub(crate) fn run_embed(mut paths: Paths, args: EmbedArgs) -> Result<ExitCode> {
    if let Some(dir) = args.artifact {
        paths.artifact = dir;
    }
    let (embedder, model) = HttpEmbedder::from_env(args.model.as_deref())?;
    let options = EmbedOptions {
        model,
        batch: args.batch,
        out: args.out,
        doc_prefix: args.doc_prefix,
        query_prefix: args.query_prefix,
    };
    let outcome = commands::embed(&paths, &options, &embedder)?;
    eprintln!(
        "{}: embedded {} units ({} dims) -> {} and {}",
        paths.artifact.display(),
        outcome.units,
        outcome.dimension,
        outcome.bin_path.display(),
        outcome.json_path.display()
    );
    if outcome.doc_prefix.is_empty() && outcome.query_prefix.is_empty() {
        eprintln!("prefixes: none");
    } else {
        eprintln!(
            "prefixes: documents {:?}, queries {:?}",
            outcome.doc_prefix, outcome.query_prefix
        );
    }
    Ok(ExitCode::SUCCESS)
}
