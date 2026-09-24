//! Retriever shapes: a common [`Backend`] trait behind `bm25` (pinakes's reference index,
//! wrapped rather than rewritten), `bm25-tantivy` (the same units scored by tantivy's own BM25),
//! `dense` (an embeddings file), `hybrid` (reciprocal rank fusion of `bm25` and
//! `dense`) and `external` (a consumer's own store over HTTP).
//!
//! `eval` selects one with `--backend NAME`, or several at once with `--compare a,b,c`; the
//! numbers this produces decide which shape to run in production, not an argument from
//! architecture. A name is a built-in kind ([`BackendKind`]) or an entry of the config's
//! `backends:`; either way the command resolves it to a [`BackendSpec`], the name plus the
//! kind and the settings that came with it, and the result records the name.

use std::fmt;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::str::FromStr;

use thiserror::Error;

use crate::contracts::ContractError;
use crate::embed::{EmbedError, Embedder};
use pinakes::corpus::CorpusError;
use pinakes::index::{Hit, IndexError, Priorities};

mod bm25;
mod dense;
mod external;
mod hybrid;
mod tantivy;
#[cfg(test)]
mod testing;

pub use bm25::Bm25Backend;
pub use dense::DenseBackend;
pub use external::ExternalBackend;
pub use hybrid::{HybridBackend, RRF_DEPTH, RRF_K, reciprocal_rank_fusion};
pub use tantivy::TantivyBackend;

/// Errors raised while building or querying a backend.
#[derive(Debug, Error)]
pub enum BackendError {
    /// The `bm25` or `bm25-tantivy` index could not be built or searched.
    #[error(transparent)]
    Index(#[from] IndexError),
    /// Embedding, or reading/writing the embeddings file pair, failed.
    #[error(transparent)]
    Embed(#[from] EmbedError),
    /// tantivy failed.
    #[error("index: {0}")]
    Tantivy(#[from] ::tantivy::TantivyError),
    /// The backend is missing configuration it needs (a URL, an embedder, …).
    #[error("backend {backend}: {message}")]
    Config {
        /// The backend name.
        backend: String,
        /// What is missing.
        message: String,
    },
    /// `--with` / `--without` is not supported for this backend.
    #[error(
        "backend {0}: --with/--without needs a backend that indexes pages directly (bm25, bm25-tantivy)"
    )]
    UnsupportedAdjustment(String),
    /// The external backend's HTTP request failed.
    #[error("{url}: {message}")]
    Http {
        /// The requested URL.
        url: String,
        /// What went wrong.
        message: String,
    },
    /// The external backend's response was not the expected shape.
    #[error("{url}: unexpected response: {message}")]
    BadResponse {
        /// The requested URL.
        url: String,
        /// What was wrong with it.
        message: String,
    },
    /// The external backend answered with a contract version newer than this kanon reads.
    #[error(transparent)]
    Contract(#[from] ContractError),
    /// The requested backend name is neither one of the five kinds nor a configured name.
    #[error(
        "unknown backend {0:?}: expected bm25, bm25-tantivy, dense, hybrid, external or a name \
         from the config's backends"
    )]
    UnknownBackend(String),
}

/// A page-loading failure is reported as the [`IndexError`] it has always been.
impl From<CorpusError> for BackendError {
    fn from(err: CorpusError) -> Self {
        BackendError::Index(IndexError::from(err))
    }
}

/// One of the five retriever shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BackendKind {
    /// pinakes's hand-rolled `BM25Okapi` reference index (the default).
    #[default]
    Bm25,
    /// The same units, scored by tantivy's own BM25 (`k1` 1.2, `b` 0.75).
    Bm25Tantivy,
    /// An embeddings file, queried by cosine similarity.
    Dense,
    /// Reciprocal rank fusion of `bm25` and `dense`.
    Hybrid,
    /// A consumer's own store, over HTTP.
    External,
}

impl BackendKind {
    /// The name used on the command line and recorded in eval results.
    pub fn name(self) -> &'static str {
        match self {
            BackendKind::Bm25 => "bm25",
            BackendKind::Bm25Tantivy => "bm25-tantivy",
            BackendKind::Dense => "dense",
            BackendKind::Hybrid => "hybrid",
            BackendKind::External => "external",
        }
    }

    /// Whether this backend needs an embedder to run at all (`dense`, `hybrid`).
    pub fn needs_embedder(self) -> bool {
        matches!(self, BackendKind::Dense | BackendKind::Hybrid)
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for BackendKind {
    type Err = BackendError;

    fn from_str(name: &str) -> Result<BackendKind, BackendError> {
        match name {
            "bm25" => Ok(BackendKind::Bm25),
            "bm25-tantivy" => Ok(BackendKind::Bm25Tantivy),
            "dense" => Ok(BackendKind::Dense),
            "hybrid" => Ok(BackendKind::Hybrid),
            "external" => Ok(BackendKind::External),
            other => Err(BackendError::UnknownBackend(other.to_string())),
        }
    }
}

/// A backend as a command selects it: the name a result records, the kind that runs, and the
/// settings that came with the name.
///
/// A built-in kind ([`BackendSpec::builtin`]) is named after itself and carries the command's
/// `--backend-url`/`--embeddings`; a configured name (`backends:` in `kanon.yaml`) carries its
/// own `url` and `embeddings`, so two names of the same kind can point at two services or two
/// embeddings files and be compared in one run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendSpec {
    /// The name `--backend`/`--compare` was given and the result records.
    pub name: String,
    /// The shape that runs.
    pub kind: BackendKind,
    /// The search endpoint base URL (`external`).
    pub url: Option<String>,
    /// `embeddings.bin` (`dense`, `hybrid`); `None` for the workspace default.
    pub embeddings: Option<PathBuf>,
}

impl BackendSpec {
    /// A built-in kind under its own name, with no settings of its own.
    pub fn builtin(kind: BackendKind) -> BackendSpec {
        BackendSpec {
            name: kind.name().to_string(),
            kind,
            url: None,
            embeddings: None,
        }
    }

    /// Whether this backend needs an embedder to run at all (`dense`, `hybrid`).
    pub fn needs_embedder(&self) -> bool {
        self.kind.needs_embedder()
    }
}

/// `bm25` under its own name.
impl Default for BackendSpec {
    fn default() -> BackendSpec {
        BackendSpec::builtin(BackendKind::default())
    }
}

/// The name.
impl fmt::Display for BackendSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

/// Configuration shared by every backend's `build`: the pieces a plain `Index` does not need.
#[derive(Clone, Default)]
pub struct BackendConfig {
    /// Source priorities for the mirror rule (as `Index` uses).
    pub priorities: Priorities,
    /// `embeddings.bin` path (`dense`, `hybrid`).
    pub embeddings_bin: PathBuf,
    /// `embeddings.json` path (`dense`, `hybrid`).
    pub embeddings_json: PathBuf,
    /// Use embeddings even when their recorded manifest hash does not match the artifact.
    pub allow_stale: bool,
    /// Where query embeddings come from (`dense`, `hybrid`); required for those backends.
    pub embedder: Option<Rc<dyn Embedder>>,
    /// The consumer's search endpoint base URL (`external`).
    pub backend_url: Option<String>,
}

impl BackendConfig {
    fn embedder(&self, backend: &str) -> Result<Rc<dyn Embedder>, BackendError> {
        self.embedder.clone().ok_or_else(|| BackendError::Config {
            backend: backend.to_string(),
            message: "no embedder configured (KANON_EMBED_URL not set?)".to_string(),
        })
    }

    fn backend_url(&self) -> Result<&str, BackendError> {
        self.backend_url
            .as_deref()
            .ok_or_else(|| BackendError::Config {
                backend: BackendKind::External.name().to_string(),
                message: "--backend-url is required".to_string(),
            })
    }
}

/// A retriever shape: built once from an artifact, then searched repeatedly.
///
/// `build` is generic over the concrete backend (`Self`) and so is not part of the trait's
/// object-safe surface; `search`, `page_count` and `searchable_count` are, so `eval --compare`
/// can hold a `Vec<Box<dyn Backend>>` of backends chosen at run time by name.
pub trait Backend {
    /// Build the backend from an artifact directory.
    fn build(artifact: &Path, config: &BackendConfig) -> Result<Self, BackendError>
    where
        Self: Sized;

    /// The best `k` pages for `query`, optionally restricted to `module`.
    fn search(&self, query: &str, k: usize, module: Option<&str>)
    -> Result<Vec<Hit>, BackendError>;

    /// Pages read from the artifact, mirrors included.
    fn page_count(&self) -> usize;

    /// Pages in the search corpus (mirrors excluded).
    fn searchable_count(&self) -> usize;
}

/// Build the named backend from an artifact.
pub fn build(
    kind: BackendKind,
    artifact: &Path,
    config: &BackendConfig,
) -> Result<Box<dyn Backend>, BackendError> {
    Ok(match kind {
        BackendKind::Bm25 => Box::new(Bm25Backend::build(artifact, config)?),
        BackendKind::Bm25Tantivy => Box::new(TantivyBackend::build(artifact, config)?),
        BackendKind::Dense => Box::new(DenseBackend::build(artifact, config)?),
        BackendKind::Hybrid => Box::new(HybridBackend::build(artifact, config)?),
        BackendKind::External => Box::new(ExternalBackend::build(artifact, config)?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_kind_round_trips_through_its_name() {
        for kind in [
            BackendKind::Bm25,
            BackendKind::Bm25Tantivy,
            BackendKind::Dense,
            BackendKind::Hybrid,
            BackendKind::External,
        ] {
            assert_eq!(kind.name().parse::<BackendKind>().unwrap(), kind);
        }
        assert!(matches!(
            "nope".parse::<BackendKind>().unwrap_err(),
            BackendError::UnknownBackend(_)
        ));
    }

    #[test]
    fn the_config_lists_exactly_the_built_in_kinds() {
        for name in crate::config::BUILTIN_BACKENDS {
            assert!(name.parse::<BackendKind>().is_ok(), "{name}");
        }
        for kind in [
            BackendKind::Bm25,
            BackendKind::Bm25Tantivy,
            BackendKind::Dense,
            BackendKind::Hybrid,
            BackendKind::External,
        ] {
            assert!(crate::config::BUILTIN_BACKENDS.contains(&kind.name()));
        }
    }

    #[test]
    fn a_built_in_spec_is_named_after_its_kind() {
        let spec = BackendSpec::default();
        assert_eq!(spec, BackendSpec::builtin(BackendKind::Bm25));
        assert_eq!(spec.to_string(), "bm25");
        assert!(spec.url.is_none() && spec.embeddings.is_none());
        assert!(!spec.needs_embedder());
        assert!(BackendSpec::builtin(BackendKind::Hybrid).needs_embedder());
    }
}
