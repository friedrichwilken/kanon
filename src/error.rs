//! [`CommandError`]: the error type shared by every command.

use std::path::{Path, PathBuf};

use pinakes::corpus::CorpusError;
use pinakes::index::IndexError;
use pinakes::manifest::ManifestError;
use pinakes::trail::TrailError;
use thiserror::Error;

use crate::backend::BackendError;
use crate::config::ConfigError;
use crate::embed::EmbedError;
use crate::eval::EvalError;
use crate::grade::GradeError;
use crate::history::HistoryError;
use crate::llm::LlmEnvError;
use crate::queries::QueriesError;

/// Errors raised by any command.
#[derive(Debug, Error)]
pub enum CommandError {
    /// Bad or unreadable config.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// Bad or unreadable manifest.
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    /// A filesystem operation failed.
    #[error("{path}: {source}")]
    Io {
        /// The path involved.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// A JSON value could not be produced or parsed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Bad or unreadable queries or eval result.
    #[error(transparent)]
    Eval(#[from] EvalError),
    /// The artifact could not be indexed.
    #[error(transparent)]
    Index(#[from] IndexError),
    /// No query file: none given and no `queries` in the config.
    #[error("no query file: pass --queries or set queries in kanon.yaml")]
    NoQueries,
    /// `queries add`, `queries check` or `queries import` failed.
    #[error(transparent)]
    Queries(#[from] QueriesError),
    /// A retriever backend failed to build or search.
    #[error(transparent)]
    Backend(#[from] BackendError),
    /// Embedding, or reading/writing the embeddings file pair, failed.
    #[error(transparent)]
    Embed(#[from] EmbedError),
    /// `eval --compare` was given no backend names.
    #[error("--compare needs at least one backend name")]
    EmptyCompare,
    /// Building the model configuration failed (e.g. `KANON_LLM_URL` is not set).
    #[error(transparent)]
    Llm(#[from] LlmEnvError),
    /// Bad or unreadable `trail.jsonl`.
    #[error(transparent)]
    Trail(#[from] TrailError),
    /// `grade` failed, including talking to the model or the backend.
    #[error(transparent)]
    Grade(#[from] GradeError),
    /// A run file could not be written or read.
    #[error(transparent)]
    History(#[from] HistoryError),
}

/// A page-loading failure is reported as the [`IndexError`] it has always been.
impl From<CorpusError> for CommandError {
    fn from(err: CorpusError) -> Self {
        CommandError::Index(IndexError::from(err))
    }
}

/// Map an I/O error to [`CommandError::Io`] for `path`.
pub(crate) fn io_err(path: &Path) -> impl FnOnce(std::io::Error) -> CommandError + '_ {
    move |source| CommandError::Io {
        path: path.to_path_buf(),
        source,
    }
}
