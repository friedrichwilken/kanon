//! Command orchestration: the library side of every `kanon` subcommand.
//!
//! `main.rs` only parses arguments and maps results to exit codes; everything that reads or
//! writes files lives here so it can be tested without spawning the binary.

mod embed;
mod eval;
mod grade;
mod queries;
mod report;

use pinakes::index::Priorities;

use crate::config::{self, Config};

pub use crate::error::CommandError;
pub use crate::workspace::Paths;

pub use embed::{EmbedOptions, EmbedOutcome, embed};
pub use eval::{
    BackendEvalOptions, BackendEvalOutcome, EvalFlags, EvalOptions, EvalOutcome, EvalPlan,
    apply_eval_config_defaults, eval, eval_backend, eval_compare, eval_embedder_from_env,
    eval_plan,
};
pub use grade::{GradeOptions, GradeOutcome, grade};
pub use queries::{
    QueriesAddOptions, QueriesImportOptions, QueriesImportOutcome, queries_add, queries_check,
    queries_import,
};
pub use report::{ReportOptions, report};

/// What every command reads before it starts: the config, when there is one, and the source
/// priorities for the mirror rule.
#[derive(Debug, Clone, Default)]
pub struct Settings {
    /// `kanon.yaml`, or `pinakes.yaml`'s `eval:` block; `None` without either.
    pub config: Option<Config>,
    /// Source priorities from `pinakes.yaml`, when one sits next to the config.
    pub priorities: Priorities,
}

/// Read the workspace's [`Settings`]. A missing config file is fine (every command has flags
/// for what it needs); a present but invalid one is an error.
pub fn settings(paths: &Paths) -> Result<Settings, CommandError> {
    let config = if paths.config.is_file() {
        config::load(&paths.config)?
    } else {
        None
    };
    let priorities = config::priorities(&paths.pinakes_config())?;
    Ok(Settings { config, priorities })
}
