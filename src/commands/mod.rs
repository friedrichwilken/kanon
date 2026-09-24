//! Command orchestration: the library side of every `kanon` subcommand.
//!
//! `main.rs` only parses arguments and maps results to exit codes; everything that reads or
//! writes files lives here so it can be tested without spawning the binary.

mod embed;
mod eval;
mod grade;
mod history;
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
pub use history::{DEFAULT_RUNS_DIR, HistoryOptions, HistoryOutcome, history};
pub use queries::{
    QueriesAcceptOptions, QueriesAddOptions, QueriesImportOptions, QueriesImportOutcome,
    QueriesSuggestOptions, QueriesSuggestOutcome, queries_accept, queries_add, queries_check,
    queries_import, queries_suggest,
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
/// for what it needs); a present but invalid one is an error. Priorities come from the config
/// itself when it is a `pinakes.yaml` under any name, else from the `pinakes.yaml` next to it.
pub fn settings(paths: &Paths) -> Result<Settings, CommandError> {
    let document = if paths.config.is_file() {
        config::load_document(&paths.config)?
    } else {
        config::Document::default()
    };
    let priorities = match document.priorities {
        Some(priorities) => priorities,
        None => config::priorities(&paths.pinakes_config())?,
    };
    Ok(Settings {
        config: document.config,
        priorities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::eval_workspace;

    const PINAKES: &str = "version: 1\nsources:\n  - name: handbook\n    repo: https://github.com/o/handbook.git\n    \
                           ref: main\n    priority: 7\n    resolver:\n      type: glob\n      include: ['**/*.md']\n\
                           eval:\n  queries: queries.jsonl\n  k: 4\n";

    #[test]
    fn settings_take_priorities_from_the_named_file_before_the_sibling() {
        let (dir, paths) = eval_workspace();
        assert!(settings(&paths).unwrap().config.is_none());
        assert_eq!(settings(&paths).unwrap().priorities.of("handbook"), 1);

        // A pinakes config under another name carries its own priorities.
        let prod = dir.path().join("prod.yaml");
        std::fs::write(&prod, PINAKES).unwrap();
        let from_prod = settings(&Paths::for_config(&prod)).unwrap();
        assert_eq!(from_prod.config.unwrap().k, 4);
        assert_eq!(from_prod.priorities.of("handbook"), 7);

        // A kanon.yaml takes them from the pinakes.yaml next to it, even one with no eval block.
        std::fs::write(&paths.config, "queries: queries.jsonl\n").unwrap();
        std::fs::write(
            dir.path().join("pinakes.yaml"),
            PINAKES
                .split("eval:")
                .next()
                .unwrap()
                .replace("priority: 7", "priority: 3"),
        )
        .unwrap();
        let from_kanon = settings(&paths).unwrap();
        assert_eq!(from_kanon.config.unwrap().k, crate::config::DEFAULT_K);
        assert_eq!(from_kanon.priorities.of("handbook"), 3);
    }
}
