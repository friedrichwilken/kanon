//! `kanon` measures a retriever against a corpus, and keeps measuring it as both change.
//!
//! The corpus is an artifact directory as [pinakes](https://github.com/friedrichwilken/pinakes)
//! compiles it; this crate depends on pinakes as a library for the reference BM25 index
//! ([`pinakes::index`]), the artifact reader, `manifest.json` and the shared model client.
//! Everything about measuring lives here.
//!
//! Four modules depend on nothing else in the crate: [`config`] (`kanon.yaml`, or the `eval:`
//! block of `pinakes.yaml`), [`workspace`] ([`workspace::Paths`], the file locations every
//! command uses), `env` (`KANON_*` variables with a `PINAKES_*` fallback) and `num` (the one
//! `usize -> f64` cast); [`llm`] (the model endpoint from the environment) sits on `env`.
//! [`contracts`] sits on pinakes alone: the serde types of the backend contract, the trail and
//! the unit, with their versions, that every reader and writer of those documents goes through.
//!
//! [`eval`] is the judge (`queries.jsonl`), the metrics and the result file; [`embed`] writes
//! and reads the embeddings file pair; [`backend`] is the [`backend::Backend`] trait and its
//! five shapes (`bm25`, `bm25-tantivy`, `dense`, `hybrid`, `external`); [`grade`] replays a
//! trail against the index and asks a model to grade what came back; [`queries`] grows and
//! validates the judge; [`report`] renders the evaluation sections of a Markdown report;
//! [`history`] is the numbered run files `eval --out` writes and the rows `history` reads back.
//!
//! [`error`] holds [`error::CommandError`], the error type every command returns. [`commands`]
//! is one file per subcommand, each owning its options and outcome. The `kanon` binary's own
//! `src/cli/` (arguments, dispatch and printing) is not part of this library.

pub mod backend;
pub mod commands;
pub mod config;
pub mod contracts;
pub mod embed;
mod env;
pub mod error;
pub mod eval;
pub mod grade;
pub mod history;
pub mod llm;
mod num;
pub mod queries;
pub mod report;
#[cfg(test)]
pub(crate) mod testing;
pub mod workspace;
