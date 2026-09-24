//! Run history: the numbered run files `eval --out DIR` writes and the rows `kanon history`
//! and `report --runs DIR` read back from them.
//!
//! A run file is `DIR/NNN-<label>.json`: the eval result JSON as `--json` writes it, plus a
//! `run` object that ties the numbers to what they measured (label, time, backend, manifest
//! hash, query-set hash, `k`). `NNN` is the next zero-padded sequence number after the highest
//! one already in the directory, so a run directory reads in the order it was written and a
//! reviewer can see how a corpus or a retriever got to its current numbers. A plain eval result
//! dropped into the directory under a matching name is still a run; its facts are simply
//! missing.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::eval::{EvalSummary, Metrics};

/// The label a run gets when none is given and the config directory is not in a git
/// repository.
pub const FALLBACK_LABEL: &str = "run";

/// Errors raised while writing or reading run files.
#[derive(Debug, Error)]
pub enum HistoryError {
    /// The directory or file could not be read or written.
    #[error("{path}: {source}")]
    Io {
        /// The path involved.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The file is not a valid run (or eval result).
    #[error("{path}: invalid run file: {source}")]
    Json {
        /// The run file path.
        path: PathBuf,
        /// Underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> HistoryError + '_ {
    move |source| HistoryError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// What a run measured: the `run` object inside a run file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunInfo {
    /// The label in the file name (already sanitised).
    pub label: String,
    /// When the run was written, RFC 3339 UTC.
    pub at: String,
    /// The backend that was measured.
    pub backend: String,
    /// `sha256` of `<artifact>/manifest.json`, or `"none"` when the artifact has no manifest.
    pub manifest_sha256: String,
    /// `sha256` of the query file's bytes.
    pub queries_sha256: String,
    /// Result list length used.
    pub k: usize,
}

/// A run file: the eval result plus, when it was written by `eval --out`, the [`RunInfo`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Run {
    /// The eval result, exactly as `eval --json` writes it.
    #[serde(flatten)]
    pub summary: EvalSummary,
    /// What the run measured; `None` for a plain eval result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<RunInfo>,
}

impl Run {
    /// Read a run file. A plain eval result reads as a run without a [`RunInfo`].
    pub fn load(path: &Path) -> Result<Run, HistoryError> {
        let text = std::fs::read_to_string(path).map_err(io(path))?;
        serde_json::from_str(&text).map_err(|source| HistoryError::Json {
            path: path.to_path_buf(),
            source,
        })
    }

    /// The run as pretty JSON with a trailing newline.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let mut text = serde_json::to_string_pretty(self)?;
        text.push('\n');
        Ok(text)
    }
}

/// Keep `[A-Za-z0-9_.-]`; every other character becomes `-`. An empty label becomes
/// [`FALLBACK_LABEL`], so the file name always has one.
pub fn sanitise_label(label: &str) -> String {
    let clean: String = label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect();
    if clean.is_empty() {
        FALLBACK_LABEL.to_string()
    } else {
        clean
    }
}

/// The short SHA of the git repository containing `dir`, when `git rev-parse --short HEAD`
/// succeeds there; `None` otherwise (no git, no repository, no commit).
pub fn git_short_sha(dir: &Path) -> Option<String> {
    let dir = if dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        dir
    };
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let sha = String::from_utf8(output.stdout).ok()?;
    let sha = sha.trim();
    (!sha.is_empty()).then(|| sha.to_string())
}

/// The label for a run: `given` when there is one, else the git short SHA of the repository
/// containing `config_dir`, else [`FALLBACK_LABEL`]; sanitised either way.
pub fn run_label(given: Option<&str>, config_dir: &Path) -> String {
    let label = match given {
        Some(label) => label.to_string(),
        None => git_short_sha(config_dir).unwrap_or_else(|| FALLBACK_LABEL.to_string()),
    };
    sanitise_label(&label)
}

/// The sequence number and label of a run file name, `NNN-<label>.json`. Anything else in
/// the directory is not a run file.
fn parse_run_name(name: &str) -> Option<(u32, &str)> {
    let stem = name.strip_suffix(".json")?;
    let (digits, label) = stem.split_once('-')?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((digits.parse().ok()?, label))
}

/// The run files in `dir`, as `(sequence, path)`, sorted by sequence and then by name.
fn run_files(dir: &Path) -> Result<Vec<(u32, PathBuf)>, HistoryError> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(io(dir))? {
        let entry = entry.map_err(io(dir))?;
        let name = entry.file_name();
        if let Some((seq, _)) = name.to_str().and_then(parse_run_name) {
            files.push((seq, entry.path()));
        }
    }
    files.sort();
    Ok(files)
}

/// The next sequence number for `dir`: one more than the highest `NNN-*.json` in it, or 1 for
/// an empty or missing directory. Gaps are not filled.
pub fn next_sequence(dir: &Path) -> Result<u32, HistoryError> {
    if !dir.exists() {
        return Ok(1);
    }
    Ok(run_files(dir)?
        .last()
        .map_or(1, |(seq, _)| seq.saturating_add(1)))
}

/// The file name for sequence `seq` and `label` (already sanitised): `NNN-<label>.json`.
pub fn run_file_name(seq: u32, label: &str) -> String {
    format!("{seq:03}-{label}.json")
}

/// Write `run` as the next numbered file in `dir` (created when missing), named after
/// `run.run.label` (or [`FALLBACK_LABEL`] without a [`RunInfo`]). Returns the path written.
pub fn write_run(dir: &Path, run: &Run) -> Result<PathBuf, HistoryError> {
    std::fs::create_dir_all(dir).map_err(io(dir))?;
    let label = run
        .run
        .as_ref()
        .map_or(FALLBACK_LABEL, |info| info.label.as_str());
    let path = dir.join(run_file_name(next_sequence(dir)?, label));
    let text = run.to_json().map_err(|source| HistoryError::Json {
        path: path.clone(),
        source,
    })?;
    std::fs::write(&path, text).map_err(io(&path))?;
    Ok(path)
}

/// One run in the history: what `kanon history` prints per row and writes with `--json`.
/// Every `run`-object fact is optional, since a plain eval result in the directory is still
/// a row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryRow {
    /// The sequence number from the file name.
    pub seq: u32,
    /// The run file's name.
    pub file: String,
    /// The label from the `run` object.
    pub label: Option<String>,
    /// The backend from the `run` object, else the result's own `backend` when it has one.
    pub backend: Option<String>,
    /// Overall tuning metrics.
    pub tuning: Metrics,
    /// Overall held-out metrics, when the result has a held-out split.
    pub holdout: Option<Metrics>,
    /// `sha256` of the artifact's manifest at the time (`"none"` for no manifest).
    pub manifest_sha256: Option<String>,
    /// `sha256` of the query file at the time.
    pub queries_sha256: Option<String>,
    /// Result list length used.
    pub k: Option<usize>,
    /// When the run was written, RFC 3339 UTC.
    pub at: Option<String>,
}

impl HistoryRow {
    /// The row for a run read from `path` with sequence `seq`.
    pub fn of(seq: u32, path: &Path, run: &Run) -> HistoryRow {
        let info = run.run.as_ref();
        let backend = info.map(|i| i.backend.clone()).or_else(|| {
            let own = &run.summary.backend;
            (!own.is_empty()).then(|| own.clone())
        });
        HistoryRow {
            seq,
            file: path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            label: info.map(|i| i.label.clone()),
            backend,
            tuning: run.summary.tuning.overall,
            holdout: run.summary.holdout.as_ref().map(|s| s.overall),
            manifest_sha256: info.map(|i| i.manifest_sha256.clone()),
            queries_sha256: info.map(|i| i.queries_sha256.clone()),
            k: info.map(|i| i.k),
            at: info.map(|i| i.at.clone()),
        }
    }
}

/// Read every run file in `dir` in sequence order, one [`HistoryRow`] each. A file that is
/// not a valid result is an error naming the file; a missing directory is an error naming it.
pub fn read_history(dir: &Path) -> Result<Vec<HistoryRow>, HistoryError> {
    run_files(dir)?
        .into_iter()
        .map(|(seq, path)| Ok(HistoryRow::of(seq, &path, &Run::load(&path)?)))
        .collect()
}

/// The history as a Markdown table, one row per run.
pub fn render_table(rows: &[HistoryRow]) -> String {
    let mut out = String::from(
        "| # | label | backend | tuning recall@5 | recall@10 | MRR | n \
         | held-out recall@5 | recall@10 | MRR | n | manifest | at |\n\
         |---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    for row in rows {
        let dash = || "–".to_string();
        let text = |value: Option<&String>| value.cloned().unwrap_or_else(dash);
        let metric = |m: Option<&Metrics>, f: fn(&Metrics) -> f64| {
            m.map_or_else(dash, |m| format!("{:.3}", f(m)))
        };
        let n = |m: Option<&Metrics>| m.map_or_else(dash, |m| m.n.to_string());
        let tuning = Some(&row.tuning);
        let holdout = row.holdout.as_ref();
        let manifest = row
            .manifest_sha256
            .as_deref()
            .map_or_else(dash, |sha| sha.chars().take(8).collect());
        let _ = writeln!(
            out,
            "| {:03} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {manifest} | {} |",
            row.seq,
            text(row.label.as_ref()),
            text(row.backend.as_ref()),
            metric(tuning, |m| m.recall5),
            metric(tuning, |m| m.recall10),
            metric(tuning, |m| m.mrr),
            n(tuning),
            metric(holdout, |m| m.recall5),
            metric(holdout, |m| m.recall10),
            metric(holdout, |m| m.mrr),
            n(holdout),
            text(row.at.as_ref()),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;

    use super::*;
    use crate::eval::Split;

    fn metrics(r5: f64, r10: f64, mrr: f64, n: usize) -> Metrics {
        Metrics {
            recall5: r5,
            recall10: r10,
            mrr,
            n,
        }
    }

    fn summary(with_holdout: bool) -> EvalSummary {
        EvalSummary {
            tuning: Split {
                overall: metrics(0.8, 0.85, 0.66, 40),
                per_kind: BTreeMap::new(),
            },
            holdout: with_holdout.then(|| Split {
                overall: metrics(0.7, 0.8, 0.6, 10),
                per_kind: BTreeMap::new(),
            }),
            queries: vec![],
            backend: "bm25".to_string(),
        }
    }

    fn info(label: &str) -> RunInfo {
        RunInfo {
            label: label.to_string(),
            at: "2026-09-16T12:00:00Z".to_string(),
            backend: "bm25".to_string(),
            manifest_sha256: "ab".repeat(32),
            queries_sha256: "cd".repeat(32),
            k: 10,
        }
    }

    #[test]
    fn sequence_starts_at_one_and_follows_the_highest_existing_number() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(next_sequence(&dir.path().join("missing")).unwrap(), 1);
        assert_eq!(next_sequence(dir.path()).unwrap(), 1, "empty dir");

        // Gaps are not filled; files that are not runs are ignored.
        for name in [
            "001-a.json",
            "007-b.json",
            "003-c.json",
            "notes.md",
            "baseline.json",
            "x-1.json",
            "010-d.txt",
        ] {
            fs::write(dir.path().join(name), "{}").unwrap();
        }
        assert_eq!(next_sequence(dir.path()).unwrap(), 8);
        assert_eq!(run_file_name(8, "e"), "008-e.json");
        assert_eq!(run_file_name(1000, "e"), "1000-e.json");
        assert_eq!(parse_run_name("1000-e.json"), Some((1000, "e")));
        assert_eq!(parse_run_name("001-a-b.json"), Some((1, "a-b")));
        assert_eq!(parse_run_name("001.json"), None);
        assert_eq!(parse_run_name("-a.json"), None);
    }

    #[test]
    fn labels_keep_only_safe_characters() {
        assert_eq!(sanitise_label("abc12_.-"), "abc12_.-");
        assert_eq!(sanitise_label("my label/v2 ü"), "my-label-v2--");
        assert_eq!(sanitise_label(""), FALLBACK_LABEL);
        assert_eq!(run_label(Some("x y"), Path::new("")), "x-y");
    }

    #[test]
    fn run_files_round_trip_and_still_read_as_eval_results() {
        let dir = tempfile::tempdir().unwrap();
        let run = Run {
            summary: summary(true),
            run: Some(info("first")),
        };
        let first = write_run(dir.path(), &run).unwrap();
        assert_eq!(first, dir.path().join("001-first.json"));
        let second = write_run(&dir.path().join("nested/runs"), &run).unwrap();
        assert!(second.ends_with("nested/runs/001-first.json"));
        assert_eq!(Run::load(&first).unwrap(), run);
        // `--gate runs/001-first.json` keeps working: the `run` object is ignored.
        assert_eq!(EvalSummary::load(&first).unwrap(), run.summary);
        let text = fs::read_to_string(&first).unwrap();
        assert!(text.starts_with("{\n  \"tuning\":"), "{text}");
        assert!(text.contains("\"run\": {"), "{text}");
        assert!(text.contains("\"label\": \"first\""), "{text}");

        let bare = Run {
            summary: summary(false),
            run: None,
        };
        let third = write_run(dir.path(), &bare).unwrap();
        assert_eq!(third, dir.path().join("002-run.json"));
        assert!(!fs::read_to_string(&third).unwrap().contains("\"run\""));
    }

    #[test]
    fn history_rows_with_and_without_the_run_object() {
        let dir = tempfile::tempdir().unwrap();
        let run = Run {
            summary: summary(true),
            run: Some(info("first")),
        };
        write_run(dir.path(), &run).unwrap();
        // A plain eval result dropped into the directory under a run name.
        summary(false)
            .save(&dir.path().join("002-plain.json"))
            .unwrap();
        // A plain result with no backend at all.
        EvalSummary {
            backend: String::new(),
            ..summary(false)
        }
        .save(&dir.path().join("003-legacy.json"))
        .unwrap();
        fs::write(dir.path().join("README.md"), "not a run").unwrap();

        let rows = read_history(dir.path()).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].seq, 1);
        assert_eq!(rows[0].file, "001-first.json");
        assert_eq!(rows[0].label.as_deref(), Some("first"));
        assert_eq!(rows[0].backend.as_deref(), Some("bm25"));
        assert_eq!(rows[0].k, Some(10));
        assert!(rows[0].holdout.is_some());
        assert_eq!(rows[1].seq, 2);
        assert_eq!(rows[1].label, None);
        assert_eq!(rows[1].backend.as_deref(), Some("bm25"), "from the result");
        assert_eq!(rows[1].manifest_sha256, None);
        assert_eq!(rows[1].holdout, None);
        assert_eq!(rows[2].backend, None);

        let table = render_table(&rows);
        assert!(table.starts_with("| # | label | backend | tuning recall@5 |"));
        assert!(
            table.contains(
                "| 001 | first | bm25 | 0.800 | 0.850 | 0.660 | 40 | 0.700 | 0.800 | 0.600 | 10 \
                 | abababab | 2026-09-16T12:00:00Z |"
            ),
            "{table}"
        );
        assert!(
            table.contains(
                "| 002 | – | bm25 | 0.800 | 0.850 | 0.660 | 40 | – | – | – | – | – | – |"
            ),
            "{table}"
        );
        assert!(table.contains("| 003 | – | – | 0.800 |"), "{table}");

        // An invalid file is an error naming it.
        fs::write(dir.path().join("004-broken.json"), "{\"nope\": 1}").unwrap();
        let err = read_history(dir.path()).unwrap_err();
        assert!(matches!(err, HistoryError::Json { .. }));
        assert!(err.to_string().contains("004-broken.json"), "{err}");

        // A missing directory is an error naming it.
        let err = read_history(&dir.path().join("missing")).unwrap_err();
        assert!(err.to_string().contains("missing"), "{err}");
    }
}
