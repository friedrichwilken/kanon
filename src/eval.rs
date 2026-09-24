//! Retrieval evaluation against `queries.jsonl`.
//!
//! [`load_queries`] reads the judge, [`evaluate`] runs every query through an [`Index`] and
//! [`summarise`] folds the per-query rows into recall@5, recall@10 (or @k when `k < 10`), MRR,
//! nDCG@5, nDCG@10 and n, overall and per `kind`, for tuning rows and `holdout: true` rows
//! separately. [`gate`] compares one tuning metric with a baseline and [`delta`] compares two
//! results for `--with` / `--without`. The result types are what `report` renders.
//!
//! `expected` entries come in two forms. `<source>::<path>` names a page and `<source>::<dir>/`
//! any page under a directory. The legacy `<source>/<path>` form (no `::`) matches the page or
//! any page under that directory prefix, on a path-segment boundary, so `handbook/docs/user` does
//! not match `handbook/docs/super-user/x.md`.
//!
//! `graded` is the graded view of the same row: the same kind of key, each with a relevance of
//! 0 to 3 ([`MAX_GRADE`]). Recall and MRR read `expected` alone, nDCG reads `graded`, and a row
//! without `graded` gets relevance 1 for every `expected` entry, so its nDCG is defined but
//! binary. Each key's gain counts once, at the first returned page that matches it (a page that
//! matches several keys takes the highest grade among them), and the ideal ordering of all the
//! row's grades is the IDCG, so nDCG never exceeds 1 when a directory key matches several pages.
//! Precision@k stays out: with "any of these" expected lists it is not well defined.
//!
//! A row with `kind: "negative"` ([`NEGATIVE_KIND`]) and an empty `expected` says the corpus
//! does not answer the query. Such rows are left out of every recall, MRR and nDCG aggregate and
//! never gate; a split's [`Negative`] block counts how many of them the retriever rejected: top
//! score under `negative_threshold`, or, without one, no hit at all.

use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::num::float;
use pinakes::index::{Hit, Index, IndexError};
use pinakes::jsonl::{self, JsonlError};

pub use crate::config::GateMetric;

/// The `kind` of a negative query: the corpus does not answer it, so `expected` is empty.
pub const NEGATIVE_KIND: &str = "negative";
/// The highest relevance a `graded` entry may carry.
pub const MAX_GRADE: u8 = 3;

/// Errors raised while reading queries or evaluation results.
#[derive(Debug, Error)]
pub enum EvalError {
    /// The file could not be read.
    #[error("{path}: {source}")]
    Io {
        /// The file path.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The file is not a valid result.
    #[error("{path}: invalid eval result: {source}")]
    Json {
        /// The result file path.
        path: PathBuf,
        /// Underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
    /// A query line is not valid JSON.
    #[error("{path}:{line}: invalid query: {source}")]
    Query {
        /// The query file path.
        path: PathBuf,
        /// 1-based line number.
        line: usize,
        /// Underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
    /// Searching failed.
    #[error(transparent)]
    Index(#[from] IndexError),
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> EvalError + '_ {
    move |source| EvalError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// One row of `queries.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Query {
    /// Query id.
    pub id: String,
    /// The query text.
    pub query: String,
    /// Page ids or id prefixes that count as a hit.
    #[serde(default)]
    pub expected: Vec<String>,
    /// Page ids or id prefixes with their relevance, 0 to [`MAX_GRADE`], for nDCG. Empty for a
    /// row with no grades (nDCG then reads `expected` with relevance 1) and left out of the JSON
    /// then, so rows written before the field existed round-trip byte for byte.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub graded: BTreeMap<String, u8>,
    /// Query kind, possibly empty; [`NEGATIVE_KIND`] marks a query the corpus does not answer.
    #[serde(default)]
    pub kind: String,
    /// Held out from tuning decisions.
    #[serde(default)]
    pub holdout: bool,
    /// Where the row came from: `"suggested"` for a row accepted from `queries suggest`, absent
    /// for a hand-written one. Left out of the JSON when absent, so rows written before the
    /// field existed round-trip byte for byte.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
}

impl Query {
    /// Whether the row is a negative query (`kind: "negative"`).
    pub fn is_negative(&self) -> bool {
        self.kind == NEGATIVE_KIND
    }
}

/// Read `queries.jsonl`; blank lines are skipped.
pub fn load_queries(path: &Path) -> Result<Vec<Query>, EvalError> {
    jsonl::read(path).map_err(|err| match err {
        JsonlError::Io { path, source } => EvalError::Io { path, source },
        JsonlError::Json { path, line, source } => EvalError::Query { path, line, source },
    })
}

/// Whether `page_id` (`<source>::<path>`) satisfies an `expected` entry.
pub fn matches(page_id: &str, expected: &str) -> bool {
    let page = page_id.replacen("::", "/", 1);
    let (prefix, any_below) = match expected.split_once("::") {
        Some((source, rest)) => (format!("{source}/{rest}"), rest.ends_with('/')),
        None => (expected.to_string(), true),
    };
    let prefix = prefix.trim_end_matches('/');
    if prefix.is_empty() {
        return false;
    }
    page == prefix
        || (any_below && page.starts_with(prefix) && page[prefix.len()..].starts_with('/'))
}

/// The retrieval metrics over a set of queries.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// Fraction of queries with an expected page in the top 5.
    #[serde(rename = "recall@5")]
    pub recall5: f64,
    /// Fraction of queries with an expected page in the top 10.
    #[serde(rename = "recall@10")]
    pub recall10: f64,
    /// Mean reciprocal rank of the first expected hit.
    pub mrr: f64,
    /// Mean nDCG over the top 5. Zero in a result written before the metric existed.
    #[serde(default, rename = "ndcg@5")]
    pub ndcg5: f64,
    /// Mean nDCG over the top 10. Zero in a result written before the metric existed.
    #[serde(default, rename = "ndcg@10")]
    pub ndcg10: f64,
    /// Number of queries.
    pub n: usize,
}

impl Metrics {
    /// Aggregate per-query rows; all zero for no rows.
    pub fn of(rows: &[&QueryResult]) -> Metrics {
        let n = rows.len();
        if n == 0 {
            return Metrics {
                recall5: 0.0,
                recall10: 0.0,
                mrr: 0.0,
                ndcg5: 0.0,
                ndcg10: 0.0,
                n: 0,
            };
        }
        let count = |pred: fn(&QueryResult) -> bool| float(rows.iter().filter(|r| pred(r)).count());
        let mean = |f: fn(&QueryResult) -> f64| rows.iter().map(|r| f(r)).sum::<f64>() / float(n);
        Metrics {
            recall5: count(|r| r.hit5) / float(n),
            recall10: count(|r| r.hit10) / float(n),
            mrr: mean(|r| r.rr),
            ndcg5: mean(|r| r.ndcg5),
            ndcg10: mean(|r| r.ndcg10),
            n,
        }
    }

    /// The value of one metric, for the gate.
    pub fn get(&self, metric: GateMetric) -> f64 {
        match metric {
            GateMetric::Recall5 => self.recall5,
            GateMetric::Recall10 => self.recall10,
            GateMetric::Mrr => self.mrr,
            GateMetric::Ndcg5 => self.ndcg5,
            GateMetric::Ndcg10 => self.ndcg10,
        }
    }
}

/// The negative queries of a split: how many the retriever rejected.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Negative {
    /// Number of negative queries.
    pub n: usize,
    /// Queries whose top score stayed under the threshold (or, without one, that got no hit).
    pub rejected: usize,
    /// `rejected / n`.
    pub share: f64,
}

impl Negative {
    /// Count the rejected rows among `rows` (negative rows only); `None` for no rows.
    pub fn of(rows: &[&QueryResult], threshold: Option<f64>) -> Option<Negative> {
        let n = rows.len();
        if n == 0 {
            return None;
        }
        let rejected = rows.iter().filter(|r| r.rejected(threshold)).count();
        Some(Negative {
            n,
            rejected,
            share: float(rejected) / float(n),
        })
    }
}

/// Metrics for one split (tuning or held-out), overall and per query kind.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Split {
    /// All queries in the split, negative ones left out.
    pub overall: Metrics,
    /// Queries grouped by `kind`, negative ones left out.
    #[serde(default)]
    pub per_kind: BTreeMap<String, Metrics>,
    /// The split's negative queries, when it has any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative: Option<Negative>,
}

impl Split {
    /// Aggregate per-query rows overall and per kind; negative rows go to their own block,
    /// rejected when their top score stays under `negative_threshold` (or, without one, when
    /// they got no hit at all).
    pub fn of(rows: &[&QueryResult], negative_threshold: Option<f64>) -> Split {
        let (negative, scored): (Vec<&QueryResult>, Vec<&QueryResult>) =
            rows.iter().partition(|r| r.is_negative());
        let mut by_kind: BTreeMap<String, Vec<&QueryResult>> = BTreeMap::new();
        for row in &scored {
            by_kind.entry(row.kind.clone()).or_default().push(row);
        }
        Split {
            overall: Metrics::of(&scored),
            per_kind: by_kind
                .into_iter()
                .map(|(kind, rows)| (kind, Metrics::of(&rows)))
                .collect(),
            negative: Negative::of(&negative, negative_threshold),
        }
    }
}

/// One query's outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryResult {
    /// Query id.
    pub id: String,
    /// Query kind.
    #[serde(default)]
    pub kind: String,
    /// Whether the query is held out from tuning.
    #[serde(default)]
    pub holdout: bool,
    /// The query's `origin` (`"suggested"` for an accepted suggestion), so a reader can tell
    /// generated queries from real ones; left out when the query has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    /// An expected page was in the top 5.
    pub hit5: bool,
    /// An expected page was in the top 10.
    pub hit10: bool,
    /// Reciprocal rank of the first expected hit (0 when none).
    pub rr: f64,
    /// nDCG over the top 5 (0 when the row has nothing relevant to find).
    #[serde(default)]
    pub ndcg5: f64,
    /// nDCG over the top 10 (0 when the row has nothing relevant to find).
    #[serde(default)]
    pub ndcg10: f64,
    /// The relevance of each returned page, same length as `top`: the grade of the `graded`
    /// key it counted for (1 for an `expected` entry when the row has no grades), else 0.
    #[serde(default)]
    pub rels: Vec<u8>,
    /// The page ids returned, best first.
    #[serde(default)]
    pub top: Vec<String>,
    /// The backend's score of the first returned page, rounded to six decimals; absent when
    /// nothing came back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_score: Option<f64>,
}

impl QueryResult {
    /// Score `top` (the `k` page ids returned, best first, with the first one's score) against
    /// the query's expectations. The score is kept to six decimals: it only ever meets a
    /// threshold, and the reference index sums term scores in hash order, so the last digits
    /// differ from run to run and would make a result file impossible to pin.
    pub fn score(query: &Query, top: Vec<String>, top_score: Option<f64>, k: usize) -> QueryResult {
        let top_score = top_score.map(|score| (score * 1e6).round() / 1e6);
        let rank = top
            .iter()
            .position(|id| query.expected.iter().any(|exp| matches(id, exp)));
        let (rels, ideal) = relevance(query, &top);
        QueryResult {
            id: query.id.clone(),
            kind: query.kind.clone(),
            holdout: query.holdout,
            origin: query.origin.clone(),
            hit5: rank.is_some_and(|r| r < 5),
            hit10: rank.is_some_and(|r| r < k.min(10)),
            rr: rank.map_or(0.0, |r| 1.0 / float(r + 1)),
            ndcg5: ndcg(&rels, &ideal, 5),
            ndcg10: ndcg(&rels, &ideal, k.min(10)),
            rels,
            top,
            top_score,
        }
    }

    /// Score a backend's hits (best first) against the query's expectations.
    pub fn of_hits(query: &Query, hits: &[Hit], k: usize) -> QueryResult {
        let top = hits.iter().map(|hit| hit.page_id.clone()).collect();
        QueryResult::score(query, top, hits.first().map(|hit| hit.score), k)
    }

    /// Whether the row is a negative query (`kind: "negative"`).
    pub fn is_negative(&self) -> bool {
        self.kind == NEGATIVE_KIND
    }

    /// Whether the retriever rejected the query: its top score stays under `threshold`, or,
    /// without one, nothing came back at all.
    pub fn rejected(&self, threshold: Option<f64>) -> bool {
        match threshold {
            Some(threshold) => self.top_score.is_none_or(|score| score < threshold),
            None => self.top.is_empty(),
        }
    }
}

/// The relevance of each page in `top` and, sorted best first, the row's ideal gains.
///
/// The keys are `graded` with their grades, or `expected` with grade 1 when the row has no
/// grades. A page takes the most specific key that matches it: its own id first, else the
/// longest directory prefix (the longer key on a tie, then the first in key order), and it takes
/// that key's grade even when it is 0. A key graded above 0 counts once, at the first page that
/// takes it; a later page under a taken prefix falls back to the next less specific key that is
/// still free, else 0. A key graded 0 is never used up, so every page it names stays at 0, even
/// under a graded directory.
fn relevance(query: &Query, top: &[String]) -> (Vec<u8>, Vec<u8>) {
    let keys: Vec<(&str, u8)> = if query.graded.is_empty() {
        query.expected.iter().map(|key| (key.as_str(), 1)).collect()
    } else {
        query
            .graded
            .iter()
            .map(|(key, grade)| (key.as_str(), *grade))
            .collect()
    };
    let path = |id: &str| id.replacen("::", "/", 1);
    let mut taken = vec![false; keys.len()];
    let rels = top
        .iter()
        .map(|page| {
            // The keys that match this page, most specific first.
            let mut candidates: Vec<(usize, bool)> = keys
                .iter()
                .enumerate()
                .filter(|(_, (key, _))| matches(page, key))
                .map(|(i, (key, _))| (i, path(page) == path(key).trim_end_matches('/')))
                .collect();
            candidates.sort_by(|&(a, a_exact), &(b, b_exact)| {
                b_exact
                    .cmp(&a_exact)
                    .then(keys[b].0.len().cmp(&keys[a].0.len()))
                    .then(keys[a].0.cmp(keys[b].0))
            });
            let free = |&&(i, _): &&(usize, bool)| keys[i].1 == 0 || !taken[i];
            candidates.iter().find(free).map_or(0, |&(i, _)| {
                taken[i] = true;
                keys[i].1
            })
        })
        .collect();
    let mut ideal: Vec<u8> = keys.iter().map(|(_, grade)| *grade).collect();
    ideal.sort_unstable_by_key(|grade| Reverse(*grade));
    (rels, ideal)
}

/// Discounted cumulative gain: Σ (2^rel − 1) / log2(rank + 1), rank from 1. Folded from `0.0`
/// rather than summed, so a row with no gain reads `0.000`, not `-0.000`.
fn dcg(rels: impl Iterator<Item = u8>) -> f64 {
    rels.enumerate().fold(0.0, |acc, (i, rel)| {
        acc + (2f64.powi(i32::from(rel)) - 1.0) / float(i + 2).log2()
    })
}

/// nDCG@k: the DCG of the first `k` of `rels` over the DCG of the first `k` of `ideal`; 0 when
/// the ideal is empty.
fn ndcg(rels: &[u8], ideal: &[u8], k: usize) -> f64 {
    let idcg = dcg(ideal.iter().take(k).copied());
    if idcg > 0.0 {
        dcg(rels.iter().take(k).copied()) / idcg
    } else {
        0.0
    }
}

/// The JSON `eval` writes with `--json` and `report` reads with `--eval-before/--eval-after`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalSummary {
    /// Queries with `holdout: false`.
    pub tuning: Split,
    /// Queries with `holdout: true`, when there are any.
    #[serde(default)]
    pub holdout: Option<Split>,
    /// Per-query rows.
    #[serde(default)]
    pub queries: Vec<QueryResult>,
    /// The backend that produced this result, e.g. `"bm25"` or `"dense"`. Left
    /// empty (and omitted from the JSON) by the plain BM25 `eval` path, which predates backend
    /// selection; only `eval --backend`/`--compare` set it.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub backend: String,
}

impl EvalSummary {
    /// Record which backend produced this result.
    #[must_use]
    pub fn with_backend(mut self, name: &str) -> EvalSummary {
        self.backend = name.to_string();
        self
    }
}

impl EvalSummary {
    /// Read a result file.
    pub fn load(path: &Path) -> Result<EvalSummary, EvalError> {
        let text = std::fs::read_to_string(path).map_err(io(path))?;
        serde_json::from_str(&text).map_err(|source| EvalError::Json {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Write the result as pretty JSON with a trailing newline.
    pub fn save(&self, path: &Path) -> Result<(), EvalError> {
        let text = self.to_json().map_err(|source| EvalError::Json {
            path: path.to_path_buf(),
            source,
        })?;
        std::fs::write(path, text).map_err(io(path))
    }

    /// The result as pretty JSON with a trailing newline.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let mut text = serde_json::to_string_pretty(self)?;
        text.push('\n');
        Ok(text)
    }

    /// The row for a query id.
    pub fn query(&self, id: &str) -> Option<&QueryResult> {
        self.queries.iter().find(|q| q.id == id)
    }
}

/// Run every query against `index` with a result list of `k` pages; see [`summarise`] for
/// `negative_threshold`.
pub fn evaluate(
    index: &Index,
    queries: &[Query],
    k: usize,
    negative_threshold: Option<f64>,
) -> Result<EvalSummary, EvalError> {
    let mut rows = Vec::with_capacity(queries.len());
    for query in queries {
        let hits = index.search(&query.query, k, None)?;
        rows.push(QueryResult::of_hits(query, &hits, k));
    }
    Ok(summarise(rows, negative_threshold))
}

/// Fold per-query rows into tuning and held-out splits. A negative row counts as rejected when
/// its top score stays under `negative_threshold`, or, without one, when it got no hit at all.
pub fn summarise(rows: Vec<QueryResult>, negative_threshold: Option<f64>) -> EvalSummary {
    let tuning: Vec<&QueryResult> = rows.iter().filter(|r| !r.holdout).collect();
    let holdout: Vec<&QueryResult> = rows.iter().filter(|r| r.holdout).collect();
    EvalSummary {
        tuning: Split::of(&tuning, negative_threshold),
        holdout: (!holdout.is_empty()).then(|| Split::of(&holdout, negative_threshold)),
        queries: rows,
        backend: String::new(),
    }
}

/// The outcome of `--gate`: one tuning metric now and in the baseline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gate {
    /// Which tuning metric is compared.
    pub metric: GateMetric,
    /// The metric in the baseline.
    pub baseline: f64,
    /// The metric now.
    pub current: f64,
    /// Largest tolerated drop (`max_recall_drop`).
    pub max_drop: f64,
    /// The baseline has queries but its value for the metric is exactly 0: most likely a
    /// result written before the metric existed, which reads as 0 and would pass silently.
    pub baseline_unset: bool,
}

impl Gate {
    /// How much the metric dropped (negative when it rose).
    pub fn drop(&self) -> f64 {
        self.baseline - self.current
    }

    /// Whether the drop is within tolerance.
    pub fn passed(&self) -> bool {
        self.drop() <= self.max_drop
    }
}

/// Compare the tuning `metric` of `current` with `baseline`. [`Gate::baseline_unset`] flags a
/// baseline that has queries but no value for the metric (see there).
pub fn gate(
    current: &EvalSummary,
    baseline: &EvalSummary,
    max_drop: f64,
    metric: GateMetric,
) -> Gate {
    let baseline_value = baseline.tuning.overall.get(metric);
    Gate {
        metric,
        baseline: baseline_value,
        current: current.tuning.overall.get(metric),
        max_drop,
        baseline_unset: baseline.tuning.overall.n > 0 && baseline_value == 0.0,
    }
}

/// A query whose reciprocal rank changed between two results.
#[derive(Debug, Clone, PartialEq)]
pub struct RankChange {
    /// Query id.
    pub id: String,
    /// Reciprocal rank before (`None` when the query was not evaluated).
    pub before: Option<f64>,
    /// Reciprocal rank after (`None` when the query was not evaluated).
    pub after: Option<f64>,
}

/// The difference between two results (`--with` / `--without`).
#[derive(Debug, Clone, PartialEq)]
pub struct Delta {
    /// Tuning metrics before and after.
    pub tuning: (Metrics, Metrics),
    /// Held-out metrics before and after, when either side has them.
    pub holdout: Option<(Metrics, Metrics)>,
    /// Queries whose reciprocal rank changed, in `after` order.
    pub changed: Vec<RankChange>,
}

/// Compare two results.
pub fn delta(before: &EvalSummary, after: &EvalSummary) -> Delta {
    let holdout = match (&before.holdout, &after.holdout) {
        (None, None) => None,
        (b, a) => Some((
            b.as_ref().map_or_else(|| Metrics::of(&[]), |s| s.overall),
            a.as_ref().map_or_else(|| Metrics::of(&[]), |s| s.overall),
        )),
    };
    let mut changed = Vec::new();
    for row in &after.queries {
        let old = before.query(&row.id).map(|q| q.rr);
        if old != Some(row.rr) {
            changed.push(RankChange {
                id: row.id.clone(),
                before: old,
                after: Some(row.rr),
            });
        }
    }
    for row in &before.queries {
        if after.query(&row.id).is_none() {
            changed.push(RankChange {
                id: row.id.clone(),
                before: Some(row.rr),
                after: None,
            });
        }
    }
    Delta {
        tuning: (before.tuning.overall, after.tuning.overall),
        holdout,
        changed,
    }
}

fn table_rows(out: &mut String, label: &str, split: &Split) {
    let row = |out: &mut String, kind: &str, m: &Metrics| {
        let _ = writeln!(
            out,
            "| {label} | {kind} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
            m.n, m.recall5, m.recall10, m.mrr, m.ndcg5, m.ndcg10
        );
    };
    row(out, "overall", &split.overall);
    for (kind, metrics) in &split.per_kind {
        row(out, kind, metrics);
    }
}

/// The line `eval` prints for a split's negative queries, e.g.
/// `tuning: 3 negative queries, 2 rejected (0.667)`.
fn negative_line(out: &mut String, label: &str, negative: &Negative) {
    let _ = writeln!(
        out,
        "{label}: {} negative queries, {} rejected ({:.3})",
        negative.n, negative.rejected, negative.share
    );
}

/// The Markdown table `eval` prints on stderr, followed by one line per split that has
/// negative queries.
pub fn render_table(summary: &EvalSummary) -> String {
    let mut out = String::from(
        "| split | kind | n | recall@5 | recall@10 | MRR | nDCG@5 | nDCG@10 |\n\
         |---|---|---|---|---|---|---|---|\n",
    );
    // A split of negative queries alone has no metric to show; its line below says it all.
    let rows = |out: &mut String, label: &str, split: &Split| {
        if split.overall.n > 0 || split.negative.is_none() {
            table_rows(out, label, split);
        }
    };
    rows(&mut out, "tuning", &summary.tuning);
    if let Some(holdout) = &summary.holdout {
        rows(&mut out, "held-out", holdout);
    }
    if let Some(negative) = &summary.tuning.negative {
        negative_line(&mut out, "tuning", negative);
    }
    if let Some(negative) = summary.holdout.as_ref().and_then(|s| s.negative.as_ref()) {
        negative_line(&mut out, "held-out", negative);
    }
    out
}

fn delta_line(out: &mut String, label: &str, (before, after): (Metrics, Metrics)) {
    let cell = |name: &str, b: f64, a: f64| format!("{name} {b:.3} → {a:.3} ({:+.3})", a - b);
    let _ = writeln!(
        out,
        "{label}: {}; {}; {}; {}; {}; n {} → {}",
        cell("recall@5", before.recall5, after.recall5),
        cell("recall@10", before.recall10, after.recall10),
        cell("MRR", before.mrr, after.mrr),
        cell("nDCG@5", before.ndcg5, after.ndcg5),
        cell("nDCG@10", before.ndcg10, after.ndcg10),
        before.n,
        after.n
    );
}

/// The human-readable delta `eval --with/--without` prints on stderr.
pub fn render_delta(delta: &Delta) -> String {
    let mut out = String::new();
    delta_line(&mut out, "tuning", delta.tuning);
    if let Some(holdout) = delta.holdout {
        delta_line(&mut out, "held-out", holdout);
    }
    if delta.changed.is_empty() {
        out.push_str("no query changed rank\n");
    } else {
        let _ = writeln!(out, "{} queries changed rank:", delta.changed.len());
        for change in &delta.changed {
            let fmt = |rr: Option<f64>| rr.map_or_else(|| "–".to_string(), |v| format!("{v:.3}"));
            let _ = writeln!(
                out,
                "  {}: rr {} → {}",
                change.id,
                fmt(change.before),
                fmt(change.after)
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{SourceSpec, write_artifact};
    use pinakes::index::Priorities;

    #[test]
    fn result_json_round_trips() {
        let text = r#"{"tuning":{"overall":{"recall@5":0.75,"recall@10":0.875,"mrr":0.6,"n":8},
            "per_kind":{"howto":{"recall@5":1.0,"recall@10":1.0,"mrr":0.9,"n":3}}},
            "queries":[{"id":"q1","kind":"howto","holdout":false,"hit5":true,"hit10":true,"rr":1.0,"top":["a::b.md"]}]}"#;
        let summary: EvalSummary = serde_json::from_str(text).unwrap();
        assert!((summary.tuning.overall.recall5 - 0.75).abs() < 1e-9);
        assert_eq!(summary.tuning.per_kind["howto"].n, 3);
        assert!(summary.holdout.is_none());
        assert_eq!(summary.queries[0].top, ["a::b.md"]);
        let again: EvalSummary =
            serde_json::from_str(&serde_json::to_string(&summary).unwrap()).unwrap();
        assert_eq!(again, summary);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("eval.json");
        std::fs::write(&path, text).unwrap();
        assert_eq!(EvalSummary::load(&path).unwrap(), summary);
        assert!(matches!(
            EvalSummary::load(&dir.path().join("nope.json")).unwrap_err(),
            EvalError::Io { .. }
        ));
        let saved = dir.path().join("saved.json");
        summary.save(&saved).unwrap();
        assert!(std::fs::read_to_string(&saved).unwrap().ends_with("}\n"));
        assert_eq!(EvalSummary::load(&saved).unwrap(), summary);
    }

    #[test]
    fn expected_matches_both_id_forms() {
        let page = "handbook::docs/user/tutorials/01-40-caching.md";
        assert!(matches(
            page,
            "handbook::docs/user/tutorials/01-40-caching.md"
        ));
        assert!(matches(page, "handbook::docs/user/"));
        assert!(matches(page, "handbook::docs/"));
        assert!(
            !matches(page, "handbook::docs/user"),
            "no trailing slash: exact only"
        );
        assert!(!matches(
            page,
            "handbook::docs/user/tutorials/01-40-caching"
        ));
        assert!(matches(
            page,
            "handbook/docs/user/tutorials/01-40-caching.md"
        ));
        assert!(matches(page, "handbook/docs/user"));
        assert!(matches(page, "handbook/docs/user/"));
        assert!(matches(page, "handbook"));
        assert!(!matches(page, "handbook/docs/use"));
        assert!(!matches(
            "handbook::docs/super-user/x.md",
            "handbook/docs/user"
        ));
        assert!(!matches(page, "guides/docs/user"));
        assert!(!matches(page, ""));
        assert!(!matches(page, "::"));
        assert!(
            matches("handbook/docs/user/a.md", "handbook/docs/user"),
            "legacy ids too"
        );
    }

    #[test]
    fn queries_load_with_defaults_and_report_bad_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("queries.jsonl");
        std::fs::write(
            &path,
            "{\"id\": \"q1\", \"query\": \"x\", \"expected\": [\"a/b\"]}\n\n\
             {\"id\": \"q2\", \"kind\": \"howto\", \"query\": \"y\", \"expected\": [], \"holdout\": true}\n",
        )
        .unwrap();
        let queries = load_queries(&path).unwrap();
        assert_eq!(queries.len(), 2);
        assert_eq!(queries[0].kind, "");
        assert!(!queries[0].holdout);
        assert!(queries[1].holdout);
        std::fs::write(&path, "{\"id\": \"q1\"}\nnot json\n").unwrap();
        let err = load_queries(&path).unwrap_err();
        assert!(matches!(err, EvalError::Query { line: 1, .. }), "{err}");
        assert!(matches!(
            load_queries(&dir.path().join("missing.jsonl")).unwrap_err(),
            EvalError::Io { .. }
        ));
    }

    #[test]
    fn scoring_uses_rank_and_k() {
        let query = Query {
            id: "q".into(),
            query: String::new(),
            expected: vec!["s::docs/".into()],
            graded: BTreeMap::new(),
            kind: "howto".into(),
            holdout: true,
            origin: Some("suggested".into()),
        };
        let top: Vec<String> = (0..12).map(|i| format!("s::other/{i}.md")).collect();
        let miss = QueryResult::score(&query, top.clone(), Some(1.5), 12);
        assert!(!miss.hit5 && !miss.hit10 && miss.rr == 0.0);
        assert!(miss.ndcg5 == 0.0 && miss.ndcg10 == 0.0);
        assert_eq!(miss.rels, [0; 12]);
        assert_eq!(miss.top_score, Some(1.5));
        let mut hit = top.clone();
        hit[7] = "s::docs/x.md".into();
        let row = QueryResult::score(&query, hit.clone(), None, 12);
        assert!(!row.hit5 && row.hit10 && (row.rr - 0.125).abs() < 1e-12);
        assert!(row.holdout && row.kind == "howto" && row.top.len() == 12);
        assert_eq!(row.origin.as_deref(), Some("suggested"));
        assert_eq!(row.rels.iter().position(|r| *r == 1), Some(7));
        assert!(row.ndcg5 == 0.0, "{}", row.ndcg5);
        assert!(
            (row.ndcg10 - 1.0 / 9f64.log2()).abs() < 1e-12,
            "{}",
            row.ndcg10
        );
        let mut late = top;
        late[10] = "s::docs/x.md".into();
        let row = QueryResult::score(&query, late, None, 12);
        assert!(!row.hit10 && (row.rr - 1.0 / 11.0).abs() < 1e-12);
        assert!(row.ndcg10 == 0.0, "rank 11 is past the cut-off");
        // With k below 10, recall@10 is recall@k.
        let mut short: Vec<String> = (0..3).map(|i| format!("s::other/{i}.md")).collect();
        short[2] = "s::docs/x.md".into();
        let row = QueryResult::score(&query, short, None, 3);
        assert!(row.hit10);
        assert!((row.ndcg10 - 0.5).abs() < 1e-12, "{}", row.ndcg10);

        let hits = vec![
            Hit {
                page_id: "s::docs/x.md".into(),
                score: 4.25,
                heading: String::new(),
            },
            Hit {
                page_id: "s::other/1.md".into(),
                score: 1.0,
                heading: String::new(),
            },
        ];
        let row = QueryResult::of_hits(&query, &hits, 10);
        assert_eq!(row.top, ["s::docs/x.md", "s::other/1.md"]);
        assert_eq!(row.top_score, Some(4.25));
        assert!(row.hit5 && (row.ndcg5 - 1.0).abs() < 1e-12);
        let none = QueryResult::of_hits(&query, &[], 10);
        assert!(none.top.is_empty() && none.top_score.is_none() && none.rels.is_empty());
    }

    fn graded_query(expected: &[&str], graded: &[(&str, u8)]) -> Query {
        Query {
            id: "q".into(),
            query: String::new(),
            expected: expected.iter().map(|s| (*s).to_string()).collect(),
            graded: graded
                .iter()
                .map(|(key, grade)| ((*key).to_string(), *grade))
                .collect(),
            kind: String::new(),
            holdout: false,
            origin: None,
        }
    }

    fn ids(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn ndcg_by_hand_on_a_graded_row() {
        // Grades 3, 2, 1 and a 0 that carries no gain; ideal = [3, 2, 1, 0].
        let query = graded_query(
            &["s::a.md"],
            &[
                ("s::a.md", 3),
                ("s::b.md", 2),
                ("s::c.md", 1),
                ("s::d.md", 0),
            ],
        );
        let log2 = |n: f64| n.log2();
        let idcg = 7.0 + 3.0 / log2(3.0) + 1.0 / log2(4.0);

        // Perfect order.
        let row = QueryResult::score(&query, ids(&["s::a.md", "s::b.md", "s::c.md"]), None, 10);
        assert_eq!(row.rels, [3, 2, 1]);
        assert!((row.ndcg5 - 1.0).abs() < 1e-12 && (row.ndcg10 - 1.0).abs() < 1e-12);

        // b, then an unrelated page, then a: DCG = 3 + 7/log2(4).
        let row = QueryResult::score(
            &query,
            ids(&["s::b.md", "s::x.md", "s::a.md", "s::d.md"]),
            None,
            10,
        );
        assert_eq!(row.rels, [2, 0, 3, 0]);
        let dcg = 3.0 + 7.0 / log2(4.0);
        assert!((row.ndcg5 - dcg / idcg).abs() < 1e-12, "{}", row.ndcg5);
        assert!((row.ndcg10 - dcg / idcg).abs() < 1e-12);
        // recall and MRR still read `expected` alone: b is not a hit, a is at rank 3.
        assert!(row.hit5 && (row.rr - 1.0 / 3.0).abs() < 1e-12);

        // The cut-off applies to the ideal as well: at k = 2 the ideal is [3, 2].
        let row = QueryResult::score(&query, ids(&["s::c.md", "s::a.md"]), None, 2);
        let expected = (1.0 + 7.0 / log2(3.0)) / (7.0 + 3.0 / log2(3.0));
        assert!((row.ndcg10 - expected).abs() < 1e-12, "{}", row.ndcg10);

        // A directory key counts once, at the first page under it, and a page that matches
        // both a directory key and its own key takes the higher grade.
        let query = graded_query(&["s::docs/"], &[("s::docs/", 1), ("s::docs/x.md", 3)]);
        let row = QueryResult::score(
            &query,
            ids(&["s::docs/y.md", "s::docs/x.md", "s::docs/z.md"]),
            None,
            10,
        );
        assert_eq!(row.rels, [1, 3, 0]);
        let idcg = 7.0 + 1.0 / log2(3.0);
        assert!((row.ndcg5 - (1.0 + 7.0 / log2(3.0)) / idcg).abs() < 1e-12);
        let row = QueryResult::score(&query, ids(&["s::docs/x.md", "s::docs/y.md"]), None, 10);
        assert_eq!(row.rels, [3, 1]);
        assert!(
            (row.ndcg5 - 1.0).abs() < 1e-12,
            "never above 1: {}",
            row.ndcg5
        );

        // Two keys at the same grade: the page takes its own id, the next page the directory.
        let query = graded_query(&["s::docs/"], &[("s::docs/", 2), ("s::docs/x.md", 2)]);
        let row = QueryResult::score(&query, ids(&["s::docs/x.md", "s::docs/y.md"]), None, 10);
        assert_eq!(row.rels, [2, 2]);
        assert!((row.ndcg5 - 1.0).abs() < 1e-12, "{}", row.ndcg5);

        // An explicit 0 under a graded directory holds, for a page or a subdirectory, and is
        // never used up; the directory's gain goes to the first page not graded 0.
        let query = graded_query(&["s::docs/"], &[("s::docs/", 2), ("s::docs/bad.md", 0)]);
        let row = QueryResult::score(&query, ids(&["s::docs/bad.md", "s::docs/y.md"]), None, 10);
        assert_eq!(row.rels, [0, 2]);
        assert!((row.ndcg5 - 1.0 / log2(3.0)).abs() < 1e-12, "{}", row.ndcg5);
        let query = graded_query(&["s::docs/"], &[("s::docs/", 2), ("s::docs/old/", 0)]);
        let row = QueryResult::score(
            &query,
            ids(&["s::docs/old/a.md", "s::docs/old/b.md", "s::docs/y.md"]),
            None,
            10,
        );
        assert_eq!(row.rels, [0, 0, 2]);

        // A taken specific prefix falls back to the wider one, which then counts once.
        let query = graded_query(&["s::docs/"], &[("s::docs/", 2), ("s::docs/sub/", 3)]);
        let row = QueryResult::score(
            &query,
            ids(&["s::docs/sub/a.md", "s::docs/sub/b.md", "s::docs/c.md"]),
            None,
            10,
        );
        assert_eq!(row.rels, [3, 2, 0]);

        // With grades, an `expected` entry that has no grade carries no gain.
        let query = graded_query(&["s::a.md", "s::b.md"], &[("s::a.md", 2)]);
        let row = QueryResult::score(&query, ids(&["s::b.md", "s::a.md"]), None, 10);
        assert_eq!(row.rels, [0, 2]);
        assert!((row.ndcg5 - 1.0 / log2(3.0)).abs() < 1e-12);
        assert!(
            row.hit5 && (row.rr - 1.0).abs() < 1e-12,
            "recall still counts b"
        );
    }

    #[test]
    fn binary_fallback_equals_relevance_one_per_expected_entry() {
        let binary = graded_query(&["s::a.md", "s::b.md"], &[]);
        let as_ones = graded_query(&["s::a.md", "s::b.md"], &[("s::a.md", 1), ("s::b.md", 1)]);
        for top in [
            ids(&["s::a.md", "s::b.md"]),
            ids(&["s::b.md", "s::x.md", "s::a.md"]),
            ids(&["s::x.md", "s::y.md"]),
            ids(&["s::a.md"]),
        ] {
            let b = QueryResult::score(&binary, top.clone(), None, 10);
            let g = QueryResult::score(&as_ones, top, None, 10);
            assert_eq!(b.rels, g.rels);
            assert!((b.ndcg5 - g.ndcg5).abs() < 1e-12 && (b.ndcg10 - g.ndcg10).abs() < 1e-12);
        }
        // Only one of two expected pages found, at rank 1: IDCG has two ones.
        let row = QueryResult::score(&binary, ids(&["s::a.md"]), None, 10);
        assert!((row.ndcg5 - 1.0 / (1.0 + 1.0 / 3f64.log2())).abs() < 1e-12);
        // A directory entry counts once even when several pages fall under it.
        let dir = graded_query(&["s::docs/"], &[]);
        let row = QueryResult::score(&dir, ids(&["s::docs/a.md", "s::docs/b.md"]), None, 10);
        assert_eq!(row.rels, [1, 0]);
        assert!((row.ndcg5 - 1.0).abs() < 1e-12);
        // Nothing expected at all: nDCG is 0, not NaN.
        let none = graded_query(&[], &[]);
        let row = QueryResult::score(&none, ids(&["s::a.md"]), None, 10);
        assert!(row.ndcg5 == 0.0 && row.ndcg10 == 0.0 && row.rels == [0]);
    }

    fn negative_row(id: &str, top: &[&str], top_score: Option<f64>, holdout: bool) -> QueryResult {
        let query = Query {
            id: id.into(),
            query: String::new(),
            expected: vec![],
            graded: BTreeMap::new(),
            kind: NEGATIVE_KIND.into(),
            holdout,
            origin: None,
        };
        QueryResult::score(&query, ids(top), top_score, 10)
    }

    #[test]
    fn negative_rows_have_their_own_block_and_leave_the_metrics_alone() {
        let positive = graded_query(&["s::a.md"], &[]);
        let rows = vec![
            QueryResult::score(&positive, ids(&["s::a.md"]), Some(5.0), 10),
            negative_row("n1", &[], None, false),
            negative_row("n2", &["s::x.md"], Some(0.4), false),
            negative_row("n3", &["s::y.md"], Some(2.0), false),
            negative_row("n4", &["s::y.md"], Some(0.1), true),
        ];
        // Without a threshold only the empty result list rejects.
        let summary = summarise(rows.clone(), None);
        assert_eq!(summary.tuning.overall.n, 1, "negative rows are not scored");
        assert!((summary.tuning.overall.recall5 - 1.0).abs() < 1e-12);
        assert!(!summary.tuning.per_kind.contains_key(NEGATIVE_KIND));
        let negative = summary.tuning.negative.unwrap();
        assert_eq!((negative.n, negative.rejected), (3, 1));
        assert!((negative.share - 1.0 / 3.0).abs() < 1e-12);
        let holdout = summary.holdout.as_ref().unwrap();
        assert_eq!(holdout.overall.n, 0);
        assert_eq!(holdout.negative.map(|n| n.rejected), Some(0));
        let table = render_table(&summary);
        assert!(
            table.contains("tuning: 3 negative queries, 1 rejected (0.333)\n"),
            "{table}"
        );
        assert!(
            table.contains("held-out: 1 negative queries, 0 rejected (0.000)\n"),
            "{table}"
        );
        assert!(!table.contains("| tuning | negative |"), "{table}");
        assert!(
            !table.contains("| held-out |"),
            "a split of negative rows alone shows no zero metric rows: {table}"
        );
        assert!(table.contains("| tuning | overall | 1 |"), "{table}");

        // With a threshold, a top score under it rejects too.
        let summary = summarise(rows, Some(0.5));
        let negative = summary.tuning.negative.unwrap();
        assert_eq!((negative.n, negative.rejected), (3, 2));
        assert_eq!(
            summary.holdout.unwrap().negative.map(|n| n.rejected),
            Some(1)
        );
        // No negative rows: no block, and nothing printed.
        let summary = summarise(
            vec![QueryResult::score(&positive, ids(&["s::a.md"]), None, 10)],
            Some(0.5),
        );
        assert!(summary.tuning.negative.is_none());
        assert!(!render_table(&summary).contains("negative"));
        assert!(!summary.to_json().unwrap().contains("negative"));
    }

    fn artifact() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        write_artifact(
            dir.path(),
            &[
                SourceSpec {
                    name: "handbook",
                    repo: "example-org/handbook",
                    pages: &[
                        (
                            "docs/user/README.md",
                            "Storage Module",
                            "# Storage\n\nThe storage module keeps uploaded files.\n\n## Upload caching\n\nEnable upload caching with a bucket label.\n",
                        ),
                        (
                            "docs/user/quotas.md",
                            "",
                            "# Configure Quotas\n\nRate limits in strict mode.\n",
                        ),
                    ],
                    residue: &[(
                        "docs/user/billing-note.md",
                        "# Billing invoices scale\n\nScale billing invoices.\n",
                    )],
                },
                SourceSpec {
                    name: "billing",
                    repo: "example-org/billing",
                    pages: &[(
                        "docs/user/README.md",
                        "Billing",
                        "# Billing\n\nInvoices scale to zero.\n",
                    )],
                    residue: &[],
                },
            ],
        );
        let queries = dir.path().join("queries.jsonl");
        std::fs::write(
            &queries,
            concat!(
                "{\"id\": \"caching\", \"kind\": \"howto\", \"query\": \"enable upload caching\", \"expected\": [\"handbook/docs/user\"]}\n",
                "{\"id\": \"quotas\", \"kind\": \"howto\", \"query\": \"quotas rate limits\", \"expected\": [\"handbook::docs/user/quotas.md\"]}\n",
                "{\"id\": \"invoices\", \"kind\": \"concept\", \"query\": \"billing invoices scale\", \"expected\": [\"billing::docs/user/\"]}\n",
                "{\"id\": \"held\", \"kind\": \"concept\", \"query\": \"uploaded files\", \"expected\": [\"billing/docs/user\"], \"holdout\": true}\n",
            ),
        )
        .unwrap();
        (dir, queries)
    }

    #[test]
    fn metrics_on_a_synthetic_artifact_separate_holdout_rows() {
        let (dir, queries) = artifact();
        let index = Index::build(dir.path(), &Priorities::default()).unwrap();
        let queries = load_queries(&queries).unwrap();
        let summary = evaluate(&index, &queries, 10, None).unwrap();
        let tuning = summary.tuning.overall;
        assert_eq!(tuning.n, 3);
        assert!((tuning.recall5 - 1.0).abs() < 1e-12 && (tuning.mrr - 1.0).abs() < 1e-12);
        assert!((tuning.ndcg5 - 1.0).abs() < 1e-12 && (tuning.ndcg10 - 1.0).abs() < 1e-12);
        assert!(summary.query("caching").unwrap().top_score.unwrap() > 0.0);
        assert_eq!(summary.tuning.per_kind["howto"].n, 2);
        assert_eq!(summary.tuning.per_kind["concept"].n, 1);
        let holdout = summary.holdout.as_ref().expect("held-out split");
        assert_eq!(holdout.overall.n, 1);
        assert!(holdout.overall.recall5 == 0.0 && holdout.overall.mrr == 0.0);
        assert_eq!(summary.queries.len(), 4);
        assert_eq!(
            summary.query("caching").unwrap().top[0],
            "handbook::docs/user/README.md"
        );
        let table = render_table(&summary);
        assert!(
            table.starts_with(
                "| split | kind | n | recall@5 | recall@10 | MRR | nDCG@5 | nDCG@10 |\n"
            )
        );
        assert!(table.contains("| tuning | overall | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |"));
        assert!(
            table.contains("| held-out | overall | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |")
        );
        assert!(table.contains("| tuning | concept | 1 |"));

        let none = summarise(Vec::new(), None);
        assert_eq!(none.tuning.overall.n, 0);
        assert!(none.holdout.is_none());
        assert!(!render_table(&none).contains("held-out"));
    }

    #[test]
    fn with_and_without_change_the_delta() {
        let (dir, queries) = artifact();
        let queries = load_queries(&queries).unwrap();
        let priorities = Priorities::default();
        let before = evaluate(
            &Index::build(dir.path(), &priorities).unwrap(),
            &queries,
            10,
            None,
        )
        .unwrap();

        // --with: the residue page outscores the billing page for the "invoices" query.
        let mut pages = pinakes::index::load_pages(dir.path(), &priorities).unwrap();
        pages.push(
            pinakes::index::load_residue_page(
                dir.path(),
                "handbook::docs/user/billing-note.md",
                &priorities,
            )
            .unwrap(),
        );
        let after = evaluate(&Index::from_pages(pages).unwrap(), &queries, 10, None).unwrap();
        let d = delta(&before, &after);
        assert!(d.tuning.1.mrr < d.tuning.0.mrr, "{d:?}");
        assert!(d.tuning.1.ndcg5 < d.tuning.0.ndcg5, "{d:?}");
        assert_eq!(d.changed.len(), 1);
        assert_eq!(d.changed[0].id, "invoices");
        assert_eq!(d.changed[0].before, Some(1.0));
        assert_eq!(d.changed[0].after, Some(0.5));
        let text = render_delta(&d);
        assert!(
            text.contains("tuning: recall@5 1.000 → 1.000 (+0.000)"),
            "{text}"
        );
        assert!(text.contains("MRR 1.000 → 0.833 (-0.167)"), "{text}");
        // invoices drops to rank 2: nDCG 1/log2(3) = 0.631, mean over three queries 0.877.
        assert!(text.contains("nDCG@5 1.000 → 0.877 (-0.123)"), "{text}");
        assert!(text.contains("nDCG@10 1.000 → 0.877 (-0.123)"), "{text}");
        assert!(text.contains("  invoices: rr 1.000 → 0.500"), "{text}");
        assert!(text.contains("held-out: "), "{text}");

        // --without: dropping the quotas page loses that query entirely.
        let mut pages = pinakes::index::load_pages(dir.path(), &priorities).unwrap();
        pages.retain(|p| p.id != "handbook::docs/user/quotas.md");
        let after = evaluate(&Index::from_pages(pages).unwrap(), &queries, 10, None).unwrap();
        let d = delta(&before, &after);
        assert!((d.tuning.1.recall5 - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(d.changed.len(), 1);
        assert_eq!(d.changed[0].id, "quotas");
        assert_eq!(d.changed[0].after, Some(0.0));

        assert_eq!(delta(&before, &before).changed, []);
        assert!(render_delta(&delta(&before, &before)).contains("no query changed rank"));
        // A query missing on one side is reported too.
        let mut fewer = before.clone();
        fewer.queries.retain(|q| q.id != "quotas");
        assert_eq!(delta(&before, &fewer).changed[0].after, None);
        assert_eq!(delta(&fewer, &before).changed[0].before, None);
    }

    #[test]
    fn gate_compares_the_named_tuning_metric() {
        let (dir, queries) = artifact();
        let queries = load_queries(&queries).unwrap();
        let index = Index::build(dir.path(), &Priorities::default()).unwrap();
        let summary = evaluate(&index, &queries, 10, None).unwrap();
        let mut baseline = summary.clone();
        let g = gate(&summary, &baseline, 0.0, GateMetric::Recall5);
        assert!(g.passed());
        assert_eq!(g.metric, GateMetric::Recall5);
        baseline.tuning.overall.recall5 = 1.0;
        let mut current = summary;
        current.tuning.overall.recall5 = 0.9;
        let g = gate(&current, &baseline, 0.05, GateMetric::Recall5);
        assert!(!g.passed());
        assert!((g.drop() - 0.1).abs() < 1e-12);
        assert!(gate(&current, &baseline, 0.1, GateMetric::Recall5).passed());
        // Held-out rows never gate.
        current.holdout = Some(Split::of(&[], None));
        current.tuning.overall.recall5 = 1.0;
        assert!(gate(&current, &baseline, 0.0, GateMetric::Recall5).passed());

        // A baseline written before nDCG existed reads the metric as 0 and is flagged; a
        // baseline with no queries at all is not, nor is a metric the baseline carries.
        assert!(!gate(&current, &baseline, 0.0, GateMetric::Ndcg5).baseline_unset);
        let legacy: EvalSummary = serde_json::from_str(
            r#"{"tuning":{"overall":{"recall@5":1.0,"recall@10":1.0,"mrr":1.0,"n":3}}}"#,
        )
        .unwrap();
        let g = gate(&current, &legacy, 0.0, GateMetric::Ndcg10);
        assert!(g.baseline_unset && g.passed(), "{g:?}");
        assert!(!gate(&current, &legacy, 0.0, GateMetric::Recall5).baseline_unset);
        assert!(!gate(&current, &summarise(vec![], None), 0.0, GateMetric::Ndcg5).baseline_unset);

        // Another metric reads its own column and leaves recall alone.
        current.tuning.overall.ndcg5 = 0.7;
        current.tuning.overall.mrr = 0.95;
        current.tuning.overall.recall10 = 0.8;
        current.tuning.overall.ndcg10 = 0.75;
        for (metric, expected) in [
            (GateMetric::Recall5, 1.0),
            (GateMetric::Recall10, 0.8),
            (GateMetric::Mrr, 0.95),
            (GateMetric::Ndcg5, 0.7),
            (GateMetric::Ndcg10, 0.75),
        ] {
            let g = gate(&current, &baseline, 0.0, metric);
            assert_eq!(g.metric, metric);
            assert!((g.current - expected).abs() < 1e-12, "{metric}: {g:?}");
            assert!((g.baseline - 1.0).abs() < 1e-12, "{metric}: {g:?}");
        }
        assert!(!gate(&current, &baseline, 0.25, GateMetric::Ndcg5).passed());
        assert!(gate(&current, &baseline, 0.31, GateMetric::Ndcg5).passed());
        // Negative rows never move the gated metric.
        let plain = summarise(current.queries.clone(), None);
        current
            .queries
            .push(negative_row("n", &["s::x.md"], Some(9.0), false));
        let with_negative = summarise(current.queries.clone(), None);
        assert_eq!(with_negative.tuning.overall, plain.tuning.overall);
        assert_eq!(
            gate(&with_negative, &plain, 0.0, GateMetric::Ndcg10),
            gate(&plain, &plain, 0.0, GateMetric::Ndcg10)
        );
    }
}
