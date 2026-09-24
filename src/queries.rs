//! `kanon queries add`, `queries check`, `queries import` and `queries suggest`: growing and
//! validating `queries.jsonl`.
//!
//! `add` appends one row after checking that every `expected` entry names at least one page in
//! the manifest; the legacy `<source>/<path>` form (no `::`) is normalised to the canonical
//! `<source>::<path>` form, or `<source>::<path>/` when it only matches pages as a directory
//! prefix. `add --from suggestions.jsonl --accept ID…` appends chosen rows of a suggestions
//! file through the same check, keeping their `origin`. `check` re-validates the whole file:
//! unknown expected and graded ids, grades above [`MAX_GRADE`], an empty `expected` on any row
//! that is not a negative query, duplicate query ids, and the held-out share against
//! `holdout_min`. `import` turns `kanon grade`'s `graded.jsonl` into query rows, one per
//! distinct query, `expected` being the ids graded at or above a threshold and `graded` every
//! candidate with its grade. [`suggest`] asks a model for questions a sample of pages answers,
//! for a human to accept.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Serialize;
use thiserror::Error;

use crate::eval::{self, MAX_GRADE, Query};
use crate::grade::GradedRow;
use crate::num::float;
use crate::rng::SplitMix64;
use pinakes::jsonl::{self, JsonlError, KeyOrder};
use pinakes::llm::ChatError;
use pinakes::manifest::{self, Manifest};
use pinakes::text::sha256_hex;

pub mod suggest;

pub use crate::config::DEFAULT_HOLDOUT_MIN;

/// Default `--min-grade` for `queries import`.
pub const DEFAULT_MIN_GRADE: u8 = 2;

/// Errors raised while adding to or checking `queries.jsonl`.
#[derive(Debug, Error)]
pub enum QueriesError {
    /// The file could not be read or written.
    #[error("{path}: {source}")]
    Io {
        /// The query file path.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The row could not be serialised.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// An `expected` entry names no page in the manifest.
    #[error("expected {0:?} matches no page in the manifest")]
    UnknownExpected(String),
    /// `--accept` named an id that is not in the suggestions file.
    #[error("--accept {0:?} names no row in the suggestions file")]
    UnknownSuggestion(String),
    /// An accepted suggestion's id is already in the query file.
    #[error("query id {0:?} is already in the query file")]
    DuplicateId(String),
    /// Talking to the model failed (`queries suggest`).
    #[error(transparent)]
    Llm(#[from] ChatError),
}

/// A row to append with `queries add`, before its `expected` ids are normalised.
#[derive(Debug, Clone)]
pub struct NewQuery {
    /// Query id.
    pub id: String,
    /// The query text.
    pub query: String,
    /// Page ids, id prefixes, or the legacy `<source>/<path>` form.
    pub expected: Vec<String>,
    /// Query kind, possibly empty.
    pub kind: String,
    /// Held out from tuning decisions.
    pub holdout: bool,
    /// Provenance kept on the row (`"suggested"` for an accepted suggestion), `None` for a
    /// hand-written one.
    pub origin: Option<String>,
}

/// Normalise one `expected` entry against `manifest`, erroring when it matches no page.
///
/// `<source>::<path>` and `<source>::<dir>/` are validated as given. The legacy
/// `<source>/<path>` form (no `::`) is rewritten to the canonical form: `<source>::<path>` when
/// it names a page exactly, else `<source>::<path>/` when pages exist under it as a directory.
pub fn normalize_expected(expected: &str, manifest: &Manifest) -> Result<String, QueriesError> {
    if expected.contains("::") {
        return if manifest
            .pages()
            .any(|(id, ..)| eval::matches(&id, expected))
        {
            Ok(expected.to_string())
        } else {
            Err(QueriesError::UnknownExpected(expected.to_string()))
        };
    }
    let Some((source, rest)) = expected.split_once('/') else {
        return Err(QueriesError::UnknownExpected(expected.to_string()));
    };
    let exact = manifest::page_id(source, rest.trim_end_matches('/'));
    if manifest.page(&exact).is_some() {
        return Ok(exact);
    }
    let prefix = format!("{exact}/");
    if manifest.pages().any(|(id, ..)| eval::matches(&id, &prefix)) {
        Ok(prefix)
    } else {
        Err(QueriesError::UnknownExpected(expected.to_string()))
    }
}

/// Append one row to `path`, normalising and validating `expected` against `manifest` first.
pub fn add(path: &Path, manifest: &Manifest, new: &NewQuery) -> Result<Query, QueriesError> {
    let mut rows = add_all(path, manifest, std::slice::from_ref(new))?;
    Ok(rows.remove(0))
}

/// Append every row to `path`, normalising and validating each one's `expected` against
/// `manifest` first: nothing is written when any row fails.
pub fn add_all(
    path: &Path,
    manifest: &Manifest,
    new: &[NewQuery],
) -> Result<Vec<Query>, QueriesError> {
    let rows = new
        .iter()
        .map(|new| {
            let expected = new
                .expected
                .iter()
                .map(|entry| normalize_expected(entry, manifest))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Query {
                id: new.id.clone(),
                query: new.query.clone(),
                expected,
                graded: BTreeMap::new(),
                kind: new.kind.clone(),
                holdout: new.holdout,
                origin: new.origin.clone(),
            })
        })
        .collect::<Result<Vec<_>, QueriesError>>()?;
    jsonl::append(path, &rows, KeyOrder::Declared).map_err(jsonl_error)?;
    Ok(rows)
}

fn jsonl_error(err: JsonlError) -> QueriesError {
    match err {
        JsonlError::Io { path, source } => QueriesError::Io { path, source },
        JsonlError::Json { source, .. } => QueriesError::Json(source),
    }
}

/// The outcome of `queries check`; exit 4 when [`CheckReport::ok`] is false.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct CheckReport {
    /// `(query id, expected entry)` pairs that match no page in the manifest.
    pub unknown: Vec<(String, String)>,
    /// `(query id, graded entry)` pairs that match no page in the manifest.
    pub unknown_graded: Vec<(String, String)>,
    /// `(query id, graded entry, grade)` triples whose grade is above [`MAX_GRADE`].
    pub bad_grade: Vec<(String, String, u8)>,
    /// Ids of rows with an empty `expected` list that are not negative queries.
    pub empty_expected: Vec<String>,
    /// Ids of negative queries that name expected pages (a negative row must have none).
    pub negative_with_expected: Vec<String>,
    /// Query ids that appear more than once.
    pub duplicate_ids: Vec<String>,
    /// Held-out fraction of all queries (`0.0` when there are none).
    pub holdout_share: f64,
    /// The configured minimum (`holdout_min`).
    pub holdout_min: f64,
}

impl CheckReport {
    /// Whether the query set passes every check.
    pub fn ok(&self) -> bool {
        self.unknown.is_empty()
            && self.unknown_graded.is_empty()
            && self.bad_grade.is_empty()
            && self.empty_expected.is_empty()
            && self.negative_with_expected.is_empty()
            && self.duplicate_ids.is_empty()
            && self.holdout_share >= self.holdout_min
    }
}

fn share(count: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        float(count) / float(total)
    }
}

/// Validate `queries` against `manifest`: unknown expected and graded ids, grades above
/// [`MAX_GRADE`], an empty `expected` on a row that is not a negative query, a negative query
/// that names expected pages, duplicate query ids, and the held-out share against
/// `holdout_min`.
pub fn check(queries: &[Query], manifest: &Manifest, holdout_min: f64) -> CheckReport {
    let mut unknown = Vec::new();
    let mut unknown_graded = Vec::new();
    let mut bad_grade = Vec::new();
    let mut empty_expected = Vec::new();
    let mut negative_with_expected = Vec::new();
    for query in queries {
        let known = |entry: &str| manifest.pages().any(|(id, ..)| eval::matches(&id, entry));
        for expected in &query.expected {
            if !known(expected) {
                unknown.push((query.id.clone(), expected.clone()));
            }
        }
        for (entry, grade) in &query.graded {
            if !known(entry) {
                unknown_graded.push((query.id.clone(), entry.clone()));
            }
            if *grade > MAX_GRADE {
                bad_grade.push((query.id.clone(), entry.clone(), *grade));
            }
        }
        match (query.is_negative(), query.expected.is_empty()) {
            (false, true) => empty_expected.push(query.id.clone()),
            (true, false) => negative_with_expected.push(query.id.clone()),
            _ => {}
        }
    }
    let mut counts = std::collections::BTreeMap::new();
    for query in queries {
        *counts.entry(query.id.clone()).or_insert(0usize) += 1;
    }
    let duplicate_ids = counts
        .into_iter()
        .filter(|(_, n)| *n > 1)
        .map(|(id, _)| id)
        .collect();
    let holdout_share = share(queries.iter().filter(|q| q.holdout).count(), queries.len());
    CheckReport {
        unknown,
        unknown_graded,
        bad_grade,
        empty_expected,
        negative_with_expected,
        duplicate_ids,
        holdout_share,
        holdout_min,
    }
}

/// A `queries.jsonl` row produced by `queries import`: the same shape as
/// [`Query`] plus a provenance `by`. Every reader of the file (`Query`'s `Deserialize` has no
/// `deny_unknown_fields`) simply ignores the extra key, so the file's contract stays backwards
/// compatible.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GradedQuery {
    /// Query id, derived from the query text so re-importing the same query is stable.
    pub id: String,
    /// The query text.
    pub query: String,
    /// Ids graded at or above `--min-grade` for this query.
    pub expected: Vec<String>,
    /// Every candidate with its grade, 0 and 1 included, for nDCG.
    pub graded: BTreeMap<String, u8>,
    /// Always empty: grading carries no `kind`.
    pub kind: String,
    /// Assigned at random (seeded by `--seed`) to approximate `--holdout-share`.
    pub holdout: bool,
    /// `"grader:<model>"`.
    pub by: String,
}

/// A short, stable, human-scannable id for a query: a slug of its text plus a hash suffix so
/// two different queries that slugify the same never collide, and so importing or suggesting
/// the same query text always produces the same id.
pub(crate) fn query_id(query: &str) -> String {
    let mut slug = String::new();
    let mut last_dash = false;
    for ch in query.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch);
            last_dash = false;
        } else if !last_dash && !slug.is_empty() {
            slug.push('-');
            last_dash = true;
        }
    }
    while slug.ends_with('-') {
        slug.pop();
    }
    let hash = sha256_hex(query.as_bytes());
    if slug.is_empty() {
        format!("query-{}", &hash[..8])
    } else {
        format!("{slug}-{}", &hash[..8])
    }
}

/// Turn graded rows into query rows: one per distinct query, `expected` the ids
/// graded at or above `min_grade`, `graded` every candidate with its grade (the highest when
/// an id was graded more than once), `holdout` drawn from a `seed`-ed PRNG to approximate
/// `holdout_share`. Returns the rows to append and the query texts with no passing candidate
/// (skipped, for the caller to report).
pub fn import_graded(
    graded: &[GradedRow],
    min_grade: u8,
    holdout_share: f64,
    seed: u64,
) -> (Vec<GradedQuery>, Vec<String>) {
    let mut by_query: BTreeMap<&str, Vec<&GradedRow>> = BTreeMap::new();
    for row in graded {
        by_query.entry(row.query.as_str()).or_default().push(row);
    }
    let mut rng = SplitMix64::new(seed);
    let mut rows = Vec::new();
    let mut skipped = Vec::new();
    for (query, group) in &by_query {
        let mut expected: Vec<String> = group
            .iter()
            .filter(|row| row.grade >= min_grade)
            .map(|row| row.id.clone())
            .collect();
        expected.sort();
        expected.dedup();
        if expected.is_empty() {
            skipped.push((*query).to_string());
            continue;
        }
        let mut graded: BTreeMap<String, u8> = BTreeMap::new();
        for row in group {
            let grade = graded.entry(row.id.clone()).or_default();
            *grade = (*grade).max(row.grade);
        }
        let model = group.first().map_or("unknown", |row| row.model.as_str());
        rows.push(GradedQuery {
            id: query_id(query),
            query: (*query).to_string(),
            expected,
            graded,
            kind: String::new(),
            holdout: rng.next_f64() < holdout_share,
            by: format!("grader:{model}"),
        });
    }
    (rows, skipped)
}

/// Append every row to `path`, creating the file when needed.
pub fn append_graded(path: &Path, rows: &[GradedQuery]) -> Result<(), QueriesError> {
    jsonl::append(path, rows, KeyOrder::Declared).map_err(jsonl_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pinakes::manifest::{PageEntry, SelectedBy};
    use std::collections::BTreeMap;

    fn manifest() -> Manifest {
        let mut m = Manifest::new("2026-09-16T12:00:00Z".to_string());
        let mut pages = BTreeMap::new();
        pages.insert(
            "docs/user/caching.md".to_string(),
            PageEntry {
                sha256: "aa".repeat(32),
                title: "Caching".to_string(),
                doc_type: "howto".to_string(),
                section: String::new(),
                selected_by: SelectedBy::Include,
                rendered_from: None,
            },
        );
        pages.insert(
            "docs/user/quotas.md".to_string(),
            PageEntry {
                sha256: "bb".repeat(32),
                title: "Quotas".to_string(),
                doc_type: "howto".to_string(),
                section: String::new(),
                selected_by: SelectedBy::Include,
                rendered_from: None,
            },
        );
        m.sources.insert(
            "handbook".to_string(),
            pinakes::manifest::ManifestSource {
                repo: "example-org/handbook".to_string(),
                repo_url: "https://github.com/example-org/handbook.git".to_string(),
                git_ref: "main".to_string(),
                commit: "a".repeat(40),
                archived: Some(false),
                resolver: "glob".to_string(),
                unrendered: Vec::new(),
                pages,
                residue: vec![],
                unresolved: vec![],
                render: None,
            },
        );
        m
    }

    #[test]
    fn normalises_the_legacy_form_to_an_exact_or_prefix_id() {
        let m = manifest();
        assert_eq!(
            normalize_expected("handbook/docs/user/caching.md", &m).unwrap(),
            "handbook::docs/user/caching.md"
        );
        assert_eq!(
            normalize_expected("handbook/docs/user", &m).unwrap(),
            "handbook::docs/user/"
        );
        assert_eq!(
            normalize_expected("handbook::docs/user/", &m).unwrap(),
            "handbook::docs/user/"
        );
        assert_eq!(
            normalize_expected("handbook::docs/user/caching.md", &m).unwrap(),
            "handbook::docs/user/caching.md"
        );
        assert!(matches!(
            normalize_expected("handbook/nope", &m).unwrap_err(),
            QueriesError::UnknownExpected(_)
        ));
        assert!(matches!(
            normalize_expected("handbook::nope.md", &m).unwrap_err(),
            QueriesError::UnknownExpected(_)
        ));
        assert!(matches!(
            normalize_expected("nope-no-slash", &m).unwrap_err(),
            QueriesError::UnknownExpected(_)
        ));
    }

    #[test]
    fn add_appends_a_normalised_row() {
        let m = manifest();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("queries.jsonl");
        let new = NewQuery {
            id: "caching".to_string(),
            query: "how do I enable caching".to_string(),
            expected: vec!["handbook/docs/user/caching.md".to_string()],
            kind: "howto".to_string(),
            holdout: false,
            origin: None,
        };
        let query = add(&path, &m, &new).unwrap();
        assert_eq!(query.expected, ["handbook::docs/user/caching.md"]);
        let loaded = eval::load_queries(&path).unwrap();
        assert_eq!(loaded, [query]);
        // A row without an origin carries no `origin` key at all.
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("origin"), "{text}");

        // A second row appends rather than overwriting; its origin is kept.
        let new2 = NewQuery {
            id: "quotas".to_string(),
            query: "quotas".to_string(),
            expected: vec!["handbook::docs/user/".to_string()],
            kind: String::new(),
            holdout: true,
            origin: Some("suggested".to_string()),
        };
        add(&path, &m, &new2).unwrap();
        let loaded = eval::load_queries(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[1].origin.as_deref(), Some("suggested"));

        // An unknown expected id is rejected before anything is written.
        let bad = NewQuery {
            id: "nope".to_string(),
            query: "x".to_string(),
            expected: vec!["handbook::missing.md".to_string()],
            kind: String::new(),
            holdout: false,
            origin: None,
        };
        assert!(add(&path, &m, &bad).is_err());
        assert_eq!(eval::load_queries(&path).unwrap().len(), 2, "not appended");

        // add_all is all or nothing: one bad row in a batch and none of it is appended.
        let err = add_all(&path, &m, &[new.clone(), bad]).unwrap_err();
        assert!(matches!(err, QueriesError::UnknownExpected(_)));
        assert_eq!(eval::load_queries(&path).unwrap().len(), 2, "not appended");
        assert_eq!(add_all(&path, &m, &[new.clone(), new]).unwrap().len(), 2);
        assert_eq!(eval::load_queries(&path).unwrap().len(), 4);
    }

    fn query(id: &str, expected: &[&str], holdout: bool) -> Query {
        Query {
            id: id.to_string(),
            query: "q".to_string(),
            expected: expected.iter().map(|s| (*s).to_string()).collect(),
            graded: BTreeMap::new(),
            kind: String::new(),
            holdout,
            origin: None,
        }
    }

    #[test]
    fn check_reports_unknown_ids_duplicates_and_the_holdout_share() {
        let m = manifest();
        let queries = vec![
            query("caching", &["handbook::docs/user/caching.md"], false),
            query("quotas", &["handbook::docs/user/quotas.md"], true),
        ];
        let report = check(&queries, &m, DEFAULT_HOLDOUT_MIN);
        assert!(report.ok(), "{report:?}");
        assert!((report.holdout_share - 0.5).abs() < 1e-12);

        // Below the held-out minimum.
        let report = check(&queries, &m, 0.6);
        assert!(!report.ok());
        assert!(report.unknown.is_empty() && report.duplicate_ids.is_empty());

        // Unknown expected id and a duplicate query id.
        let queries = vec![
            query("caching", &["handbook::missing.md"], false),
            query("caching", &["handbook::docs/user/quotas.md"], true),
        ];
        let report = check(&queries, &m, 0.0);
        assert!(!report.ok());
        assert_eq!(
            report.unknown,
            [("caching".to_string(), "handbook::missing.md".to_string())]
        );
        assert_eq!(report.duplicate_ids, ["caching"]);

        // No queries at all: a zero share, never a divide by zero.
        let report = check(&[], &m, 0.0);
        assert!(report.ok());
        assert!(report.holdout_share.abs() < 1e-12);
    }

    #[test]
    fn check_rejects_empty_expected_lists_except_on_negative_rows() {
        let m = manifest();
        let mut negative = query("nothing", &[], false);
        negative.kind = eval::NEGATIVE_KIND.to_string();
        let queries = vec![
            query("caching", &["handbook::docs/user/caching.md"], true),
            negative,
            query("empty", &[], false),
        ];
        let report = check(&queries, &m, 0.0);
        assert!(!report.ok());
        assert_eq!(report.empty_expected, ["empty"]);
        assert!(report.negative_with_expected.is_empty());
        assert!(report.unknown.is_empty() && report.bad_grade.is_empty());
        // The negative row alone passes.
        let report = check(&queries[..2], &m, 0.0);
        assert!(report.ok(), "{report:?}");

        // The mirror rule: a negative row that names expected pages fails too.
        let mut contradiction = query("both", &["handbook::docs/user/quotas.md"], false);
        contradiction.kind = eval::NEGATIVE_KIND.to_string();
        let report = check(&[contradiction], &m, 0.0);
        assert!(!report.ok());
        assert_eq!(report.negative_with_expected, ["both"]);
        assert!(report.empty_expected.is_empty() && report.unknown.is_empty());
    }

    #[test]
    fn check_validates_graded_keys_and_grades() {
        let m = manifest();
        let mut graded = query("caching", &["handbook::docs/user/caching.md"], true);
        graded.graded = BTreeMap::from([
            ("handbook::docs/user/caching.md".to_string(), 3),
            ("handbook::docs/user/".to_string(), 1),
            ("handbook::docs/user/quotas.md".to_string(), 0),
        ]);
        let report = check(std::slice::from_ref(&graded), &m, 0.0);
        assert!(report.ok(), "{report:?}");

        graded.graded.insert("handbook::missing.md".to_string(), 2);
        graded
            .graded
            .insert("handbook::docs/user/quotas.md".to_string(), 4);
        let report = check(std::slice::from_ref(&graded), &m, 0.0);
        assert!(!report.ok());
        assert!(report.unknown.is_empty(), "expected ids are all known");
        assert_eq!(
            report.unknown_graded,
            [("caching".to_string(), "handbook::missing.md".to_string())]
        );
        assert_eq!(
            report.bad_grade,
            [(
                "caching".to_string(),
                "handbook::docs/user/quotas.md".to_string(),
                4
            )]
        );
    }

    fn graded(query: &str, id: &str, grade: u8) -> GradedRow {
        GradedRow {
            query: query.to_string(),
            id: id.to_string(),
            grade,
            model: "grader-model".to_string(),
            at: "2026-09-16T12:00:00Z".to_string(),
        }
    }

    #[test]
    fn import_keeps_ids_at_or_above_the_threshold_per_query() {
        let rows = vec![
            graded("caching", "h::a.md", 3),
            graded("caching", "h::b.md", 1),
            graded("caching", "h::c.md", 2),
            graded("caching", "h::b.md", 0),
            graded("caching", "h::e.md", 0),
            graded("quotas", "h::d.md", 0),
        ];
        let (imported, skipped) = import_graded(&rows, 2, 0.0, 42);
        assert_eq!(skipped, ["quotas"]);
        assert_eq!(imported.len(), 1);
        assert_eq!(imported[0].query, "caching");
        assert_eq!(imported[0].expected, ["h::a.md", "h::c.md"]);
        // Every candidate keeps its grade, 0 and 1 included; a repeat keeps the higher one.
        assert_eq!(
            imported[0].graded,
            BTreeMap::from([
                ("h::a.md".to_string(), 3),
                ("h::b.md".to_string(), 1),
                ("h::c.md".to_string(), 2),
                ("h::e.md".to_string(), 0),
            ])
        );
        assert_eq!(imported[0].by, "grader:grader-model");
        assert_eq!(imported[0].kind, "");
    }

    #[test]
    fn import_id_is_stable_and_url_safe() {
        let rows = vec![graded("How do I enable caching?", "h::a.md", 3)];
        let (imported, _) = import_graded(&rows, 2, 0.0, 1);
        assert!(imported[0].id.starts_with("how-do-i-enable-caching-"));
        // Re-importing the same query text yields the same id.
        let (imported2, _) = import_graded(&rows, 2, 0.0, 99);
        assert_eq!(imported[0].id, imported2[0].id);
    }

    #[test]
    fn import_holdout_share_is_approximated_by_the_seeded_rng() {
        let rows: Vec<GradedRow> = (0..200)
            .map(|i| graded(&format!("query {i}"), "h::a.md", 3))
            .collect();
        let (imported, skipped) = import_graded(&rows, 2, 0.25, 7);
        assert!(skipped.is_empty());
        let holdout = imported.iter().filter(|q| q.holdout).count();
        let share = f64::from(u32::try_from(holdout).unwrap()) / 200.0;
        assert!((share - 0.25).abs() < 0.1, "share was {share}");

        // Deterministic: the same seed reproduces the same assignment.
        let (imported_again, _) = import_graded(&rows, 2, 0.25, 7);
        assert_eq!(imported, imported_again);
    }

    #[test]
    fn import_appends_and_is_readable_as_ordinary_queries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("queries.jsonl");
        let rows = vec![graded("caching", "h::a.md", 3)];
        let (imported, _) = import_graded(&rows, 2, 0.0, 0);
        append_graded(&path, &imported).unwrap();
        let loaded = eval::load_queries(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].query, "caching");
        assert_eq!(loaded[0].expected, ["h::a.md"]);
        assert_eq!(
            loaded[0].graded,
            BTreeMap::from([("h::a.md".to_string(), 3)])
        );
        assert!(!loaded[0].holdout);
    }
}
