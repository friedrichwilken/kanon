//! The three contracts that cross `kanon`'s boundaries, as serde types: the backend contract
//! ([`SearchRequest`] and [`SearchResponse`], what `POST URL/search` exchanges), the trail
//! ([`TrailEntry`], one line of `trail.jsonl`) and the retrieval [`Unit`] a hit refers to.
//!
//! Every document carries a `version`; a missing one means 1. Changes within a version are
//! additive (a reader ignores fields it does not know), a removal or rename is a new version,
//! and a reader rejects a version newer than the one it knows with one line
//! ([`check_version`]). The JSON Schemas under `docs/schemas/` are generated from these types
//! by `tests/schemas.rs`, so a consumer in another language pins the same definitions.
//!
//! The two readers in this crate validate the version: the external backend on the response
//! it gets ([`document_version`] then [`check_version`]) and [`read_trail`] on every line.
//! `pinakes usage` reads the trail without this crate; the schema is the contract between them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use pinakes::index::{Page, iter_units};
use pinakes::manifest::split_page_id;
use pinakes::text::sha256_hex;

/// The backend contract version this crate writes and reads.
pub const BACKEND_VERSION: u32 = 1;
/// The trail version this crate reads.
pub const TRAIL_VERSION: u32 = 1;
/// The unit version this crate writes.
pub const UNIT_VERSION: u32 = 1;

/// Errors raised while reading a contract document.
#[derive(Debug, Error)]
pub enum ContractError {
    /// The document's version is newer than the one this crate reads.
    #[error("{contract}: version {found} is newer than the version {known} this kanon reads")]
    Newer {
        /// What was being read: a file path, or the URL that answered.
        contract: String,
        /// The version the document declares.
        found: u32,
        /// The version this crate reads.
        known: u32,
    },
    /// The trail file could not be read.
    #[error("{path}: {source}")]
    Io {
        /// The trail file path.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// A trail line is not a valid entry.
    #[error("{path}:{line}: invalid trail entry: {source}")]
    Json {
        /// The trail file path.
        path: PathBuf,
        /// One-based line number.
        line: usize,
        /// Underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
    /// A trail line names an id that is not `<source>::<path>`.
    #[error("{path}:{line}: {field} contains {id:?}, which is not a <source>::<path> id")]
    BadId {
        /// The trail file path.
        path: PathBuf,
        /// One-based line number.
        line: usize,
        /// `retrieved` or `cited`.
        field: &'static str,
        /// The offending id.
        id: String,
    },
}

/// The version a document without a `version` field has.
fn default_version() -> u32 {
    1
}

/// Reject a document whose version is newer than the one this crate reads. `contract` names
/// what was read (the trail path, the backend URL) and opens the one-line message.
pub fn check_version(found: u32, known: u32, contract: &str) -> Result<(), ContractError> {
    if found > known {
        return Err(ContractError::Newer {
            contract: contract.to_string(),
            found,
            known,
        });
    }
    Ok(())
}

/// Only the `version` of a document, read before the rest so a newer version is reported as
/// such rather than as whatever field it changed.
#[derive(Debug, Deserialize)]
struct DocumentVersion {
    #[serde(default = "default_version")]
    version: u32,
}

/// The `version` a JSON document declares, 1 when it has none.
pub fn document_version(text: &str) -> Result<u32, serde_json::Error> {
    serde_json::from_str::<DocumentVersion>(text).map(|v| v.version)
}

/// What `kanon` sends to `POST URL/search`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SearchRequest {
    /// Backend contract version; missing means 1.
    #[serde(default = "default_version")]
    pub version: u32,
    /// The query text.
    pub query: String,
    /// How many hits to return at most.
    pub k: usize,
    /// Restrict the search to one module (a source name), when given.
    pub module: Option<String>,
}

/// What a backend answers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SearchResponse {
    /// Backend contract version; missing means 1.
    #[serde(default = "default_version")]
    pub version: u32,
    /// The hits, best first.
    pub hits: Vec<SearchHit>,
}

/// One hit of a search response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SearchHit {
    /// `<source>::<path>` of the page.
    pub page_id: String,
    /// The backend's score; only the order matters to `kanon`.
    pub score: f64,
    /// Heading of the section that matched, empty for the page's intro.
    #[serde(default)]
    pub heading: String,
    /// The unit that matched (`<page_id>#<ordinal>`), when the backend retrieves units.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit_id: Option<String>,
}

/// What the consumer recorded happened with the response, when it recorded anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// The response was judged good.
    Ok,
    /// The response was judged bad.
    Bad,
    /// Not recorded, or recorded as unknown.
    #[default]
    Unknown,
}

/// One served query, as a consumer recorded it: one line of `trail.jsonl`. Every field but `at` and `query` is optional, since consumers vary in how much they log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct TrailEntry {
    /// Trail version; missing means 1.
    #[serde(default = "default_version")]
    pub version: u32,
    /// RFC 3339 UTC time the query was served.
    pub at: String,
    /// The query text.
    pub query: String,
    /// Page ids (`<source>::<path>`) returned, in retrieval order.
    #[serde(default)]
    pub retrieved: Vec<String>,
    /// Rank shown to the user for each entry of `retrieved` (same length, when given).
    #[serde(default)]
    pub ranks: Vec<u32>,
    /// Page ids (`<source>::<path>`) the consumer says were actually cited or used.
    #[serde(default)]
    pub cited: Vec<String>,
    /// How the response fared, when the consumer recorded it.
    #[serde(default)]
    pub outcome: Outcome,
    /// An opaque session identifier, when the consumer gave one.
    #[serde(default)]
    pub session: String,
}

/// One retrieval unit: a section of a page, the thing a hit refers to and `embed` embeds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Unit {
    /// Unit version; missing means 1.
    #[serde(default = "default_version")]
    pub version: u32,
    /// `<page_id>#<ordinal>`.
    pub id: String,
    /// `<source>::<path>` of the page the unit belongs to.
    pub page_id: String,
    /// The unit's heading, empty for the page's intro.
    pub heading: String,
    /// 0-based position of the unit within its page, in document order.
    pub ordinal: usize,
    /// Title, heading and body joined into one text worth embedding.
    pub text: String,
    /// SHA-256 of `text`, lower-case hex.
    pub sha256: String,
}

/// The unit id for `ordinal` within `page_id`.
pub fn unit_id(page_id: &str, ordinal: usize) -> String {
    format!("{page_id}#{ordinal}")
}

/// The retrieval units of the searchable pages, in page then section order, with their ids
/// and hashes. `pages` must already have [`pinakes::index::mark_mirrors`] applied, as
/// [`iter_units`] requires.
pub fn units(pages: &[Page]) -> Vec<Unit> {
    let mut next_ordinal: BTreeMap<String, usize> = BTreeMap::new();
    iter_units(pages)
        .into_iter()
        .map(|unit| {
            let ordinal = next_ordinal.entry(unit.page_id.clone()).or_default();
            let built = Unit {
                version: UNIT_VERSION,
                id: unit_id(&unit.page_id, *ordinal),
                page_id: unit.page_id,
                heading: unit.heading,
                ordinal: *ordinal,
                sha256: sha256_hex(unit.text.as_bytes()),
                text: unit.text,
            };
            *ordinal += 1;
            built
        })
        .collect()
}

fn check_ids<'a>(
    path: &Path,
    line: usize,
    field: &'static str,
    ids: impl IntoIterator<Item = &'a String>,
) -> Result<(), ContractError> {
    for id in ids {
        if split_page_id(id).is_none() {
            return Err(ContractError::BadId {
                path: path.to_path_buf(),
                line,
                field,
                id: id.clone(),
            });
        }
    }
    Ok(())
}

/// Read `trail.jsonl` at `path`: every non-blank line is one [`TrailEntry`]. Each line's
/// version is checked against [`TRAIL_VERSION`] before the line is parsed, and every
/// `retrieved`/`cited` id must be `<source>::<path>`.
pub fn read_trail(path: &Path) -> Result<Vec<TrailEntry>, ContractError> {
    let text = std::fs::read_to_string(path).map_err(|source| ContractError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let json_err = |line: usize| {
        move |source| ContractError::Json {
            path: path.to_path_buf(),
            line,
            source,
        }
    };
    let mut entries = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let line_no = index + 1;
        let version = document_version(line).map_err(json_err(line_no))?;
        check_version(version, TRAIL_VERSION, &path.display().to_string())?;
        let entry: TrailEntry = serde_json::from_str(line).map_err(json_err(line_no))?;
        check_ids(path, line_no, "retrieved", &entry.retrieved)?;
        check_ids(path, line_no, "cited", &entry.cited)?;
        entries.push(entry);
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{SourceSpec, write_artifact};
    use pinakes::index::{Priorities, load_pages, mark_mirrors};

    fn entry(query: &str) -> TrailEntry {
        TrailEntry {
            version: TRAIL_VERSION,
            at: "2026-09-16T12:00:00Z".to_string(),
            query: query.to_string(),
            retrieved: vec![
                "handbook::docs/a.md".to_string(),
                "handbook::docs/b.md".to_string(),
            ],
            ranks: vec![1, 2],
            cited: vec!["handbook::docs/a.md".to_string()],
            outcome: Outcome::Ok,
            session: "s1".to_string(),
        }
    }

    #[test]
    fn a_missing_version_means_one_and_a_present_one_is_kept() {
        let request: SearchRequest = serde_json::from_str(r#"{"query": "q", "k": 3}"#).unwrap();
        assert_eq!(request.version, 1);
        assert_eq!(request.module, None);
        let response: SearchResponse = serde_json::from_str(r#"{"hits": []}"#).unwrap();
        assert_eq!(response.version, 1);
        let entry: TrailEntry = serde_json::from_str(r#"{"at": "t", "query": "q"}"#).unwrap();
        assert_eq!(entry.version, 1);
        assert_eq!(entry.outcome, Outcome::Unknown);
        let unit: Unit = serde_json::from_str(
            r#"{"id": "h::a.md#0", "page_id": "h::a.md", "heading": "", "ordinal": 0, "text": "t", "sha256": "x"}"#,
        )
        .unwrap();
        assert_eq!(unit.version, 1);
        let entry: TrailEntry =
            serde_json::from_str(r#"{"version": 7, "at": "t", "query": "q"}"#).unwrap();
        assert_eq!(entry.version, 7);
        assert_eq!(document_version(r#"{"version": 7}"#).unwrap(), 7);
        assert_eq!(document_version("{}").unwrap(), 1);
    }

    #[test]
    fn the_version_is_always_written() {
        let request = SearchRequest {
            version: BACKEND_VERSION,
            query: "q".to_string(),
            k: 3,
            module: None,
        };
        assert_eq!(
            serde_json::to_string(&request).unwrap(),
            r#"{"version":1,"query":"q","k":3,"module":null}"#
        );
        let line = serde_json::to_string(&entry("q")).unwrap();
        assert!(line.starts_with(r#"{"version":1,"at":"#), "{line}");
        let hit = SearchHit {
            page_id: "handbook::docs/a.md".to_string(),
            score: 1.5,
            heading: String::new(),
            unit_id: None,
        };
        assert_eq!(
            serde_json::to_string(&hit).unwrap(),
            r#"{"page_id":"handbook::docs/a.md","score":1.5,"heading":""}"#
        );
    }

    #[test]
    fn a_higher_version_is_rejected_with_one_line() {
        assert!(check_version(1, 1, "trail.jsonl").is_ok());
        assert!(check_version(0, 1, "trail.jsonl").is_ok());
        let err = check_version(2, 1, "trail.jsonl").unwrap_err();
        assert_eq!(
            err.to_string(),
            "trail.jsonl: version 2 is newer than the version 1 this kanon reads"
        );
    }

    #[test]
    fn a_hit_carries_its_unit_id_when_given() {
        let response: SearchResponse = serde_json::from_str(
            r#"{"version": 1, "hits": [
                {"page_id": "handbook::docs/a.md", "score": 2.0, "heading": "Setup", "unit_id": "handbook::docs/a.md#1"},
                {"page_id": "handbook::docs/b.md", "score": 1.0}
            ]}"#,
        )
        .unwrap();
        assert_eq!(
            response.hits[0].unit_id.as_deref(),
            Some("handbook::docs/a.md#1")
        );
        assert_eq!(response.hits[1].unit_id, None);
        assert_eq!(response.hits[1].heading, "");
    }

    #[test]
    fn units_are_numbered_per_page_and_hashed() {
        let dir = tempfile::tempdir().unwrap();
        write_artifact(
            dir.path(),
            &[SourceSpec {
                name: "handbook",
                repo: "example-org/handbook",
                pages: &[
                    (
                        "docs/a.md",
                        "Storage",
                        "# Storage\n\nKeeps files.\n\n## Caching\n\nEnable caching.\n",
                    ),
                    ("docs/b.md", "Billing", "# Billing\n\nInvoices.\n"),
                ],
                residue: &[],
            }],
        );
        let mut pages = load_pages(dir.path(), &Priorities::default()).unwrap();
        mark_mirrors(&mut pages);
        let units = units(&pages);
        let ids: Vec<&str> = units.iter().map(|u| u.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "handbook::docs/a.md#0",
                "handbook::docs/a.md#1",
                "handbook::docs/b.md#0"
            ]
        );
        assert_eq!(units[1].page_id, "handbook::docs/a.md");
        assert_eq!(units[1].ordinal, 1);
        assert_eq!(units[1].heading, "Caching");
        assert_eq!(units[1].version, UNIT_VERSION);
        assert_eq!(units[1].sha256, sha256_hex(units[1].text.as_bytes()));
        assert_eq!(units[1].sha256.len(), 64);
        // The same text as pinakes cuts, in the same order.
        let raw = iter_units(&pages);
        assert_eq!(raw.len(), units.len());
        for (raw, unit) in raw.iter().zip(&units) {
            assert_eq!(raw.text, unit.text);
        }
    }

    #[test]
    fn read_trail_reads_a_fully_populated_line_and_skips_blanks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trail.jsonl");
        let line = serde_json::to_string(&entry("q")).unwrap();
        std::fs::write(&path, format!("{line}\n\n")).unwrap();
        assert_eq!(read_trail(&path).unwrap(), [entry("q")]);
    }

    #[test]
    fn read_trail_tolerates_missing_optional_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trail.jsonl");
        std::fs::write(
            &path,
            "{\"at\": \"2026-09-16T12:00:00Z\", \"query\": \"q\"}\n",
        )
        .unwrap();
        let entries = read_trail(&path).unwrap();
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        assert_eq!(e.version, 1);
        assert!(e.retrieved.is_empty());
        assert!(e.ranks.is_empty());
        assert!(e.cited.is_empty());
        assert_eq!(e.outcome, Outcome::Unknown);
        assert_eq!(e.session, "");
    }

    #[test]
    fn read_trail_rejects_a_newer_version_before_parsing_the_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trail.jsonl");
        // A version 2 line that would not even parse as version 1 (`query` is an object).
        std::fs::write(
            &path,
            "{\"at\": \"t\", \"query\": \"q\"}\n{\"version\": 2, \"at\": \"t\", \"query\": {\"text\": \"q\"}}\n",
        )
        .unwrap();
        let err = read_trail(&path).unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::Newer {
                    found: 2,
                    known: 1,
                    ..
                }
            ),
            "{err}"
        );
        assert_eq!(
            err.to_string(),
            format!(
                "{}: version 2 is newer than the version 1 this kanon reads",
                path.display()
            )
        );
    }

    #[test]
    fn read_trail_rejects_ids_that_are_not_source_path_shaped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trail.jsonl");
        std::fs::write(
            &path,
            "{\"at\": \"t\", \"query\": \"q\", \"retrieved\": [\"not-an-id\"]}\n",
        )
        .unwrap();
        let err = read_trail(&path).unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::BadId {
                    field: "retrieved",
                    line: 1,
                    ..
                }
            ),
            "{err}"
        );
        std::fs::write(
            &path,
            "{\"at\": \"t\", \"query\": \"q\"}\n{\"at\": \"t\", \"query\": \"q\", \"cited\": [\"handbook::\"]}\n",
        )
        .unwrap();
        let err = read_trail(&path).unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::BadId {
                    field: "cited",
                    line: 2,
                    ..
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn read_trail_reports_a_bad_line_by_number_and_a_missing_file_as_io() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trail.jsonl");
        let line = serde_json::to_string(&entry("q")).unwrap();
        std::fs::write(&path, format!("{line}\n{{bad\n")).unwrap();
        let err = read_trail(&path).unwrap_err();
        assert!(matches!(err, ContractError::Json { line: 2, .. }), "{err}");
        let err = read_trail(&dir.path().join("nope.jsonl")).unwrap_err();
        assert!(matches!(err, ContractError::Io { .. }), "{err}");
    }
}
