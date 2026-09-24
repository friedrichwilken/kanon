//! The reference fixtures under `examples/` read through the contract types: `trail.jsonl`
//! against the golden fixture's page ids, and a `search-response.json` as an external backend
//! would answer it.

use std::path::{Path, PathBuf};

use kanon::contracts::{BACKEND_VERSION, Outcome, SearchResponse, TRAIL_VERSION, read_trail};

fn examples() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("examples")
}

/// Every `<source>::<path>` of the golden fixture's artifact.
fn golden_page_ids() -> Vec<String> {
    let artifact = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden/artifact");
    let mut ids = Vec::new();
    for source in std::fs::read_dir(&artifact).unwrap() {
        let source = source.unwrap();
        let name = source.file_name().to_string_lossy().into_owned();
        if name.starts_with('_') || !source.path().is_dir() {
            continue;
        }
        let mut stack = vec![source.path()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "md") {
                    let rel = path.strip_prefix(source.path()).unwrap();
                    ids.push(format!("{name}::{}", rel.display()));
                }
            }
        }
    }
    ids
}

#[test]
fn the_example_trail_reads_and_names_golden_pages() {
    let entries = read_trail(&examples().join("trail.jsonl")).unwrap();
    assert_eq!(entries.len(), 12);
    let pages = golden_page_ids();
    for entry in &entries {
        assert_eq!(entry.version, TRAIL_VERSION);
        assert!(!entry.retrieved.is_empty(), "{}", entry.query);
        for id in entry.retrieved.iter().chain(&entry.cited) {
            assert!(pages.contains(id), "{id} is not a golden fixture page");
        }
        if !entry.ranks.is_empty() {
            assert_eq!(entry.ranks.len(), entry.retrieved.len(), "{}", entry.query);
        }
    }
    // One line spells its version out, the rest rely on the default.
    let text = std::fs::read_to_string(examples().join("trail.jsonl")).unwrap();
    assert_eq!(text.matches("\"version\": 1").count(), 1);
    // Every outcome shows up, including a line that leaves it out.
    assert!(entries.iter().any(|e| e.outcome == Outcome::Ok));
    assert!(entries.iter().any(|e| e.outcome == Outcome::Bad));
    assert!(entries.iter().any(|e| e.outcome == Outcome::Unknown));
}

#[test]
fn the_example_search_response_parses_with_and_without_unit_ids() {
    let text = std::fs::read_to_string(examples().join("search-response.json")).unwrap();
    let response: SearchResponse = serde_json::from_str(&text).unwrap();
    assert_eq!(response.version, BACKEND_VERSION);
    assert_eq!(response.hits.len(), 3);
    assert_eq!(
        response.hits[0].unit_id.as_deref(),
        Some("guides::docs/enable-caching.md#1")
    );
    assert_eq!(response.hits[2].unit_id, None);
    assert_eq!(response.hits[2].heading, "");
    // Written back, the version is always there, an absent `unit_id` stays absent and the
    // defaulted heading is spelled out; reading that again gives the same response.
    let again = serde_json::to_string_pretty(&response).unwrap();
    assert!(again.starts_with("{\n  \"version\": 1,"), "{again}");
    assert_eq!(again.matches("\"unit_id\"").count(), 2);
    assert_eq!(again.matches("\"heading\"").count(), 3);
    let reread: SearchResponse = serde_json::from_str(&again).unwrap();
    assert_eq!(reread, response);
}
