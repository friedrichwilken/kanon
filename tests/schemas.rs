//! The JSON Schemas under `docs/schemas/` are generated from the contract types and committed,
//! so a consumer in another language pins the same definitions. This test keeps them from
//! drifting: it asserts each committed file equals what the types generate now, and rewrites
//! them when `UPDATE_SCHEMAS=1` is set (`just update-golden` does both refreshes).

use std::path::{Path, PathBuf};

use kanon::contracts::{SearchRequest, SearchResponse, TrailEntry, Unit};
use schemars::{JsonSchema, schema_for};

fn schemas_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/schemas")
}

/// The schema of `T` as pretty JSON with a trailing newline, the form committed.
fn generated<T: JsonSchema>() -> String {
    let mut text = serde_json::to_string_pretty(&schema_for!(T)).unwrap();
    text.push('\n');
    text
}

fn check(name: &str, text: &str) {
    let path = schemas_dir().join(format!("{name}.schema.json"));
    if std::env::var_os("UPDATE_SCHEMAS").is_some() {
        std::fs::write(&path, text).unwrap();
    }
    let committed =
        std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    assert_eq!(
        committed,
        text,
        "{} differs from the contract types; run `UPDATE_SCHEMAS=1 cargo test --test schemas` if the change is intended",
        path.display()
    );
}

#[test]
fn committed_schemas_match_the_contract_types() {
    check("backend-request", &generated::<SearchRequest>());
    check("backend-response", &generated::<SearchResponse>());
    check("trail-entry", &generated::<TrailEntry>());
    check("unit", &generated::<Unit>());
}

#[test]
fn every_schema_declares_a_defaulted_version() {
    for name in ["backend-request", "backend-response", "trail-entry", "unit"] {
        let path = schemas_dir().join(format!("{name}.schema.json"));
        let text = std::fs::read_to_string(&path).unwrap();
        let schema: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(schema["properties"]["version"]["default"], 1, "{name}");
        assert!(
            text.ends_with("}\n"),
            "{name}: pretty JSON with a trailing newline"
        );
    }
}
