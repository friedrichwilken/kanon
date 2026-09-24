//! Test-only helpers shared by the backend tests: a small artifact fixture and an embeddings
//! file pair for it.

use std::path::Path;
use std::rc::Rc;

use super::BackendConfig;
use crate::embed::Embedder;
use crate::testing::{SourceSpec, write_artifact};
use pinakes::index::{Page, Priorities, iter_units, load_pages, mark_mirrors};

pub(super) fn fixture_pages() -> (tempfile::TempDir, Vec<Page>) {
    let dir = tempfile::tempdir().unwrap();
    write_artifact(
        dir.path(),
        &[SourceSpec {
            name: "handbook",
            repo: "example-org/handbook",
            pages: &[
                (
                    "docs/user/README.md",
                    "Storage Module",
                    "# Storage\n\nThe storage module keeps uploaded files.\n\n## Upload caching\n\nEnable upload caching with a bucket label.\n",
                ),
                (
                    "docs/user/billing.md",
                    "Billing",
                    "# Billing\n\nInvoices scale to zero.\n",
                ),
            ],
            residue: &[],
        }],
    );
    let pages = load_pages(dir.path(), &Priorities::default()).unwrap();
    (dir, pages)
}

pub(super) fn dense_config(
    dir: &Path,
    embedder: Rc<dyn Embedder>,
) -> (tempfile::TempDir, BackendConfig) {
    let mut pages = load_pages(dir, &Priorities::default()).unwrap();
    mark_mirrors(&mut pages);
    let units = iter_units(&pages);
    let texts: Vec<String> = units.iter().map(|u| u.text.clone()).collect();
    let vectors = crate::embed::embed_units(embedder.as_ref(), "fake", &texts, 64).unwrap();
    let out = tempfile::tempdir().unwrap();
    let bin = out.path().join("embeddings.bin");
    let json = out.path().join("embeddings.json");
    let manifest = crate::embed::EmbeddingsManifest {
        model: "fake".to_string(),
        dimension: crate::embed::testing::FAKE_DIMENSION,
        unit_ids: units.iter().map(|u| u.page_id.clone()).collect(),
        manifest_sha256: crate::embed::artifact_manifest_hash(dir).unwrap(),
    };
    crate::embed::write_embeddings(&bin, &json, &manifest, &vectors).unwrap();
    let config = BackendConfig {
        embeddings_bin: bin,
        embeddings_json: json,
        embedder: Some(embedder),
        ..BackendConfig::default()
    };
    (out, config)
}
