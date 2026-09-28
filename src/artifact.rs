//! The artifact's own `manifest.json`, read through pinakes's loader so the artifact contract
//! version (pinakes SPEC §2.8) is enforced and never re-parsed here.
//!
//! `kanon` defines no struct for `manifest.json` or a source's `meta.json`: pages come from
//! [`pinakes::corpus::load_pages`], which rejects a `meta.json` of a newer major, and the
//! manifest from [`pinakes::manifest::Manifest::load`], which rejects the same in `manifest.json`.
//! Both print the same one line (`artifact version 2 is newer than this pinakes supports (1);
//! upgrade pinakes`), which `kanon` passes on unchanged. A consumer in another language pins
//! pinakes's `docs/schemas/manifest.schema.json`.

use std::path::Path;

use pinakes::layout::MANIFEST_FILE;
use pinakes::manifest::{Manifest, ManifestError};

/// The `artifact_version` recorded in `<artifact>/manifest.json`, or `None` when the artifact
/// has no manifest.
///
/// A manifest that cannot be read, is not valid, or follows a newer artifact contract than
/// pinakes reads is the loader's [`ManifestError`], so a caller that goes on to measure the
/// artifact has checked it first.
pub fn manifest_artifact_version(artifact: &Path) -> Result<Option<u32>, ManifestError> {
    match Manifest::load(&artifact.join(MANIFEST_FILE)) {
        Ok(manifest) => Ok(Some(manifest.artifact_version)),
        Err(ManifestError::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
            Ok(None)
        }
        Err(err) => Err(err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, text: &str) {
        std::fs::write(dir.join(MANIFEST_FILE), text).unwrap();
    }

    #[test]
    fn no_manifest_is_no_version() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(manifest_artifact_version(dir.path()).unwrap(), None);
    }

    #[test]
    fn the_recorded_version_is_read_and_a_missing_field_means_one() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            r#"{"version": 1, "artifact_version": 1, "generated_at": "t", "sources": {}}"#,
        );
        assert_eq!(manifest_artifact_version(dir.path()).unwrap(), Some(1));
        write(
            dir.path(),
            r#"{"version": 1, "generated_at": "t", "sources": {}}"#,
        );
        assert_eq!(manifest_artifact_version(dir.path()).unwrap(), Some(1));
    }

    #[test]
    fn a_newer_version_is_the_loaders_own_line_with_the_path() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            r#"{"version": 1, "artifact_version": 2, "generated_at": "t", "sources": {}}"#,
        );
        let err = manifest_artifact_version(dir.path()).unwrap_err();
        assert!(
            matches!(err, ManifestError::ArtifactVersion { .. }),
            "{err}"
        );
        assert_eq!(
            err.to_string(),
            format!(
                "{}: artifact version 2 is newer than this pinakes supports (1); upgrade pinakes",
                dir.path().join(MANIFEST_FILE).display()
            )
        );
    }

    #[test]
    fn a_manifest_that_is_not_one_is_an_error_not_a_missing_manifest() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "{}");
        let err = manifest_artifact_version(dir.path()).unwrap_err();
        assert!(matches!(err, ManifestError::Json { .. }), "{err}");
    }
}
