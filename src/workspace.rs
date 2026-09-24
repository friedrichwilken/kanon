//! [`Paths`]: the file locations shared by every command.

use std::path::{Path, PathBuf};

/// The `pinakes.yaml` a workspace may carry next to `kanon.yaml`; read for source priorities
/// (the mirror rule) and, when there is no `kanon.yaml`, for its `eval:` block.
pub const PINAKES_CONFIG: &str = "pinakes.yaml";
/// The default config file name.
pub const KANON_CONFIG: &str = "kanon.yaml";

/// File locations shared by the commands; every path is taken as given (no implicit cwd magic
/// beyond the defaults the CLI fills in).
#[derive(Debug, Clone)]
pub struct Paths {
    /// `kanon.yaml`, or a `pinakes.yaml` whose `eval:` block stands in for it.
    pub config: PathBuf,
    /// The committed `manifest.json`, written by pinakes; `queries add|check` validate
    /// against it.
    pub manifest: PathBuf,
    /// The artifact directory.
    pub artifact: PathBuf,
    /// `embeddings.bin`, written by `embed`; `embeddings.json` sits next to it.
    pub embeddings: PathBuf,
}

impl Paths {
    /// Defaults relative to the config file's directory.
    pub fn for_config(config: &Path) -> Paths {
        let dir = config.parent().map(Path::to_path_buf).unwrap_or_default();
        Paths {
            config: config.to_path_buf(),
            manifest: dir.join("manifest.json"),
            artifact: dir.join("artifact"),
            embeddings: dir.join("embeddings.bin"),
        }
    }

    /// Directory of the config file.
    pub fn config_dir(&self) -> PathBuf {
        self.config
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default()
    }

    /// The `pinakes.yaml` next to the config (or the config itself when that is what it is).
    pub fn pinakes_config(&self) -> PathBuf {
        if self.config.file_name().is_some_and(|n| n == PINAKES_CONFIG) {
            self.config.clone()
        } else {
            self.config_dir().join(PINAKES_CONFIG)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_sit_next_to_the_config() {
        let paths = Paths::for_config(Path::new("proj/kanon.yaml"));
        assert_eq!(paths.manifest, Path::new("proj/manifest.json"));
        assert_eq!(paths.artifact, Path::new("proj/artifact"));
        assert_eq!(paths.embeddings, Path::new("proj/embeddings.bin"));
        assert_eq!(paths.config_dir(), Path::new("proj"));
        assert_eq!(paths.pinakes_config(), Path::new("proj/pinakes.yaml"));
        let pinakes = Paths::for_config(Path::new("proj/pinakes.yaml"));
        assert_eq!(pinakes.pinakes_config(), Path::new("proj/pinakes.yaml"));
        assert_eq!(
            Paths::for_config(Path::new("kanon.yaml")).config_dir(),
            Path::new("")
        );
    }
}
