//! `kanon.yaml`: what a bare `kanon eval` measures, and the judge's rules.
//!
//! The file holds exactly the keys of the `eval:` block that `pinakes.yaml` used to carry
//! before evaluation moved here, at the top level, plus a `version`. A workspace that still
//! keeps those keys under `eval:` in `pinakes.yaml` needs no `kanon.yaml`: [`load`] reads
//! either form, and the CLI falls back to `pinakes.yaml` when `kanon.yaml` is absent.
//!
//! `backends:` names retrievers beyond the five built-in kinds ([`BackendKind`]): two running
//! services, or two embeddings files, each under a name that `--backend` and `--compare` accept
//! and the result records ([`NamedBackend`]). The entries are checked here, at parse time, so a
//! config that loads is a config every command can use.
//!
//! Source priorities for the mirror rule ([`pinakes::index::Priorities`]) come from
//! `pinakes.yaml` alone; [`priorities`] reads just the `sources[].name` and `priority` keys,
//! so a `pinakes.yaml` written for a newer or older pinakes still yields them.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use pinakes::index::Priorities;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The only configuration version understood by this iteration.
pub const CONFIG_VERSION: u32 = 1;
/// Result list length when neither `--k` nor the config gives one.
pub const DEFAULT_K: usize = 10;
/// Gate tolerance when there is no config at all.
pub const DEFAULT_MAX_RECALL_DROP: f64 = 0.05;
/// Default minimum share of queries that must be held out (`holdout_min`).
pub const DEFAULT_HOLDOUT_MIN: f64 = 0.2;
/// One of the five retriever shapes, as `backend`, `--backend`, `--compare` and `type:` under
/// `backends:` name them (the `backend` module re-exports it and builds them).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BackendKind {
    /// pinakes's hand-rolled `BM25Okapi` reference index (the default).
    #[default]
    Bm25,
    /// The same units, scored by tantivy's own BM25 (`k1` 1.2, `b` 0.75).
    Bm25Tantivy,
    /// An embeddings file, queried by cosine similarity.
    Dense,
    /// Reciprocal rank fusion of `bm25` and `dense`.
    Hybrid,
    /// A consumer's own store, over HTTP.
    External,
}

impl BackendKind {
    /// The name used on the command line, in the config and in eval results.
    pub fn name(self) -> &'static str {
        match self {
            BackendKind::Bm25 => "bm25",
            BackendKind::Bm25Tantivy => "bm25-tantivy",
            BackendKind::Dense => "dense",
            BackendKind::Hybrid => "hybrid",
            BackendKind::External => "external",
        }
    }

    /// Whether this backend needs an embedder to run at all (`dense`, `hybrid`).
    pub fn needs_embedder(self) -> bool {
        matches!(self, BackendKind::Dense | BackendKind::Hybrid)
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A backend name that is neither one of the five kinds nor a configured name.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error(
    "unknown backend {0:?}: expected bm25, bm25-tantivy, dense, hybrid, external or a name \
     from the config's backends"
)]
pub struct UnknownBackend(pub String);

impl FromStr for BackendKind {
    type Err = UnknownBackend;

    fn from_str(name: &str) -> Result<BackendKind, UnknownBackend> {
        match name {
            "bm25" => Ok(BackendKind::Bm25),
            "bm25-tantivy" => Ok(BackendKind::Bm25Tantivy),
            "dense" => Ok(BackendKind::Dense),
            "hybrid" => Ok(BackendKind::Hybrid),
            "external" => Ok(BackendKind::External),
            other => Err(UnknownBackend(other.to_string())),
        }
    }
}

/// The tuning metric `eval --gate` compares with the baseline (`gate_metric`, `--gate-metric`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GateMetric {
    /// recall@5 (the default).
    #[default]
    Recall5,
    /// recall@10.
    Recall10,
    /// MRR.
    Mrr,
    /// nDCG@5.
    Ndcg5,
    /// nDCG@10.
    Ndcg10,
}

impl GateMetric {
    /// The name on the command line and in the config: `recall5`, `recall10`, `mrr`, `ndcg5`
    /// or `ndcg10`.
    pub fn name(self) -> &'static str {
        match self {
            GateMetric::Recall5 => "recall5",
            GateMetric::Recall10 => "recall10",
            GateMetric::Mrr => "mrr",
            GateMetric::Ndcg5 => "ndcg5",
            GateMetric::Ndcg10 => "ndcg10",
        }
    }

    /// The column label, as the eval table prints it: `recall@5`, `recall@10`, `MRR`, `nDCG@5`
    /// or `nDCG@10`.
    pub fn label(self) -> &'static str {
        match self {
            GateMetric::Recall5 => "recall@5",
            GateMetric::Recall10 => "recall@10",
            GateMetric::Mrr => "MRR",
            GateMetric::Ndcg5 => "nDCG@5",
            GateMetric::Ndcg10 => "nDCG@10",
        }
    }
}

impl fmt::Display for GateMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A gate metric name that is none of the five.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("unknown gate metric {0:?}: expected recall5, recall10, mrr, ndcg5 or ndcg10")]
pub struct UnknownGateMetric(pub String);

impl FromStr for GateMetric {
    type Err = UnknownGateMetric;

    fn from_str(name: &str) -> Result<GateMetric, UnknownGateMetric> {
        match name {
            "recall5" => Ok(GateMetric::Recall5),
            "recall10" => Ok(GateMetric::Recall10),
            "mrr" => Ok(GateMetric::Mrr),
            "ndcg5" => Ok(GateMetric::Ndcg5),
            "ndcg10" => Ok(GateMetric::Ndcg10),
            other => Err(UnknownGateMetric(other.to_string())),
        }
    }
}

/// Errors raised while reading `kanon.yaml` or the `eval:` block of `pinakes.yaml`.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The file could not be read.
    #[error("cannot read config {path}: {source}")]
    Io {
        /// Path that failed to open.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The YAML did not parse or did not match the expected shape.
    #[error("{path}: invalid config: {source}")]
    Yaml {
        /// The config path.
        path: PathBuf,
        /// Underlying YAML error.
        #[source]
        source: serde_yaml_ng::Error,
    },
    /// The `version` field is not supported.
    #[error("{path}: unsupported config version {version}; expected {CONFIG_VERSION}")]
    Version {
        /// The config path.
        path: PathBuf,
        /// The version found.
        version: u32,
    },
    /// An entry under `backends:` is not usable: a bad name, a name that shadows a built-in
    /// kind, an unknown type, or keys that do not fit the type.
    #[error("{path}: backend {name:?}: {message}")]
    Backend {
        /// The config path.
        path: PathBuf,
        /// The entry's name.
        name: String,
        /// What is wrong with it.
        message: String,
    },
}

/// One entry of `backends:`: a retriever under a name of the user's choosing.
///
/// `type` is a [`BackendKind`]. `url` is required for `external` and rejected for every other
/// type; `embeddings` (relative to the config file) is accepted for `dense` and `hybrid` only,
/// where it overrides the top-level `embeddings`. This is checked when the config is parsed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    expecting = "a map with type, and url or embeddings"
)]
pub struct NamedBackend {
    /// The built-in kind this name runs as.
    #[serde(rename = "type")]
    pub kind: BackendKind,
    /// The search endpoint base URL (`external`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// `embeddings.bin` for this name (`dense`, `hybrid`), relative to the config file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embeddings: Option<PathBuf>,
}

/// The evaluation settings (`kanon.yaml`, or `pinakes.yaml`'s `eval:` block).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Config format version; must equal [`CONFIG_VERSION`]. Absent inside a `pinakes.yaml`
    /// `eval:` block, where it defaults.
    #[serde(
        default = "default_version",
        skip_serializing_if = "is_default_version"
    )]
    pub version: u32,
    /// Path to `queries.jsonl`, relative to the config file.
    pub queries: PathBuf,
    /// Cut-off for the result list.
    #[serde(default = "default_k")]
    pub k: usize,
    /// `eval --gate` exits 2 when the gated tuning metric (`gate_metric`) drops by more than
    /// this. Zero when the config omits it (the tolerance is a decision, not a default); a
    /// workspace with no config at all gets [`DEFAULT_MAX_RECALL_DROP`].
    #[serde(default)]
    pub max_recall_drop: f64,
    /// The tuning metric `eval --gate` compares: `recall5` (default), `recall10`, `mrr`, `ndcg5`
    /// or `ndcg10`; `--gate-metric` overrides it. `max_recall_drop` is the tolerance whichever
    /// metric is gated.
    #[serde(default)]
    pub gate_metric: GateMetric,
    /// A negative query (`kind: "negative"`) counts as rejected when its top score stays under
    /// this; without it only an empty result list rejects. `--negative-threshold` overrides it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_threshold: Option<f64>,
    /// `queries check` fails when the held-out share of `queries.jsonl` falls below this.
    #[serde(default = "default_holdout_min")]
    pub holdout_min: f64,
    /// The backend a bare `eval` measures: `bm25` (default), `bm25-tantivy`, `dense`, `hybrid`
    /// or `external`; `--backend` overrides it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    /// The consumer's search endpoint base URL for the `external` backend; `--backend-url`
    /// overrides it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_url: Option<String>,
    /// `embeddings.bin` for the `dense` and `hybrid` backends, relative to the config file
    /// (default: `embeddings.bin` next to it); `--embeddings` overrides it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embeddings: Option<PathBuf>,
    /// Backends a bare `eval` compares over the same query set, one table each; when set it
    /// wins over `backend`. `--compare` or `--backend` on the command line overrides it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub compare: Vec<String>,
    /// Named backends (`old`, `new`, …) that `backend`, `compare`, `--backend` and `--compare`
    /// accept next to the built-in kinds; a result records the name. See [`NamedBackend`].
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub backends: BTreeMap<String, NamedBackend>,
}

fn default_version() -> u32 {
    CONFIG_VERSION
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_default_version(version: &u32) -> bool {
    *version == CONFIG_VERSION
}

fn default_k() -> usize {
    DEFAULT_K
}

fn default_holdout_min() -> f64 {
    DEFAULT_HOLDOUT_MIN
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> ConfigError + '_ {
    move |source| ConfigError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn yaml(path: &Path) -> impl FnOnce(serde_yaml_ng::Error) -> ConfigError + '_ {
    move |source| ConfigError::Yaml {
        path: path.to_path_buf(),
        source,
    }
}

/// Whether a parsed YAML document is a `pinakes.yaml` (it declares `sources`) rather than a
/// `kanon.yaml`.
fn is_pinakes_config(value: &serde_yaml_ng::Value) -> bool {
    value.get("sources").is_some()
}

/// What one config file yields: the evaluation settings, and the source priorities when the
/// file is a `pinakes.yaml` (decided by its content, not its name).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Document {
    /// `kanon.yaml` as is, or the `eval:` block of a `pinakes.yaml`; `None` when a
    /// `pinakes.yaml` has no such block.
    pub config: Option<Config>,
    /// `sources[].priority` when the file is a `pinakes.yaml`.
    pub priorities: Option<Priorities>,
}

/// Read a config file: `kanon.yaml` as is, or the `eval:` block of a `pinakes.yaml` (`None`
/// when that file has no such block).
pub fn load(path: &Path) -> Result<Option<Config>, ConfigError> {
    Ok(load_document(path)?.config)
}

/// Read a config file with everything it carries; see [`Document`].
pub fn load_document(path: &Path) -> Result<Document, ConfigError> {
    let text = std::fs::read_to_string(path).map_err(io(path))?;
    parse_document(path, &text)
}

/// Parse config text; `path` only labels errors.
pub fn from_yaml(path: &Path, text: &str) -> Result<Option<Config>, ConfigError> {
    Ok(parse_document(path, text)?.config)
}

/// Parse config text with everything it carries; `path` only labels errors.
pub fn parse_document(path: &Path, text: &str) -> Result<Document, ConfigError> {
    let value: serde_yaml_ng::Value = serde_yaml_ng::from_str(text).map_err(yaml(path))?;
    let (block, priorities) = if is_pinakes_config(&value) {
        (value.get("eval").cloned(), Some(priorities_of(&value)))
    } else {
        (Some(value), None)
    };
    let config = block
        .map(|block| serde_yaml_ng::from_value::<Config>(block).map_err(yaml(path)))
        .transpose()?;
    if let Some(config) = &config {
        if config.version != CONFIG_VERSION {
            return Err(ConfigError::Version {
                path: path.to_path_buf(),
                version: config.version,
            });
        }
        validate_backends(path, &config.backends)?;
    }
    Ok(Document { config, priorities })
}

/// Whether `name` may name a backend: `[A-Za-z0-9_-]+`.
fn is_backend_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Check every `backends:` entry: the name's characters, that it shadows no built-in kind, and
/// that it carries exactly the keys its type uses.
fn validate_backends(
    path: &Path,
    backends: &BTreeMap<String, NamedBackend>,
) -> Result<(), ConfigError> {
    let err = |name: &str, message: String| ConfigError::Backend {
        path: path.to_path_buf(),
        name: name.to_string(),
        message,
    };
    for (name, backend) in backends {
        if !is_backend_name(name) {
            return Err(err(
                name,
                "invalid name: expected [A-Za-z0-9_-]+".to_string(),
            ));
        }
        if name.parse::<BackendKind>().is_ok() {
            return Err(err(name, format!("shadows the built-in backend {name}")));
        }
        let kind = backend.kind;
        let external = kind == BackendKind::External;
        if external && backend.url.is_none() {
            return Err(err(name, "type external needs a url".to_string()));
        }
        if !external && backend.url.is_some() {
            return Err(err(
                name,
                format!("url is only for type external, not {kind}"),
            ));
        }
        if backend.embeddings.is_some() && !kind.needs_embedder() {
            return Err(err(
                name,
                format!("embeddings is only for type dense or hybrid, not {kind}"),
            ));
        }
    }
    Ok(())
}

/// `sources[].priority` of a parsed `pinakes.yaml` (default 1); only `name` and `priority`
/// are read, so a file written for another pinakes version still yields them.
fn priorities_of(value: &serde_yaml_ng::Value) -> Priorities {
    let mut priorities = Priorities::default();
    if let Some(sources) = value.get("sources").and_then(|s| s.as_sequence()) {
        for source in sources {
            let Some(name) = source.get("name").and_then(|n| n.as_str()) else {
                continue;
            };
            let priority = source
                .get("priority")
                .and_then(serde_yaml_ng::Value::as_i64)
                .unwrap_or(pinakes::index::DEFAULT_PRIORITY);
            priorities.explicit.insert(name.to_string(), priority);
        }
    }
    priorities
}

/// Source priorities from a `pinakes.yaml` (`sources[].priority`, default 1); every source is
/// equal when the file does not exist, so the mirror rule collapses nothing.
pub fn priorities(pinakes_config: &Path) -> Result<Priorities, ConfigError> {
    let text = match std::fs::read_to_string(pinakes_config) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Priorities::default());
        }
        Err(err) => return Err(io(pinakes_config)(err)),
    };
    let value: serde_yaml_ng::Value =
        serde_yaml_ng::from_str(&text).map_err(yaml(pinakes_config))?;
    Ok(priorities_of(&value))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PINAKES: &str = "\
version: 1
sources:
  - name: handbook
    repo: https://github.com/example-org/handbook.git
    ref: main
    priority: 10
    resolver:
      type: glob
      include: ['**/*.md']
  - name: guides
    repo: https://github.com/example-org/guides.git
    ref: main
    resolver:
      type: glob
      include: ['**/*.md']
eval:
  queries: queries.jsonl
  k: 10
  max_recall_drop: 0.05
";

    #[test]
    fn kanon_yaml_parses_with_defaults() {
        let path = Path::new("kanon.yaml");
        let config = from_yaml(path, "queries: queries.jsonl\n")
            .unwrap()
            .expect("a config");
        assert_eq!(config.version, CONFIG_VERSION);
        assert_eq!(config.queries, Path::new("queries.jsonl"));
        assert_eq!(config.k, DEFAULT_K);
        assert!(config.max_recall_drop.abs() < f64::EPSILON);
        assert!((config.holdout_min - DEFAULT_HOLDOUT_MIN).abs() < f64::EPSILON);
        assert_eq!(config.gate_metric, GateMetric::Recall5);
        assert!(config.negative_threshold.is_none());
        assert!(config.backend.is_none() && config.backend_url.is_none());
        assert!(config.embeddings.is_none() && config.compare.is_empty());
        assert!(config.backends.is_empty());

        let full = "version: 1\nqueries: q.jsonl\nk: 5\nmax_recall_drop: 0.1\nholdout_min: 0.3\n\
                    gate_metric: ndcg10\nnegative_threshold: 2.5\n\
                    backend: hybrid\nbackend_url: http://localhost:8080\n\
                    embeddings: vectors/embeddings.bin\ncompare: [bm25, dense]\n";
        let config = from_yaml(path, full).unwrap().unwrap();
        assert_eq!(config.k, 5);
        assert_eq!(config.gate_metric, GateMetric::Ndcg10);
        assert_eq!(config.negative_threshold, Some(2.5));
        assert_eq!(config.backend.as_deref(), Some("hybrid"));
        assert_eq!(config.backend_url.as_deref(), Some("http://localhost:8080"));
        assert_eq!(
            config.embeddings.as_deref(),
            Some(Path::new("vectors/embeddings.bin"))
        );
        assert_eq!(config.compare, ["bm25", "dense"]);
        let text = serde_yaml_ng::to_string(&config).unwrap();
        assert_eq!(
            from_yaml(path, &text).unwrap().unwrap(),
            config,
            "round trips"
        );
    }

    #[test]
    fn pinakes_yaml_yields_its_eval_block_or_nothing() {
        let path = Path::new("pinakes.yaml");
        let config = from_yaml(path, PINAKES).unwrap().expect("the eval block");
        assert_eq!(config.queries, Path::new("queries.jsonl"));
        assert!((config.max_recall_drop - 0.05).abs() < f64::EPSILON);
        let without = PINAKES.split("eval:").next().unwrap();
        assert!(from_yaml(path, without).unwrap().is_none());
    }

    #[test]
    fn rejects_unknown_keys_missing_queries_and_other_versions() {
        let path = Path::new("kanon.yaml");
        assert!(matches!(
            from_yaml(path, "queries: q\ntypo: 1\n").unwrap_err(),
            ConfigError::Yaml { .. }
        ));
        assert!(matches!(
            from_yaml(path, "k: 3\n").unwrap_err(),
            ConfigError::Yaml { .. }
        ));
        assert!(matches!(
            from_yaml(path, "version: 2\nqueries: q\n").unwrap_err(),
            ConfigError::Version { version: 2, .. }
        ));
        assert!(matches!(
            from_yaml(path, "queries: [\n").unwrap_err(),
            ConfigError::Yaml { .. }
        ));
    }

    const BACKENDS: &str = "\
queries: queries.jsonl
compare: [old, new]
backends:
  old:
    type: external
    url: http://localhost:8080
  new:
    type: external
    url: http://localhost:8081
  vectors-v2:
    type: dense
    embeddings: v2/embeddings.bin
  fused_v2:
    type: hybrid
";

    #[test]
    fn backends_parse_in_both_forms_and_round_trip() {
        let path = Path::new("kanon.yaml");
        let config = from_yaml(path, BACKENDS).unwrap().unwrap();
        assert_eq!(config.compare, ["old", "new"]);
        assert_eq!(config.backends.len(), 4);
        let old = &config.backends["old"];
        assert_eq!(old.kind, BackendKind::External);
        assert_eq!(old.url.as_deref(), Some("http://localhost:8080"));
        assert!(old.embeddings.is_none());
        assert_eq!(
            config.backends["vectors-v2"].embeddings.as_deref(),
            Some(Path::new("v2/embeddings.bin"))
        );
        assert_eq!(config.backends["fused_v2"].kind, BackendKind::Hybrid);
        let text = serde_yaml_ng::to_string(&config).unwrap();
        assert_eq!(
            from_yaml(path, &text).unwrap().unwrap(),
            config,
            "round trips"
        );

        // The same block under `eval:` in a pinakes.yaml.
        let pinakes = format!(
            "{}eval:\n{}",
            PINAKES.split("eval:").next().unwrap(),
            BACKENDS
                .lines()
                .fold(String::new(), |text, line| text + "  " + line + "\n")
        );
        let config = from_yaml(Path::new("pinakes.yaml"), &pinakes)
            .unwrap()
            .unwrap();
        assert_eq!(
            config.backends["new"].url.as_deref(),
            Some("http://localhost:8081")
        );
    }

    #[test]
    fn backends_are_checked_when_the_config_is_parsed() {
        let path = Path::new("kanon.yaml");
        let backend_error = |yaml: &str| match from_yaml(path, yaml).unwrap_err() {
            ConfigError::Backend { name, message, .. } => (name, message),
            other => panic!("expected a backend error, got {other}"),
        };
        let cases = [
            (
                "queries: q\nbackends:\n  external:\n    type: external\n    url: http://x\n",
                "external",
                "shadows the built-in backend external",
            ),
            (
                "queries: q\nbackends:\n  old:\n    type: external\n",
                "old",
                "type external needs a url",
            ),
            (
                "queries: q\nbackends:\n  'a b':\n    type: bm25\n",
                "a b",
                "invalid name: expected [A-Za-z0-9_-]+",
            ),
            (
                "queries: q\nbackends:\n  old:\n    type: bm25\n    url: http://x\n",
                "old",
                "url is only for type external, not bm25",
            ),
            (
                "queries: q\nbackends:\n  old:\n    type: external\n    url: http://x\n    \
                 embeddings: e.bin\n",
                "old",
                "embeddings is only for type dense or hybrid, not external",
            ),
        ];
        for (yaml, name, message) in cases {
            assert_eq!(
                backend_error(yaml),
                (name.to_string(), message.to_string()),
                "{yaml}"
            );
        }
        let err = from_yaml(path, cases[1].0).unwrap_err().to_string();
        assert_eq!(
            err,
            "kanon.yaml: backend \"old\": type external needs a url"
        );
        // An unknown type, an unknown key under an entry, a missing type, or a scalar where the
        // entry should be a map, is a YAML shape error.
        let err = from_yaml(path, "queries: q\nbackends:\n  old:\n    type: sparse\n").unwrap_err();
        assert!(matches!(err, ConfigError::Yaml { .. }), "{err}");
        assert!(
            err.to_string()
                .contains("expected one of `bm25`, `bm25-tantivy`, `dense`, `hybrid`, `external`"),
            "{err}"
        );
        let err = from_yaml(path, "queries: q\nbackends:\n  old: external\n").unwrap_err();
        assert!(matches!(err, ConfigError::Yaml { .. }), "{err}");
        assert!(
            err.to_string()
                .contains("a map with type, and url or embeddings"),
            "{err}"
        );
        assert!(matches!(
            from_yaml(
                path,
                "queries: q\nbackends:\n  old:\n    type: bm25\n    k: 3\n"
            )
            .unwrap_err(),
            ConfigError::Yaml { .. }
        ));
        assert!(matches!(
            from_yaml(path, "queries: q\nbackends:\n  old:\n    url: http://x\n").unwrap_err(),
            ConfigError::Yaml { .. }
        ));
    }

    #[test]
    fn gate_metric_names_parse_and_print() {
        for (name, metric, label) in [
            ("recall5", GateMetric::Recall5, "recall@5"),
            ("recall10", GateMetric::Recall10, "recall@10"),
            ("mrr", GateMetric::Mrr, "MRR"),
            ("ndcg5", GateMetric::Ndcg5, "nDCG@5"),
            ("ndcg10", GateMetric::Ndcg10, "nDCG@10"),
        ] {
            assert_eq!(name.parse::<GateMetric>().unwrap(), metric);
            assert_eq!(metric.to_string(), name);
            assert_eq!(metric.label(), label);
        }
        let err = "recall@5".parse::<GateMetric>().unwrap_err();
        assert_eq!(err, UnknownGateMetric("recall@5".to_string()));
        assert!(err.to_string().contains("expected recall5"), "{err}");
        assert!(matches!(
            from_yaml(
                Path::new("kanon.yaml"),
                "queries: q\ngate_metric: precision5\n"
            )
            .unwrap_err(),
            ConfigError::Yaml { .. }
        ));
    }

    #[test]
    fn backend_kind_round_trips_through_its_name() {
        for kind in [
            BackendKind::Bm25,
            BackendKind::Bm25Tantivy,
            BackendKind::Dense,
            BackendKind::Hybrid,
            BackendKind::External,
        ] {
            assert_eq!(kind.name().parse::<BackendKind>().unwrap(), kind);
            assert_eq!(kind.to_string(), kind.name());
            let yaml = serde_yaml_ng::to_string(&kind).unwrap();
            assert_eq!(yaml.trim(), kind.name(), "serde uses the same name");
        }
        assert_eq!(
            "nope".parse::<BackendKind>().unwrap_err(),
            UnknownBackend("nope".to_string())
        );
        assert!(BackendKind::Hybrid.needs_embedder() && !BackendKind::Bm25.needs_embedder());
    }

    #[test]
    fn priorities_come_from_pinakes_yaml_or_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pinakes.yaml");
        assert_eq!(priorities(&path).unwrap(), Priorities::default());
        std::fs::write(&path, PINAKES).unwrap();
        let p = priorities(&path).unwrap();
        assert_eq!(p.of("handbook"), 10);
        assert_eq!(p.of("guides"), pinakes::index::DEFAULT_PRIORITY);
        assert_eq!(p.of("unknown"), pinakes::index::DEFAULT_PRIORITY);
        // Keys this tool does not know are ignored, so a newer pinakes.yaml still works.
        std::fs::write(
            &path,
            "version: 9\nsources:\n  - name: a\n    priority: 3\n    future: x\n",
        )
        .unwrap();
        assert_eq!(priorities(&path).unwrap().of("a"), 3);
        std::fs::write(&path, "not: [valid\n").unwrap();
        assert!(matches!(
            priorities(&path).unwrap_err(),
            ConfigError::Yaml { .. }
        ));
    }

    #[test]
    fn a_pinakes_config_carries_priorities_under_any_name() {
        let doc = parse_document(Path::new("configs/prod.yaml"), PINAKES).unwrap();
        assert_eq!(doc.config.unwrap().queries, Path::new("queries.jsonl"));
        assert_eq!(doc.priorities.as_ref().unwrap().of("handbook"), 10);
        let doc = parse_document(Path::new("kanon.yaml"), "queries: q.jsonl\n").unwrap();
        assert!(
            doc.priorities.is_none(),
            "a kanon.yaml never carries priorities"
        );
        let without = PINAKES.split("eval:").next().unwrap();
        let doc = parse_document(Path::new("pinakes.yaml"), without).unwrap();
        assert!(doc.config.is_none() && doc.priorities.is_some());
    }

    #[test]
    fn load_reads_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kanon.yaml");
        assert!(matches!(load(&path).unwrap_err(), ConfigError::Io { .. }));
        std::fs::write(&path, "queries: queries.jsonl\n").unwrap();
        assert_eq!(load(&path).unwrap().unwrap().k, DEFAULT_K);
    }
}
