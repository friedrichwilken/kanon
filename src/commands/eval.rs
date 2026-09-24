use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::backend::{self, Backend, BackendConfig, BackendError, BackendKind, BackendSpec};
use crate::config::{
    Config, DEFAULT_K, DEFAULT_MAX_RECALL_DROP, GateMetric, NamedBackend, UnknownBackend,
};
use crate::embed::{EmbedError, Embedder, HttpEmbedder};
use crate::error::CommandError;
use crate::eval::{self, Delta, EvalSummary, Gate};
use crate::history::{self, Run, RunInfo};
use crate::workspace::Paths;
use pinakes::index::{self, Index, IndexError, Page, Priorities};

use super::settings;

/// Options for `eval`.
#[derive(Debug, Clone, Default)]
pub struct EvalOptions {
    /// Query file (default: `queries` from the config, relative to it).
    pub queries: Option<PathBuf>,
    /// Result list length (default: `k` from the config, else [`DEFAULT_K`]).
    pub k: Option<usize>,
    /// Write the result JSON here.
    pub json: Option<PathBuf>,
    /// Baseline result to gate the tuning metric against.
    pub gate: Option<PathBuf>,
    /// The tuning metric `gate` compares (default: `gate_metric` from the config, else
    /// recall@5).
    pub gate_metric: Option<GateMetric>,
    /// Residue pages (`<source>::<path>`) to add from `_residue` before measuring.
    pub with: Vec<String>,
    /// Pages to remove before measuring.
    pub without: Vec<String>,
    /// A negative query counts as rejected when its top score stays under this (default:
    /// `negative_threshold` from the config, else only an empty result list rejects).
    pub negative_threshold: Option<f64>,
    /// Write a numbered run file into this directory (see [`crate::history`]).
    pub out: Option<PathBuf>,
    /// The run file's label; default: the git short SHA, else `run`.
    pub label: Option<String>,
}

/// The judge's rules for one run, from the options and the config: the gate tolerance and
/// metric, and the negative threshold. An option always wins over the config.
#[derive(Debug, Clone, Copy)]
struct Rules {
    max_drop: f64,
    gate_metric: GateMetric,
    negative_threshold: Option<f64>,
}

impl Rules {
    fn of(
        eval_config: Option<&Config>,
        gate_metric: Option<GateMetric>,
        negative_threshold: Option<f64>,
    ) -> Rules {
        Rules {
            max_drop: eval_config.map_or(DEFAULT_MAX_RECALL_DROP, |e| e.max_recall_drop),
            gate_metric: gate_metric
                .or_else(|| eval_config.map(|e| e.gate_metric))
                .unwrap_or_default(),
            negative_threshold: negative_threshold
                .or_else(|| eval_config.and_then(|e| e.negative_threshold)),
        }
    }

    fn gate(
        &self,
        current: &EvalSummary,
        baseline: Option<&Path>,
    ) -> Result<Option<Gate>, CommandError> {
        match baseline {
            Some(path) => Ok(Some(eval::gate(
                current,
                &EvalSummary::load(path)?,
                self.max_drop,
                self.gate_metric,
            ))),
            None => Ok(None),
        }
    }
}

/// What `eval` produced.
#[derive(Debug)]
pub struct EvalOutcome {
    /// The result (after `--with`/`--without` when given).
    pub summary: EvalSummary,
    /// Pages in the measured corpus, mirrors included.
    pub page_count: usize,
    /// Pages in the search corpus.
    pub searchable_count: usize,
    /// Result list length used.
    pub k: usize,
    /// The gate outcome when `--gate` was given.
    pub gate: Option<Gate>,
    /// The delta when `--with`/`--without` was given.
    pub delta: Option<Delta>,
    /// The run file written when `--out` was given.
    pub run_file: Option<PathBuf>,
}

/// The query file: `--queries` when given, else `queries` from the config.
pub(super) fn resolve_queries_path(
    paths: &Paths,
    eval_config: Option<&Config>,
    queries: Option<&Path>,
) -> Result<PathBuf, CommandError> {
    match queries {
        Some(path) => Ok(path.to_path_buf()),
        None => eval_config
            .map(|e| paths.config_dir().join(&e.queries))
            .ok_or(CommandError::NoQueries),
    }
}

/// Run `eval`: index the artifact, run the queries, optionally gate against a baseline and
/// measure the effect of adding residue pages or removing pages.
///
/// The config is optional: without one, the query file must be given, every source gets the
/// default priority of [`Priorities`] (so no page is a mirror), `k` defaults to [`DEFAULT_K`],
/// the gate tolerance to [`DEFAULT_MAX_RECALL_DROP`] and the gated metric to recall@5.
pub fn eval(paths: &Paths, options: &EvalOptions) -> Result<EvalOutcome, CommandError> {
    let settings = settings(paths)?;
    let eval_config = settings.config.as_ref();
    let queries_path = resolve_queries_path(paths, eval_config, options.queries.as_deref())?;
    let k = options
        .k
        .or_else(|| eval_config.map(|e| e.k))
        .unwrap_or(DEFAULT_K);
    let rules = Rules::of(eval_config, options.gate_metric, options.negative_threshold);
    let priorities = settings.priorities;
    let queries = eval::load_queries(&queries_path)?;
    let pages = index::load_pages(&paths.artifact, &priorities)?;

    let (summary, page_count, searchable_count, delta) =
        if options.with.is_empty() && options.without.is_empty() {
            let index = Index::from_pages(pages)?;
            let summary = eval::evaluate(&index, &queries, k, rules.negative_threshold)?;
            (summary, index.page_count(), index.searchable_count(), None)
        } else {
            let (summary, page_count, searchable_count, delta) = evaluate_adjusted(
                pages,
                &paths.artifact,
                &priorities,
                &queries,
                k,
                rules.negative_threshold,
                &options.with,
                &options.without,
            )?;
            (summary, page_count, searchable_count, Some(delta))
        };
    let gate = rules.gate(&summary, options.gate.as_deref())?;
    if let Some(path) = &options.json {
        summary.save(path)?;
    }
    let run_file = match &options.out {
        Some(dir) => Some(write_run(
            paths,
            dir,
            options.label.as_deref(),
            &queries_path,
            k,
            &summary,
        )?),
        None => None,
    };
    Ok(EvalOutcome {
        summary,
        page_count,
        searchable_count,
        k,
        gate,
        delta,
        run_file,
    })
}

/// Write `summary` as the next run file in `dir`, with a [`RunInfo`] tying it to the artifact's
/// manifest, the query file and `k`. The backend recorded is the result's own, or `bm25` for
/// the plain path, which measures the reference index and leaves the field empty.
fn write_run(
    paths: &Paths,
    dir: &Path,
    label: Option<&str>,
    queries_path: &Path,
    k: usize,
    summary: &EvalSummary,
) -> Result<PathBuf, CommandError> {
    let backend = if summary.backend.is_empty() {
        BackendKind::Bm25.name().to_string()
    } else {
        summary.backend.clone()
    };
    let queries_bytes = std::fs::read(queries_path).map_err(crate::error::io_err(queries_path))?;
    let run = Run {
        summary: summary.clone(),
        run: Some(RunInfo {
            label: history::run_label(label, &paths.config_dir()),
            at: pinakes::manifest::now_rfc3339(),
            backend,
            manifest_sha256: crate::embed::artifact_manifest_hash(&paths.artifact)?,
            queries_sha256: pinakes::text::sha256_hex(&queries_bytes),
            k,
        }),
    };
    Ok(history::write_run(dir, &run)?)
}

/// Adjust `pages` by `with`/`without`, index it, and return the before/after summaries' delta
/// alongside the after index's counts — the `--with`/`--without` computation shared by [`eval`]
/// and [`eval_backend`]'s `bm25` adjusting branch.
#[allow(clippy::too_many_arguments)]
fn evaluate_adjusted(
    pages: Vec<Page>,
    artifact: &Path,
    priorities: &Priorities,
    queries: &[eval::Query],
    k: usize,
    negative_threshold: Option<f64>,
    with: &[String],
    without: &[String],
) -> Result<(EvalSummary, usize, usize, Delta), CommandError> {
    let before = eval::evaluate(
        &Index::from_pages(pages.clone())?,
        queries,
        k,
        negative_threshold,
    )?;
    let index = Index::from_pages(adjust_pages(pages, artifact, priorities, with, without)?)?;
    let after = eval::evaluate(&index, queries, k, negative_threshold)?;
    let delta = eval::delta(&before, &after);
    Ok((after, index.page_count(), index.searchable_count(), delta))
}

/// Apply `--with` (add residue pages) and `--without` (remove pages) to the page list.
fn adjust_pages(
    mut pages: Vec<Page>,
    artifact: &Path,
    priorities: &Priorities,
    with: &[String],
    without: &[String],
) -> Result<Vec<Page>, CommandError> {
    for id in with {
        if pages.iter().any(|p| &p.id == id) {
            return Err(IndexError::AlreadyPresent(id.clone()).into());
        }
        pages.push(index::load_residue_page(artifact, id, priorities)?);
    }
    for id in without {
        let before = pages.len();
        pages.retain(|p| &p.id != id);
        if pages.len() == before {
            return Err(IndexError::UnknownPage(id.clone()).into());
        }
    }
    Ok(pages)
}

// -------------------------------------------------------------------------------------------
// eval --backend / --compare
// -------------------------------------------------------------------------------------------

/// Options for `eval --backend` (any backend other than the plain default).
#[derive(Clone, Default)]
pub struct BackendEvalOptions {
    /// Query file (default: `queries` from the config, relative to it).
    pub queries: Option<PathBuf>,
    /// Result list length (default: `k` from the config, else [`DEFAULT_K`]).
    pub k: Option<usize>,
    /// Write the result JSON here.
    pub json: Option<PathBuf>,
    /// Baseline result to gate the tuning metric against.
    pub gate: Option<PathBuf>,
    /// The tuning metric `gate` compares (default: `gate_metric` from the config, else
    /// recall@5).
    pub gate_metric: Option<GateMetric>,
    /// Residue pages to add before measuring; only supported for `bm25`.
    pub with: Vec<String>,
    /// Pages to remove before measuring; only supported for `bm25`.
    pub without: Vec<String>,
    /// A negative query counts as rejected when its top score stays under this (default:
    /// `negative_threshold` from the config, else only an empty result list rejects).
    pub negative_threshold: Option<f64>,
    /// The backend to measure: its name, kind, and the URL or embeddings file that came with
    /// the name (`embeddings` defaults to `paths.embeddings`).
    pub backend: BackendSpec,
    /// Use embeddings even when their recorded manifest hash does not match the artifact.
    pub allow_stale: bool,
    /// Where query embeddings come from (`dense`, `hybrid`).
    pub embedder: Option<std::rc::Rc<dyn Embedder>>,
    /// Write a numbered run file into this directory (see [`crate::history`]).
    pub out: Option<PathBuf>,
    /// The run file's label; default: the git short SHA, else `run`. `eval --compare` suffixes
    /// it with `-<backend>`.
    pub label: Option<String>,
}

/// What `eval --backend` produced.
#[derive(Debug)]
pub struct BackendEvalOutcome {
    /// The result (after `--with`/`--without` when given), with [`EvalSummary::backend`] set.
    pub summary: EvalSummary,
    /// Pages in the measured corpus, mirrors included.
    pub page_count: usize,
    /// Pages in the search corpus.
    pub searchable_count: usize,
    /// Result list length used.
    pub k: usize,
    /// The gate outcome when `--gate` was given.
    pub gate: Option<Gate>,
    /// The delta when `--with`/`--without` was given.
    pub delta: Option<Delta>,
    /// The run file written when `--out` was given.
    pub run_file: Option<PathBuf>,
}

/// Run every query against `backend` with a result list of `k` pages.
fn evaluate_backend(
    backend: &dyn Backend,
    queries: &[eval::Query],
    k: usize,
    negative_threshold: Option<f64>,
) -> Result<EvalSummary, CommandError> {
    let mut rows = Vec::with_capacity(queries.len());
    for query in queries {
        let hits = backend.search(&query.query, k, None)?;
        rows.push(eval::QueryResult::of_hits(query, &hits, k));
    }
    Ok(eval::summarise(rows, negative_threshold))
}

/// The [`BackendConfig`] for a command's backend, shared by `eval --backend` and `grade`:
/// the spec's `embeddings` defaults to the workspace's `embeddings.bin`, and `embeddings.json`
/// sits next to whichever file is used.
pub(super) fn backend_config(
    paths: &Paths,
    priorities: Priorities,
    spec: &BackendSpec,
    allow_stale: bool,
    embedder: Option<Rc<dyn Embedder>>,
) -> BackendConfig {
    let embeddings_bin = spec
        .embeddings
        .clone()
        .unwrap_or_else(|| paths.embeddings.clone());
    let embeddings_json = embeddings_bin.with_extension("json");
    BackendConfig {
        priorities,
        embeddings_bin,
        embeddings_json,
        allow_stale,
        embedder,
        backend_url: spec.url.clone(),
    }
}

/// Run `eval --backend NAME`: like [`eval()`], but through the [`Backend`] trait,
/// recording the backend's name (a configured name, or the kind's) in the result.
/// `--with`/`--without` only work for `bm25`, which indexes a page list directly; every other
/// backend rejects them with [`BackendError::UnsupportedAdjustment`].
pub fn eval_backend(
    paths: &Paths,
    options: &BackendEvalOptions,
) -> Result<BackendEvalOutcome, CommandError> {
    let settings = settings(paths)?;
    let eval_config = settings.config.as_ref();
    let queries_path = resolve_queries_path(paths, eval_config, options.queries.as_deref())?;
    let k = options
        .k
        .or_else(|| eval_config.map(|e| e.k))
        .unwrap_or(DEFAULT_K);
    let rules = Rules::of(eval_config, options.gate_metric, options.negative_threshold);
    let priorities = settings.priorities;
    let queries = eval::load_queries(&queries_path)?;
    let adjusting = !options.with.is_empty() || !options.without.is_empty();
    if adjusting && options.backend.kind != BackendKind::Bm25 {
        return Err(BackendError::UnsupportedAdjustment(options.backend.name.clone()).into());
    }

    let (summary, page_count, searchable_count, delta) = if adjusting {
        let pages = index::load_pages(&paths.artifact, &priorities)?;
        let (summary, page_count, searchable_count, delta) = evaluate_adjusted(
            pages,
            &paths.artifact,
            &priorities,
            &queries,
            k,
            rules.negative_threshold,
            &options.with,
            &options.without,
        )?;
        (
            summary.with_backend(&options.backend.name),
            page_count,
            searchable_count,
            Some(delta),
        )
    } else {
        let config = backend_config(
            paths,
            priorities,
            &options.backend,
            options.allow_stale,
            options.embedder.clone(),
        );
        let built = backend::build(options.backend.kind, &paths.artifact, &config)?;
        let summary = evaluate_backend(built.as_ref(), &queries, k, rules.negative_threshold)?
            .with_backend(&options.backend.name);
        (summary, built.page_count(), built.searchable_count(), None)
    };

    let gate = rules.gate(&summary, options.gate.as_deref())?;
    if let Some(path) = &options.json {
        summary.save(path)?;
    }
    let run_file = match &options.out {
        Some(dir) => Some(write_run(
            paths,
            dir,
            options.label.as_deref(),
            &queries_path,
            k,
            &summary,
        )?),
        None => None,
    };
    Ok(BackendEvalOutcome {
        summary,
        page_count,
        searchable_count,
        k,
        gate,
        delta,
        run_file,
    })
}

/// Run `eval --compare a,b,c`: [`eval_backend`] once per backend, over the same query set;
/// `common.backend` is ignored. With `out`, each backend gets its own run file, labelled
/// `<label>-<backend name>`.
pub fn eval_compare(
    paths: &Paths,
    backends: &[BackendSpec],
    common: &BackendEvalOptions,
) -> Result<Vec<(BackendSpec, BackendEvalOutcome)>, CommandError> {
    if backends.is_empty() {
        return Err(CommandError::EmptyCompare);
    }
    let label = common
        .out
        .is_some()
        .then(|| history::run_label(common.label.as_deref(), &paths.config_dir()));
    backends
        .iter()
        .map(|spec| {
            let options = BackendEvalOptions {
                backend: spec.clone(),
                json: None,
                label: label.as_ref().map(|label| format!("{label}-{spec}")),
                ..common.clone()
            };
            Ok((spec.clone(), eval_backend(paths, &options)?))
        })
        .collect()
}

// -------------------------------------------------------------------------------------------
// Decision layer: which of eval / eval_backend / eval_compare a set of flags selects.
// -------------------------------------------------------------------------------------------

/// The backend-selecting flags `eval` and `grade` share (`--backend`, `--backend-url`,
/// `--embeddings`, `--allow-stale`), as given on the command line and before the config's
/// defaults are applied, plus the config's named backends once they are.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BackendFlags {
    /// `--backend`.
    pub backend: Option<String>,
    /// `--backend-url`.
    pub backend_url: Option<String>,
    /// `--embeddings`.
    pub embeddings: Option<PathBuf>,
    /// `--allow-stale`.
    pub allow_stale: bool,
    /// The config's `backends:`, with each `embeddings` made relative to the config file;
    /// [`apply_backend_config_defaults`] fills it, and `--backend`/`--compare` may name any
    /// entry.
    pub backends: BTreeMap<String, NamedBackend>,
}

impl BackendFlags {
    /// The backend `--backend` selects, `bm25` when none was named; see [`BackendFlags::resolve`].
    pub fn spec(&self) -> Result<BackendSpec, BackendError> {
        self.backend
            .as_deref()
            .map_or_else(|| Ok(BackendSpec::default()), |name| self.resolve(name))
    }

    /// One backend by name, as `--backend NAME` or one entry of `--compare` selects it. A
    /// built-in kind takes the command's `--backend-url` and `--embeddings` (or the config's);
    /// a configured name brings its own `url`, and its own `embeddings` when it has one, else
    /// `--embeddings`. Any other name is [`BackendError::UnknownBackend`].
    pub fn resolve(&self, name: &str) -> Result<BackendSpec, BackendError> {
        if let Ok(kind) = name.parse::<BackendKind>() {
            return Ok(BackendSpec {
                url: self.backend_url.clone(),
                embeddings: self.embeddings.clone(),
                ..BackendSpec::builtin(kind)
            });
        }
        let named = self
            .backends
            .get(name)
            .ok_or_else(|| UnknownBackend(name.to_string()))?;
        Ok(BackendSpec {
            name: name.to_string(),
            kind: named.kind,
            url: named.url.clone(),
            embeddings: named.embeddings.clone().or_else(|| self.embeddings.clone()),
        })
    }

    /// Whether none of the flags was given (the config's named backends do not count).
    pub(crate) fn is_empty(&self) -> bool {
        self.backend.is_none()
            && self.backend_url.is_none()
            && self.embeddings.is_none()
            && !self.allow_stale
    }
}

/// Fill the config's `backend`, `backend_url` and `embeddings` (relative to the config file)
/// into whichever flags were left unset, and its `backends` for [`BackendFlags::resolve`]. A
/// flag always wins. `eval` and `grade` both go through here, so the two commands read the
/// config the same way.
pub fn apply_backend_config_defaults(
    paths: &Paths,
    flags: &mut BackendFlags,
) -> Result<(), CommandError> {
    if let Some(config) = load_config(paths)? {
        fill_backend_defaults(paths, config, flags);
    }
    Ok(())
}

/// The config when there is a config file (and it has one); `None` otherwise.
fn load_config(paths: &Paths) -> Result<Option<Config>, CommandError> {
    if !paths.config.is_file() {
        return Ok(None);
    }
    Ok(crate::config::load(&paths.config)?)
}

fn fill_backend_defaults(paths: &Paths, config: Config, flags: &mut BackendFlags) {
    let dir = paths.config.parent().unwrap_or(Path::new("."));
    let relative_to_config = |path: PathBuf| {
        if path.is_absolute() {
            path
        } else {
            dir.join(path)
        }
    };
    if flags.backend.is_none() {
        flags.backend = config.backend;
    }
    if flags.backend_url.is_none() {
        flags.backend_url = config.backend_url;
    }
    if flags.embeddings.is_none() {
        flags.embeddings = config.embeddings.map(relative_to_config);
    }
    flags.backends = config
        .backends
        .into_iter()
        .map(|(name, mut backend)| {
            backend.embeddings = backend.embeddings.map(relative_to_config);
            (name, backend)
        })
        .collect();
}

/// Eval flags as given on the command line, before the config's defaults are applied
/// and before `--backend`/`--compare` names are parsed. One field per CLI flag.
#[derive(Debug, Clone, Default)]
pub struct EvalFlags {
    /// `--queries`.
    pub queries: Option<PathBuf>,
    /// `--k`.
    pub k: Option<usize>,
    /// `--json`.
    pub json: Option<PathBuf>,
    /// `--gate`.
    pub gate: Option<PathBuf>,
    /// `--gate-metric`.
    pub gate_metric: Option<GateMetric>,
    /// `--with`.
    pub with: Vec<String>,
    /// `--without`.
    pub without: Vec<String>,
    /// `--negative-threshold`.
    pub negative_threshold: Option<f64>,
    /// `--backend`, `--backend-url`, `--embeddings`, `--allow-stale`.
    pub backend: BackendFlags,
    /// `--compare`.
    pub compare: Vec<String>,
    /// `--out`.
    pub out: Option<PathBuf>,
    /// `--label`.
    pub label: Option<String>,
}

/// Fill the config's defaults (`backend`, `backend_url`, `embeddings`, `compare`)
/// into whichever flags were left unset. A flag always wins; a configured `compare`
/// applies only to a bare `eval`, so `--backend NAME` still measures that one backend.
pub fn apply_eval_config_defaults(
    paths: &Paths,
    flags: &mut EvalFlags,
) -> Result<(), CommandError> {
    let Some(eval_config) = load_config(paths)? else {
        return Ok(());
    };
    if flags.compare.is_empty() && flags.backend.backend.is_none() {
        flags.compare.clone_from(&eval_config.compare);
    }
    fill_backend_defaults(paths, eval_config, &mut flags.backend);
    Ok(())
}

/// Which of the three eval paths a (config-defaulted) set of flags selects.
pub enum EvalPlan {
    /// The legacy path, through [`eval()`].
    Plain(EvalOptions),
    /// A single named backend, through [`eval_backend()`].
    Backend(BackendEvalOptions),
    /// Every named backend over the same query set, through [`eval_compare()`]; `json` is the
    /// combined result's destination (each individual backend call always gets `json: None`).
    Compare {
        /// The backends to compare.
        backends: Vec<BackendSpec>,
        /// Options common to every backend in the comparison.
        common: BackendEvalOptions,
        /// Where to write the combined JSON, if anywhere.
        json: Option<PathBuf>,
    },
}

impl EvalPlan {
    /// Whether an embedder must be built before running this plan (`dense`/`hybrid`).
    pub fn needs_embedder(&self) -> bool {
        match self {
            EvalPlan::Plain(_) => false,
            EvalPlan::Backend(options) => options.backend.needs_embedder(),
            EvalPlan::Compare { backends, .. } => backends.iter().any(BackendSpec::needs_embedder),
        }
    }

    /// Attach an embedder to the plan (`Backend`/`Compare` only; a no-op on `Plain`).
    #[must_use]
    pub fn with_embedder(mut self, embedder: Rc<dyn Embedder>) -> EvalPlan {
        match &mut self {
            EvalPlan::Plain(_) => {}
            EvalPlan::Backend(options) => options.embedder = Some(embedder),
            EvalPlan::Compare { common, .. } => common.embedder = Some(embedder),
        }
        self
    }
}

/// `run_eval`'s dispatch, minus execution: `--compare` wins; else any backend-only flag
/// (`--backend`/`--backend-url`/`--embeddings`/`--allow-stale`) selects that one backend; else
/// the legacy plain path. Resolves backend names ([`BackendFlags::resolve`]), so this can fail.
pub fn eval_plan(flags: EvalFlags) -> Result<EvalPlan, CommandError> {
    if !flags.compare.is_empty() {
        let backends: Vec<BackendSpec> = flags
            .compare
            .iter()
            .map(|name| flags.backend.resolve(name.trim()))
            .collect::<Result<_, _>>()?;
        let common = BackendEvalOptions {
            queries: flags.queries,
            k: flags.k,
            json: None,
            gate: flags.gate,
            gate_metric: flags.gate_metric,
            with: flags.with,
            without: flags.without,
            negative_threshold: flags.negative_threshold,
            backend: BackendSpec::default(),
            allow_stale: flags.backend.allow_stale,
            embedder: None,
            out: flags.out,
            label: flags.label,
        };
        return Ok(EvalPlan::Compare {
            backends,
            common,
            json: flags.json,
        });
    }
    if flags.backend.is_empty() {
        return Ok(EvalPlan::Plain(EvalOptions {
            queries: flags.queries,
            k: flags.k,
            json: flags.json,
            gate: flags.gate,
            gate_metric: flags.gate_metric,
            with: flags.with,
            without: flags.without,
            negative_threshold: flags.negative_threshold,
            out: flags.out,
            label: flags.label,
        }));
    }
    let backend = flags.backend.spec()?;
    Ok(EvalPlan::Backend(BackendEvalOptions {
        queries: flags.queries,
        k: flags.k,
        json: flags.json,
        gate: flags.gate,
        gate_metric: flags.gate_metric,
        with: flags.with,
        without: flags.without,
        negative_threshold: flags.negative_threshold,
        backend,
        allow_stale: flags.backend.allow_stale,
        embedder: None,
        out: flags.out,
        label: flags.label,
    }))
}

/// An embedder for `KANON_EMBED_URL`/`KANON_EMBED_KEY`, for backends that embed queries
/// (`dense`, `hybrid`). The model itself comes from `embeddings.json`, not from here.
pub fn eval_embedder_from_env() -> Result<Rc<dyn Embedder>, EmbedError> {
    let base = crate::env::var("KANON_EMBED_URL").ok_or(EmbedError::MissingEmbedUrl)?;
    let key = crate::env::var("KANON_EMBED_KEY");
    Ok(Rc::new(HttpEmbedder::new(base, key)))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::commands::embed::{EmbedOptions, embed};
    use crate::testing::{eval_workspace, fake_embedder};

    #[test]
    fn eval_reads_defaults_from_the_config_or_the_options() {
        let (dir, paths) = eval_workspace();
        assert!(matches!(
            eval(&paths, &EvalOptions::default()).unwrap_err(),
            CommandError::NoQueries
        ));
        let options = EvalOptions {
            queries: Some(dir.path().join("queries.jsonl")),
            json: Some(dir.path().join("out.json")),
            ..EvalOptions::default()
        };
        let outcome = eval(&paths, &options).unwrap();
        assert_eq!(
            (outcome.page_count, outcome.searchable_count, outcome.k),
            (1, 1, 10)
        );
        assert!(
            (outcome.summary.tuning.overall.recall5 - 0.5).abs() < 1e-12,
            "{:?}",
            outcome.summary.queries
        );
        assert!(outcome.gate.is_none() && outcome.delta.is_none());
        assert_eq!(
            EvalSummary::load(&dir.path().join("out.json")).unwrap(),
            outcome.summary
        );

        // With a config: queries and k come from its eval section.
        fs::write(
            &paths.config,
            "queries: queries.jsonl\nk: 3\nmax_recall_drop: 0.1\n",
        )
        .unwrap();
        let outcome = eval(&paths, &EvalOptions::default()).unwrap();
        assert_eq!(outcome.k, 3);
        assert_eq!(outcome.summary.query("caching").unwrap().top.len(), 1);
    }

    #[test]
    fn eval_gates_and_measures_with_and_without() {
        let (dir, paths) = eval_workspace();
        let queries = dir.path().join("queries.jsonl");
        let baseline = dir.path().join("baseline.json");
        let options = EvalOptions {
            queries: Some(queries.clone()),
            json: Some(baseline.clone()),
            ..EvalOptions::default()
        };
        eval(&paths, &options).unwrap();

        // Same corpus: the gate passes even with zero tolerance from the config.
        fs::write(
            &paths.config,
            "queries: queries.jsonl\nmax_recall_drop: 0\n",
        )
        .unwrap();
        let options = EvalOptions {
            queries: Some(queries.clone()),
            gate: Some(baseline.clone()),
            ..EvalOptions::default()
        };
        assert!(eval(&paths, &options).unwrap().gate.unwrap().passed());

        // Adding the residue page lifts recall; removing the only page drops it to zero, which
        // fails the gate.
        let options = EvalOptions {
            queries: Some(queries.clone()),
            with: vec!["handbook::docs/user/quotas.md".into()],
            ..EvalOptions::default()
        };
        let outcome = eval(&paths, &options).unwrap();
        assert_eq!((outcome.page_count, outcome.searchable_count), (2, 2));
        let delta = outcome.delta.unwrap();
        assert!((delta.tuning.1.recall5 - 1.0).abs() < 1e-12);
        assert_eq!(delta.changed.len(), 1);
        let options = EvalOptions {
            queries: Some(queries.clone()),
            gate: Some(baseline),
            without: vec!["handbook::docs/user/README.md".into()],
            ..EvalOptions::default()
        };
        let outcome = eval(&paths, &options).unwrap();
        assert!(!outcome.gate.unwrap().passed());
        assert!(outcome.delta.unwrap().tuning.1.recall5.abs() < 1e-12);

        for (with, without) in [
            (vec!["handbook::docs/user/README.md".to_string()], vec![]),
            (vec!["handbook::nope.md".to_string()], vec![]),
            (vec![], vec!["handbook::nope.md".to_string()]),
        ] {
            let options = EvalOptions {
                queries: Some(queries.clone()),
                with,
                without,
                ..EvalOptions::default()
            };
            assert!(matches!(
                eval(&paths, &options).unwrap_err(),
                CommandError::Index(_)
            ));
        }
    }

    #[test]
    fn eval_backend_bm25_matches_the_plain_eval_path() {
        let (dir, paths) = eval_workspace();
        let queries = dir.path().join("queries.jsonl");
        let plain = eval(
            &paths,
            &EvalOptions {
                queries: Some(queries.clone()),
                ..EvalOptions::default()
            },
        )
        .unwrap();
        let via_backend = eval_backend(
            &paths,
            &BackendEvalOptions {
                queries: Some(queries),
                backend: BackendSpec::builtin(BackendKind::Bm25),
                ..BackendEvalOptions::default()
            },
        )
        .unwrap();
        assert_eq!(via_backend.page_count, plain.page_count);
        assert_eq!(via_backend.searchable_count, plain.searchable_count);
        assert_eq!(via_backend.summary.tuning, plain.summary.tuning);
        assert_eq!(via_backend.summary.backend, "bm25");
        assert_eq!(plain.summary.backend, "", "the legacy path never sets it");
    }

    #[test]
    fn eval_backend_rejects_with_without_for_non_bm25_backends() {
        let (dir, paths) = eval_workspace();
        let options = BackendEvalOptions {
            queries: Some(dir.path().join("queries.jsonl")),
            backend: BackendSpec::builtin(BackendKind::Bm25Tantivy),
            with: vec!["handbook::docs/user/quotas.md".to_string()],
            ..BackendEvalOptions::default()
        };
        assert!(matches!(
            eval_backend(&paths, &options).unwrap_err(),
            CommandError::Backend(BackendError::UnsupportedAdjustment(_))
        ));
    }

    #[test]
    fn eval_compare_runs_every_backend_over_the_same_queries() {
        let (dir, paths) = eval_workspace();
        let common = BackendEvalOptions {
            queries: Some(dir.path().join("queries.jsonl")),
            ..BackendEvalOptions::default()
        };
        let results = eval_compare(
            &paths,
            &[
                BackendSpec::builtin(BackendKind::Bm25),
                BackendSpec::builtin(BackendKind::Bm25Tantivy),
            ],
            &common,
        )
        .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.kind, BackendKind::Bm25);
        assert_eq!(results[0].1.summary.backend, "bm25");
        assert_eq!(results[1].0.kind, BackendKind::Bm25Tantivy);
        assert_eq!(results[1].1.summary.backend, "bm25-tantivy");
        assert!(matches!(
            eval_compare(&paths, &[], &common).unwrap_err(),
            CommandError::EmptyCompare
        ));
    }

    #[test]
    fn eval_backend_dense_uses_the_injected_embedder() {
        let (dir, paths) = eval_workspace();
        let embed_options = EmbedOptions {
            model: "fake".to_string(),
            batch: 64,
            out: None,
        };
        embed(&paths, &embed_options, fake_embedder().as_ref()).unwrap();
        let options = BackendEvalOptions {
            queries: Some(dir.path().join("queries.jsonl")),
            backend: BackendSpec::builtin(BackendKind::Dense),
            embedder: Some(fake_embedder()),
            ..BackendEvalOptions::default()
        };
        let outcome = eval_backend(&paths, &options).unwrap();
        assert_eq!(outcome.summary.backend, "dense");
        assert_eq!(outcome.page_count, 1);
    }

    #[test]
    fn eval_backend_records_a_configured_name_and_uses_its_embeddings() {
        let (dir, paths) = eval_workspace();
        let embed_options = EmbedOptions {
            model: "fake".to_string(),
            batch: 64,
            out: Some(dir.path().join("v2/embeddings.bin")),
        };
        fs::create_dir_all(dir.path().join("v2")).unwrap();
        embed(&paths, &embed_options, fake_embedder().as_ref()).unwrap();
        assert!(!paths.embeddings.exists(), "only the named file exists");
        let options = BackendEvalOptions {
            queries: Some(dir.path().join("queries.jsonl")),
            backend: BackendSpec {
                name: "vectors-v2".to_string(),
                kind: BackendKind::Dense,
                url: None,
                embeddings: Some(dir.path().join("v2/embeddings.bin")),
            },
            embedder: Some(fake_embedder()),
            ..BackendEvalOptions::default()
        };
        let outcome = eval_backend(&paths, &options).unwrap();
        assert_eq!(outcome.summary.backend, "vectors-v2");
        assert_eq!(outcome.page_count, 1);
    }

    #[test]
    fn eval_reads_the_eval_block_and_priorities_of_a_pinakes_yaml() {
        let (dir, _) = eval_workspace();
        let pinakes = dir.path().join("pinakes.yaml");
        fs::write(
            &pinakes,
            "version: 1\nsources:\n  - name: handbook\n    repo: https://github.com/o/handbook.git\n    \
             ref: main\n    priority: 7\n    resolver:\n      type: glob\n      include: ['**/*.md']\n\
             eval:\n  queries: queries.jsonl\n  k: 4\n",
        )
        .unwrap();
        let paths = Paths::for_config(&pinakes);
        let settings = super::super::settings(&paths).unwrap();
        assert_eq!(settings.priorities.of("handbook"), 7);
        let outcome = eval(&paths, &EvalOptions::default()).unwrap();
        assert_eq!(outcome.k, 4);
        assert_eq!(outcome.page_count, 1);
    }

    /// `--backend NAME` alone.
    fn named(backend: &str) -> BackendFlags {
        BackendFlags {
            backend: Some(backend.to_string()),
            ..BackendFlags::default()
        }
    }

    // -----------------------------------------------------------------------------------------
    // BackendFlags / apply_backend_config_defaults
    // -----------------------------------------------------------------------------------------

    #[test]
    fn backend_flags_spec_defaults_to_bm25_and_rejects_an_unknown_name() {
        assert_eq!(
            BackendFlags::default().spec().unwrap(),
            BackendSpec::default()
        );
        assert_eq!(named("hybrid").spec().unwrap().kind, BackendKind::Hybrid);
        assert!(matches!(
            named("nope").spec().unwrap_err(),
            BackendError::UnknownBackend(UnknownBackend(name)) if name == "nope"
        ));
    }

    /// The config's `backends:` as the flags carry them after the defaults are applied.
    fn configured() -> BackendFlags {
        let entry = |kind: BackendKind, url: Option<&str>, embeddings: Option<&str>| NamedBackend {
            kind,
            url: url.map(str::to_string),
            embeddings: embeddings.map(PathBuf::from),
        };
        BackendFlags {
            backend_url: Some("https://flag.test".to_string()),
            embeddings: Some(PathBuf::from("/flag/embeddings.bin")),
            backends: [
                (
                    "old",
                    entry(BackendKind::External, Some("https://old.test"), None),
                ),
                (
                    "v2",
                    entry(BackendKind::Dense, None, Some("/cfg/v2/embeddings.bin")),
                ),
                ("fused", entry(BackendKind::Hybrid, None, None)),
            ]
            .into_iter()
            .map(|(name, backend)| (name.to_string(), backend))
            .collect(),
            ..BackendFlags::default()
        }
    }

    #[test]
    fn resolve_gives_a_built_in_kind_the_flags_and_a_configured_name_its_own_settings() {
        let flags = configured();
        assert_eq!(
            flags.resolve("external").unwrap(),
            BackendSpec {
                name: "external".to_string(),
                kind: BackendKind::External,
                url: Some("https://flag.test".to_string()),
                embeddings: Some(PathBuf::from("/flag/embeddings.bin")),
            }
        );
        assert_eq!(
            flags.resolve("old").unwrap(),
            BackendSpec {
                name: "old".to_string(),
                kind: BackendKind::External,
                url: Some("https://old.test".to_string()),
                embeddings: Some(PathBuf::from("/flag/embeddings.bin")),
            },
            "--backend-url applies to the built-in external only"
        );
        let v2 = flags.resolve("v2").unwrap();
        assert_eq!((v2.name.as_str(), v2.kind), ("v2", BackendKind::Dense));
        assert_eq!(
            v2.embeddings.as_deref(),
            Some(Path::new("/cfg/v2/embeddings.bin")),
            "the name's own embeddings win"
        );
        let fused = flags.resolve("fused").unwrap();
        assert_eq!(fused.kind, BackendKind::Hybrid);
        assert_eq!(
            fused.embeddings.as_deref(),
            Some(Path::new("/flag/embeddings.bin")),
            "a name without embeddings falls back to --embeddings"
        );
        assert!(matches!(
            flags.resolve("nope").unwrap_err(),
            BackendError::UnknownBackend(UnknownBackend(name)) if name == "nope"
        ));
        assert!(!flags.is_empty());
        assert!(
            BackendFlags {
                backends: flags.backends.clone(),
                ..BackendFlags::default()
            }
            .is_empty(),
            "named backends alone do not select the backend path"
        );
    }

    #[test]
    fn apply_backend_config_defaults_fills_what_the_flags_leave_unset() {
        let (dir, paths) = eval_workspace();
        let mut flags = BackendFlags::default();
        apply_backend_config_defaults(&paths, &mut flags).unwrap();
        assert!(flags.is_empty(), "no config file, nothing to fill");

        fs::write(
            &paths.config,
            "queries: queries.jsonl\nbackend: external\nbackend_url: https://example.test\n\
             embeddings: custom/embeddings.bin\n",
        )
        .unwrap();
        let mut flags = named("bm25-tantivy");
        apply_backend_config_defaults(&paths, &mut flags).unwrap();
        assert_eq!(
            flags,
            BackendFlags {
                backend: Some("bm25-tantivy".to_string()),
                backend_url: Some("https://example.test".to_string()),
                embeddings: Some(dir.path().join("custom/embeddings.bin")),
                allow_stale: false,
                backends: BTreeMap::new(),
            },
            "the flag wins, the config fills the rest"
        );
    }

    #[test]
    fn apply_backend_config_defaults_carries_named_backends_with_paths_relative_to_the_config() {
        let (dir, paths) = eval_workspace();
        fs::write(
            &paths.config,
            "queries: queries.jsonl\nbackend: old\nbackends:\n  old:\n    type: external\n    \
             url: https://old.test\n  v2:\n    type: dense\n    embeddings: v2/embeddings.bin\n",
        )
        .unwrap();
        let mut flags = BackendFlags::default();
        apply_backend_config_defaults(&paths, &mut flags).unwrap();
        assert_eq!(flags.backend.as_deref(), Some("old"));
        assert_eq!(flags.backends.len(), 2);
        assert_eq!(
            flags.backends["v2"].embeddings.as_deref(),
            Some(dir.path().join("v2/embeddings.bin").as_path())
        );
        let spec = flags.spec().unwrap();
        assert_eq!(spec.name, "old");
        assert_eq!(spec.kind, BackendKind::External);
        assert_eq!(spec.url.as_deref(), Some("https://old.test"));
        // A config with a bad entry is refused before anything runs.
        fs::write(
            &paths.config,
            "queries: queries.jsonl\nbackends:\n  old:\n    type: external\n",
        )
        .unwrap();
        let err = apply_backend_config_defaults(&paths, &mut BackendFlags::default()).unwrap_err();
        assert!(
            err.to_string().contains("type external needs a url"),
            "{err}"
        );
    }

    // -----------------------------------------------------------------------------------------
    // apply_eval_config_defaults
    // -----------------------------------------------------------------------------------------

    #[test]
    fn apply_eval_config_defaults_is_a_noop_without_a_config_file() {
        let (_dir, paths) = eval_workspace();
        assert!(!paths.config.is_file());
        let mut flags = EvalFlags::default();
        apply_eval_config_defaults(&paths, &mut flags).unwrap();
        assert!(flags.backend.is_empty() && flags.compare.is_empty());
    }

    #[test]
    fn apply_eval_config_defaults_fills_unset_flags_and_resolves_relative_embeddings() {
        let (dir, paths) = eval_workspace();
        fs::write(
            &paths.config,
            "queries: queries.jsonl\nbackend: dense\nbackend_url: https://example.test\n\
             embeddings: custom/embeddings.bin\n",
        )
        .unwrap();
        let mut flags = EvalFlags::default();
        apply_eval_config_defaults(&paths, &mut flags).unwrap();
        assert_eq!(
            flags.backend,
            BackendFlags {
                backend: Some("dense".to_string()),
                backend_url: Some("https://example.test".to_string()),
                embeddings: Some(dir.path().join("custom/embeddings.bin")),
                allow_stale: false,
                backends: BTreeMap::new(),
            }
        );
    }

    #[test]
    fn apply_eval_config_defaults_keeps_an_already_set_flag() {
        let (_dir, paths) = eval_workspace();
        fs::write(
            &paths.config,
            "queries: queries.jsonl\nbackend: bm25-tantivy\n",
        )
        .unwrap();
        let mut flags = EvalFlags {
            backend: named("dense"),
            ..EvalFlags::default()
        };
        apply_eval_config_defaults(&paths, &mut flags).unwrap();
        assert_eq!(
            flags.backend.backend.as_deref(),
            Some("dense"),
            "the flag wins"
        );
    }

    #[test]
    fn apply_eval_config_defaults_fills_compare_only_for_a_bare_eval() {
        let (_dir, paths) = eval_workspace();
        fs::write(
            &paths.config,
            "queries: queries.jsonl\ncompare: [bm25, dense]\n",
        )
        .unwrap();

        let mut bare = EvalFlags::default();
        apply_eval_config_defaults(&paths, &mut bare).unwrap();
        assert_eq!(bare.compare, vec!["bm25".to_string(), "dense".to_string()]);

        let mut with_backend_flag = EvalFlags {
            backend: named("bm25"),
            ..EvalFlags::default()
        };
        apply_eval_config_defaults(&paths, &mut with_backend_flag).unwrap();
        assert!(
            with_backend_flag.compare.is_empty(),
            "a configured compare is ignored once --backend is given"
        );
    }

    // -----------------------------------------------------------------------------------------
    // eval_plan
    // -----------------------------------------------------------------------------------------

    #[test]
    fn eval_plan_is_plain_with_no_backend_selecting_flag() {
        assert!(matches!(
            eval_plan(EvalFlags::default()).unwrap(),
            EvalPlan::Plain(_)
        ));
    }

    #[test]
    fn eval_plan_selects_the_named_backend() {
        let flags = EvalFlags {
            backend: named("bm25-tantivy"),
            ..EvalFlags::default()
        };
        match eval_plan(flags).unwrap() {
            EvalPlan::Backend(options) => {
                assert_eq!(options.backend.kind, BackendKind::Bm25Tantivy);
            }
            _ => panic!("expected Backend, got a different plan"),
        }
    }

    #[test]
    fn eval_plan_resolves_configured_names_in_backend_and_compare() {
        let flags = EvalFlags {
            backend: BackendFlags {
                backend: Some("old".to_string()),
                ..configured()
            },
            ..EvalFlags::default()
        };
        match eval_plan(flags).unwrap() {
            EvalPlan::Backend(options) => {
                assert_eq!(options.backend.name, "old");
                assert_eq!(options.backend.kind, BackendKind::External);
                assert_eq!(options.backend.url.as_deref(), Some("https://old.test"));
            }
            _ => panic!("expected Backend"),
        }
        let flags = EvalFlags {
            compare: vec!["old".to_string(), " external".to_string(), "v2".to_string()],
            backend: configured(),
            ..EvalFlags::default()
        };
        match eval_plan(flags).unwrap() {
            EvalPlan::Compare { backends, .. } => {
                let names: Vec<&str> = backends.iter().map(|b| b.name.as_str()).collect();
                assert_eq!(names, ["old", "external", "v2"]);
                assert_eq!(backends[0].url.as_deref(), Some("https://old.test"));
                assert_eq!(backends[1].url.as_deref(), Some("https://flag.test"));
                assert!(backends.iter().any(BackendSpec::needs_embedder));
            }
            _ => panic!("expected Compare"),
        }
        let flags = EvalFlags {
            compare: vec!["old".to_string(), "nope".to_string()],
            backend: configured(),
            ..EvalFlags::default()
        };
        match eval_plan(flags) {
            Err(CommandError::Backend(BackendError::UnknownBackend(UnknownBackend(name)))) => {
                assert_eq!(name, "nope");
            }
            _ => panic!("expected an unknown-backend error"),
        }
    }

    #[test]
    fn eval_plan_allow_stale_alone_still_selects_bm25_through_the_backend_path() {
        let flags = EvalFlags {
            backend: BackendFlags {
                allow_stale: true,
                ..BackendFlags::default()
            },
            ..EvalFlags::default()
        };
        match eval_plan(flags).unwrap() {
            EvalPlan::Backend(options) => {
                assert_eq!(options.backend, BackendSpec::default());
                assert!(options.allow_stale);
            }
            _ => panic!("--allow-stale alone must still route through eval_backend"),
        }
    }

    #[test]
    fn eval_plan_compare_wins_over_a_backend_flag() {
        let flags = EvalFlags {
            compare: vec!["bm25".to_string(), "dense".to_string()],
            backend: named("bm25-tantivy"),
            ..EvalFlags::default()
        };
        match eval_plan(flags).unwrap() {
            EvalPlan::Compare { backends, .. } => {
                assert_eq!(
                    backends,
                    vec![
                        BackendSpec::builtin(BackendKind::Bm25),
                        BackendSpec::builtin(BackendKind::Dense)
                    ]
                );
            }
            _ => panic!("--compare must win over --backend"),
        }
    }

    #[test]
    fn eval_plan_rejects_an_unknown_backend_name() {
        // `EvalPlan` holds an `Rc<dyn Embedder>` (via `BackendEvalOptions`), which is not
        // `Debug`, so match manually instead of `unwrap_err()`.
        let flags = EvalFlags {
            backend: named("nope"),
            ..EvalFlags::default()
        };
        match eval_plan(flags) {
            Err(CommandError::Backend(BackendError::UnknownBackend(_))) => {}
            _ => panic!("expected an unknown-backend error"),
        }
        let flags = EvalFlags {
            compare: vec!["nope".to_string()],
            ..EvalFlags::default()
        };
        match eval_plan(flags) {
            Err(CommandError::Backend(BackendError::UnknownBackend(_))) => {}
            _ => panic!("expected an unknown-backend error"),
        }
    }

    #[test]
    fn eval_plan_needs_embedder_only_for_dense_or_hybrid() {
        assert!(!eval_plan(EvalFlags::default()).unwrap().needs_embedder());
        let bm25 = EvalFlags {
            backend: named("bm25"),
            ..EvalFlags::default()
        };
        assert!(!eval_plan(bm25).unwrap().needs_embedder());
        let dense = EvalFlags {
            backend: named("dense"),
            ..EvalFlags::default()
        };
        assert!(eval_plan(dense).unwrap().needs_embedder());
        let compare_with_hybrid = EvalFlags {
            compare: vec!["bm25".to_string(), "hybrid".to_string()],
            ..EvalFlags::default()
        };
        assert!(eval_plan(compare_with_hybrid).unwrap().needs_embedder());
        let compare_without = EvalFlags {
            compare: vec!["bm25".to_string(), "bm25-tantivy".to_string()],
            ..EvalFlags::default()
        };
        assert!(!eval_plan(compare_without).unwrap().needs_embedder());
    }

    #[test]
    fn eval_plan_with_embedder_attaches_to_backend_and_compare_but_not_plain() {
        let plain = eval_plan(EvalFlags::default())
            .unwrap()
            .with_embedder(fake_embedder());
        assert!(matches!(plain, EvalPlan::Plain(_)));

        let dense = EvalFlags {
            backend: named("dense"),
            ..EvalFlags::default()
        };
        match eval_plan(dense).unwrap().with_embedder(fake_embedder()) {
            EvalPlan::Backend(options) => assert!(options.embedder.is_some()),
            _ => panic!("expected Backend"),
        }

        let compare = EvalFlags {
            compare: vec!["dense".to_string()],
            ..EvalFlags::default()
        };
        match eval_plan(compare).unwrap().with_embedder(fake_embedder()) {
            EvalPlan::Compare { common, .. } => assert!(common.embedder.is_some()),
            _ => panic!("expected Compare"),
        }
    }

    // -----------------------------------------------------------------------------------------
    // eval_embedder_from_env
    // -----------------------------------------------------------------------------------------

    #[test]
    fn eval_embedder_from_env_reports_the_exact_original_wording_when_unset() {
        let _guard = crate::testing::ENV_LOCK.lock().unwrap();
        // SAFETY: serialised by ENV_LOCK; no other test observes these vars concurrently.
        unsafe {
            std::env::remove_var("KANON_EMBED_URL");
            std::env::remove_var("PINAKES_EMBED_URL");
        }
        // `Rc<dyn Embedder>` is not `Debug`, so match manually instead of `unwrap_err()`.
        let Err(err) = eval_embedder_from_env() else {
            panic!("expected an error")
        };
        assert_eq!(
            err.to_string(),
            "KANON_EMBED_URL is not set (needed for --backend dense/hybrid)"
        );
    }
}
