use std::path::{Path, PathBuf};

use crate::error::{CommandError, io_err};
use crate::eval;
use crate::grade::GradedRow;
use crate::queries::suggest::{self, Suggestion};
use crate::queries::{self, CheckReport, GradedQuery, NewQuery};
use crate::workspace::Paths;
use pinakes::index;
use pinakes::jsonl;
use pinakes::llm::ChatTransport;
use pinakes::manifest::Manifest;

use super::eval::resolve_queries_path;
use super::settings;

/// Options for `queries add`.
#[derive(Debug, Clone, Default)]
pub struct QueriesAddOptions {
    /// Query file (default: `queries` from the config, relative to it).
    pub queries: Option<PathBuf>,
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
}

/// Run `queries add`: append a row after checking `expected` against the committed manifest.
pub fn queries_add(
    paths: &Paths,
    options: &QueriesAddOptions,
) -> Result<eval::Query, CommandError> {
    let settings = settings(paths)?;
    let eval_config = settings.config.as_ref();
    let queries_path = resolve_queries_path(paths, eval_config, options.queries.as_deref())?;
    let manifest = Manifest::load(&paths.manifest)?;
    let new = NewQuery {
        id: options.id.clone(),
        query: options.query.clone(),
        expected: options.expected.clone(),
        kind: options.kind.clone(),
        holdout: options.holdout,
        origin: None,
    };
    Ok(queries::add(&queries_path, &manifest, &new)?)
}

/// Options for `queries add --from`: accepting rows of a suggestions file.
#[derive(Debug, Clone)]
pub struct QueriesAcceptOptions {
    /// Query file (default: `queries` from the config, relative to it).
    pub queries: Option<PathBuf>,
    /// The suggestions file written by `queries suggest`.
    pub from: PathBuf,
    /// Ids of the rows to accept, in this order; ignored with `accept_all`.
    pub accept: Vec<String>,
    /// Accept every row, in file order.
    pub accept_all: bool,
    /// Hold the accepted rows out of tuning decisions.
    pub holdout: bool,
}

/// Run `queries add --from`: append the chosen suggestions through the same check as a
/// hand-written row, keeping their `origin`. Nothing is written when an id is not in the file,
/// is already in the query file, or an expected id is not in the manifest.
pub fn queries_accept(
    paths: &Paths,
    options: &QueriesAcceptOptions,
) -> Result<Vec<eval::Query>, CommandError> {
    let settings = settings(paths)?;
    let eval_config = settings.config.as_ref();
    let queries_path = resolve_queries_path(paths, eval_config, options.queries.as_deref())?;
    let manifest = Manifest::load(&paths.manifest)?;
    let rows = suggest::load(&options.from)?;
    let chosen: Vec<&Suggestion> = if options.accept_all {
        rows.iter().collect()
    } else {
        let mut chosen = Vec::new();
        for id in &options.accept {
            let row = rows
                .iter()
                .find(|row| &row.id == id)
                .ok_or_else(|| queries::QueriesError::UnknownSuggestion(id.clone()))?;
            if !chosen.iter().any(|seen: &&Suggestion| seen.id == row.id) {
                chosen.push(row);
            }
        }
        chosen
    };
    let existing = if queries_path.is_file() {
        eval::load_queries(&queries_path)?
    } else {
        Vec::new()
    };
    if let Some(row) = chosen
        .iter()
        .find(|row| existing.iter().any(|query| query.id == row.id))
    {
        return Err(queries::QueriesError::DuplicateId(row.id.clone()).into());
    }
    let new: Vec<NewQuery> = chosen
        .into_iter()
        .map(|row| NewQuery {
            id: row.id.clone(),
            query: row.query.clone(),
            expected: row.expected.clone(),
            kind: row.kind.clone(),
            holdout: options.holdout,
            origin: (!row.origin.is_empty()).then(|| row.origin.clone()),
        })
        .collect();
    Ok(queries::add_all(&queries_path, &manifest, &new)?)
}

/// Options for `queries suggest`.
#[derive(Debug, Clone)]
pub struct QueriesSuggestOptions {
    /// Pages to sample (`--n`).
    pub n: usize,
    /// Most pages any one source contributes (`--per-source`).
    pub per_source: Option<usize>,
    /// The suggestions file to write (`--out`); replaced when it exists.
    pub out: PathBuf,
    /// `--model`; falls back to `KANON_LLM_MODEL`.
    pub model: Option<String>,
    /// Seed for the deterministic sampling (`--seed`).
    pub seed: u64,
}

/// What `queries suggest` produced.
#[derive(Debug)]
pub struct QueriesSuggestOutcome {
    /// The rows written to the suggestions file.
    pub suggestions: Vec<Suggestion>,
    /// Pages sampled and sent to the model.
    pub pages: usize,
    /// Model answers dropped for quoting the page title.
    pub title_quotes: usize,
    /// Model answers dropped for being empty.
    pub empty: usize,
    /// Model answers dropped as a repeat of a query already suggested.
    pub repeats: usize,
    /// Pages skipped because the reply was not the expected JSON.
    pub failed: usize,
}

/// Run `queries suggest`: sample pages of the artifact, ask the model for the questions each
/// one answers and write them to the suggestions file, never to `queries.jsonl`. The endpoint
/// and the output directory are checked before the first model call.
pub fn queries_suggest(
    paths: &Paths,
    options: &QueriesSuggestOptions,
    transport: &dyn ChatTransport,
) -> Result<QueriesSuggestOutcome, CommandError> {
    let config = crate::llm::config_from_env(options.model.clone())?;
    if let Some(dir) = options
        .out
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
    {
        std::fs::create_dir_all(dir).map_err(io_err(dir))?;
    }
    let priorities = settings(paths)?.priorities;
    let mut pages = index::load_pages(&paths.artifact, &priorities)?;
    index::mark_mirrors(&mut pages);
    let sampled = suggest::sample(&pages, options.n, options.per_source, options.seed);
    let suggested = suggest::suggest(transport, &config, &sampled)?;
    suggest::write(&options.out, &suggested.rows)?;
    Ok(QueriesSuggestOutcome {
        suggestions: suggested.rows,
        pages: sampled.len(),
        title_quotes: suggested.title_quotes,
        empty: suggested.empty,
        repeats: suggested.repeats,
        failed: suggested.failed,
    })
}

/// Run `queries check`: validate `queries.jsonl` against the committed manifest.
pub fn queries_check(
    paths: &Paths,
    queries_override: Option<&Path>,
) -> Result<CheckReport, CommandError> {
    let settings = settings(paths)?;
    let eval_config = settings.config.as_ref();
    let queries_path = resolve_queries_path(paths, eval_config, queries_override)?;
    let manifest = Manifest::load(&paths.manifest)?;
    let rows = eval::load_queries(&queries_path)?;
    let holdout_min = eval_config.map_or(queries::DEFAULT_HOLDOUT_MIN, |e| e.holdout_min);
    Ok(queries::check(&rows, &manifest, holdout_min))
}

/// Options for `queries import`.
#[derive(Debug, Clone)]
pub struct QueriesImportOptions {
    /// Query file to append to (default: `queries` from the config, relative to it).
    pub queries: Option<PathBuf>,
    /// `graded.jsonl` to read.
    pub graded: PathBuf,
    /// Minimum grade for an id to enter `expected`.
    pub min_grade: u8,
    /// Target held-out share for the random assignment.
    pub holdout_share: f64,
    /// Seed for the deterministic PRNG that assigns `holdout`.
    pub seed: u64,
}

/// What `queries import` produced.
#[derive(Debug)]
pub struct QueriesImportOutcome {
    /// Rows appended to the query file.
    pub imported: Vec<GradedQuery>,
    /// Query texts with no candidate at or above `min_grade` (not appended).
    pub skipped: Vec<String>,
}

/// Run `queries import`: turn `grade`'s output into query rows.
pub fn queries_import(
    paths: &Paths,
    options: &QueriesImportOptions,
) -> Result<QueriesImportOutcome, CommandError> {
    let settings = settings(paths)?;
    let eval_config = settings.config.as_ref();
    let queries_path = resolve_queries_path(paths, eval_config, options.queries.as_deref())?;
    let graded_text = std::fs::read_to_string(&options.graded).map_err(io_err(&options.graded))?;
    let graded: Vec<GradedRow> = jsonl::parse(&graded_text).map_err(|err| err.source)?;
    let (imported, skipped) = queries::import_graded(
        &graded,
        options.min_grade,
        options.holdout_share,
        options.seed,
    );
    queries::append_graded(&queries_path, &imported)?;
    Ok(QueriesImportOutcome { imported, skipped })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::testing::{eval_workspace, manifest_workspace};

    #[test]
    fn queries_add_and_check_validate_against_the_manifest() {
        let (dir, paths) = manifest_workspace();
        let queries_path = dir.path().join("queries.jsonl");

        let add_options = QueriesAddOptions {
            queries: Some(queries_path.clone()),
            id: "a".to_string(),
            query: "what is a".to_string(),
            expected: vec!["handbook/docs/a.md".to_string()],
            kind: "howto".to_string(),
            holdout: false,
        };
        let query = queries_add(&paths, &add_options).unwrap();
        assert_eq!(query.expected, ["handbook::docs/a.md"]);

        // An unknown expected id is rejected and nothing is appended.
        let bad = QueriesAddOptions {
            id: "bad".to_string(),
            expected: vec!["handbook::nope.md".to_string()],
            ..add_options.clone()
        };
        assert!(matches!(
            queries_add(&paths, &bad).unwrap_err(),
            CommandError::Queries(_)
        ));
        assert_eq!(eval::load_queries(&queries_path).unwrap().len(), 1);

        // A held-out row so the share check has something to pass.
        let held = QueriesAddOptions {
            id: "b".to_string(),
            query: "what is b".to_string(),
            expected: vec!["handbook::docs/b.md".to_string()],
            holdout: true,
            ..add_options
        };
        queries_add(&paths, &held).unwrap();

        let report = queries_check(&paths, Some(&queries_path)).unwrap();
        assert!(report.ok(), "{report:?}");
        assert!((report.holdout_share - 0.5).abs() < 1e-12);

        // Below the configured minimum: same rows, a stricter holdout_min.
        fs::write(&paths.config, "queries: queries.jsonl\nholdout_min: 0.6\n").unwrap();
        let report = queries_check(&paths, Some(&queries_path)).unwrap();
        assert!(!report.ok());
        assert!((report.holdout_min - 0.6).abs() < 1e-12);
        fs::remove_file(&paths.config).unwrap();

        // A raw duplicate id and an unknown expected id, appended straight to the file.
        let mut text = fs::read_to_string(&queries_path).unwrap();
        text.push_str(
            "{\"id\": \"a\", \"query\": \"dup\", \"expected\": [\"handbook::nope.md\"]}\n",
        );
        fs::write(&queries_path, text).unwrap();
        let report = queries_check(&paths, Some(&queries_path)).unwrap();
        assert!(!report.ok());
        assert_eq!(report.duplicate_ids, ["a"]);
        assert_eq!(
            report.unknown,
            [("a".to_string(), "handbook::nope.md".to_string())]
        );

        assert!(matches!(
            queries_check(&paths, None).unwrap_err(),
            CommandError::NoQueries
        ));
    }

    #[test]
    fn queries_import_appends_rows_from_graded_jsonl() {
        let (dir, paths) = manifest_workspace();
        let graded_path = dir.path().join("graded.jsonl");
        fs::write(
            &graded_path,
            "{\"query\": \"a\", \"id\": \"handbook::docs/a.md\", \"grade\": 3, \
             \"model\": \"m\", \"at\": \"t\"}\n",
        )
        .unwrap();
        let queries_path = dir.path().join("queries.jsonl");
        let options = QueriesImportOptions {
            queries: Some(queries_path.clone()),
            graded: graded_path,
            min_grade: 2,
            holdout_share: 0.0,
            seed: 0,
        };
        let outcome = queries_import(&paths, &options).unwrap();
        assert_eq!(outcome.imported.len(), 1);
        assert!(outcome.skipped.is_empty());
        let loaded = eval::load_queries(&queries_path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].expected, ["handbook::docs/a.md"]);
    }

    #[test]
    fn queries_suggest_samples_pages_asks_the_model_and_writes_the_suggestions_file() {
        use crate::testing::with_llm_url;
        use pinakes::llm::testing::{Scripted, ScriptedTransport, completion};

        let (dir, paths) = eval_workspace();
        // A directory that does not exist yet: created before the first model call.
        let out = dir.path().join("scratch").join("suggestions.jsonl");
        let options = QueriesSuggestOptions {
            n: 10,
            per_source: None,
            out: out.clone(),
            model: Some("suggest-model".to_string()),
            seed: 0,
        };

        // Without an endpoint the command fails before touching the artifact or the file.
        {
            let _guard = crate::testing::ENV_LOCK.lock().unwrap();
            // SAFETY: serialised by ENV_LOCK; no other test observes these vars concurrently.
            unsafe {
                std::env::remove_var("KANON_LLM_URL");
                std::env::remove_var("PINAKES_LLM_URL");
            }
            let transport = ScriptedTransport::new(vec![]);
            let err = queries_suggest(&paths, &options, &transport).unwrap_err();
            assert!(matches!(err, CommandError::Llm(_)), "{err}");
            assert!(err.to_string().contains("queries suggest"), "{err}");
            assert!(!out.parent().unwrap().exists());
        }

        with_llm_url(|| {
            // The one searchable page ("Storage Module"): two kept, a title quote, an empty
            // query and a repeat rejected.
            let reply = completion(
                &serde_json::to_string(&serde_json::json!([
                    {"query": "how do I cache uploads", "kind": "howto"},
                    {"query": "bucket label for caching", "kind": "reference"},
                    {"query": "what is the storage module", "kind": "concept"},
                    {"query": "", "kind": "howto"},
                    {"query": "how do I cache uploads", "kind": "howto"},
                ]))
                .unwrap(),
            );
            let transport = ScriptedTransport::new(vec![Scripted::Ok(reply)]);
            let outcome = queries_suggest(&paths, &options, &transport).unwrap();
            assert_eq!(transport.requests.lock().unwrap().len(), 1);
            assert_eq!(
                transport.requests.lock().unwrap()[0]["model"],
                "suggest-model"
            );
            assert_eq!(outcome.pages, 1);
            assert_eq!(
                (
                    outcome.title_quotes,
                    outcome.empty,
                    outcome.repeats,
                    outcome.failed
                ),
                (1, 1, 1, 0)
            );
            assert_eq!(outcome.suggestions.len(), 2);

            let written = suggest::load(&out).unwrap();
            assert_eq!(written, outcome.suggestions);
            assert_eq!(written[0].query, "how do I cache uploads");
            assert_eq!(written[0].id, queries::query_id("how do I cache uploads"));
            assert_eq!(written[0].expected, ["handbook::docs/user/README.md"]);
            assert_eq!(written[0].kind, "howto");
            assert_eq!(written[0].origin, "suggested");
            assert_eq!(written[0].page_title, "Storage Module");
            // queries.jsonl is untouched.
            assert_eq!(
                eval::load_queries(&paths.config_dir().join("queries.jsonl"))
                    .unwrap()
                    .len(),
                2
            );
        });
    }

    #[test]
    fn queries_accept_validates_against_the_manifest_and_refuses_unknown_ids() {
        let (dir, paths) = manifest_workspace();
        let from = dir.path().join("suggestions.jsonl");
        fs::write(
            &from,
            "{\"id\": \"what-is-a-1\", \"query\": \"what is a\", \"expected\": [\"handbook::docs/a.md\"], \
             \"kind\": \"concept\", \"origin\": \"suggested\", \"page_title\": \"A\"}\n\
             {\"id\": \"what-is-b-2\", \"query\": \"what is b\", \"expected\": [\"handbook::docs/b.md\"], \
             \"kind\": \"concept\", \"origin\": \"suggested\", \"page_title\": \"B\"}\n\
             {\"id\": \"gone-3\", \"query\": \"gone\", \"expected\": [\"handbook::docs/gone.md\"], \
             \"kind\": \"howto\", \"origin\": \"suggested\", \"page_title\": \"Gone\"}\n",
        )
        .unwrap();
        let queries_path = dir.path().join("queries.jsonl");
        let options = QueriesAcceptOptions {
            queries: Some(queries_path.clone()),
            from: from.clone(),
            accept: vec!["what-is-a-1".to_string()],
            accept_all: false,
            holdout: false,
        };

        let added = queries_accept(&paths, &options).unwrap();
        assert_eq!(added.len(), 1);
        assert_eq!(added[0].id, "what-is-a-1");
        assert_eq!(added[0].origin.as_deref(), Some("suggested"));
        let loaded = eval::load_queries(&queries_path).unwrap();
        assert_eq!(loaded, added);
        let text = fs::read_to_string(&queries_path).unwrap();
        assert!(text.contains("\"origin\":\"suggested\""), "{text}");
        assert!(!text.contains("page_title"), "{text}");

        // An id that is not in the suggestions file: nothing appended.
        let unknown = QueriesAcceptOptions {
            accept: vec!["what-is-b-2".to_string(), "nope".to_string()],
            ..options.clone()
        };
        let err = queries_accept(&paths, &unknown).unwrap_err();
        assert!(
            matches!(&err, CommandError::Queries(queries::QueriesError::UnknownSuggestion(id)) if id == "nope"),
            "{err}"
        );
        assert_eq!(eval::load_queries(&queries_path).unwrap().len(), 1);

        // An expected id the manifest no longer has: rejected, nothing appended.
        let stale = QueriesAcceptOptions {
            accept: vec!["gone-3".to_string()],
            ..options.clone()
        };
        let err = queries_accept(&paths, &stale).unwrap_err();
        assert!(
            matches!(&err, CommandError::Queries(queries::QueriesError::UnknownExpected(id)) if id == "handbook::docs/gone.md"),
            "{err}"
        );
        assert_eq!(eval::load_queries(&queries_path).unwrap().len(), 1);

        // --accept-all is all or nothing too: the stale row blocks the whole file.
        let all = QueriesAcceptOptions {
            accept: vec![],
            accept_all: true,
            ..options.clone()
        };
        assert!(queries_accept(&paths, &all).is_err());
        assert_eq!(eval::load_queries(&queries_path).unwrap().len(), 1);

        // An id already in the query file is refused before anything is appended, so
        // `queries check` never sees a duplicate.
        let again = QueriesAcceptOptions {
            accept: vec!["what-is-b-2".to_string(), "what-is-a-1".to_string()],
            ..options.clone()
        };
        let err = queries_accept(&paths, &again).unwrap_err();
        assert!(
            matches!(&err, CommandError::Queries(queries::QueriesError::DuplicateId(id)) if id == "what-is-a-1"),
            "{err}"
        );
        assert_eq!(eval::load_queries(&queries_path).unwrap().len(), 1);

        // Held out on request; a repeated id is appended once.
        let held = QueriesAcceptOptions {
            accept: vec!["what-is-b-2".to_string(), "what-is-b-2".to_string()],
            holdout: true,
            ..options
        };
        let added = queries_accept(&paths, &held).unwrap();
        assert_eq!(added.len(), 1);
        assert!(added[0].holdout);
        let report = queries_check(&paths, Some(&queries_path)).unwrap();
        assert!(report.ok(), "{report:?}");
    }
}
