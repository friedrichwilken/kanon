use std::path::PathBuf;

use crate::error::{CommandError, io_err};
use crate::grade::{self, GradedRow};
use crate::workspace::Paths;
use pinakes::index::Index;
use pinakes::llm::ChatTransport;
use pinakes::manifest::now_rfc3339;
use pinakes::trail;

use super::settings;

/// Options for `grade`.
#[derive(Debug, Clone)]
pub struct GradeOptions {
    /// `trail.jsonl` to replay.
    pub trail: PathBuf,
    /// `--backend` (only `"bm25"` is available; see [`grade::check_backend`]).
    pub backend: Option<String>,
    /// `--k` (default [`grade::DEFAULT_K`]).
    pub k: Option<usize>,
    /// `--model`; falls back to `KANON_LLM_MODEL`.
    pub model: Option<String>,
    /// Write the graded rows here instead of returning them for the caller to print.
    pub out: Option<PathBuf>,
}

/// What `grade` produced.
#[derive(Debug)]
pub struct GradeOutcome {
    /// One row per (query, candidate) the model graded.
    pub rows: Vec<GradedRow>,
    /// Distinct queries replayed.
    pub queries: usize,
}

/// Run `grade`: replay every distinct trail query against the backend and ask the model to
/// grade each candidate.
pub fn grade(
    paths: &Paths,
    options: &GradeOptions,
    transport: &dyn ChatTransport,
) -> Result<GradeOutcome, CommandError> {
    let backend = options.backend.as_deref().unwrap_or(grade::BM25_BACKEND);
    grade::check_backend(backend)?;
    let config = crate::llm::config_from_env(options.model.clone())?;
    let entries = trail::read_jsonl(&options.trail)?;
    let queries = grade::distinct_queries(&entries);
    let priorities = settings(paths)?.priorities;
    let index = Index::build(&paths.artifact, &priorities)?;
    let k = options.k.unwrap_or(grade::DEFAULT_K);
    let at = now_rfc3339();
    let mut rows = Vec::new();
    for query in &queries {
        let hits = grade::bm25_search(&index, query, k)?;
        rows.extend(grade::grade_query(
            transport, &config, &index, query, &hits, &at,
        )?);
    }
    if let Some(path) = &options.out {
        std::fs::write(path, grade::to_jsonl(&rows)?).map_err(io_err(path))?;
    }
    Ok(GradeOutcome {
        rows,
        queries: queries.len(),
    })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::grade::GradeError;
    use crate::testing::{eval_workspace, with_llm_url};

    #[test]
    fn grade_rejects_a_backend_other_than_bm25() {
        let (_dir, paths) = eval_workspace();
        let trail_path = paths.config_dir().join("trail.jsonl");
        fs::write(&trail_path, "").unwrap();
        let transport = pinakes::llm::testing::ScriptedTransport::new(vec![]);
        let options = GradeOptions {
            trail: trail_path,
            backend: Some("dense".to_string()),
            k: None,
            model: None,
            out: None,
        };
        let err = grade(&paths, &options, &transport).unwrap_err();
        assert!(
            matches!(&err, CommandError::Grade(GradeError::UnknownBackend(b)) if b == "dense"),
            "{err}"
        );
    }

    #[test]
    fn grade_replays_distinct_trail_queries_against_bm25_and_writes_graded_rows() {
        let (dir, paths) = eval_workspace();
        let artifact_page = paths.artifact.join("handbook/docs/caching.md");
        fs::create_dir_all(artifact_page.parent().unwrap()).unwrap();
        fs::write(
            &artifact_page,
            "# Caching\n\nEnable upload caching with a label on the bucket.\n",
        )
        .unwrap();
        let trail_path = dir.path().join("trail.jsonl");
        fs::write(
            &trail_path,
            "{\"at\": \"t\", \"query\": \"enable caching\", \"retrieved\": [], \"cited\": []}\n\
             {\"at\": \"t\", \"query\": \"enable caching\", \"retrieved\": [], \"cited\": []}\n",
        )
        .unwrap();

        with_llm_url(|| {
            let reply = pinakes::llm::testing::completion(
                &serde_json::to_string(&serde_json::json!([
                    {"id": "handbook::docs/caching.md", "grade": 3}
                ]))
                .unwrap(),
            );
            let transport = pinakes::llm::testing::ScriptedTransport::new(vec![
                pinakes::llm::testing::Scripted::Ok(reply),
            ]);
            let out_path = dir.path().join("graded.jsonl");
            let options = GradeOptions {
                trail: trail_path,
                backend: None,
                k: None,
                model: Some("grader".to_string()),
                out: Some(out_path.clone()),
            };
            let outcome = grade(&paths, &options, &transport).unwrap();
            // Only one request even though the query appears twice in the trail.
            assert_eq!(transport.requests.lock().unwrap().len(), 1);
            assert_eq!(outcome.queries, 1);
            assert_eq!(outcome.rows.len(), 1);
            assert_eq!(outcome.rows[0].id, "handbook::docs/caching.md");
            assert_eq!(outcome.rows[0].grade, 3);
            let written = fs::read_to_string(&out_path).unwrap();
            assert_eq!(written.lines().count(), 1);
        });
    }
}
