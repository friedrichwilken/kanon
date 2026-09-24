use std::path::PathBuf;
use std::rc::Rc;

use crate::backend::{self, BackendKind};
use crate::contracts;
use crate::embed::Embedder;
use crate::error::{CommandError, io_err};
use crate::grade::{self, GradedRow, PageLookup};
use crate::workspace::Paths;
use pinakes::index;
use pinakes::llm::ChatTransport;
use pinakes::manifest::now_rfc3339;

use super::eval::backend_config;
use super::settings;

/// Options for `grade`. The backend fields are the same as `eval --backend`'s
/// ([`super::BackendEvalOptions`]); the CLI fills them from [`super::BackendFlags`] after
/// [`super::apply_backend_config_defaults`].
#[derive(Clone, Default)]
pub struct GradeOptions {
    /// `trail.jsonl` to replay.
    pub trail: PathBuf,
    /// The backend to fetch candidates from (`--backend`; `bm25` when not given).
    pub backend: BackendKind,
    /// The consumer's search endpoint (`external`).
    pub backend_url: Option<String>,
    /// `embeddings.bin` path (`dense`, `hybrid`); default: `paths.embeddings`.
    pub embeddings: Option<PathBuf>,
    /// Use embeddings even when their recorded manifest hash does not match the artifact.
    pub allow_stale: bool,
    /// Where query embeddings come from (`dense`, `hybrid`).
    pub embedder: Option<Rc<dyn Embedder>>,
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
    /// The backend the candidates came from.
    pub backend: BackendKind,
}

/// Run `grade`: replay every distinct trail query against the backend and ask the model to
/// grade each candidate. The candidates are whatever the backend returns; their title and
/// excerpt come from the artifact's pages, and a hit the artifact does not have is graded
/// with its id alone (see [`PageLookup`]).
pub fn grade(
    paths: &Paths,
    options: &GradeOptions,
    transport: &dyn ChatTransport,
) -> Result<GradeOutcome, CommandError> {
    let config = crate::llm::config_from_env(options.model.clone())?;
    let entries = contracts::read_trail(&options.trail)?;
    let queries = grade::distinct_queries(&entries);
    let priorities = settings(paths)?.priorities;
    let mut pages = index::load_pages(&paths.artifact, &priorities)?;
    index::mark_mirrors(&mut pages);
    let lookup = PageLookup::new(&pages);
    let backend_config = backend_config(
        paths,
        priorities,
        options.backend_url.as_deref(),
        options.embeddings.as_deref(),
        options.allow_stale,
        options.embedder.clone(),
    );
    let backend = backend::build(options.backend, &paths.artifact, &backend_config)?;
    let k = options.k.unwrap_or(grade::DEFAULT_K);
    let at = now_rfc3339();
    let mut rows = Vec::new();
    for query in &queries {
        let hits = backend.search(query, k, None)?;
        let candidates = lookup.candidates(&hits);
        rows.extend(grade::grade_query(
            transport,
            &config,
            query,
            &candidates,
            &at,
        )?);
    }
    if let Some(path) = &options.out {
        std::fs::write(path, grade::to_jsonl(&rows)?).map_err(io_err(path))?;
    }
    Ok(GradeOutcome {
        rows,
        queries: queries.len(),
        backend: options.backend,
    })
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::Write as _;
    use std::net::TcpListener;
    use std::path::Path;

    use super::*;
    use crate::backend::BackendError;
    use crate::backend::testing::read_http_request;
    use crate::contracts::{BACKEND_VERSION, SearchHit, SearchResponse};
    use crate::testing::{eval_workspace, with_llm_url};
    use pinakes::llm::testing::{Scripted, ScriptedTransport, completion};

    const PAGE_ID: &str = "handbook::docs/user/README.md";
    const UNKNOWN_ID: &str = "elsewhere::docs/unknown.md";

    /// A trail with the same query twice, next to the workspace.
    fn trail(dir: &Path) -> PathBuf {
        let trail_path = dir.join("trail.jsonl");
        fs::write(
            &trail_path,
            "{\"at\": \"t\", \"query\": \"enable upload caching\", \"retrieved\": [], \"cited\": []}\n\
             {\"at\": \"t\", \"query\": \"enable upload caching\", \"retrieved\": [], \"cited\": []}\n",
        )
        .unwrap();
        trail_path
    }

    /// One scripted reply grading `ids` 3, 2, 1, … in order.
    fn grader(ids: &[&str]) -> ScriptedTransport {
        let grades: Vec<serde_json::Value> = ids
            .iter()
            .zip((1..=3).rev())
            .map(|(id, grade)| serde_json::json!({"id": id, "grade": grade}))
            .collect();
        let reply = completion(&serde_json::to_string(&grades).unwrap());
        ScriptedTransport::new(vec![Scripted::Ok(reply)])
    }

    /// Answer one `POST /search` with `ids` as hits, best first.
    fn serve_one_search(listener: &TcpListener, ids: &[&str]) {
        let (mut stream, _) = listener.accept().unwrap();
        let request = read_http_request(&mut stream);
        assert!(request.starts_with("POST /search"), "{request}");
        assert!(request.contains("enable upload caching"), "{request}");
        let hits = ids
            .iter()
            .zip([2.0, 1.0])
            .map(|(id, score)| SearchHit {
                page_id: (*id).to_string(),
                score,
                heading: String::new(),
                unit_id: None,
            })
            .collect();
        let body = serde_json::to_string(&SearchResponse {
            version: BACKEND_VERSION,
            hits,
        })
        .unwrap();
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(response.as_bytes()).unwrap();
    }

    #[test]
    fn grade_replays_distinct_trail_queries_against_bm25_and_writes_graded_rows() {
        let (dir, paths) = eval_workspace();
        let trail_path = trail(dir.path());
        with_llm_url(|| {
            let transport = grader(&[PAGE_ID]);
            let out_path = dir.path().join("graded.jsonl");
            let options = GradeOptions {
                trail: trail_path,
                model: Some("grader".to_string()),
                out: Some(out_path.clone()),
                ..GradeOptions::default()
            };
            let outcome = grade(&paths, &options, &transport).unwrap();
            // Only one request even though the query appears twice in the trail.
            assert_eq!(transport.requests.lock().unwrap().len(), 1);
            assert_eq!(outcome.queries, 1);
            assert_eq!(outcome.backend, BackendKind::Bm25);
            assert_eq!(outcome.rows.len(), 1);
            assert_eq!(outcome.rows[0].id, PAGE_ID);
            assert_eq!(outcome.rows[0].grade, 3);
            let written = fs::read_to_string(&out_path).unwrap();
            assert_eq!(written.lines().count(), 1);
        });
    }

    #[test]
    fn grade_runs_through_bm25_tantivy() {
        let (dir, paths) = eval_workspace();
        let trail_path = trail(dir.path());
        with_llm_url(|| {
            let transport = grader(&[PAGE_ID]);
            let options = GradeOptions {
                trail: trail_path,
                backend: BackendKind::Bm25Tantivy,
                model: Some("grader".to_string()),
                ..GradeOptions::default()
            };
            let outcome = grade(&paths, &options, &transport).unwrap();
            assert_eq!(outcome.backend, BackendKind::Bm25Tantivy);
            assert_eq!(outcome.rows.len(), 1);
            assert_eq!(outcome.rows[0].id, PAGE_ID);
            // The model saw the page's title and text, looked up from the artifact.
            let request = transport.requests.lock().unwrap()[0].to_string();
            assert!(request.contains("Storage Module"), "{request}");
            assert!(request.contains("Enable upload caching"), "{request}");
        });
    }

    #[test]
    fn grade_takes_candidates_from_an_external_backend_and_grades_an_unknown_id() {
        let (dir, paths) = eval_workspace();
        let trail_path = trail(dir.path());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        // Best first: an id the artifact does not have, then one it does.
        let handle =
            std::thread::spawn(move || serve_one_search(&listener, &[UNKNOWN_ID, PAGE_ID]));
        with_llm_url(|| {
            let transport = grader(&[UNKNOWN_ID, PAGE_ID]);
            let options = GradeOptions {
                trail: trail_path,
                backend: BackendKind::External,
                backend_url: Some(format!("http://{addr}")),
                model: Some("grader".to_string()),
                ..GradeOptions::default()
            };
            let outcome = grade(&paths, &options, &transport).unwrap();
            handle.join().unwrap();
            assert_eq!(outcome.backend, BackendKind::External);
            let ids: Vec<&str> = outcome.rows.iter().map(|r| r.id.as_str()).collect();
            assert_eq!(
                ids,
                [UNKNOWN_ID, PAGE_ID],
                "the endpoint's list, in its order"
            );
            assert_eq!(outcome.rows[0].grade, 3);

            // The unknown id went to the model as a candidate with its id for a title and no
            // excerpt; the known one with its page.
            let requests = transport.requests.lock().unwrap();
            let user = requests[0]["messages"][1]["content"].as_str().unwrap();
            let body: serde_json::Value = serde_json::from_str(user).unwrap();
            let candidates = body["candidates"].as_array().unwrap();
            assert_eq!(candidates.len(), 2);
            assert_eq!(candidates[0]["id"], UNKNOWN_ID);
            assert_eq!(candidates[0]["title"], UNKNOWN_ID);
            assert_eq!(candidates[0]["excerpt"], "");
            assert_eq!(candidates[1]["id"], PAGE_ID);
            assert_eq!(candidates[1]["title"], "Storage Module");
        });
    }

    #[test]
    fn grade_with_external_and_no_url_is_a_config_error() {
        let (dir, paths) = eval_workspace();
        let trail_path = trail(dir.path());
        with_llm_url(|| {
            let transport = ScriptedTransport::new(vec![]);
            let options = GradeOptions {
                trail: trail_path,
                backend: BackendKind::External,
                model: Some("grader".to_string()),
                ..GradeOptions::default()
            };
            let err = grade(&paths, &options, &transport).unwrap_err();
            assert!(
                matches!(&err, CommandError::Backend(BackendError::Config { .. })),
                "{err}"
            );
            assert!(transport.requests.lock().unwrap().is_empty());
        });
    }
}
