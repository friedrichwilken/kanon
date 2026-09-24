//! `kanon grade`: replay distinct trail queries against a backend and ask the model to grade
//! each candidate 0-3 for relevance.
//!
//! The candidates come from whichever [`crate::backend::Backend`] the command built: `bm25`
//! by default, or the retriever that actually served the trail, so `queries import` learns
//! from the list the consumer showed. What the model sees of each candidate is the title and
//! an excerpt of its artifact page, looked up by id through [`PageLookup`]. A hit whose id is
//! not in the artifact (an external backend may return anything) is still graded, with the id
//! as its title and no excerpt, rather than dropped. Grading itself is one model call per
//! distinct query ([`grade_query`]).

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use pinakes::index::{Hit, Page};
use pinakes::jsonl::{self, KeyOrder};
use pinakes::llm::{self, ChatError, ChatTransport, LlmConfig};
use pinakes::residue;

use crate::contracts::TrailEntry;

/// Default `--k`.
pub const DEFAULT_K: usize = 20;
/// Tokens of page content shown to the model per candidate.
const EXCERPT_TOKENS: usize = 300;
/// Highest grade the model may give.
const MAX_GRADE: u8 = 3;

/// Errors raised while grading.
#[derive(Debug, Error)]
pub enum GradeError {
    /// Talking to the model failed.
    #[error(transparent)]
    Llm(#[from] ChatError),
}

/// One graded row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GradedRow {
    /// The query text.
    pub query: String,
    /// `<source>::<path>`.
    pub id: String,
    /// Relevance grade, 0 (irrelevant) to 3 (fully relevant).
    pub grade: u8,
    /// The model that produced the grade.
    pub model: String,
    /// RFC 3339 UTC time the grade was produced.
    pub at: String,
}

/// Serialise rows as JSONL with sorted keys, one object per line.
pub fn to_jsonl(rows: &[GradedRow]) -> Result<String, serde_json::Error> {
    jsonl::to_string(rows, KeyOrder::Sorted)
}

/// Distinct `query` values from a trail, in first-occurrence order.
pub fn distinct_queries(entries: &[TrailEntry]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for entry in entries {
        if seen.insert(entry.query.clone()) {
            out.push(entry.query.clone());
        }
    }
    out
}

/// One candidate shown to the model for grading: a hit's page id, with the title and an
/// excerpt of its artifact page when the artifact has one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Candidate {
    /// `<source>::<path>`, as the backend returned it.
    pub id: String,
    /// The page's title; the id itself when the page is not in the artifact.
    pub title: String,
    /// The first 300 tokens of the page; empty when the page is not in the
    /// artifact.
    pub excerpt: String,
}

/// The artifact's pages keyed by id, for turning a backend's hits into [`Candidate`]s.
#[derive(Debug, Clone)]
pub struct PageLookup<'a>(BTreeMap<&'a str, &'a Page>);

impl<'a> PageLookup<'a> {
    /// Index `pages` by id (the artifact as `pinakes::index::load_pages` reads it, mirrors
    /// included).
    pub fn new(pages: &'a [Page]) -> PageLookup<'a> {
        PageLookup(pages.iter().map(|page| (page.id.as_str(), page)).collect())
    }

    /// The candidates for `hits`, in hit order. A hit whose page is not in the artifact is
    /// kept, with its id as the title and an empty excerpt: the retriever showed it, so it
    /// gets graded.
    pub fn candidates(&self, hits: &[Hit]) -> Vec<Candidate> {
        hits.iter()
            .map(|hit| match self.0.get(hit.page_id.as_str()) {
                Some(page) => Candidate {
                    id: hit.page_id.clone(),
                    title: page.title.clone(),
                    excerpt: residue::excerpt(&page.content, EXCERPT_TOKENS),
                },
                None => Candidate {
                    id: hit.page_id.clone(),
                    title: hit.page_id.clone(),
                    excerpt: String::new(),
                },
            })
            .collect()
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ModelGrade {
    id: String,
    grade: u8,
}

const SYSTEM_PROMPT: &str = "You are grading search results for a documentation retrieval \
system. You will be given a query and a JSON array of candidate pages, each with an id, title \
and excerpt. For every candidate, grade how relevant it is to the query from 0 (irrelevant) to \
3 (fully relevant and directly answers the query). Respond with a JSON array only - no prose, \
no markdown code fences, one object per candidate id you were given, in this exact shape: \
[{\"id\": \"...\", \"grade\": 0}]";

fn user_prompt(query: &str, candidates: &[Candidate]) -> String {
    let body = serde_json::json!({"query": query, "candidates": candidates});
    serde_json::to_string_pretty(&body).unwrap_or_default()
}

/// Ask the model to grade every candidate of one query, dropping any id it returns that was
/// not among the candidates it was given. No candidates, no model call.
pub fn grade_query(
    transport: &dyn ChatTransport,
    config: &LlmConfig,
    query: &str,
    candidates: &[Candidate],
    at: &str,
) -> Result<Vec<GradedRow>, GradeError> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let known: BTreeSet<&str> = candidates.iter().map(|c| c.id.as_str()).collect();
    let user = user_prompt(query, candidates);
    let grades: Vec<ModelGrade> = llm::chat(transport, config, SYSTEM_PROMPT, &user)?;
    Ok(grades
        .into_iter()
        .filter(|g| known.contains(g.id.as_str()))
        .map(|g| GradedRow {
            query: query.to_string(),
            id: g.id,
            grade: g.grade.min(MAX_GRADE),
            model: config.model.clone(),
            at: at.to_string(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pinakes::llm::testing::{Scripted, ScriptedTransport, completion};

    fn page(id: &str, source: &str, path: &str, title: &str, content: &str) -> Page {
        Page {
            id: id.to_string(),
            source: source.to_string(),
            path: path.to_string(),
            repo: source.to_string(),
            module: source.to_string(),
            title: title.to_string(),
            heading: title.to_string(),
            doc_type: String::new(),
            section: String::new(),
            priority: pinakes::index::DEFAULT_PRIORITY,
            content: content.to_string(),
            mirror_of: None,
        }
    }

    fn pages() -> Vec<Page> {
        vec![
            page(
                "handbook::docs/caching.md",
                "handbook",
                "docs/caching.md",
                "Caching",
                "Enable upload caching with a label on the bucket.",
            ),
            page(
                "handbook::docs/quotas.md",
                "handbook",
                "docs/quotas.md",
                "Quotas",
                "Quota limits apply per project.",
            ),
        ]
    }

    fn hit(page_id: &str) -> Hit {
        Hit {
            page_id: page_id.to_string(),
            score: 1.0,
            heading: String::new(),
        }
    }

    fn caching_candidates() -> Vec<Candidate> {
        vec![Candidate {
            id: "handbook::docs/caching.md".to_string(),
            title: "Caching".to_string(),
            excerpt: "Enable upload caching with a label on the bucket.".to_string(),
        }]
    }

    fn config() -> LlmConfig {
        LlmConfig {
            url: "https://example.test".to_string(),
            key: None,
            model: "grader-model".to_string(),
        }
    }

    #[test]
    fn distinct_queries_preserves_first_occurrence_order() {
        let entries = vec![
            TrailEntry {
                query: "b".to_string(),
                ..entries_base()
            },
            TrailEntry {
                query: "a".to_string(),
                ..entries_base()
            },
            TrailEntry {
                query: "b".to_string(),
                ..entries_base()
            },
        ];
        assert_eq!(distinct_queries(&entries), ["b", "a"]);
    }

    fn entries_base() -> TrailEntry {
        TrailEntry {
            version: crate::contracts::TRAIL_VERSION,
            at: "t".to_string(),
            query: String::new(),
            retrieved: vec![],
            ranks: vec![],
            cited: vec![],
            outcome: crate::contracts::Outcome::Unknown,
            session: String::new(),
        }
    }

    #[test]
    fn candidates_carry_the_page_title_and_excerpt_in_hit_order() {
        let pages = pages();
        let lookup = PageLookup::new(&pages);
        let candidates = lookup.candidates(&[
            hit("handbook::docs/quotas.md"),
            hit("handbook::docs/caching.md"),
        ]);
        assert_eq!(
            candidates,
            [
                Candidate {
                    id: "handbook::docs/quotas.md".to_string(),
                    title: "Quotas".to_string(),
                    excerpt: "Quota limits apply per project.".to_string(),
                },
                caching_candidates()[0].clone(),
            ]
        );
    }

    #[test]
    fn a_hit_outside_the_artifact_is_kept_with_its_id_as_title_and_no_excerpt() {
        let pages = pages();
        let lookup = PageLookup::new(&pages);
        let candidates = lookup.candidates(&[hit("elsewhere::docs/unknown.md")]);
        assert_eq!(
            candidates,
            [Candidate {
                id: "elsewhere::docs/unknown.md".to_string(),
                title: "elsewhere::docs/unknown.md".to_string(),
                excerpt: String::new(),
            }]
        );
    }

    #[test]
    fn grade_query_asks_the_model_and_drops_unknown_ids() {
        let reply = completion(
            &serde_json::to_string(&serde_json::json!([
                {"id": "handbook::docs/caching.md", "grade": 3},
                {"id": "handbook::docs/not-a-candidate.md", "grade": 2},
            ]))
            .unwrap(),
        );
        let transport = ScriptedTransport::new(vec![Scripted::Ok(reply)]);
        let rows =
            grade_query(&transport, &config(), "caching", &caching_candidates(), "t").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, "handbook::docs/caching.md");
        assert_eq!(rows[0].grade, 3);
        assert_eq!(rows[0].model, "grader-model");
    }

    #[test]
    fn grade_query_clamps_an_out_of_range_grade() {
        let reply = completion(
            &serde_json::to_string(&serde_json::json!([
                {"id": "handbook::docs/caching.md", "grade": 9},
            ]))
            .unwrap(),
        );
        let transport = ScriptedTransport::new(vec![Scripted::Ok(reply)]);
        let rows =
            grade_query(&transport, &config(), "caching", &caching_candidates(), "t").unwrap();
        assert_eq!(rows[0].grade, MAX_GRADE);
    }

    #[test]
    fn grade_query_with_no_candidates_never_calls_the_model() {
        let transport = ScriptedTransport::new(vec![]);
        let rows = grade_query(&transport, &config(), "nothing matches at all", &[], "t").unwrap();
        assert!(rows.is_empty());
        assert!(transport.requests.lock().unwrap().is_empty());
    }

    #[test]
    fn to_jsonl_round_trips() {
        let rows = vec![GradedRow {
            query: "q".to_string(),
            id: "handbook::docs/caching.md".to_string(),
            grade: 2,
            model: "m".to_string(),
            at: "t".to_string(),
        }];
        let text = to_jsonl(&rows).unwrap();
        assert_eq!(text.lines().count(), 1);
        assert!(text.contains("\"grade\":2"));
    }
}
