//! `kanon queries suggest`: bootstrap a query set from the corpus.
//!
//! [`sample`] draws pages stratified by source and by section (a page's `section`, else the
//! first segment of its path) with a seeded PRNG, so the same seed on the same artifact picks
//! the same pages. [`suggest`] asks the model, one request per page, for two or three
//! realistic questions the page answers, and turns them into [`Suggestion`] rows with the page
//! as the expected id. A suggestion that quotes the page title verbatim (see [`quotes_title`])
//! is dropped, as it would inflate BM25 recall without measuring anything; so are empty
//! queries and repeats of a query already suggested for another page, and a `kind` outside
//! the prompt's vocabulary is emptied. A page whose reply is not the expected JSON is skipped
//! and counted, not fatal, unless every page failed that way. The rows go to a suggestions
//! file, never to `queries.jsonl`: `queries add --from suggestions.jsonl --accept ID…` is the
//! human step in between.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::{QueriesError, jsonl_error, query_id};
use crate::rng::SplitMix64;
use pinakes::index::Page;
use pinakes::jsonl::{self, KeyOrder};
use pinakes::llm::{self, ChatError, ChatTransport, LlmConfig};
use pinakes::residue;
use pinakes::tokenizer::title_key;

/// Default `--n`: pages sampled.
pub const DEFAULT_N: usize = 50;
/// Default `--out`.
pub const DEFAULT_OUT: &str = "suggestions.jsonl";
/// The `origin` every suggested row carries, in the suggestions file and in `queries.jsonl`
/// once accepted.
pub const ORIGIN: &str = "suggested";
/// Tokens of page content shown to the model.
const EXCERPT_TOKENS: usize = 400;

/// One row of the suggestions file: a [`crate::eval::Query`] without `holdout`, plus the
/// provenance `origin` and the sampled page's title for the human reading the file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Suggestion {
    /// Stable id derived from the query text: a slug plus a hash suffix, the same scheme
    /// `queries import` uses, so the same text always gets the same id.
    pub id: String,
    /// The suggested query.
    pub query: String,
    /// The sampled page, `<source>::<path>`.
    pub expected: Vec<String>,
    /// `howto`, `reference`, `troubleshooting` or `concept`, as the model named it.
    #[serde(default)]
    pub kind: String,
    /// Always [`ORIGIN`] when written by `queries suggest`.
    #[serde(default)]
    pub origin: String,
    /// The sampled page's title.
    #[serde(default)]
    pub page_title: String,
}

/// The stratum a page is sampled in within its source: its navigation section, else the first
/// segment of its path (`docs` for `docs/install.md`, the file name for a top-level page).
pub fn stratum(page: &Page) -> String {
    if !page.section.trim().is_empty() {
        return page.section.trim().to_string();
    }
    page.path.split('/').next().unwrap_or_default().to_string()
}

/// Draw up to `n` searchable pages (mirrors are skipped), stratified by source and by
/// [`stratum`]: sources take turns, and within a source its strata take turns, each handing
/// out its pages in an order shuffled by `seed`. Every source gets at least one page when
/// `n` allows; `per_source` caps what any one source contributes. The result is fixed by
/// `seed` alone: sources and strata are visited in sorted order and pages are sorted by id
/// before the shuffle.
pub fn sample<'a>(
    pages: &'a [Page],
    n: usize,
    per_source: Option<usize>,
    seed: u64,
) -> Vec<&'a Page> {
    let mut strata: BTreeMap<&str, BTreeMap<String, Vec<&'a Page>>> = BTreeMap::new();
    for page in pages.iter().filter(|page| page.mirror_of.is_none()) {
        strata
            .entry(page.source.as_str())
            .or_default()
            .entry(stratum(page))
            .or_default()
            .push(page);
    }
    let mut rng = SplitMix64::new(seed);
    // Per source: pages taken so far, the next stratum to ask, and one stack per stratum.
    let mut sources: Vec<(usize, usize, Vec<Vec<&'a Page>>)> = strata
        .into_values()
        .map(|by_stratum| {
            let stacks = by_stratum
                .into_values()
                .map(|mut pages| {
                    pages.sort_by(|a, b| a.id.cmp(&b.id));
                    rng.shuffle(&mut pages);
                    pages
                })
                .collect();
            (0, 0, stacks)
        })
        .collect();
    let cap = per_source.unwrap_or(usize::MAX);
    let mut out = Vec::new();
    let mut progress = true;
    while out.len() < n && progress {
        progress = false;
        for (taken, cursor, stacks) in &mut sources {
            if out.len() >= n || *taken >= cap {
                continue;
            }
            for _ in 0..stacks.len() {
                let index = *cursor % stacks.len();
                *cursor += 1;
                if let Some(page) = stacks[index].pop() {
                    out.push(page);
                    *taken += 1;
                    progress = true;
                    break;
                }
            }
        }
    }
    out
}

/// Whether `query` quotes `title` verbatim: the tokenised title (see [`title_key`]) appears
/// whole, on token boundaries, in the tokenised query. A one-token title ("Install") is quoted
/// only by a query that is nothing but that token, as [`title_key`] drops stopwords and
/// "how do I install the service" is a fair question, not a quote. An untitled page quotes
/// nothing.
pub fn quotes_title(query: &str, title: &str) -> bool {
    let title = title_key(title);
    if title.is_empty() {
        return false;
    }
    let query = title_key(query);
    if title.contains(' ') {
        format!(" {query} ").contains(&format!(" {title} "))
    } else {
        query == title
    }
}

/// The kinds the model may name; anything else becomes an empty kind rather than a row of its
/// own in `eval`'s per-kind table.
const KINDS: [&str; 4] = ["howto", "reference", "troubleshooting", "concept"];

/// The model's `kind`, normalised to one of [`KINDS`] or empty.
fn kind_of(reply: &str) -> String {
    let kind = reply.trim().to_lowercase();
    if KINDS.contains(&kind.as_str()) {
        kind
    } else {
        String::new()
    }
}

const SYSTEM_PROMPT: &str = "You are helping build a query set for evaluating a documentation \
search system. You will be given one documentation page as JSON: its id, title, section and an \
excerpt of its content. Write 2 to 3 realistic queries a user would type into search when this \
page is the answer they need. Use the user's own words for their situation, not the page's \
title or headings; never repeat the title. Vary the form: some short keyword queries, some full \
questions. Name each query's kind: howto (how to do something), reference (a fact, a value, a \
name), troubleshooting (something is failing) or concept (what something is or why). Respond \
with a JSON array only - no prose, no markdown code fences, in this exact shape: \
[{\"query\": \"...\", \"kind\": \"howto|reference|troubleshooting|concept\"}]";

fn user_prompt(page: &Page) -> String {
    let body = serde_json::json!({
        "id": page.id,
        "title": page.title,
        "section": page.section,
        "excerpt": residue::excerpt(&page.content, EXCERPT_TOKENS),
    });
    serde_json::to_string_pretty(&body).unwrap_or_default()
}

#[derive(Debug, Clone, Deserialize)]
struct ModelSuggestion {
    #[serde(default)]
    query: String,
    #[serde(default)]
    kind: String,
}

/// What [`suggest`] produced.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Suggested {
    /// The rows to write, in page order.
    pub rows: Vec<Suggestion>,
    /// Model answers dropped for quoting the page title.
    pub title_quotes: usize,
    /// Model answers dropped for being empty.
    pub empty: usize,
    /// Model answers dropped as a repeat of a query already suggested.
    pub repeats: usize,
    /// Pages whose reply was not the expected JSON (skipped; see [`suggest`]).
    pub failed: usize,
}

impl Suggested {
    /// Every model answer dropped, whatever the reason.
    pub fn rejected(&self) -> usize {
        self.title_quotes + self.empty + self.repeats
    }
}

/// Ask the model for queries each of `pages` answers, one request per page, and keep the ones
/// that pass the checks in the module docs. A reply that is not the expected JSON (a fenced
/// block, prose) skips that page and counts in [`Suggested::failed`]; it is the error only
/// when every page failed that way. Any other model error stops the run.
pub fn suggest(
    transport: &dyn ChatTransport,
    config: &LlmConfig,
    pages: &[&Page],
) -> Result<Suggested, QueriesError> {
    let mut out = Suggested::default();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut first_failure = None;
    for page in pages {
        let answers: Vec<ModelSuggestion> =
            match llm::chat(transport, config, SYSTEM_PROMPT, &user_prompt(page)) {
                Ok(answers) => answers,
                Err(err @ ChatError::Json { .. }) => {
                    out.failed += 1;
                    first_failure.get_or_insert(err);
                    continue;
                }
                Err(err) => return Err(err.into()),
            };
        for answer in answers {
            let query = answer.query.trim();
            if query.is_empty() {
                out.empty += 1;
                continue;
            }
            if quotes_title(query, &page.title) {
                out.title_quotes += 1;
                continue;
            }
            let id = query_id(query);
            if !seen.insert(id.clone()) {
                out.repeats += 1;
                continue;
            }
            out.rows.push(Suggestion {
                id,
                query: query.to_string(),
                expected: vec![page.id.clone()],
                kind: kind_of(&answer.kind),
                origin: ORIGIN.to_string(),
                page_title: page.title.clone(),
            });
        }
    }
    match first_failure {
        Some(err) if out.failed == pages.len() => Err(err.into()),
        _ => Ok(out),
    }
}

/// Replace the suggestions file at `path` with `rows`.
pub fn write(path: &Path, rows: &[Suggestion]) -> Result<(), QueriesError> {
    jsonl::write(path, rows, KeyOrder::Declared).map_err(jsonl_error)
}

/// Read a suggestions file; blank lines are skipped.
pub fn load(path: &Path) -> Result<Vec<Suggestion>, QueriesError> {
    jsonl::read(path).map_err(jsonl_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pinakes::llm::testing::{Scripted, ScriptedTransport, completion};

    fn page(source: &str, path: &str, title: &str, section: &str) -> Page {
        Page {
            id: format!("{source}::{path}"),
            source: source.to_string(),
            path: path.to_string(),
            repo: source.to_string(),
            module: source.to_string(),
            title: title.to_string(),
            heading: title.to_string(),
            doc_type: String::new(),
            section: section.to_string(),
            priority: pinakes::index::DEFAULT_PRIORITY,
            content: format!("# {title}\n\nSome words about {title}.\n"),
            mirror_of: None,
        }
    }

    fn corpus() -> Vec<Page> {
        let mut pages = vec![
            page("handbook", "docs/install.md", "Install", "Start"),
            page(
                "handbook",
                "docs/configuration.md",
                "Configuration",
                "Start",
            ),
            page(
                "handbook",
                "docs/reference/cli.md",
                "Command Line",
                "Reference",
            ),
            page(
                "handbook",
                "docs/reference/keys.md",
                "Configuration Keys",
                "Reference",
            ),
            page("guides", "docs/rotate-keys.md", "Rotate Keys", ""),
            page("guides", "docs/enable-caching.md", "Enable Caching", ""),
            page("guides", "recipes/cron.md", "Cron Jobs", ""),
            page("cookbook", "README.md", "Cookbook", ""),
        ];
        let mut mirror = page("cookbook", "docs/enable-caching.md", "Enable Caching", "");
        mirror.mirror_of = Some("guides::docs/enable-caching.md".to_string());
        pages.push(mirror);
        pages
    }

    fn config() -> LlmConfig {
        LlmConfig {
            url: "https://example.test".to_string(),
            key: None,
            model: "suggest-model".to_string(),
        }
    }

    #[test]
    fn stratum_is_the_section_else_the_first_path_segment() {
        assert_eq!(stratum(&page("h", "docs/a.md", "A", "Start")), "Start");
        assert_eq!(stratum(&page("h", "docs/a.md", "A", "  ")), "docs");
        assert_eq!(stratum(&page("h", "README.md", "A", "")), "README.md");
    }

    #[test]
    fn sample_is_deterministic_and_covers_every_source() {
        let pages = corpus();
        let first = sample(&pages, 5, None, 42);
        assert_eq!(first.len(), 5);
        let sources: BTreeSet<&str> = first.iter().map(|p| p.source.as_str()).collect();
        assert_eq!(
            sources,
            BTreeSet::from(["cookbook", "guides", "handbook"]),
            "{first:?}"
        );
        let again = sample(&pages, 5, None, 42);
        assert_eq!(first, again, "the same seed picks the same pages");
        // Sources take turns, so the first three draws are one per source (sorted order).
        let head: Vec<&str> = first.iter().take(3).map(|p| p.source.as_str()).collect();
        assert_eq!(head, ["cookbook", "guides", "handbook"]);
        // Within a source its strata take turns: handbook's two draws come from two sections.
        let handbook: BTreeSet<String> = first
            .iter()
            .filter(|p| p.source == "handbook")
            .map(|p| stratum(p))
            .collect();
        assert_eq!(
            handbook,
            BTreeSet::from(["Reference".to_string(), "Start".to_string()])
        );
    }

    #[test]
    fn sample_respects_the_per_source_cap_and_skips_mirrors() {
        let pages = corpus();
        let capped = sample(&pages, 50, Some(1), 1);
        assert_eq!(capped.len(), 3);
        let sources: Vec<&str> = capped.iter().map(|p| p.source.as_str()).collect();
        assert_eq!(sources, ["cookbook", "guides", "handbook"]);
        // More than there are pages: every searchable page, the mirror never.
        let all = sample(&pages, 50, None, 1);
        assert_eq!(all.len(), 8);
        assert!(all.iter().all(|p| p.mirror_of.is_none()));
        assert!(sample(&pages, 0, None, 1).is_empty());
        // Different seeds visit the pages in a different order somewhere.
        let other = sample(&pages, 50, None, 2);
        assert_ne!(all, other);
    }

    #[test]
    fn quotes_title_is_case_insensitive_and_token_bounded() {
        assert!(quotes_title(
            "Where is the Command Line reference?",
            "Command line"
        ));
        assert!(quotes_title("install", "Install"));
        assert!(quotes_title("Install?", "Install"));
        assert!(!quotes_title("how do I set up the cli", "Command Line"));
        // A whole token only: "caches" does not quote "cache".
        assert!(!quotes_title("enable caches for uploads", "Cache"));
        // A one-token title is quoted only by a query that is nothing but that token, since
        // title_key drops stopwords and every question about installing contains "install".
        assert!(!quotes_title("how do I install the service", "Install"));
        assert!(!quotes_title(
            "permissions for a read-only user",
            "Permissions"
        ));
        assert!(!quotes_title("anything", ""));
    }

    #[test]
    fn kind_is_one_of_the_prompt_vocabulary_or_empty() {
        assert_eq!(kind_of(" HowTo "), "howto");
        assert_eq!(kind_of("troubleshooting"), "troubleshooting");
        assert_eq!(kind_of("tutorial"), "");
        assert_eq!(kind_of(""), "");
    }

    #[test]
    fn suggest_keeps_good_queries_and_rejects_title_quotes_empties_and_repeats() {
        let pages = corpus();
        let install = &pages[0];
        let keys = &pages[3];
        let first = completion(
            &serde_json::to_string(&serde_json::json!([
                {"query": "how do I set the service up on a fresh machine", "kind": "HowTo "},
                {"query": "Install", "kind": "howto"},
                {"query": "  ", "kind": "howto"},
            ]))
            .unwrap(),
        );
        let second = completion(
            &serde_json::to_string(&serde_json::json!([
                {"query": "which port does TLS use", "kind": "faq"},
                {"query": "how do I set the service up on a fresh machine", "kind": "howto"},
            ]))
            .unwrap(),
        );
        let transport = ScriptedTransport::new(vec![Scripted::Ok(first), Scripted::Ok(second)]);
        let out = suggest(&transport, &config(), &[install, keys]).unwrap();
        assert_eq!(
            (out.title_quotes, out.empty, out.repeats, out.failed),
            (1, 1, 1, 0)
        );
        assert_eq!(out.rejected(), 3);
        assert_eq!(out.rows.len(), 2);
        assert_eq!(out.rows[1].kind, "", "off-vocabulary kind is emptied");
        assert_eq!(
            out.rows[0].query,
            "how do I set the service up on a fresh machine"
        );
        assert_eq!(out.rows[0].id, query_id(&out.rows[0].query));
        assert_eq!(out.rows[0].expected, ["handbook::docs/install.md"]);
        assert_eq!(out.rows[0].kind, "howto");
        assert_eq!(out.rows[0].origin, ORIGIN);
        assert_eq!(out.rows[0].page_title, "Install");
        assert_eq!(out.rows[1].expected, ["handbook::docs/reference/keys.md"]);

        // One request per page, temperature 0, the page in the user message.
        let requests = transport.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0]["temperature"], 0);
        let system = requests[0]["messages"][0]["content"].as_str().unwrap();
        assert!(system.contains("howto|reference|troubleshooting|concept"));
        let user = requests[0]["messages"][1]["content"].as_str().unwrap();
        assert!(user.contains("handbook::docs/install.md"), "{user}");
    }

    #[test]
    fn suggest_skips_a_page_whose_reply_is_not_json_and_fails_only_when_every_page_did() {
        let pages = corpus();
        let good = |query: &str| {
            completion(
                &serde_json::to_string(&serde_json::json!([{"query": query, "kind": "howto"}]))
                    .unwrap(),
            )
        };
        let fenced = completion("```json\n[{\"query\": \"lost\", \"kind\": \"howto\"}]\n```");
        let transport = ScriptedTransport::new(vec![
            Scripted::Ok(good("set up a fresh machine")),
            Scripted::Ok(fenced.clone()),
            Scripted::Ok(good("change the listening port")),
        ]);
        let out = suggest(&transport, &config(), &[&pages[0], &pages[1], &pages[2]]).unwrap();
        assert_eq!(out.failed, 1);
        assert_eq!(out.rows.len(), 2, "the pages around the failure are kept");
        assert_eq!(out.rows[1].expected, ["handbook::docs/reference/cli.md"]);

        // Every page failing that way is the error.
        let transport = ScriptedTransport::new(vec![Scripted::Ok(fenced)]);
        let err = suggest(&transport, &config(), &[&pages[0]]).unwrap_err();
        assert!(
            matches!(err, QueriesError::Llm(ChatError::Json { .. })),
            "{err}"
        );

        // Any other model error still stops the run.
        let transport = ScriptedTransport::new(vec![
            Scripted::Ok(good("set up a fresh machine")),
            Scripted::Err(pinakes::llm::TransportError::Status(
                401,
                "unauthorized".to_string(),
            )),
        ]);
        let err = suggest(&transport, &config(), &[&pages[0], &pages[1]]).unwrap_err();
        assert!(
            matches!(err, QueriesError::Llm(ChatError::Http { status: 401, .. })),
            "{err}"
        );
    }

    #[test]
    fn suggestions_round_trip_in_declared_key_order() {
        let rows = vec![Suggestion {
            id: "q-1".to_string(),
            query: "q".to_string(),
            expected: vec!["h::a.md".to_string()],
            kind: "howto".to_string(),
            origin: ORIGIN.to_string(),
            page_title: "A".to_string(),
        }];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("suggestions.jsonl");
        write(&path, &rows).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "{\"id\":\"q-1\",\"query\":\"q\",\"expected\":[\"h::a.md\"],\"kind\":\"howto\",\
             \"origin\":\"suggested\",\"page_title\":\"A\"}\n"
        );
        assert_eq!(load(&path).unwrap(), rows);
        assert!(matches!(
            load(&dir.path().join("missing.jsonl")).unwrap_err(),
            QueriesError::Io { .. }
        ));
    }
}
