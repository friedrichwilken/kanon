//! `external`: a consumer's own store, over HTTP.

use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;

use super::{Backend, BackendConfig, BackendError};
use crate::contracts::{
    self, BACKEND_VERSION, SearchHit, SearchRequest, SearchResponse, split_unit_id,
};
use crate::hit::Hit;
use pinakes::index::{load_pages, mark_mirrors};

/// External backend timeout.
const EXTERNAL_TIMEOUT: Duration = Duration::from_secs(30);

/// `external`: `POST {backend_url}/search` with a [`SearchRequest`], expecting a
/// [`SearchResponse`] (the backend contract, `docs/manual/contracts.md`). Used to evaluate a
/// store a consumer already runs; a failed request, a malformed response or a response of a
/// newer contract version fails the whole `eval`. `eval` scores pages: a hit that names a
/// `unit_id` is checked against its `page_id` and the unit is recorded, and a page a backend
/// returns more than once counts once, at its best rank.
pub struct ExternalBackend {
    url: String,
    agent: ureq::Agent,
    page_count: usize,
    searchable_count: usize,
}

impl std::fmt::Debug for ExternalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalBackend")
            .field("url", &self.url)
            .finish_non_exhaustive()
    }
}

impl Backend for ExternalBackend {
    fn build(artifact: &Path, config: &BackendConfig) -> Result<ExternalBackend, BackendError> {
        let url = config.backend_url()?.to_string();
        let mut pages = load_pages(artifact, &config.priorities)?;
        mark_mirrors(&mut pages);
        let searchable_count = pages.iter().filter(|p| p.mirror_of.is_none()).count();
        let agent = ureq::Agent::config_builder()
            .user_agent(concat!("kanon/", env!("CARGO_PKG_VERSION")))
            .build()
            .new_agent();
        Ok(ExternalBackend {
            url,
            agent,
            page_count: pages.len(),
            searchable_count,
        })
    }

    fn search(
        &self,
        query: &str,
        k: usize,
        module: Option<&str>,
    ) -> Result<Vec<Hit>, BackendError> {
        let url = format!("{}/search", self.url.trim_end_matches('/'));
        let body = SearchRequest {
            version: BACKEND_VERSION,
            query: query.to_string(),
            k,
            module: module.map(str::to_string),
        };
        let response = self
            .agent
            .post(&url)
            .config()
            .timeout_global(Some(EXTERNAL_TIMEOUT))
            .build()
            .header("content-type", "application/json")
            .send_json(&body)
            .map_err(|err| BackendError::Http {
                url: url.clone(),
                message: err.to_string(),
            })?;
        let bad_response = |message: String| BackendError::BadResponse {
            url: url.clone(),
            message,
        };
        let text = response
            .into_body()
            .read_to_string()
            .map_err(|err| bad_response(err.to_string()))?;
        let version =
            contracts::document_version(&text).map_err(|err| bad_response(err.to_string()))?;
        contracts::check_version(version, BACKEND_VERSION, &url)?;
        let parsed: SearchResponse =
            serde_json::from_str(&text).map_err(|err| bad_response(err.to_string()))?;
        page_hits(parsed.hits, k).map_err(bad_response)
    }

    fn page_count(&self) -> usize {
        self.page_count
    }

    fn searchable_count(&self) -> usize {
        self.searchable_count
    }
}

/// The first `k` distinct pages of a response's hits, best first.
///
/// A backend that retrieves units may return several units of one page; `eval` scores pages,
/// so the best-ranked hit of a page stands for it, as the built-in backends do. A hit's
/// `unit_id` must be `<page_id>#<ordinal>` (see [`split_unit_id`]) for the page it names as
/// `page_id`; a hit that says otherwise is the backend's bug and fails the response with the
/// offending id, rather than being scored against a page it did not retrieve.
fn page_hits(hits: Vec<SearchHit>, k: usize) -> Result<Vec<Hit>, String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for hit in hits {
        if let Some(unit_id) = &hit.unit_id {
            match split_unit_id(unit_id) {
                Some((page_id, _)) if page_id == hit.page_id => {}
                Some((page_id, _)) => {
                    return Err(format!(
                        "unit_id {unit_id:?} belongs to page {page_id:?}, not to the hit's page_id {:?}",
                        hit.page_id
                    ));
                }
                None => {
                    return Err(format!(
                        "unit_id {unit_id:?} is not <source>::<path>#<ordinal>, as `pinakes chunks` numbers units"
                    ));
                }
            }
        }
        if out.len() < k && seen.insert(hit.page_id.clone()) {
            out.push(Hit {
                page_id: hit.page_id,
                score: hit.score,
                heading: hit.heading,
                unit_id: hit.unit_id,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::testing::fixture_pages;
    use crate::testing::read_http_request;

    fn search_hit(page_id: &str, unit_id: Option<&str>) -> SearchHit {
        SearchHit {
            page_id: page_id.to_string(),
            score: 1.0,
            heading: String::new(),
            unit_id: unit_id.map(str::to_string),
        }
    }

    #[test]
    fn a_unit_hit_is_scored_as_its_page_and_keeps_its_unit() {
        let hits = page_hits(
            vec![
                search_hit("handbook::docs/a.md", Some("handbook::docs/a.md#2")),
                search_hit("handbook::docs/b.md", None),
            ],
            10,
        )
        .unwrap();
        assert_eq!(hits[0].page_id, "handbook::docs/a.md");
        assert_eq!(hits[0].unit_id.as_deref(), Some("handbook::docs/a.md#2"));
        assert_eq!(hits[1].unit_id, None);
    }

    #[test]
    fn a_page_returned_twice_counts_once_at_its_best_rank() {
        let hits = page_hits(
            vec![
                search_hit("handbook::docs/a.md", Some("handbook::docs/a.md#1")),
                search_hit("handbook::docs/b.md", None),
                search_hit("handbook::docs/a.md", Some("handbook::docs/a.md#0")),
                search_hit("handbook::docs/c.md", None),
            ],
            2,
        )
        .unwrap();
        let pages: Vec<&str> = hits.iter().map(|h| h.page_id.as_str()).collect();
        assert_eq!(pages, ["handbook::docs/a.md", "handbook::docs/b.md"]);
        assert_eq!(hits[0].unit_id.as_deref(), Some("handbook::docs/a.md#1"));
    }

    #[test]
    fn a_unit_id_that_does_not_belong_to_its_page_fails_the_response() {
        let err = page_hits(
            vec![search_hit(
                "handbook::docs/a.md",
                Some("handbook::docs/b.md#0"),
            )],
            10,
        )
        .unwrap_err();
        assert!(
            err.contains("belongs to page \"handbook::docs/b.md\""),
            "{err}"
        );
        for bad in ["handbook::docs/a.md", "handbook::docs/a.md#x", "#0"] {
            let err =
                page_hits(vec![search_hit("handbook::docs/a.md", Some(bad))], 10).unwrap_err();
            assert!(err.contains(&format!("{bad:?}")), "{err}");
        }
        // Past the first `k` distinct pages a bad hit still fails: the response is checked whole.
        let err = page_hits(
            vec![
                search_hit("handbook::docs/a.md", None),
                search_hit("handbook::docs/b.md", Some("nope")),
            ],
            1,
        )
        .unwrap_err();
        assert!(err.contains("\"nope\""), "{err}");
    }

    #[test]
    fn external_backend_sends_the_request_and_parses_the_response() {
        use std::io::Write;
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let request = read_http_request(&mut stream);
            assert!(request.starts_with("POST /search"), "{request}");
            assert!(request.contains("\"query\""), "{request}");
            assert!(request.contains("\"caching\""), "{request}");
            let body_start = request.find("\r\n\r\n").unwrap() + 4;
            let sent: SearchRequest = serde_json::from_str(&request[body_start..]).unwrap();
            assert_eq!(sent.version, BACKEND_VERSION);
            assert_eq!(sent.k, 5);
            assert_eq!(sent.module, None);
            let body = r#"{"version":1,"hits":[{"page_id":"handbook::docs/user/README.md","score":1.5,"heading":"Upload caching","unit_id":"handbook::docs/user/README.md#1"}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        let (dir, _pages) = fixture_pages();
        let config = BackendConfig {
            backend_url: Some(format!("http://{addr}")),
            ..BackendConfig::default()
        };
        let backend = ExternalBackend::build(dir.path(), &config).unwrap();
        assert_eq!(backend.page_count(), 2);
        let hits = backend.search("caching", 5, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].page_id, "handbook::docs/user/README.md");
        assert!((hits[0].score - 1.5).abs() < 1e-12);
        assert_eq!(hits[0].heading, "Upload caching");
        assert_eq!(
            hits[0].unit_id.as_deref(),
            Some("handbook::docs/user/README.md#1")
        );
        handle.join().unwrap();
    }

    #[test]
    fn external_backend_rejects_a_response_of_a_newer_version() {
        use std::io::Write;
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _request = read_http_request(&mut stream);
            // A version 2 response that would not parse as version 1 either: the version is
            // what gets reported.
            let body = r#"{"version":2,"hits":[{"page":"handbook::docs/user/README.md"}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        let (dir, _pages) = fixture_pages();
        let config = BackendConfig {
            backend_url: Some(format!("http://{addr}")),
            ..BackendConfig::default()
        };
        let backend = ExternalBackend::build(dir.path(), &config).unwrap();
        let err = backend.search("caching", 5, None).unwrap_err();
        handle.join().unwrap();
        assert!(matches!(&err, BackendError::Contract(_)), "{err}");
        assert_eq!(
            err.to_string(),
            format!("http://{addr}/search: version 2 is newer than the version 1 this kanon reads")
        );
    }

    #[test]
    fn external_backend_without_a_url_is_a_config_error() {
        let (dir, _pages) = fixture_pages();
        let err = ExternalBackend::build(dir.path(), &BackendConfig::default()).unwrap_err();
        assert!(matches!(err, BackendError::Config { .. }), "{err}");
    }
}
