//! `external`: a consumer's own store, over HTTP.

use std::path::Path;
use std::time::Duration;

use super::{Backend, BackendConfig, BackendError};
use crate::contracts::{self, BACKEND_VERSION, SearchRequest, SearchResponse};
use pinakes::index::{Hit, load_pages, mark_mirrors};

/// External backend timeout.
const EXTERNAL_TIMEOUT: Duration = Duration::from_secs(30);

/// `external`: `POST {backend_url}/search` with a [`SearchRequest`], expecting a
/// [`SearchResponse`] (the backend contract, `docs/manual/contracts.md`). Used to evaluate a
/// store a consumer already runs; a failed request, a malformed response or a response of a
/// newer contract version fails the whole `eval`. A hit's `unit_id` is parsed and, for now,
/// not used: the [`Hit`] `eval` scores is the page.
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
        Ok(parsed
            .hits
            .into_iter()
            .take(k)
            .map(|hit| Hit {
                page_id: hit.page_id,
                score: hit.score,
                heading: hit.heading,
            })
            .collect())
    }

    fn page_count(&self) -> usize {
        self.page_count
    }

    fn searchable_count(&self) -> usize {
        self.searchable_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::testing::fixture_pages;
    use crate::testing::read_http_request;

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
