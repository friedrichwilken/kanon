//! Test fixtures shared by more than one module's tests: a synthetic artifact writer, a
//! workspace around it, a fake embedder, an environment lock and a one-request HTTP server
//! reader.

use std::fs;
use std::path::Path;
use std::sync::Mutex;

use crate::embed::Embedder;
use crate::workspace::Paths;

/// Guards tests that set or read `KANON_*` / `PINAKES_*` env vars, so they cannot interleave
/// with `cargo test`'s default multi-threaded runner.
pub(crate) static ENV_LOCK: Mutex<()> = Mutex::new(());

/// A source to write: directory name, repo slug, `(path, nav title, content)` pages and
/// `(path, content)` residue pages.
pub(crate) struct SourceSpec<'a> {
    pub name: &'a str,
    pub repo: &'a str,
    pub pages: &'a [(&'a str, &'a str, &'a str)],
    pub residue: &'a [(&'a str, &'a str)],
}

/// Write `sources` as an artifact directory with a `meta.json` per source (no manifest), in
/// the layout pinakes writes.
pub(crate) fn write_artifact(dir: &Path, sources: &[SourceSpec<'_>]) {
    for source in sources {
        let root = dir.join(source.name);
        let mut pages = serde_json::Map::new();
        for (path, title, content) in source.pages {
            let file = root.join(path);
            fs::create_dir_all(file.parent().unwrap()).unwrap();
            fs::write(&file, content).unwrap();
            if !title.is_empty() {
                pages.insert(
                    (*path).to_string(),
                    serde_json::json!({"title": title, "doc_type": "", "section": ""}),
                );
            }
        }
        fs::create_dir_all(&root).unwrap();
        let meta = serde_json::json!({
            "repo": source.repo,
            "module": source.name,
            "base_url": format!("https://github.com/{}/blob/abc", source.repo),
            "commit": "abc",
            "pages": pages,
        });
        fs::write(root.join("meta.json"), meta.to_string()).unwrap();
        for (path, content) in source.residue {
            let file = dir.join("_residue").join(source.name).join(path);
            fs::create_dir_all(file.parent().unwrap()).unwrap();
            fs::write(&file, content).unwrap();
        }
    }
}

/// A workspace with a synthetic artifact, a query file and no config.
pub(crate) fn eval_workspace() -> (tempfile::TempDir, Paths) {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::for_config(&dir.path().join("kanon.yaml"));
    write_artifact(
        &paths.artifact,
        &[SourceSpec {
            name: "handbook",
            repo: "example-org/handbook",
            pages: &[(
                "docs/user/README.md",
                "Storage Module",
                "# Storage\n\nEnable upload caching with a bucket label.\n",
            )],
            residue: &[(
                "docs/user/quotas.md",
                "# Configure Quotas\n\nRate limits in strict mode.\n",
            )],
        }],
    );
    fs::write(
        dir.path().join("queries.jsonl"),
        concat!(
            "{\"id\": \"caching\", \"kind\": \"howto\", \"query\": \"enable upload caching\", \"expected\": [\"handbook/docs/user\"]}\n",
            "{\"id\": \"quotas\", \"kind\": \"howto\", \"query\": \"quotas rate limits\", \"expected\": [\"handbook::docs/user/quotas.md\"]}\n",
        ),
    )
    .unwrap();
    (dir, paths)
}

pub(crate) fn fake_embedder() -> std::rc::Rc<dyn Embedder> {
    std::rc::Rc::new(crate::embed::testing::FakeEmbedder)
}

/// Set `KANON_LLM_URL` for the duration of `body`, serialised against every other test that
/// touches the model environment, and always clean up afterwards.
pub(crate) fn with_llm_url<T>(body: impl FnOnce() -> T) -> T {
    let _guard = ENV_LOCK.lock().unwrap();
    // SAFETY: serialised by ENV_LOCK; no other test observes this var concurrently.
    unsafe {
        std::env::set_var("KANON_LLM_URL", "https://example.test");
    }
    let result = body();
    // SAFETY: serialised by ENV_LOCK; no other test observes this var concurrently.
    unsafe {
        std::env::remove_var("KANON_LLM_URL");
    }
    result
}

/// A workspace with a committed `manifest.json` (two `handbook` pages, `docs/a.md` and
/// `docs/b.md`) and no config.
pub(crate) fn manifest_workspace() -> (tempfile::TempDir, Paths) {
    use std::collections::BTreeMap;

    use pinakes::manifest::{Manifest, ManifestSource, PageEntry, SelectedBy};

    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::for_config(&dir.path().join("kanon.yaml"));
    let mut manifest = Manifest::new("2026-09-16T12:00:00Z".to_string());
    let mut pages = BTreeMap::new();
    for path in ["docs/a.md", "docs/b.md"] {
        pages.insert(
            path.to_string(),
            PageEntry {
                sha256: "aa".repeat(32),
                title: path.to_string(),
                doc_type: "howto".to_string(),
                section: String::new(),
                selected_by: SelectedBy::Include,
                rendered_from: None,
            },
        );
    }
    manifest.sources.insert(
        "handbook".to_string(),
        ManifestSource {
            repo: "example-org/handbook".to_string(),
            repo_url: "https://github.com/example-org/handbook.git".to_string(),
            git_ref: "main".to_string(),
            commit: "a".repeat(40),
            archived: Some(false),
            resolver: "glob".to_string(),
            unrendered: Vec::new(),
            pages,
            residue: vec![],
            unresolved: vec![],
            render: None,
        },
    );
    manifest.save(&paths.manifest).unwrap();
    (dir, paths)
}

/// Read one HTTP/1.1 request off `stream`, answering `Expect: 100-continue` (which ureq
/// sends before a request body) so the client proceeds to send it, then returning headers
/// and body as one string once `Content-Length` bytes of body have arrived.
pub(crate) fn read_http_request(stream: &mut std::net::TcpStream) -> String {
    use std::io::{Read as _, Write as _};

    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut answered_continue = false;
    loop {
        let n = stream.read(&mut chunk).unwrap();
        assert!(n > 0, "connection closed before a full request arrived");
        buf.extend_from_slice(&chunk[..n]);
        let Some(header_end) = find_subslice(&buf, b"\r\n\r\n") else {
            continue;
        };
        let headers = String::from_utf8_lossy(&buf[..header_end]).into_owned();
        if !answered_continue && headers.to_lowercase().contains("expect: 100-continue") {
            stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").unwrap();
            answered_continue = true;
        }
        let content_length = headers
            .lines()
            .find_map(|line| {
                line.to_lowercase()
                    .strip_prefix("content-length:")
                    .map(str::trim)
                    .map(str::to_string)
            })
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        let body_start = header_end + 4;
        if buf.len() - body_start >= content_length {
            return String::from_utf8_lossy(&buf).into_owned();
        }
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}
