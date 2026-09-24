//! Named backends end to end through the binary: a `kanon.yaml` with two `external` entries,
//! `old` and `new`, each answered by its own in-process HTTP server (the pattern of
//! `backend_external_cli.rs`). `eval --compare old,new` prints one table per name and one
//! combined JSON keyed by name; `--backend old` and `grade --backend old` take the same names.

use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::Command;

use kanon::contracts::{BACKEND_VERSION, SearchHit, SearchRequest, SearchResponse};

const PAGE_ID: &str = "handbook::docs/user/README.md";
const OTHER_ID: &str = "handbook::docs/user/quotas.md";

fn kanon(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_kanon"))
        .current_dir(dir)
        .arg("--config")
        .arg("kanon.yaml")
        .args(args)
        .output()
        .expect("kanon runs")
}

/// Answer one `POST /search` with `hits` (best first), tolerating the `Expect: 100-continue`
/// ureq sends with the request body. Returns the query the request carried.
fn serve_one_search(listener: &TcpListener, hits: &[&str]) -> String {
    let (mut stream, _) = listener.accept().unwrap();
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut answered_continue = false;
    let query = loop {
        let n = stream.read(&mut chunk).unwrap();
        assert!(n > 0, "connection closed before a full request arrived");
        buf.extend_from_slice(&chunk[..n]);
        let Some(header_end) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
            continue;
        };
        let headers = String::from_utf8_lossy(&buf[..header_end]).into_owned();
        if !answered_continue && headers.to_lowercase().contains("expect: 100-continue") {
            answer_continue(&mut stream);
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
        if buf.len() - (header_end + 4) >= content_length {
            let request: SearchRequest =
                serde_json::from_slice(&buf[header_end + 4..]).expect("a SearchRequest body");
            assert_eq!(request.version, BACKEND_VERSION);
            break request.query;
        }
    };
    let body = serde_json::to_string(&SearchResponse {
        version: BACKEND_VERSION,
        hits: hits
            .iter()
            .zip([2.0, 1.0])
            .map(|(id, score)| SearchHit {
                page_id: (*id).to_string(),
                score,
                heading: String::new(),
                unit_id: None,
            })
            .collect(),
    })
    .unwrap();
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(response.as_bytes()).unwrap();
    query
}

fn answer_continue(stream: &mut TcpStream) {
    stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").unwrap();
}

/// A one-page artifact, one query, and a `kanon.yaml` naming `old` and `new` at the two
/// servers. `old` answers with the wrong page first, `new` with the right one.
fn workspace() -> (tempfile::TempDir, TcpListener, TcpListener) {
    let dir = tempfile::tempdir().unwrap();
    let page = dir.path().join("artifact/handbook/docs/user/README.md");
    fs::create_dir_all(page.parent().unwrap()).unwrap();
    fs::write(&page, "# Storage\n\nEnable upload caching with a label.\n").unwrap();
    fs::write(
        dir.path().join("queries.jsonl"),
        "{\"id\": \"q\", \"kind\": \"howto\", \"query\": \"enable upload caching\", \
         \"expected\": [\"handbook/docs/user/README.md\"]}\n",
    )
    .unwrap();
    let old = TcpListener::bind("127.0.0.1:0").unwrap();
    let new = TcpListener::bind("127.0.0.1:0").unwrap();
    fs::write(
        dir.path().join("kanon.yaml"),
        format!(
            "queries: queries.jsonl\nbackends:\n  old:\n    type: external\n    url: http://{}\n  \
             new:\n    type: external\n    url: http://{}\n",
            old.local_addr().unwrap(),
            new.local_addr().unwrap()
        ),
    )
    .unwrap();
    (dir, old, new)
}

#[test]
fn compare_two_named_external_backends_keys_tables_and_json_by_name() {
    let (dir, old, new) = workspace();
    let old_thread = std::thread::spawn(move || serve_one_search(&old, &[OTHER_ID, PAGE_ID]));
    let new_thread = std::thread::spawn(move || serve_one_search(&new, &[PAGE_ID]));

    let out = kanon(dir.path(), &["eval", "--compare", "old,new"]);
    assert_eq!(old_thread.join().unwrap(), "enable upload caching");
    assert_eq!(new_thread.join().unwrap(), "enable upload caching");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("[old]"), "{stderr}");
    assert!(stderr.contains("[new]"), "{stderr}");
    assert_eq!(
        stderr.matches("| tuning | overall |").count(),
        2,
        "one table per name: {stderr}"
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    let combined: serde_json::Value =
        serde_json::from_str(&stdout).expect("combined JSON on stdout");
    let keys: Vec<&String> = combined.as_object().unwrap().keys().collect();
    assert_eq!(keys, ["new", "old"], "keyed by name");
    assert_eq!(combined["old"]["backend"], "old");
    assert_eq!(combined["new"]["backend"], "new");
    assert_eq!(combined["old"]["queries"][0]["top"][0], OTHER_ID);
    assert_eq!(combined["new"]["queries"][0]["top"][0], PAGE_ID);
    assert_eq!(combined["old"]["tuning"]["overall"]["mrr"], 0.5);
    assert_eq!(combined["new"]["tuning"]["overall"]["mrr"], 1.0);
}

#[test]
fn a_named_backend_works_alone_and_a_bad_name_or_config_is_exit_1() {
    let (dir, old, _new) = workspace();
    let old_thread = std::thread::spawn(move || serve_one_search(&old, &[PAGE_ID]));

    // `--backend old`: the result records the name, and no `--backend-url` is needed.
    let out = kanon(
        dir.path(),
        &["eval", "--backend", "old", "--json", "old.json"],
    );
    old_thread.join().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("[old]"));
    let text = fs::read_to_string(dir.path().join("old.json")).unwrap();
    let summary: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(summary["backend"], "old");

    // A name the config does not have.
    let out = kanon(dir.path(), &["eval", "--compare", "old,newer"]);
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unknown backend \"newer\""), "{stderr}");

    // A config whose entry shadows a built-in kind is refused, even for a bare eval.
    fs::write(
        dir.path().join("kanon.yaml"),
        "queries: queries.jsonl\nbackends:\n  bm25:\n    type: bm25\n",
    )
    .unwrap();
    let out = kanon(dir.path(), &["eval"]);
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("shadows the built-in backend bm25"),
        "{stderr}"
    );
}

/// `grade --backend NAME` resolves the same names: a trail query goes to the `new` server,
/// and the model call that follows fails fast because no `KANON_LLM_URL` is set.
#[test]
fn grade_takes_a_named_backend() {
    let (dir, _old, new) = workspace();
    fs::write(
        dir.path().join("trail.jsonl"),
        "{\"at\": \"t\", \"query\": \"enable upload caching\", \"retrieved\": [], \"cited\": []}\n",
    )
    .unwrap();
    let new_thread = std::thread::spawn(move || serve_one_search(&new, &[PAGE_ID]));
    let out = Command::new(env!("CARGO_BIN_EXE_kanon"))
        .current_dir(dir.path())
        .env_remove("KANON_LLM_URL")
        .env_remove("PINAKES_LLM_URL")
        .env("KANON_LLM_URL", "http://127.0.0.1:9")
        .args([
            "--config",
            "kanon.yaml",
            "grade",
            "--trail",
            "trail.jsonl",
            "--backend",
            "new",
            "--model",
            "grader",
        ])
        .output()
        .expect("kanon runs");
    assert_eq!(new_thread.join().unwrap(), "enable upload caching");
    assert_eq!(
        out.status.code(),
        Some(1),
        "the model endpoint is unreachable"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("unknown backend"), "{stderr}");
}
