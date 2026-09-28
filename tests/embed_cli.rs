//! `kanon embed` end to end through the binary: a tiny in-process HTTP server
//! stands in for an OpenAI-compatible embeddings endpoint.

use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::Command;

fn kanon(dir: &Path, args: &[&str], env: &[(&str, &str)]) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_kanon"));
    command
        .current_dir(dir)
        .arg("--config")
        .arg("nonexistent.yaml")
        .args(args);
    for (key, value) in env {
        command.env(key, value);
    }
    command.output().expect("kanon runs")
}

/// Answer one `POST /embeddings` on `listener` with a two-dimensional vector per input text and
/// return the texts it was asked to embed. Tolerates `Expect: 100-continue`.
fn serve_embeddings(listener: &TcpListener) -> Vec<String> {
    let (mut stream, _) = listener.accept().unwrap();
    let Some(request) = read_request(&mut stream) else {
        return Vec::new();
    };
    let body_start = request.find("\r\n\r\n").unwrap() + 4;
    let body: serde_json::Value = serde_json::from_str(&request[body_start..]).unwrap();
    let inputs: Vec<String> = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .map(|text| text.as_str().unwrap().to_string())
        .collect();
    let count = inputs.len();
    let data: Vec<_> = (0..count)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let index = i as f64;
            serde_json::json!({"embedding": [index, 1.0]})
        })
        .collect();
    let response_body = serde_json::json!({ "data": data }).to_string();
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        response_body.len(),
        response_body
    );
    stream.write_all(response.as_bytes()).unwrap();
    inputs
}

fn read_request(stream: &mut TcpStream) -> Option<String> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut answered_continue = false;
    loop {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
        let Some(header_end) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
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
        if buf.len() - (header_end + 4) >= content_length {
            return Some(String::from_utf8_lossy(&buf).into_owned());
        }
    }
}

fn workspace() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let page = dir.path().join("artifact/handbook/docs/user/README.md");
    fs::create_dir_all(page.parent().unwrap()).unwrap();
    fs::write(&page, "# Storage\n\nEnable upload caching with a label.\n").unwrap();
    dir
}

#[test]
fn embed_writes_the_embeddings_file_pair() {
    let dir = workspace();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || serve_embeddings(&listener));

    let out = kanon(
        dir.path(),
        &["embed"],
        &[
            ("KANON_EMBED_URL", &format!("http://{addr}")),
            ("KANON_EMBED_MODEL", "test-model"),
        ],
    );
    handle.join().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bin = dir.path().join("embeddings.bin");
    let json = dir.path().join("embeddings.json");
    assert!(bin.is_file());
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&json).unwrap()).unwrap();
    assert_eq!(manifest["model"], "test-model");
    assert_eq!(manifest["dimension"], 2);
    assert_eq!(manifest["unit_ids"][0], "handbook::docs/user/README.md");
    assert_eq!(manifest["manifest_sha256"], "none");
    assert_eq!(fs::metadata(&bin).unwrap().len(), 8, "one unit, 2 f32s");

    // A missing KANON_EMBED_URL is an error, not a silent skip.
    let out = Command::new(env!("CARGO_BIN_EXE_kanon"))
        .current_dir(dir.path())
        .env_remove("KANON_EMBED_URL")
        .env_remove("PINAKES_EMBED_URL")
        .arg("--config")
        .arg("nonexistent.yaml")
        .arg("embed")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("KANON_EMBED_URL"));
}

/// Run `kanon embed` with `args` against a one-shot endpoint; the texts it received, the
/// parsed `embeddings.json` and stderr.
fn embed_with(args: &[&str], model: &str) -> (Vec<String>, serde_json::Value, String) {
    let dir = workspace();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = std::thread::spawn(move || serve_embeddings(&listener));
    let mut all = vec!["embed", "--model", model];
    all.extend_from_slice(args);
    let out = kanon(
        dir.path(),
        &all,
        &[("KANON_EMBED_URL", &format!("http://{addr}"))],
    );
    let received = handle.join().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let manifest =
        serde_json::from_str(&fs::read_to_string(dir.path().join("embeddings.json")).unwrap())
            .unwrap();
    (
        received,
        manifest,
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn embed_gives_a_known_model_its_prefixes_and_records_them() {
    let (received, manifest, stderr) = embed_with(&[], "nomic-embed-text");
    assert_eq!(received.len(), 1);
    assert!(received[0].starts_with("search_document: "), "{received:?}");
    assert_eq!(manifest["doc_prefix"], "search_document: ");
    assert_eq!(manifest["query_prefix"], "search_query: ");
    assert!(
        stderr.contains("prefixes: documents \"search_document: \", queries \"search_query: \""),
        "{stderr}"
    );
}

#[test]
fn embed_flags_set_or_switch_off_the_prefixes() {
    let (received, manifest, _) = embed_with(
        &["--doc-prefix", "passage: ", "--query-prefix", "query: "],
        "some-other-model",
    );
    assert!(received[0].starts_with("passage: "), "{received:?}");
    assert_eq!(manifest["doc_prefix"], "passage: ");
    assert_eq!(manifest["query_prefix"], "query: ");

    // An empty value is a choice: a known model's prefixes are switched off.
    let (received, manifest, stderr) = embed_with(
        &["--doc-prefix", "", "--query-prefix", ""],
        "nomic-embed-text",
    );
    assert!(!received[0].starts_with("search_document"), "{received:?}");
    assert_eq!(manifest["doc_prefix"], "");
    assert_eq!(manifest["query_prefix"], "");
    assert!(stderr.contains("prefixes: none"), "{stderr}");

    // A model that is not known gets none when nothing is given.
    let (received, manifest, _) = embed_with(&[], "some-other-model");
    assert!(received[0].starts_with("Storage"), "{received:?}");
    assert_eq!(manifest["doc_prefix"], "");
}

#[test]
fn eval_embeds_each_query_with_the_prefix_embed_recorded_for_dense_and_hybrid() {
    let dir = workspace();
    fs::write(
        dir.path().join("queries.jsonl"),
        "{\"id\": \"q\", \"kind\": \"howto\", \"query\": \"enable upload caching\", \
         \"expected\": [\"handbook/docs/user\"]}\n",
    )
    .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}");
    // One request for `embed`, one query embedding each for `dense` and for `hybrid`.
    let handle = std::thread::spawn(move || {
        (0..3)
            .map(|_| serve_embeddings(&listener))
            .collect::<Vec<_>>()
    });
    let env = [("KANON_EMBED_URL", url.as_str())];

    let out = kanon(dir.path(), &["embed", "--model", "nomic-embed-text"], &env);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    for backend in ["dense", "hybrid"] {
        let out = kanon(
            dir.path(),
            &["eval", "--queries", "queries.jsonl", "--backend", backend],
            &env,
        );
        assert!(
            out.status.success(),
            "{backend}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let received = handle.join().unwrap();
    assert!(
        received[0][0].starts_with("search_document: "),
        "{received:?}"
    );
    // The model and the prefix come from embeddings.json; only the URL is from the environment.
    assert_eq!(received[1], ["search_query: enable upload caching"]);
    assert_eq!(received[2], ["search_query: enable upload caching"]);
}

#[test]
fn the_config_file_sets_the_prefixes_and_a_flag_beats_it() {
    let dir = workspace();
    fs::write(
        dir.path().join("kanon.yaml"),
        "queries: queries.jsonl\ndoc_prefix: 'cfg-doc: '\nquery_prefix: 'cfg-query: '\n",
    )
    .unwrap();
    let embed = |args: &[&str]| {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || serve_embeddings(&listener));
        let out = Command::new(env!("CARGO_BIN_EXE_kanon"))
            .current_dir(dir.path())
            .env("KANON_EMBED_URL", format!("http://{addr}"))
            .args([
                "--config",
                "kanon.yaml",
                "embed",
                "--model",
                "some-other-model",
            ])
            .args(args)
            .output()
            .unwrap();
        let received = handle.join().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.path().join("embeddings.json")).unwrap())
                .unwrap();
        (received, json)
    };
    let (received, json) = embed(&[]);
    assert!(received[0].starts_with("cfg-doc: "), "{received:?}");
    assert_eq!(json["query_prefix"], "cfg-query: ");
    let (received, json) = embed(&["--doc-prefix", "flag-doc: "]);
    assert!(received[0].starts_with("flag-doc: "), "{received:?}");
    assert_eq!(json["doc_prefix"], "flag-doc: ");
    assert_eq!(
        json["query_prefix"], "cfg-query: ",
        "the other side still comes from the config"
    );
}
