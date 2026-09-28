//! The model endpoint `grade` and `queries suggest` talk to: [`pinakes::llm`]'s
//! OpenAI-compatible chat client, configured from `KANON_LLM_URL`, `KANON_LLM_KEY` and
//! `KANON_LLM_MODEL` (the `PINAKES_LLM_*` names are accepted as a fallback).
//!
//! [`chat_json`] is how both commands ask for JSON. The hosted models `grade` was written
//! against answer with the bare value; a small local model often wraps it in a markdown fence
//! or a sentence, so the reply is read leniently ([`extract_json`]) and a reply that still is
//! not JSON gets one more request with a stricter instruction before the caller sees an error.

use pinakes::llm::{self, ChatError, ChatTransport, LlmConfig};
use serde::de::DeserializeOwned;
use thiserror::Error;

/// Added to the system prompt of the one retry [`chat_json`] makes.
const STRICTER: &str = "\n\nYour previous reply could not be parsed. Reply with the JSON value \
                        only: no text before or after it, no markdown code fences, no comments.";

/// Errors raised while building an [`LlmConfig`] from the environment.
#[derive(Debug, Error)]
pub enum LlmEnvError {
    /// `KANON_LLM_URL` is not set.
    #[error(
        "KANON_LLM_URL is not set: grade and queries suggest need an OpenAI-compatible chat \
         completions endpoint"
    )]
    MissingUrl,
    /// Neither `--model` nor `KANON_LLM_MODEL` is set.
    #[error("no model given: pass --model or set KANON_LLM_MODEL")]
    MissingModel,
}

/// Read `KANON_LLM_URL`, `KANON_LLM_KEY` and `KANON_LLM_MODEL` from the environment;
/// `model_override` (a command's `--model`) takes precedence over `KANON_LLM_MODEL`.
pub fn config_from_env(model_override: Option<String>) -> Result<LlmConfig, LlmEnvError> {
    let url = crate::env::var("KANON_LLM_URL").ok_or(LlmEnvError::MissingUrl)?;
    let key = crate::env::var("KANON_LLM_KEY");
    let model = model_override
        .filter(|m| !m.trim().is_empty())
        .or_else(|| crate::env::var("KANON_LLM_MODEL"))
        .ok_or(LlmEnvError::MissingModel)?;
    Ok(LlmConfig { url, key, model })
}

/// The longest reply [`extract_json`] searches for a value inside; a longer one has to be the
/// value. Every `[` or `{` is a place to try, so the search is not free, and no answer to
/// `grade` or `queries suggest` comes near this size.
const MAX_SCAN: usize = 64 * 1024;

/// Whether a JSON value says nothing: `null`, `{}`, `[]`, or an array of such. A value like
/// that parses as almost any list of all-default records, so found inside prose it is not an
/// answer.
fn is_blank(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => true,
        serde_json::Value::Object(fields) => fields.is_empty(),
        serde_json::Value::Array(items) => items.iter().all(is_blank),
        _ => false,
    }
}

/// The JSON value of type `T` in `text`: the whole text when it is one (an empty list is a
/// fine answer then), else the last value inside it that parses as `T`, so a fenced block
/// (```` ```json ````), a sentence around the value, a reasoning preamble, or a format example
/// the model echoed before its answer does not hide it. Blank values found inside prose
/// (`[]`, `{}`) are not answers, a value nested in one already taken is not taken again, and
/// text after the value is ignored. `None` when there is no such value.
pub fn extract_json<T: DeserializeOwned>(text: &str) -> Option<T> {
    if let Ok(value) = serde_json::from_str(text.trim()) {
        return Some(value);
    }
    if text.len() > MAX_SCAN {
        return None;
    }
    let mut found = None;
    let mut resume = 0;
    for (start, c) in text.char_indices() {
        if start < resume || !matches!(c, '[' | '{') {
            continue;
        }
        let mut values =
            serde_json::Deserializer::from_str(&text[start..]).into_iter::<serde_json::Value>();
        let Some(Ok(value)) = values.next() else {
            continue;
        };
        if is_blank(&value) {
            continue;
        }
        let end = start + values.byte_offset();
        if let Ok(parsed) = serde_json::from_value::<T>(value) {
            found = Some(parsed);
            resume = end;
        }
    }
    found
}

/// Ask the model for a JSON value of type `T` ([`pinakes::llm::chat`]) and read the reply
/// leniently ([`extract_json`]). A reply with no such value is asked for once more with a
/// stricter instruction added to `system`; if that one has none either, the error is the
/// second reply's [`ChatError::Json`], raw text included. Any other error on the first request
/// (HTTP, transport, a malformed completion) is returned at once: asking again would not change
/// it. When the retry itself fails for another reason, the caller gets the first reply's
/// [`ChatError::Json`], which is what was wrong with the answer, rather than a timeout on the
/// second try, so a command that skips a page it cannot read skips this one too.
pub fn chat_json<T: DeserializeOwned>(
    transport: &dyn ChatTransport,
    config: &LlmConfig,
    system: &str,
    user: &str,
) -> Result<T, ChatError> {
    let first = match llm::chat::<T>(transport, config, system, user) {
        Err(ChatError::Json { raw, source }) => match extract_json(&raw) {
            Some(value) => return Ok(value),
            None => ChatError::Json { raw, source },
        },
        other => return other,
    };
    let stricter = format!("{system}{STRICTER}");
    match llm::chat::<T>(transport, config, &stricter, user) {
        Err(ChatError::Json { raw, source }) => {
            extract_json(&raw).ok_or(ChatError::Json { raw, source })
        }
        Err(_) => Err(first),
        ok => ok,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::ENV_LOCK;
    use pinakes::llm::TransportError;
    use pinakes::llm::testing::{Scripted, ScriptedTransport, completion};

    #[derive(Debug, PartialEq, serde::Deserialize)]
    struct Grade {
        id: String,
        grade: u8,
    }

    /// A record whose every field has a default, so it parses from any object.
    #[derive(Debug, PartialEq, serde::Deserialize)]
    struct Loose {
        #[serde(default)]
        query: String,
    }

    fn grades(text: &str) -> Option<Vec<Grade>> {
        extract_json(text)
    }

    fn one(id: &str, grade: u8) -> Vec<Grade> {
        vec![Grade {
            id: id.to_string(),
            grade,
        }]
    }

    fn config() -> LlmConfig {
        LlmConfig {
            url: "http://model.test/v1".to_string(),
            key: None,
            model: "small".to_string(),
        }
    }

    #[test]
    fn extract_json_reads_the_bare_fenced_and_wrapped_forms_a_small_model_answers_in() {
        let bare = r#"[{"id": "a", "grade": 2}]"#;
        assert_eq!(grades(bare).unwrap(), one("a", 2));
        assert_eq!(grades(&format!("  \n{bare}\n")).unwrap(), one("a", 2));
        assert_eq!(
            grades(&format!("```json\n{bare}\n```")).unwrap(),
            one("a", 2)
        );
        assert_eq!(grades(&format!("```\n{bare}\n```\n")).unwrap(), one("a", 2));
        assert_eq!(
            grades(&format!(
                "Sure! Here are the grades:\n{bare}\nHope that helps."
            ))
            .unwrap(),
            one("a", 2)
        );
        // A reasoning preamble with brackets of its own, and prose that is not the value.
        assert_eq!(
            grades(&format!("<think>ids [a] and {{b}} first</think>\n{bare}")).unwrap(),
            one("a", 2)
        );
        // Text after the value is ignored, whatever it holds.
        assert_eq!(grades(&format!("{bare} and [1, 2]")).unwrap(), one("a", 2));
    }

    #[test]
    fn extract_json_does_not_take_a_blank_or_echoed_value_for_the_answer() {
        let real = r#"[{"id": "a", "grade": 3}]"#;
        // An empty list in the prose, and the format example the model repeated first.
        assert_eq!(
            grades(&format!("I will grade the [] candidates.\n{real}")).unwrap(),
            one("a", 3)
        );
        assert_eq!(
            grades(&format!(
                r#"Format: [{{"id": "...", "grade": 0}}] Answer: {real}"#
            ))
            .unwrap(),
            one("a", 3)
        );
        // Records whose fields all have defaults match anything; blank ones are not answers.
        let loose: Vec<Loose> =
            extract_json(r#"Refs [{}] then real [{"query": "x"}] done"#).unwrap();
        assert_eq!(
            loose,
            [Loose {
                query: "x".to_string()
            }]
        );
        // A wrapper object around the list is looked through, and an empty list inside one is
        // not an answer, so the reply is asked for again.
        assert_eq!(
            grades(&format!(r#"{{"grades": {real}}}"#)).unwrap(),
            one("a", 3)
        );
        assert_eq!(grades(r#"{"grades": []}"#), None);
        // An empty list is a fine answer when it is the whole reply.
        assert_eq!(grades("[]").unwrap(), Vec::<Grade>::new());
        assert_eq!(grades(" \n[]\n ").unwrap(), Vec::<Grade>::new());
    }

    #[test]
    fn extract_json_does_not_search_a_reply_past_the_scan_limit() {
        let value = r#"[{"id": "a", "grade": 1}]"#;
        let long = format!("{}{value}", "x ".repeat(MAX_SCAN));
        assert_eq!(grades(&long), None);
        // The whole reply is still read at any size.
        let padded = format!("{value}{}", " ".repeat(MAX_SCAN * 2));
        assert_eq!(grades(&padded).unwrap(), one("a", 1));
    }

    #[test]
    fn extract_json_gives_none_for_replies_with_no_value_of_the_type() {
        for text in [
            "",
            "I cannot grade these.",
            r#"{"id": "a", "grade": 2}"#,
            r#"[{"id": "a", "grade": "high"}]"#,
            r#"[{"id": "a", "grade": 2}, {"id": "b""#,
        ] {
            assert_eq!(grades(text), None, "{text:?}");
        }
    }

    #[test]
    fn a_wrapped_reply_is_read_without_asking_again() {
        let transport = ScriptedTransport::new(vec![Scripted::Ok(completion(
            "```json\n[{\"id\": \"a\", \"grade\": 3}]\n```",
        ))]);
        let got: Vec<Grade> = chat_json(&transport, &config(), "sys", "user").unwrap();
        assert_eq!(got, one("a", 3));
        assert_eq!(transport.requests.lock().unwrap().len(), 1);
    }

    #[test]
    fn a_reply_with_no_json_is_asked_for_once_more_with_a_stricter_instruction() {
        let transport = ScriptedTransport::new(vec![
            Scripted::Ok(completion("These look fine to me.")),
            Scripted::Ok(completion(r#"[{"id": "a", "grade": 1}]"#)),
        ]);
        let got: Vec<Grade> = chat_json(&transport, &config(), "sys", "user").unwrap();
        assert_eq!(got, one("a", 1));
        let requests = transport.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0]["messages"][0]["content"], "sys");
        let retry = requests[1]["messages"][0]["content"].as_str().unwrap();
        assert!(retry.starts_with("sys"), "{retry}");
        assert!(retry.contains("could not be parsed"), "{retry}");
        assert_eq!(
            requests[1]["messages"][1]["content"], "user",
            "same question"
        );
    }

    #[test]
    fn a_reply_that_is_still_not_json_after_the_retry_is_the_second_replys_error() {
        let transport = ScriptedTransport::new(vec![
            Scripted::Ok(completion("first")),
            Scripted::Ok(completion("second")),
            Scripted::Ok(completion(r#"[{"id": "a", "grade": 1}]"#)),
        ]);
        let err = chat_json::<Vec<Grade>>(&transport, &config(), "sys", "user").unwrap_err();
        assert!(
            matches!(&err, ChatError::Json { raw, .. } if raw == "second"),
            "{err}"
        );
        assert_eq!(
            transport.requests.lock().unwrap().len(),
            2,
            "one retry, never more"
        );
    }

    #[test]
    fn a_retry_that_answers_in_a_wrapper_is_read_too() {
        let transport = ScriptedTransport::new(vec![
            Scripted::Ok(completion("nope")),
            Scripted::Ok(completion("Here: [{\"id\": \"a\", \"grade\": 0}]")),
        ]);
        let got: Vec<Grade> = chat_json(&transport, &config(), "sys", "user").unwrap();
        assert_eq!(got, one("a", 0));
    }

    #[test]
    fn a_retry_that_fails_for_another_reason_reports_the_first_reply() {
        let transport = ScriptedTransport::new(vec![
            Scripted::Ok(completion("the first, unusable reply")),
            Scripted::Err(TransportError::Transport("timed out".to_string())),
        ]);
        let err = chat_json::<Vec<Grade>>(&transport, &config(), "sys", "user").unwrap_err();
        assert!(
            matches!(&err, ChatError::Json { raw, .. } if raw == "the first, unusable reply"),
            "{err}"
        );
        assert_eq!(transport.requests.lock().unwrap().len(), 2);
    }

    #[test]
    fn an_http_error_is_not_retried_either() {
        let transport = ScriptedTransport::new(vec![Scripted::Err(TransportError::Status(
            401,
            "unauthorized".to_string(),
        ))]);
        let err = chat_json::<Vec<Grade>>(&transport, &config(), "sys", "user").unwrap_err();
        assert!(matches!(err, ChatError::Http { status: 401, .. }), "{err}");
        assert_eq!(transport.requests.lock().unwrap().len(), 1);
    }

    #[test]
    fn an_endpoint_error_is_not_retried() {
        let transport = ScriptedTransport::new(vec![Scripted::Err(TransportError::Transport(
            "connection refused".to_string(),
        ))]);
        let err = chat_json::<Vec<Grade>>(&transport, &config(), "sys", "user").unwrap_err();
        assert!(matches!(err, ChatError::Transport { .. }), "{err}");
        assert_eq!(transport.requests.lock().unwrap().len(), 1);
    }

    #[test]
    fn missing_url_is_an_error_not_a_silent_skip() {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: serialised by ENV_LOCK; no other test observes these vars concurrently.
        unsafe {
            std::env::remove_var("KANON_LLM_URL");
            std::env::remove_var("PINAKES_LLM_URL");
        }
        assert!(matches!(
            config_from_env(None).unwrap_err(),
            LlmEnvError::MissingUrl
        ));
    }

    #[test]
    fn model_override_wins_and_pinakes_names_are_a_fallback() {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: serialised by ENV_LOCK; no other test observes these vars concurrently.
        unsafe {
            for name in [
                "KANON_LLM_URL",
                "PINAKES_LLM_URL",
                "KANON_LLM_MODEL",
                "PINAKES_LLM_MODEL",
            ] {
                std::env::remove_var(name);
            }
            std::env::set_var("PINAKES_LLM_URL", "https://fallback.test");
            std::env::set_var("KANON_LLM_MODEL", "env-model");
        }
        let config = config_from_env(Some("cli-model".to_string())).unwrap();
        assert_eq!(config.model, "cli-model");
        assert_eq!(config.url, "https://fallback.test");
        let config = config_from_env(None).unwrap();
        assert_eq!(config.model, "env-model");
        // SAFETY: serialised by ENV_LOCK; no other test observes these vars concurrently.
        unsafe {
            std::env::set_var("KANON_LLM_URL", "https://kanon.test");
        }
        assert_eq!(config_from_env(None).unwrap().url, "https://kanon.test");
        // SAFETY: serialised by ENV_LOCK; no other test observes these vars concurrently.
        unsafe {
            std::env::remove_var("KANON_LLM_URL");
            std::env::remove_var("PINAKES_LLM_URL");
            std::env::remove_var("KANON_LLM_MODEL");
        }
    }
}
