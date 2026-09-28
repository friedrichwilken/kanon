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

/// The first JSON value of type `T` in `text`: the whole text when it is one, else the first
/// value that starts at a `[` or `{` and parses, so a fenced block (```` ```json ````), a
/// sentence around the value, or a reasoning preamble does not hide it. Anything after the value
/// is ignored. `None` when no such value exists.
pub fn extract_json<T: DeserializeOwned>(text: &str) -> Option<T> {
    if let Ok(value) = serde_json::from_str(text.trim()) {
        return Some(value);
    }
    text.char_indices()
        .filter(|(_, c)| matches!(c, '[' | '{'))
        .find_map(|(start, _)| {
            serde_json::Deserializer::from_str(&text[start..])
                .into_iter::<T>()
                .next()?
                .ok()
        })
}

/// Ask the model for a JSON value of type `T` ([`pinakes::llm::chat`]) and read the reply
/// leniently ([`extract_json`]). A reply with no such value is asked for once more with a
/// stricter instruction added to `system`; if that one has none either, the error is the
/// second reply's [`ChatError::Json`], raw text included. Any other error (HTTP, transport, a
/// malformed completion) is returned at once: asking again would not change it.
pub fn chat_json<T: DeserializeOwned>(
    transport: &dyn ChatTransport,
    config: &LlmConfig,
    system: &str,
    user: &str,
) -> Result<T, ChatError> {
    match llm::chat::<T>(transport, config, system, user) {
        Err(ChatError::Json { raw, .. }) => {
            if let Some(value) = extract_json(&raw) {
                return Ok(value);
            }
        }
        other => return other,
    }
    let stricter = format!("{system}{STRICTER}");
    match llm::chat::<T>(transport, config, &stricter, user) {
        Err(ChatError::Json { raw, source }) => {
            extract_json(&raw).ok_or(ChatError::Json { raw, source })
        }
        other => other,
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
