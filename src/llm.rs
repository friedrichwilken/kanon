//! The model endpoint `grade` talks to: [`pinakes::llm`]'s OpenAI-compatible chat client,
//! configured from `KANON_LLM_URL`, `KANON_LLM_KEY` and `KANON_LLM_MODEL` (the `PINAKES_LLM_*`
//! names are accepted as a fallback).

use pinakes::llm::LlmConfig;
use thiserror::Error;

/// Errors raised while building an [`LlmConfig`] from the environment.
#[derive(Debug, Error)]
pub enum LlmEnvError {
    /// `KANON_LLM_URL` is not set.
    #[error("KANON_LLM_URL is not set: grade needs an OpenAI-compatible chat completions endpoint")]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::ENV_LOCK;

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
