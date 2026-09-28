use std::path::PathBuf;

use crate::artifact;
use crate::config::Config;
use crate::embed::{self, Embedder};
use crate::error::CommandError;
use crate::workspace::Paths;
use pinakes::index;

use super::settings;

// -------------------------------------------------------------------------------------------
// embed
// -------------------------------------------------------------------------------------------

/// Options for `embed`.
#[derive(Debug, Clone, Default)]
pub struct EmbedOptions {
    /// The embedding model name, recorded in `embeddings.json`.
    pub model: String,
    /// Texts per embedding request.
    pub batch: usize,
    /// `embeddings.bin` output path (default: `paths.embeddings`); `embeddings.json` is written
    /// next to it, with a `.json` extension.
    pub out: Option<PathBuf>,
    /// The text put in front of every unit before it is embedded. `None` takes the config's
    /// `doc_prefix`, else the model's known one ([`embed::known_prefixes`]), else nothing; an
    /// empty string means none whatever the model.
    pub doc_prefix: Option<String>,
    /// The text `dense` and `hybrid` put in front of every query, recorded in
    /// `embeddings.json`; resolved like `doc_prefix` from the config's `query_prefix`.
    pub query_prefix: Option<String>,
}

/// What `embed` wrote.
#[derive(Debug)]
pub struct EmbedOutcome {
    /// Retrieval units embedded.
    pub units: usize,
    /// Vector length.
    pub dimension: usize,
    /// `embeddings.bin` path.
    pub bin_path: PathBuf,
    /// `embeddings.json` path.
    pub json_path: PathBuf,
    /// The prefix put in front of every unit, as recorded in `embeddings.json`.
    pub doc_prefix: String,
    /// The prefix `dense` and `hybrid` put in front of every query, as recorded.
    pub query_prefix: String,
}

/// The prefixes `embed` uses, each side on its own: the option, else the config, else what
/// the model is known to want, else none. An explicit empty string is a choice, not an absence.
fn resolve_prefixes(options: &EmbedOptions, config: Option<&Config>) -> embed::Prefixes {
    let known = embed::known_prefixes(&options.model).unwrap_or_default();
    let pick = |option: &Option<String>, configured: Option<&Option<String>>, known: String| {
        option
            .clone()
            .or_else(|| configured.and_then(Clone::clone))
            .unwrap_or(known)
    };
    embed::Prefixes {
        doc: pick(
            &options.doc_prefix,
            config.map(|c| &c.doc_prefix),
            known.doc,
        ),
        query: pick(
            &options.query_prefix,
            config.map(|c| &c.query_prefix),
            known.query,
        ),
    }
}

/// Run `embed`: one embedding per retrieval unit of the artifact, through `embedder`.
///
/// The config is optional, exactly as for `eval`: priorities default to
/// [`pinakes::index::Priorities::default`] without one.
pub fn embed(
    paths: &Paths,
    options: &EmbedOptions,
    embedder: &dyn Embedder,
) -> Result<EmbedOutcome, CommandError> {
    let settings = settings(paths)?;
    let prefixes = resolve_prefixes(options, settings.config.as_ref());
    let priorities = settings.priorities;
    // The manifest is hashed below, so it is read through the loader first: a manifest of a
    // newer artifact contract stops the run before any text is embedded.
    artifact::manifest_artifact_version(&paths.artifact)?;
    let mut pages = index::load_pages(&paths.artifact, &priorities)?;
    index::mark_mirrors(&mut pages);
    let units = index::iter_units(&pages);
    let texts: Vec<String> = units
        .iter()
        .map(|u| format!("{}{}", prefixes.doc, u.text))
        .collect();
    let batch = if options.batch == 0 {
        embed::DEFAULT_BATCH
    } else {
        options.batch
    };
    let vectors = embed::embed_units(embedder, &options.model, &texts, batch)?;
    let dimension = vectors.first().map_or(0, Vec::len);
    let manifest = embed::EmbeddingsManifest {
        model: options.model.clone(),
        dimension,
        unit_ids: units.into_iter().map(|u| u.page_id).collect(),
        manifest_sha256: embed::artifact_manifest_hash(&paths.artifact)?,
        doc_prefix: prefixes.doc,
        query_prefix: prefixes.query,
    };
    let bin_path = options
        .out
        .clone()
        .unwrap_or_else(|| paths.embeddings.clone());
    let json_path = bin_path.with_extension("json");
    embed::write_embeddings(&bin_path, &json_path, &manifest, &vectors)?;
    Ok(EmbedOutcome {
        units: manifest.unit_ids.len(),
        dimension,
        bin_path,
        json_path,
        doc_prefix: manifest.doc_prefix,
        query_prefix: manifest.query_prefix,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::eval_workspace;

    fn config(yaml: &str) -> Config {
        serde_yaml_ng::from_str(&format!("queries: queries.jsonl\n{yaml}")).unwrap()
    }

    fn options(model: &str, doc: Option<&str>, query: Option<&str>) -> EmbedOptions {
        EmbedOptions {
            model: model.to_string(),
            doc_prefix: doc.map(str::to_string),
            query_prefix: query.map(str::to_string),
            ..EmbedOptions::default()
        }
    }

    fn resolved(options: &EmbedOptions, config: Option<&Config>) -> (String, String) {
        let prefixes = resolve_prefixes(options, config);
        (prefixes.doc, prefixes.query)
    }

    #[test]
    fn prefixes_come_from_the_option_then_the_config_then_the_model_then_nothing() {
        let pair = |a: &str, b: &str| (a.to_string(), b.to_string());
        let nomic = pair("search_document: ", "search_query: ");
        let cfg = config("doc_prefix: 'cfg-doc: '\nquery_prefix: 'cfg-query: '\n");
        let empty_cfg = config("doc_prefix: ''\n");

        // Nothing given: none for an unknown model, the model's own for a known one.
        assert_eq!(resolved(&options("fake", None, None), None), pair("", ""));
        assert_eq!(
            resolved(&options("nomic-embed-text", None, None), None),
            nomic
        );
        // The config beats the model, the option beats the config.
        let known = options("nomic-embed-text", None, None);
        assert_eq!(
            resolved(&known, Some(&cfg)),
            pair("cfg-doc: ", "cfg-query: ")
        );
        let flags = options("nomic-embed-text", Some("flag-doc: "), Some("flag-query: "));
        assert_eq!(
            resolved(&flags, Some(&cfg)),
            pair("flag-doc: ", "flag-query: ")
        );
        // An explicit empty string is a choice: it switches a known model's prefix off.
        let off = options("nomic-embed-text", Some(""), Some(""));
        assert_eq!(resolved(&off, None), pair("", ""));
        assert_eq!(
            resolved(&known, Some(&empty_cfg)),
            pair("", "search_query: "),
            "an empty config value beats the model, a missing one does not"
        );
        // The two sides are resolved on their own.
        let doc_only = options("nomic-embed-text", Some("only-doc: "), None);
        assert_eq!(
            resolved(&doc_only, None),
            pair("only-doc: ", "search_query: ")
        );
    }

    #[test]
    fn embed_puts_the_document_prefix_in_front_of_every_unit_and_records_both_sides() {
        use crate::embed::testing::RecordingEmbedder;

        let (_dir, paths) = eval_workspace();
        let embedder = RecordingEmbedder::default();
        let outcome = embed(&paths, &options("nomic-embed-text", None, None), &embedder).unwrap();
        let inputs = embedder.inputs.lock().unwrap();
        assert_eq!(inputs.len(), outcome.units);
        assert!(
            inputs
                .iter()
                .all(|text| text.starts_with("search_document: ")),
            "{inputs:?}"
        );
        assert_eq!(outcome.doc_prefix, "search_document: ");
        assert_eq!(outcome.query_prefix, "search_query: ");
        let (manifest, _) =
            crate::embed::read_embeddings(&outcome.bin_path, &outcome.json_path).unwrap();
        assert_eq!(manifest.doc_prefix, "search_document: ");
        assert_eq!(manifest.query_prefix, "search_query: ");

        // A model that wants none embeds the bare unit text and records none.
        let plain = RecordingEmbedder::default();
        let outcome = embed(&paths, &options("fake", None, None), &plain).unwrap();
        let bare = plain.inputs.lock().unwrap();
        assert!(
            bare.iter().all(|text| !text.starts_with("search_document")),
            "{bare:?}"
        );
        assert_eq!(
            (outcome.doc_prefix.as_str(), outcome.query_prefix.as_str()),
            ("", "")
        );
    }

    #[test]
    fn embed_writes_units_for_every_searchable_page() {
        let (_dir, paths) = eval_workspace();
        let options = EmbedOptions {
            model: "fake".to_string(),
            batch: 1,
            out: None,
            ..EmbedOptions::default()
        };
        let outcome = embed(&paths, &options, &crate::embed::testing::FakeEmbedder).unwrap();
        assert_eq!(outcome.units, 1, "one intro unit on the single page");
        assert_eq!(outcome.dimension, crate::embed::testing::FAKE_DIMENSION);
        assert_eq!(outcome.bin_path, paths.embeddings);
        assert_eq!(outcome.json_path, paths.embeddings.with_extension("json"));
        let (manifest, vectors) =
            crate::embed::read_embeddings(&outcome.bin_path, &outcome.json_path).unwrap();
        assert_eq!(manifest.unit_ids, ["handbook::docs/user/README.md"]);
        assert_eq!(vectors.len(), 1);
    }
}
