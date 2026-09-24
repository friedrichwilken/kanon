//! Environment variables: `KANON_*`, with the `PINAKES_*` names the moved commands used to
//! read accepted as a fallback so an existing setup keeps working.

/// The value of `name` (a `KANON_*` variable), or of its `PINAKES_*` counterpart when `name`
/// is unset; an empty or blank value counts as unset.
pub(crate) fn var(name: &str) -> Option<String> {
    let set = |name: &str| std::env::var(name).ok().filter(|v| !v.trim().is_empty());
    set(name).or_else(|| {
        name.strip_prefix("KANON_")
            .and_then(|rest| set(&format!("PINAKES_{rest}")))
    })
}
