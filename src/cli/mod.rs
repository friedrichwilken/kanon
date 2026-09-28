pub(crate) mod embed;
pub(crate) mod eval;
pub(crate) mod grade;
pub(crate) mod history;
pub(crate) mod queries;
pub(crate) mod report;

/// Exit code for a failed `eval --gate` or a failed cost budget (`--max-p95-ms`, `--max-tokens`).
pub(crate) const EXIT_GATE: u8 = 2;
/// Exit code for a failed `queries check`.
pub(crate) const EXIT_POLICY: u8 = 4;
