pub(crate) mod embed;
pub(crate) mod eval;
pub(crate) mod grade;
pub(crate) mod history;
pub(crate) mod queries;
pub(crate) mod report;

/// Exit code for a failed `eval --gate`.
pub(crate) const EXIT_GATE: u8 = 2;
/// Exit code for a failed `queries check`.
pub(crate) const EXIT_POLICY: u8 = 4;
