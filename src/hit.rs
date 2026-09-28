//! [`Hit`]: one page a retriever returned for a query, the row `eval` and `grade` read.

/// One page of a result list, best first.
///
/// The page is what `eval` scores. A retriever that works on units also says which unit of the
/// page matched: `unit_id` is that unit's id (`<page_id>#<ordinal>`, as `pinakes::chunks`
/// numbers them) and is kept only so a run file can show it; the built-in backends rank pages
/// and leave it empty.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    /// `<source>::<path>`.
    pub page_id: String,
    /// The backend's score of the page's best unit.
    pub score: f64,
    /// Heading of the best unit, empty for the intro.
    pub heading: String,
    /// The best unit's id, when the backend reports units.
    pub unit_id: Option<String>,
}

impl From<pinakes::index::Hit> for Hit {
    fn from(hit: pinakes::index::Hit) -> Hit {
        Hit {
            page_id: hit.page_id,
            score: hit.score,
            heading: hit.heading,
            unit_id: None,
        }
    }
}
