//! What a retriever costs its caller, next to what it gets right: how long a search takes, and
//! how many tokens the units it returns add to the reader's context.
//!
//! A [`CostMeter`] wraps each search of an `eval` run: it times the call and adds up the tokens
//! of the units the hits name, top 5 and top 10 like recall. [`CostMeter::finish`] folds that
//! into a [`Cost`], which a result carries as `cost`. A [`Budget`] turns it into an optional gate.
//!
//! Tokens are counted with the tokeniser the reference index uses ([`pinakes::index::tokenize`]),
//! over a unit's text as `embed` embeds it: lower-cased words with the common stop words left
//! out, and an identifier compound counted joined as well as split (`spec.sink` is `spec`, `sink`
//! and `specsink`). That is not a model's tokens: the counts compare retrievers with each other
//! and a corpus with itself over time, and are not a bill.
//! Latency is the wall-clock time of the search call, so for `dense`, `hybrid` and `external` it
//! includes the network round trip a caller would pay.

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::hit::Hit;
use crate::num::float;
use pinakes::index::{Page, iter_units, tokenize};

/// How many hits the smaller token count covers, as in recall@5.
const SHORT: usize = 5;
/// How many hits the larger token count covers, as in recall@10.
const LONG: usize = 10;

/// Search latency over the queries of a run, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Latency {
    /// The median search.
    pub p50_ms: f64,
    /// The 95th percentile (nearest rank), the slow tail a caller waits for now and then.
    pub p95_ms: f64,
}

/// Tokens the returned units add to the reader's context, per query.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Tokens {
    /// Mean tokens of the units of the top 5 hits.
    #[serde(rename = "mean@5")]
    pub mean5: f64,
    /// Mean tokens of the units of the top 10 hits (all of them when fewer came back).
    #[serde(rename = "mean@10")]
    pub mean10: f64,
    /// Hits whose unit could not be identified and so counted no tokens: the artifact does not
    /// have the unit (an `external` backend naming other pages), or the hit names its unit by a
    /// page and a heading that several units of the page share (the built-in backends say no
    /// more than that, so a page with two `## FAQ` sections is ambiguous). When it is not zero
    /// the means are under-counts.
    #[serde(default)]
    pub unresolved: usize,
}

/// The cost of a run: each half is absent when it was not measured.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Cost {
    /// Search latency.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency: Option<Latency>,
    /// Tokens per query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Tokens>,
}

impl Cost {
    /// The one-line form `eval` prints under its table.
    pub fn render(&self) -> String {
        let mut parts = Vec::new();
        if let Some(latency) = &self.latency {
            // Three decimals: a search of a small corpus takes a fraction of a millisecond.
            parts.push(format!(
                "latency p50 {:.3} ms, p95 {:.3} ms",
                latency.p50_ms, latency.p95_ms
            ));
        }
        if let Some(tokens) = &self.tokens {
            let uncounted = if tokens.unresolved > 0 {
                format!(" ({} hits not counted)", tokens.unresolved)
            } else {
                String::new()
            };
            parts.push(format!(
                "tokens per query {:.1} @5, {:.1} @10{uncounted}",
                tokens.mean5, tokens.mean10
            ));
        }
        format!("cost: {}", parts.join("; "))
    }
}

/// The token count of every unit of an artifact, to look a hit up by.
#[derive(Debug, Clone, Default)]
pub struct UnitTokens {
    by_id: HashMap<String, u32>,
    /// A page and heading name one unit, or `None` when several units of the page share them.
    by_heading: HashMap<(String, String), Option<u32>>,
}

impl UnitTokens {
    /// The units of the searchable pages of `pages`, which must have the mirror rule applied
    /// (an [`pinakes::index::Index`] has), counted with the index's own tokeniser.
    pub fn of(pages: &[Page]) -> UnitTokens {
        let mut units = UnitTokens::default();
        for unit in iter_units(pages) {
            let tokens = u32::try_from(tokenize(&unit.text).len()).unwrap_or(u32::MAX);
            units
                .by_heading
                .entry((unit.page_id.clone(), unit.heading))
                .and_modify(|found| *found = None)
                .or_insert(Some(tokens));
            units.by_id.insert(unit.id, tokens);
        }
        units
    }

    /// The tokens of the unit a hit names: the unit of its `unit_id` when it carries one, else
    /// the unit of its page under its heading. `None` when the artifact has no such unit, and
    /// when the page and heading fit several: guessing one would charge the hit for a unit it
    /// may not have returned. A `unit_id` the artifact does not know is not second-guessed
    /// through the heading either.
    pub fn count(&self, hit: &Hit) -> Option<u32> {
        match hit.unit_id.as_deref() {
            Some(id) => self.by_id.get(id).copied(),
            None => self
                .by_heading
                .get(&(hit.page_id.clone(), hit.heading.clone()))
                .copied()
                .flatten(),
        }
    }
}

/// Times the searches of a run and counts the tokens they return; see the module docs.
#[derive(Debug)]
pub struct CostMeter<'a> {
    units: &'a UnitTokens,
    latencies_ms: Vec<f64>,
    short: Vec<u64>,
    long: Vec<u64>,
    unresolved: usize,
}

impl<'a> CostMeter<'a> {
    /// A meter that looks hits up in `units`.
    pub fn new(units: &'a UnitTokens) -> CostMeter<'a> {
        CostMeter {
            units,
            latencies_ms: Vec::new(),
            short: Vec::new(),
            long: Vec::new(),
            unresolved: 0,
        }
    }

    /// Run one search, timed, and count what it returned. An error is passed on untouched and
    /// counts nothing.
    pub fn search<E>(
        &mut self,
        search: impl FnOnce() -> Result<Vec<Hit>, E>,
    ) -> Result<Vec<Hit>, E> {
        let started = Instant::now();
        let hits = search()?;
        self.latencies_ms
            .push(started.elapsed().as_secs_f64() * 1000.0);
        let (mut short, mut long) = (0u64, 0u64);
        for (rank, hit) in hits.iter().enumerate().take(LONG) {
            match self.units.count(hit) {
                Some(tokens) => {
                    long += u64::from(tokens);
                    if rank < SHORT {
                        short += u64::from(tokens);
                    }
                }
                None => self.unresolved += 1,
            }
        }
        self.short.push(short);
        self.long.push(long);
        Ok(hits)
    }

    /// The cost of the searches so far; `None` when there were none.
    pub fn finish(self) -> Option<Cost> {
        let n = self.latencies_ms.len();
        if n == 0 {
            return None;
        }
        let mut sorted = self.latencies_ms;
        sorted.sort_by(f64::total_cmp);
        let mean = |sums: &[u64]| sums.iter().map(|&s| float_u64(s)).sum::<f64>() / float(n);
        Some(Cost {
            latency: Some(Latency {
                p50_ms: milli(nearest_rank(&sorted, 50)),
                p95_ms: milli(nearest_rank(&sorted, 95)),
            }),
            tokens: Some(Tokens {
                mean5: mean(&self.short),
                mean10: mean(&self.long),
                unresolved: self.unresolved,
            }),
        })
    }
}

/// A token sum as a float; sums stay far below 2^53.
fn float_u64(value: u64) -> f64 {
    float(usize::try_from(value).unwrap_or(usize::MAX))
}

/// The value at the `percent`th percentile of `sorted` by the nearest-rank rule: the smallest
/// value with at least that share of the values at or below it. `sorted` is not empty.
fn nearest_rank(sorted: &[f64], percent: usize) -> f64 {
    let rank = (sorted.len() * percent).div_ceil(100).max(1);
    sorted[rank - 1]
}

/// Round to three decimals (a microsecond), which is all a wall-clock reading means.
fn milli(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}

/// Optional limits on a run's cost, the gate `eval --max-p95-ms` and `--max-tokens` set. Each
/// is an absolute ceiling, not a comparison with a baseline: latency varies from run to run and
/// machine to machine, so a ceiling is set with room to spare.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Budget {
    /// The most the 95th percentile search may take, in milliseconds.
    pub p95_ms: Option<f64>,
    /// The most tokens per query the top 5 hits may add, on average.
    pub tokens5: Option<f64>,
}

/// What a [`Budget`] limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetKind {
    /// 95th percentile search latency, in milliseconds.
    P95Latency,
    /// Mean tokens per query of the top 5 hits.
    Tokens5,
}

impl BudgetKind {
    /// How the limit is named in output.
    pub fn label(self) -> &'static str {
        match self {
            BudgetKind::P95Latency => "p95 latency (ms)",
            BudgetKind::Tokens5 => "tokens per query @5",
        }
    }
}

/// One limit of a [`Budget`] against a run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BudgetCheck {
    /// What is limited.
    pub kind: BudgetKind,
    /// The run's value, `None` when the run has none (no queries).
    pub value: Option<f64>,
    /// The ceiling.
    pub max: f64,
    /// Hits the value could not count (see [`Tokens::unresolved`]): a token value is then an
    /// under-count, which passing a ceiling does not answer for. Zero for latency.
    pub uncounted: usize,
}

impl BudgetCheck {
    /// Whether the value is within the ceiling. A run with no value has not shown that it is.
    pub fn passed(&self) -> bool {
        self.value.is_some_and(|value| value <= self.max)
    }
}

impl Budget {
    /// Whether any limit is set.
    pub fn is_set(&self) -> bool {
        self.p95_ms.is_some() || self.tokens5.is_some()
    }

    /// One [`BudgetCheck`] per limit that is set, against `cost`.
    pub fn check(&self, cost: Option<&Cost>) -> Vec<BudgetCheck> {
        let mut checks = Vec::new();
        if let Some(max) = self.p95_ms {
            checks.push(BudgetCheck {
                kind: BudgetKind::P95Latency,
                value: cost.and_then(|c| c.latency).map(|l| l.p95_ms),
                max,
                uncounted: 0,
            });
        }
        if let Some(max) = self.tokens5 {
            checks.push(BudgetCheck {
                kind: BudgetKind::Tokens5,
                value: cost.and_then(|c| c.tokens).map(|t| t.mean5),
                max,
                uncounted: cost.and_then(|c| c.tokens).map_or(0, |t| t.unresolved),
            });
        }
        checks
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::testing::{SourceSpec, write_artifact};
    use pinakes::index::{Priorities, load_pages, mark_mirrors};

    fn units() -> (UnitTokens, Vec<Page>) {
        let dir = tempfile::tempdir().unwrap();
        write_artifact(
            dir.path(),
            &[SourceSpec {
                name: "handbook",
                repo: "example-org/handbook",
                pages: &[
                    (
                        "docs/a.md",
                        "Storage",
                        "# Storage\n\nKeeps files.\n\n## Caching\n\nEnable caching for uploads today.\n",
                    ),
                    ("docs/b.md", "Billing", "# Billing\n\nInvoices.\n"),
                ],
                residue: &[],
            }],
        );
        let mut pages = load_pages(dir.path(), &Priorities::default()).unwrap();
        mark_mirrors(&mut pages);
        (UnitTokens::of(&pages), pages)
    }

    fn hit(page_id: &str, heading: &str, unit_id: Option<&str>) -> Hit {
        Hit {
            page_id: page_id.to_string(),
            score: 1.0,
            heading: heading.to_string(),
            unit_id: unit_id.map(str::to_string),
        }
    }

    #[test]
    fn a_hit_is_counted_by_its_unit_id_else_by_its_page_and_heading() {
        let (units, pages) = units();
        let expected: Vec<u32> = iter_units(&pages)
            .iter()
            .map(|u| u32::try_from(tokenize(&u.text).len()).unwrap())
            .collect();
        assert_eq!(
            expected.len(),
            3,
            "an intro and Caching on a.md, an intro on b.md"
        );
        let by_id = |id: &str| units.count(&hit("handbook::docs/a.md", "", Some(id)));
        assert_eq!(by_id("handbook::docs/a.md#1"), Some(expected[1]));
        assert_eq!(
            units.count(&hit("handbook::docs/a.md", "Caching", None)),
            Some(expected[1])
        );
        assert_eq!(
            units.count(&hit("handbook::docs/b.md", "", None)),
            Some(expected[2])
        );
        // The unit id wins over a heading that says something else.
        assert_eq!(
            units.count(&hit(
                "handbook::docs/a.md",
                "Caching",
                Some("handbook::docs/a.md#0")
            )),
            Some(expected[0])
        );
        // An id the artifact does not know is not second-guessed through the heading, and a
        // page it does not know has no unit at all.
        assert_eq!(
            units.count(&hit(
                "handbook::docs/a.md",
                "Caching",
                Some("handbook::docs/a.md#9")
            )),
            None
        );
        assert_eq!(units.count(&hit("handbook::docs/zzz.md", "", None)), None);
        assert_eq!(
            units.count(&hit("handbook::docs/a.md", "No such heading", None)),
            None
        );
        assert!(
            expected[1] > expected[0],
            "the Caching section has more words than the intro"
        );
    }

    #[test]
    fn a_heading_that_several_units_of_a_page_share_is_ambiguous_not_the_first_of_them() {
        let dir = tempfile::tempdir().unwrap();
        write_artifact(
            dir.path(),
            &[SourceSpec {
                name: "handbook",
                repo: "example-org/handbook",
                pages: &[(
                    "docs/guide.md",
                    "Guide",
                    "# Guide\n\nIntro.\n\n## FAQ\n\nShort answer.\n\n## Other\n\nElsewhere.\n\n## FAQ\n\nThe long \
                     answer has many more words than the first one did.\n",
                )],
                residue: &[],
            }],
        );
        let mut pages = load_pages(dir.path(), &Priorities::default()).unwrap();
        mark_mirrors(&mut pages);
        let units = UnitTokens::of(&pages);
        let by_page_and_heading =
            |heading: &str| units.count(&hit("handbook::docs/guide.md", heading, None));
        assert_eq!(by_page_and_heading("FAQ"), None, "two units are called FAQ");
        assert!(
            by_page_and_heading("Other").is_some(),
            "one unit is called Other"
        );
        // The unit id still names either one exactly.
        assert!(
            units
                .count(&hit(
                    "handbook::docs/guide.md",
                    "FAQ",
                    Some("handbook::docs/guide.md#3")
                ))
                .is_some_and(|long| long
                    > units
                        .count(&hit(
                            "handbook::docs/guide.md",
                            "FAQ",
                            Some("handbook::docs/guide.md#1")
                        ))
                        .unwrap())
        );

        // A meter counts such a hit as not counted, never as the wrong unit.
        let mut meter = CostMeter::new(&units);
        meter
            .search(|| Ok::<_, ()>(vec![hit("handbook::docs/guide.md", "FAQ", None)]))
            .unwrap();
        let tokens = meter.finish().unwrap().tokens.unwrap();
        assert_eq!(tokens.unresolved, 1);
        assert!(tokens.mean5.abs() < f64::EPSILON);
    }

    #[test]
    fn tokens_are_the_index_tokens_lower_cased_words_without_stop_words_and_compounds_joined() {
        // What the README and the module docs say the count is.
        assert_eq!(tokenize("The Spec of a Sink"), ["spec", "sink"]);
        let compound = tokenize("spec.sink");
        assert!(
            ["spec", "sink", "specsink"]
                .iter()
                .all(|t| compound.iter().any(|c| c == t)),
            "{compound:?}"
        );
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let values: Vec<f64> = (1..=20).map(f64::from).collect();
        assert!((nearest_rank(&values, 50) - 10.0).abs() < f64::EPSILON);
        assert!((nearest_rank(&values, 95) - 19.0).abs() < f64::EPSILON);
        assert!((nearest_rank(&[7.0], 95) - 7.0).abs() < f64::EPSILON);
        assert!((nearest_rank(&[1.0, 2.0], 50) - 1.0).abs() < f64::EPSILON);
        assert!((nearest_rank(&[1.0, 2.0], 95) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn the_meter_times_each_search_and_sums_the_tokens_of_the_top_5_and_top_10() {
        let (units, _) = units();
        let one = units.count(&hit("handbook::docs/b.md", "", None)).unwrap();
        let mut meter = CostMeter::new(&units);
        let b = hit("handbook::docs/b.md", "", None);
        // Twelve hits of the same unit: only the first ten count, the first five in `@5`.
        let hits = meter
            .search(|| {
                std::thread::sleep(Duration::from_millis(3));
                Ok::<_, ()>(vec![b.clone(); 12])
            })
            .unwrap();
        assert_eq!(hits.len(), 12, "the hits come back untouched");
        // A second query returning nothing costs no tokens but is still a query.
        meter.search(|| Ok::<_, ()>(Vec::new())).unwrap();
        // One that names a unit the artifact lacks is counted as not counted.
        meter
            .search(|| Ok::<_, ()>(vec![hit("handbook::docs/zzz.md", "", None)]))
            .unwrap();
        let cost = meter.finish().unwrap();
        let tokens = cost.tokens.unwrap();
        assert!(
            (tokens.mean5 - f64::from(5 * one) / 3.0).abs() < 1e-9,
            "{tokens:?}"
        );
        assert!(
            (tokens.mean10 - f64::from(10 * one) / 3.0).abs() < 1e-9,
            "{tokens:?}"
        );
        assert_eq!(tokens.unresolved, 1);
        let latency = cost.latency.unwrap();
        assert!(
            latency.p95_ms >= 3.0,
            "the slow search is the tail: {latency:?}"
        );
        assert!(latency.p50_ms <= latency.p95_ms);
    }

    #[test]
    fn a_failed_search_counts_nothing_and_no_search_is_no_cost() {
        let (units, _) = units();
        let mut meter = CostMeter::new(&units);
        assert_eq!(meter.search(|| Err::<Vec<Hit>, _>("down")), Err("down"));
        assert_eq!(meter.finish(), None);
    }

    #[test]
    fn cost_round_trips_and_leaves_out_what_was_not_measured() {
        let cost = Cost {
            latency: Some(Latency {
                p50_ms: 1.5,
                p95_ms: 4.25,
            }),
            tokens: Some(Tokens {
                mean5: 40.0,
                mean10: 75.5,
                unresolved: 0,
            }),
        };
        let json = serde_json::to_string(&cost).unwrap();
        assert_eq!(
            json,
            r#"{"latency":{"p50_ms":1.5,"p95_ms":4.25},"tokens":{"mean@5":40.0,"mean@10":75.5,"unresolved":0}}"#
        );
        assert_eq!(serde_json::from_str::<Cost>(&json).unwrap(), cost);
        let bare = Cost {
            latency: None,
            tokens: None,
        };
        assert_eq!(serde_json::to_string(&bare).unwrap(), "{}");
        assert_eq!(serde_json::from_str::<Cost>("{}").unwrap(), bare);
        assert_eq!(
            cost.render(),
            "cost: latency p50 1.500 ms, p95 4.250 ms; tokens per query 40.0 @5, 75.5 @10"
        );
        let uncounted = Cost {
            tokens: Some(Tokens {
                unresolved: 3,
                ..cost.tokens.unwrap()
            }),
            ..cost
        };
        assert!(
            uncounted.render().ends_with("(3 hits not counted)"),
            "{}",
            uncounted.render()
        );
    }

    #[test]
    fn a_budget_checks_each_limit_it_sets_and_a_missing_value_does_not_pass() {
        let cost = Cost {
            latency: Some(Latency {
                p50_ms: 1.0,
                p95_ms: 50.0,
            }),
            tokens: Some(Tokens {
                mean5: 300.0,
                mean10: 600.0,
                unresolved: 0,
            }),
        };
        assert!(!Budget::default().is_set());
        assert!(Budget::default().check(Some(&cost)).is_empty());

        let budget = Budget {
            p95_ms: Some(50.0),
            tokens5: Some(299.0),
        };
        assert!(budget.is_set());
        let checks = budget.check(Some(&cost));
        assert_eq!(checks.len(), 2);
        assert!(checks[0].passed(), "at the ceiling is within it");
        assert_eq!(checks[0].kind, BudgetKind::P95Latency);
        assert!(!checks[1].passed());
        assert_eq!(checks[1].value, Some(300.0));

        // A token value that could not count every hit says so; latency never does.
        let partial = Cost {
            tokens: Some(Tokens {
                unresolved: 4,
                ..cost.tokens.unwrap()
            }),
            ..cost
        };
        let checks = budget.check(Some(&partial));
        assert_eq!((checks[0].uncounted, checks[1].uncounted), (0, 4));

        // No cost at all (a run with no queries): a limit that was asked for is not met.
        let checks = budget.check(None);
        assert!(checks.iter().all(|c| c.value.is_none() && !c.passed()));
    }
}
