//! `kanon report --svg DIR`: the three charts a review reads faster than the tables, written
//! as SVG files next to the Markdown so GitHub renders them inline in a README, a pull request
//! or an issue with no tooling.
//!
//! - [`recall_over_runs`]: one line per metric over the rows of a run directory (`--runs`),
//!   tuning solid, held-out dashed.
//! - [`rank_movement`]: one row per query, an arrow from the rank of the first expected hit
//!   before to the rank after, misses at the right edge, the biggest wins on top and the
//!   biggest regressions at the bottom.
//! - [`recall_per_kind`]: recall@5 per query kind, tuning next to held-out, so overfitting to
//!   the tuning split is visible.
//!
//! Every chart is rendered from the same structures the Markdown is rendered from (the
//! [`HistoryRow`]s and the [`EvalSummary`]), never from the text, so the two cannot drift. The
//! SVG is hand-rolled: a small builder for the handful of elements the charts need, one fixed
//! palette of mid-saturation colours that reads on a light and on a dark page (GitHub's image
//! rendering does not honour `currentColor` or the page's colour scheme, so every fill is
//! explicit and no text is pure black or white), and a `<title>` on every mark for hover.
//!
//! The corpus funnel (fetched → selected → residue → excluded → duplicates) is a pinakes chart
//! and lives there. A usage chart from the trail (searches per day, top pages) comes once a
//! consumer writes one.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::eval::{EvalSummary, Metrics, Split};
use crate::history::HistoryRow;
use crate::num::float;

/// File name of the recall-over-runs chart inside the `--svg` directory.
pub const RECALL_OVER_RUNS: &str = "recall-over-runs.svg";
/// File name of the rank-movement chart inside the `--svg` directory.
pub const RANK_MOVEMENT: &str = "rank-movement.svg";
/// File name of the recall-per-kind chart inside the `--svg` directory.
pub const RECALL_PER_KIND: &str = "recall-per-kind.svg";

/// Errors raised while writing chart files.
#[derive(Debug, Error)]
pub enum SvgError {
    /// The directory or a chart file could not be written.
    #[error("{path}: {source}")]
    Io {
        /// The path involved.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

/// The chart files `--svg DIR` wrote, as paths under `DIR` exactly as given, for the report
/// to link to. Each is `None` when the chart's inputs were not given.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Charts {
    /// `DIR/recall-over-runs.svg`, from `--runs`.
    pub recall_over_runs: Option<PathBuf>,
    /// `DIR/rank-movement.svg`, from `--eval-before` and `--eval-after` together.
    pub rank_movement: Option<PathBuf>,
    /// `DIR/recall-per-kind.svg`, from `--eval-after` (or `--eval-before` alone).
    pub recall_per_kind: Option<PathBuf>,
}

impl Charts {
    /// The files written, in the order the report links them.
    pub fn paths(&self) -> impl Iterator<Item = &Path> {
        [
            &self.recall_per_kind,
            &self.rank_movement,
            &self.recall_over_runs,
        ]
        .into_iter()
        .filter_map(|p| p.as_deref())
    }
}

/// Write every chart whose inputs are given into `dir` (created when missing) and return the
/// paths. The recall-over-runs chart needs `history`, the rank-movement chart both results,
/// the recall-per-kind chart either result (`after` wins).
pub fn write_charts(
    dir: &Path,
    before: Option<&EvalSummary>,
    after: Option<&EvalSummary>,
    history: Option<&[HistoryRow]>,
) -> Result<Charts, SvgError> {
    let io = |path: &Path| {
        let path = path.to_path_buf();
        move |source| SvgError::Io { path, source }
    };
    std::fs::create_dir_all(dir).map_err(io(dir))?;
    let write = |name: &str, svg: String| -> Result<PathBuf, SvgError> {
        let path = dir.join(name);
        std::fs::write(&path, svg).map_err(io(&path))?;
        Ok(path)
    };
    let mut charts = Charts::default();
    if let Some(result) = after.or(before) {
        charts.recall_per_kind = Some(write(RECALL_PER_KIND, recall_per_kind(result))?);
    }
    if let (Some(before), Some(after)) = (before, after) {
        charts.rank_movement = Some(write(RANK_MOVEMENT, rank_movement(before, after))?);
    }
    if let Some(rows) = history {
        charts.recall_over_runs = Some(write(RECALL_OVER_RUNS, recall_over_runs(rows))?);
    }
    Ok(charts)
}

// The palette. Mid-saturation hues that keep 3:1 contrast on white and on GitHub's dark page,
// distinct under the common colour-vision deficiencies; greys for text and chrome that are
// neither black nor white so they read on both.
const TEXT: &str = "#6b7280";
const GRID: &str = "#9ca3af";
const BLUE: &str = "#3080dc";
const ORANGE: &str = "#e2602d";
const AQUA: &str = "#1aa675";
const RED: &str = "#d03b3b";
const FONT: &str = "system-ui, -apple-system, 'Segoe UI', sans-serif";
const WIDTH: f64 = 640.0;

/// Where text is anchored horizontally.
#[derive(Clone, Copy)]
enum Anchor {
    Start,
    Middle,
    End,
}

/// A minimal SVG document builder: the elements the charts need, with escaping and a
/// `<title>` on every data mark.
struct Svg {
    width: f64,
    height: f64,
    body: String,
}

impl Svg {
    fn new(width: f64, height: f64) -> Svg {
        Svg {
            width,
            height,
            body: String::new(),
        }
    }

    fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, fill: &str, title: Option<&str>) {
        let attrs = format!(
            "x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{fill}\"",
            num(x),
            num(y),
            num(w),
            num(h)
        );
        self.element("rect", &attrs, title);
    }

    fn line(
        &mut self,
        (x1, y1): (f64, f64),
        (x2, y2): (f64, f64),
        stroke: &str,
        width: f64,
        dashed: bool,
    ) {
        let attrs = format!(
            "x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{stroke}\" stroke-width=\"{}\"{}",
            num(x1),
            num(y1),
            num(x2),
            num(y2),
            num(width),
            dash(dashed)
        );
        self.element("line", &attrs, None);
    }

    fn path(
        &mut self,
        d: &str,
        stroke: &str,
        fill: &str,
        width: f64,
        dashed: bool,
        title: Option<&str>,
    ) {
        let attrs = format!(
            "d=\"{d}\" stroke=\"{stroke}\" fill=\"{fill}\" stroke-width=\"{}\" \
             stroke-linejoin=\"round\" stroke-linecap=\"round\"{}",
            num(width),
            dash(dashed)
        );
        self.element("path", &attrs, title);
    }

    fn circle(&mut self, cx: f64, cy: f64, r: f64, fill: &str, title: Option<&str>) {
        let attrs = format!(
            "cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"{fill}\"",
            num(cx),
            num(cy),
            num(r)
        );
        self.element("circle", &attrs, title);
    }

    fn text(&mut self, x: f64, y: f64, anchor: Anchor, fill: &str, content: &str) {
        let anchor = match anchor {
            Anchor::Start => "start",
            Anchor::Middle => "middle",
            Anchor::End => "end",
        };
        let _ = writeln!(
            self.body,
            "<text x=\"{}\" y=\"{}\" text-anchor=\"{anchor}\" fill=\"{fill}\">{}</text>",
            num(x),
            num(y),
            escape(content)
        );
    }

    /// One element: self-closing, or wrapping a `<title>` for hover when given.
    fn element(&mut self, tag: &str, attrs: &str, title: Option<&str>) {
        let _ = match title {
            Some(title) => writeln!(
                self.body,
                "<{tag} {attrs}><title>{}</title></{tag}>",
                escape(title)
            ),
            None => writeln!(self.body, "<{tag} {attrs}/>"),
        };
    }

    fn finish(self) -> String {
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" \
             viewBox=\"0 0 {w} {h}\" font-family=\"{FONT}\" font-size=\"12\">\n{}</svg>\n",
            self.body,
            w = num(self.width),
            h = num(self.height)
        )
    }
}

/// The dash attribute of a held-out line.
fn dash(dashed: bool) -> &'static str {
    if dashed {
        " stroke-dasharray=\"5 4\""
    } else {
        ""
    }
}

/// A coordinate: integers as such, everything else with one decimal.
fn num(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{v:.0}")
    } else {
        format!("{v:.1}")
    }
}

/// Escape text for an element body or attribute.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

/// At most `max` characters, the last one an ellipsis when cut.
fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// The plot area: the rectangle the data is drawn in, with a 0..1 y scale.
struct Plot {
    left: f64,
    right: f64,
    top: f64,
    bottom: f64,
}

impl Plot {
    fn width(&self) -> f64 {
        self.right - self.left
    }

    /// The middle of the plot area.
    fn centre(&self) -> (f64, f64) {
        (
            f64::midpoint(self.left, self.right),
            f64::midpoint(self.top, self.bottom),
        )
    }

    /// The y coordinate of a value in 0..1.
    fn y(&self, value: f64) -> f64 {
        self.bottom - value.clamp(0.0, 1.0) * (self.bottom - self.top)
    }

    /// Hairline gridlines at 0, 0.25, 0.5, 0.75 and 1 with their labels on the left.
    fn y_axis(&self, svg: &mut Svg) {
        for step in 0..=4 {
            let value = float(step) / 4.0;
            let y = self.y(value);
            svg.line((self.left, y), (self.right, y), GRID, 0.5, false);
            svg.text(
                self.left - 6.0,
                y + 4.0,
                Anchor::End,
                TEXT,
                &format!("{value:.2}"),
            );
        }
    }
}

/// One legend entry: a swatch line (dashed or not) and its label, starting at `x`; returns
/// the x after the entry.
fn legend_line(svg: &mut Svg, x: f64, y: f64, colour: &str, dashed: bool, label: &str) -> f64 {
    svg.line((x, y - 4.0), (x + 18.0, y - 4.0), colour, 2.0, dashed);
    svg.text(x + 24.0, y, Anchor::Start, TEXT, label);
    x + 24.0 + 7.0 * float(label.chars().count()) + 16.0
}

/// One legend entry: a square swatch and its label; returns the x after the entry.
fn legend_swatch(svg: &mut Svg, x: f64, y: f64, colour: &str, label: &str) -> f64 {
    svg.rect(x, y - 10.0, 12.0, 12.0, colour, None);
    svg.text(x + 18.0, y, Anchor::Start, TEXT, label);
    x + 18.0 + 7.0 * float(label.chars().count()) + 16.0
}

/// A chart line: its name, its colour and the metric it plots.
type Metric = (&'static str, &'static str, fn(&Metrics) -> f64);

/// The three metrics every chart line stands for.
const METRICS: [Metric; 3] = [
    ("recall@5", BLUE, |m| m.recall5),
    ("recall@10", ORANGE, |m| m.recall10),
    ("MRR", AQUA, |m| m.mrr),
];

/// Recall@5, recall@10 and MRR over the runs of a directory: one line per metric, tuning
/// solid, held-out dashed, x the run sequence number with the label beneath, y 0..1.
pub fn recall_over_runs(rows: &[HistoryRow]) -> String {
    let height = 320.0;
    let plot = Plot {
        left: 44.0,
        right: WIDTH - 16.0,
        top: 52.0,
        bottom: height - 48.0,
    };
    let mut svg = Svg::new(WIDTH, height);
    svg.text(
        plot.left,
        18.0,
        Anchor::Start,
        TEXT,
        "Recall and MRR over runs",
    );
    let mut x = plot.left;
    for (name, colour, _) in METRICS {
        x = legend_line(&mut svg, x, 38.0, colour, false, name);
    }
    x = legend_line(&mut svg, x + 8.0, 38.0, GRID, false, "tuning");
    legend_line(&mut svg, x, 38.0, GRID, true, "held-out");
    plot.y_axis(&mut svg);

    if rows.is_empty() {
        let (x, y) = plot.centre();
        svg.text(x, y, Anchor::Middle, TEXT, "no runs");
        return svg.finish();
    }

    let n = rows.len();
    let x_of = |i: usize| plot.left + (float(i) + 0.5) * plot.width() / float(n);
    let label_every = n.div_ceil(10).max(1);
    for (i, row) in rows.iter().enumerate() {
        if i % label_every != 0 {
            continue;
        }
        let x = x_of(i);
        svg.line((x, plot.bottom), (x, plot.bottom + 4.0), GRID, 1.0, false);
        svg.text(
            x,
            plot.bottom + 16.0,
            Anchor::Middle,
            TEXT,
            &format!("{:03}", row.seq),
        );
        if let Some(label) = &row.label {
            svg.text(
                x,
                plot.bottom + 30.0,
                Anchor::Middle,
                TEXT,
                &clip(label, 12),
            );
        }
    }

    let run_name = |row: &HistoryRow| match &row.label {
        Some(label) => format!("{:03} {label}", row.seq),
        None => format!("{:03}", row.seq),
    };
    for (split, dashed) in [("tuning", false), ("held-out", true)] {
        for (name, colour, value) in METRICS {
            // One path per unbroken stretch of rows that have the split.
            let mut d = String::new();
            let mut previous = false;
            for (i, row) in rows.iter().enumerate() {
                match split_metrics(row, dashed) {
                    Some(m) => {
                        let command = if previous { 'L' } else { 'M' };
                        let _ = write!(d, "{command}{} {} ", num(x_of(i)), num(plot.y(value(m))));
                        previous = true;
                    }
                    None => previous = false,
                }
            }
            if !d.is_empty() {
                svg.path(d.trim_end(), colour, "none", 2.0, dashed, None);
            }
            for (i, row) in rows.iter().enumerate() {
                if let Some(m) = split_metrics(row, dashed) {
                    let title = format!("{}: {split} {name} {:.3}", run_name(row), value(m));
                    svg.circle(x_of(i), plot.y(value(m)), 3.0, colour, Some(&title));
                }
            }
        }
    }
    svg.finish()
}

/// A row's held-out metrics (when it has them) or its tuning metrics.
fn split_metrics(row: &HistoryRow, holdout: bool) -> Option<&Metrics> {
    if holdout {
        row.holdout.as_ref()
    } else {
        Some(&row.tuning)
    }
}

/// The rank of the first expected hit from a reciprocal rank; `None` for a miss.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn rank_of(rr: f64) -> Option<usize> {
    (rr > 0.0).then(|| (1.0 / rr).round() as usize)
}

/// One side of a query's row in the rank-movement chart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    /// The query was not in that result.
    Absent,
    /// Evaluated, no expected page in the list.
    Miss,
    /// Evaluated, the first expected hit at this rank.
    Rank(usize),
}

impl Side {
    fn of(result: &EvalSummary, id: &str) -> Side {
        match result.query(id).map(|q| rank_of(q.rr)) {
            None => Side::Absent,
            Some(None) => Side::Miss,
            Some(Some(rank)) => Side::Rank(rank),
        }
    }

    /// The 1-based column: the rank itself, or one past `last` for a miss.
    fn column(self, last: usize) -> Option<usize> {
        match self {
            Side::Absent => None,
            Side::Miss => Some(last + 1),
            Side::Rank(rank) => Some(rank),
        }
    }

    fn label(self) -> String {
        match self {
            Side::Absent => "not evaluated".to_string(),
            Side::Miss => "miss".to_string(),
            Side::Rank(rank) => rank.to_string(),
        }
    }
}

/// One query's row in the rank-movement chart.
struct Movement {
    id: String,
    before: Side,
    after: Side,
}

impl Movement {
    /// Columns moved towards rank 1 (positive is an improvement); 0 when either side is
    /// absent.
    #[allow(clippy::cast_possible_wrap)]
    fn delta(&self, last: usize) -> i64 {
        match (self.before.column(last), self.after.column(last)) {
            (Some(old), Some(new)) => old as i64 - new as i64,
            _ => 0,
        }
    }

    fn both_evaluated(&self) -> bool {
        self.before != Side::Absent && self.after != Side::Absent
    }
}

/// The rows of the rank-movement chart: every query id in either result, sorted by delta
/// (biggest win first, ties by id), plus the last rank column and the number of unchanged
/// rows. When more than 40 rows would be drawn and more than one is unchanged, the unchanged
/// ones are dropped (`collapsed` is true) for the chart to draw as one summary row.
fn movements(before: &EvalSummary, after: &EvalSummary) -> (Vec<Movement>, usize, usize, bool) {
    let mut ids: Vec<&str> = after.queries.iter().map(|q| q.id.as_str()).collect();
    ids.extend(
        before
            .queries
            .iter()
            .map(|q| q.id.as_str())
            .filter(|id| after.query(id).is_none()),
    );
    let last = before
        .queries
        .iter()
        .chain(&after.queries)
        .map(|q| q.top.len().max(rank_of(q.rr).unwrap_or(0)))
        .max()
        .unwrap_or(0)
        .max(10);
    let mut rows: Vec<Movement> = ids
        .iter()
        .map(|id| Movement {
            id: (*id).to_string(),
            before: Side::of(before, id),
            after: Side::of(after, id),
        })
        .collect();
    rows.sort_by(|first, second| {
        second
            .delta(last)
            .cmp(&first.delta(last))
            .then_with(|| first.id.cmp(&second.id))
    });
    let unchanged = rows
        .iter()
        .filter(|r| r.delta(last) == 0 && r.both_evaluated())
        .count();
    let collapsed = rows.len() > 40 && unchanged > 1;
    if collapsed {
        rows.retain(|r| r.delta(last) != 0 || !r.both_evaluated());
    }
    (rows, last, unchanged, collapsed)
}

/// The rank of the first expected hit before and after, one row per query id in either
/// result: an arrow from the old rank to the new one, misses in a column past the last rank,
/// the biggest wins on top and the biggest regressions at the bottom, unchanged rows greyed
/// in the middle. When more than 40 queries would be drawn, the unchanged ones collapse into
/// one summary row so the chart stays the height of what moved.
pub fn rank_movement(before: &EvalSummary, after: &EvalSummary) -> String {
    let (rows, last, unchanged, collapsed) = movements(before, after);
    let row_height = 18.0;
    let top = 60.0;
    let drawn = rows.len() + usize::from(collapsed);
    let height = top + row_height * float(drawn.max(1)) + 12.0;
    let plot = Plot {
        left: 150.0,
        right: WIDTH - 16.0,
        top,
        bottom: height - 12.0,
    };
    let mut svg = Svg::new(WIDTH, height);
    svg.text(
        16.0,
        18.0,
        Anchor::Start,
        TEXT,
        "Rank of the first expected hit, before → after",
    );
    let mut legend_x = legend_swatch(&mut svg, 16.0, 38.0, BLUE, "better");
    legend_x = legend_swatch(&mut svg, legend_x, 38.0, RED, "worse");
    legend_swatch(&mut svg, legend_x, 38.0, GRID, "unchanged");

    let columns = last + 1;
    let x_of = |column: usize| plot.left + (float(column) - 0.5) * plot.width() / float(columns);
    for column in (1..=columns).filter(|c| *c == 1 || c % 5 == 0 || *c == columns) {
        let x = x_of(column);
        svg.line((x, plot.top - 4.0), (x, plot.bottom), GRID, 0.5, false);
        let label = if column == columns {
            "miss".to_string()
        } else {
            column.to_string()
        };
        svg.text(x, plot.top - 8.0, Anchor::Middle, TEXT, &label);
    }
    if drawn == 0 {
        let (x, y) = plot.centre();
        svg.text(x, y, Anchor::Middle, TEXT, "no per-query rows");
        return svg.finish();
    }

    let mut y = plot.top + row_height / 2.0;
    let mut summary_drawn = false;
    let summary = |svg: &mut Svg, y: f64| {
        let label = format!("{unchanged} unchanged");
        svg.text(plot.left - 8.0, y + 4.0, Anchor::End, TEXT, &label);
        svg.text(plot.left + 8.0, y + 4.0, Anchor::Start, GRID, "not drawn");
    };
    for row in &rows {
        if collapsed && !summary_drawn && row.delta(last) <= 0 {
            summary(&mut svg, y);
            y += row_height;
            summary_drawn = true;
        }
        let label = clip(&row.id, 20);
        svg.text(plot.left - 8.0, y + 4.0, Anchor::End, TEXT, &label);
        let title = format!("{}: {} → {}", row.id, row.before.label(), row.after.label());
        match (row.before.column(last), row.after.column(last)) {
            (Some(old), Some(new)) if old != new => {
                let colour = if new < old { BLUE } else { RED };
                arrow(&mut svg, (x_of(old), x_of(new)), y, colour, &title);
            }
            (Some(column), _) | (None, Some(column)) => {
                svg.circle(x_of(column), y, 3.0, GRID, Some(&title));
            }
            (None, None) => {}
        }
        y += row_height;
    }
    if collapsed && !summary_drawn {
        summary(&mut svg, y);
    }
    svg.finish()
}

/// A horizontal arrow at `y` from `from` to `to`: a dot at the start, a shaft and a filled
/// head, all carrying `title`.
fn arrow(svg: &mut Svg, (from, to): (f64, f64), y: f64, colour: &str, title: &str) {
    let base = if to < from { to + 6.0 } else { to - 6.0 };
    let shaft = format!("M{} {} L{} {}", num(from), num(y), num(base), num(y));
    svg.path(&shaft, colour, "none", 2.0, false, Some(title));
    svg.circle(from, y, 3.0, colour, Some(title));
    let head = format!(
        "M{} {} L{} {} L{} {} Z",
        num(to),
        num(y),
        num(base),
        num(y - 4.0),
        num(base),
        num(y + 4.0)
    );
    svg.path(&head, colour, colour, 1.0, false, Some(title));
}

/// How many 12px characters fit in `width` (at least four, so a label is never just "…").
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn label_chars(width: f64) -> usize {
    ((width / 7.0).floor() as usize).max(4)
}

/// Recall@5 per query kind as grouped bars, tuning next to held-out, kinds sorted. A result
/// with no kinds at all shows its overall numbers as the one group.
pub fn recall_per_kind(result: &EvalSummary) -> String {
    let height = 280.0;
    let plot = Plot {
        left: 44.0,
        right: WIDTH - 16.0,
        top: 56.0,
        bottom: height - 32.0,
    };
    let mut svg = Svg::new(WIDTH, height);
    svg.text(plot.left, 18.0, Anchor::Start, TEXT, "Recall@5 per kind");
    let x = legend_swatch(&mut svg, plot.left, 38.0, BLUE, "tuning");
    if result.holdout.is_some() {
        legend_swatch(&mut svg, x, 38.0, ORANGE, "held-out");
    }
    plot.y_axis(&mut svg);

    let mut kinds: BTreeSet<&str> = result.tuning.per_kind.keys().map(String::as_str).collect();
    if let Some(holdout) = &result.holdout {
        kinds.extend(holdout.per_kind.keys().map(String::as_str));
    }
    let overall_only = kinds.is_empty();
    let kinds: Vec<&str> = if overall_only {
        vec!["overall"]
    } else {
        kinds.into_iter().collect()
    };
    let metrics_of = |split: &Split, kind: &str| {
        if overall_only {
            Some(split.overall)
        } else {
            split.per_kind.get(kind).copied()
        }
    };

    let group_width = plot.width() / float(kinds.len());
    let bar_width = (group_width * 0.32).min(36.0);
    let mut splits = vec![("tuning", BLUE, &result.tuning)];
    if let Some(holdout) = &result.holdout {
        splits.push(("held-out", ORANGE, holdout));
    }
    for (i, kind) in kinds.iter().enumerate() {
        let centre = plot.left + (float(i) + 0.5) * group_width;
        svg.text(
            centre,
            plot.bottom + 18.0,
            Anchor::Middle,
            TEXT,
            &clip(kind, label_chars(group_width)),
        );
        let first = if splits.len() == 1 {
            centre - bar_width / 2.0
        } else {
            centre - bar_width - 1.0
        };
        for (j, (split, colour, metrics)) in splits.iter().enumerate() {
            let Some(m) = metrics_of(metrics, kind) else {
                continue;
            };
            let x = first + float(j) * (bar_width + 2.0);
            let y = plot.y(m.recall5);
            let title = format!("{kind} {split}: recall@5 {:.3} (n = {})", m.recall5, m.n);
            svg.rect(
                x,
                y,
                bar_width,
                (plot.bottom - y).max(1.0),
                colour,
                Some(&title),
            );
            if bar_width >= 30.0 {
                svg.text(
                    x + bar_width / 2.0,
                    y - 4.0,
                    Anchor::Middle,
                    TEXT,
                    &format!("{:.2}", m.recall5),
                );
            }
        }
    }
    svg.line(
        (plot.left, plot.bottom),
        (plot.right, plot.bottom),
        GRID,
        1.0,
        false,
    );
    svg.finish()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::Path;

    use super::*;
    use crate::eval::QueryResult;
    use crate::history::{Run, RunInfo};

    fn metrics(r5: f64, r10: f64, mrr: f64, n: usize) -> Metrics {
        Metrics {
            recall5: r5,
            recall10: r10,
            mrr,
            n,
        }
    }

    fn query(id: &str, rr: f64) -> QueryResult {
        QueryResult {
            id: id.to_string(),
            kind: "howto".to_string(),
            holdout: false,
            origin: None,
            hit5: rr >= 0.2,
            hit10: rr >= 0.1,
            rr,
            top: vec![String::new(); 10],
        }
    }

    /// The report snapshots' before/after pair, plus per-query rows for the rank chart.
    fn evals() -> (EvalSummary, EvalSummary) {
        let before = EvalSummary {
            tuning: Split {
                overall: metrics(0.8, 0.85, 0.66, 40),
                per_kind: BTreeMap::from([("howto".to_string(), metrics(0.9, 0.95, 0.8, 10))]),
            },
            holdout: Some(Split {
                overall: metrics(0.7, 0.8, 0.6, 10),
                per_kind: BTreeMap::new(),
            }),
            queries: vec![
                query("caching", 1.0),
                query("quotas", 1.0 / 12.0),
                query("labels", 0.0),
                query("retention", 0.5),
                query("dropped", 1.0 / 3.0),
            ],
            backend: String::new(),
        };
        let after = EvalSummary {
            tuning: Split {
                overall: metrics(0.85, 0.9, 0.7, 40),
                per_kind: BTreeMap::from([
                    ("concept".to_string(), metrics(0.7, 0.7, 0.5, 5)),
                    ("howto".to_string(), metrics(0.9, 1.0, 0.85, 10)),
                ]),
            },
            holdout: Some(Split {
                overall: metrics(0.75, 0.8, 0.65, 10),
                per_kind: BTreeMap::from([("howto".to_string(), metrics(0.7, 0.8, 0.6, 10))]),
            }),
            queries: vec![
                query("caching", 1.0),
                query("quotas", 0.5),
                query("labels", 1.0 / 7.0),
                query("retention", 0.0),
                query("added", 1.0 / 4.0),
            ],
            backend: "bm25".to_string(),
        };
        (before, after)
    }

    /// The report snapshots' history rows.
    fn rows() -> Vec<HistoryRow> {
        let (before, after) = evals();
        let first = Run {
            summary: before,
            run: Some(RunInfo {
                label: "a1b2c3d".to_string(),
                at: "2026-09-16T12:00:00Z".to_string(),
                backend: "bm25".to_string(),
                manifest_sha256: "0123456789abcdef".repeat(4),
                queries_sha256: "fedcba9876543210".repeat(4),
                k: 10,
            }),
        };
        let second = Run {
            summary: after,
            run: None,
        };
        vec![
            HistoryRow::of(1, Path::new("runs/001-a1b2c3d.json"), &first),
            HistoryRow::of(2, Path::new("runs/002-plain.json"), &second),
        ]
    }

    /// Compare `rendered` with `tests/snapshots/<name>` and its copy under `docs/img/`,
    /// refreshing both when `UPDATE_SNAPSHOTS` is set.
    fn snapshot(name: &str, rendered: &str) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        for dir in ["tests/snapshots", "docs/img"] {
            let path = root.join(dir).join(name);
            if std::env::var_os("UPDATE_SNAPSHOTS").is_some() {
                std::fs::write(&path, rendered).unwrap();
            }
            let expected = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                panic!("{}: {e}; run UPDATE_SNAPSHOTS=1 cargo test", path.display())
            });
            assert_eq!(
                rendered,
                expected,
                "{} changed; run `UPDATE_SNAPSHOTS=1 cargo test` if intended",
                path.display()
            );
        }
    }

    #[test]
    fn builder_escapes_and_titles() {
        let mut svg = Svg::new(10.0, 20.5);
        svg.rect(1.0, 2.0, 3.0, 4.0, "#fff", Some("a & b <c> \"d\""));
        svg.text(0.0, 1.5, Anchor::End, TEXT, "x < y");
        svg.line((0.0, 0.0), (1.0, 1.0), GRID, 0.5, true);
        let out = svg.finish();
        assert!(out.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"20.5\" viewBox=\"0 0 10 20.5\""));
        assert!(
            out.contains("<rect x=\"1\" y=\"2\" width=\"3\" height=\"4\" fill=\"#fff\"><title>a &amp; b &lt;c&gt; &quot;d&quot;</title></rect>\n"),
            "{out}"
        );
        assert!(
            out.contains("text-anchor=\"end\" fill=\"#6b7280\">x &lt; y</text>"),
            "{out}"
        );
        assert!(out.contains("stroke-dasharray=\"5 4\"/>"), "{out}");
        assert_eq!(clip("abcdef", 4), "abc…");
        assert_eq!(clip("abcd", 4), "abcd");
        assert_eq!(rank_of(0.0), None);
        assert_eq!(rank_of(1.0 / 3.0), Some(3));
        assert_eq!(rank_of(1.0 / 12.0), Some(12));
    }

    #[test]
    fn recall_over_runs_matches_snapshot() {
        let svg = recall_over_runs(&rows());
        assert!(
            svg.contains("<title>001 a1b2c3d: tuning recall@5 0.800</title>"),
            "{svg}"
        );
        assert!(
            svg.contains("<title>002: held-out MRR 0.650</title>"),
            "{svg}"
        );
        assert_eq!(
            svg.matches("stroke-dasharray").count(),
            1 + 3,
            "legend + 3 held-out lines"
        );
        assert!(svg.contains(">a1b2c3d</text>"), "{svg}");
        snapshot(RECALL_OVER_RUNS, &svg);

        let empty = recall_over_runs(&[]);
        assert!(empty.contains(">no runs</text>"), "{empty}");
    }

    #[test]
    fn recall_over_runs_breaks_the_held_out_line_where_a_run_has_none() {
        let mut rows = rows();
        let mut third = rows[1].clone();
        third.seq = 3;
        rows[1].holdout = None;
        rows.push(third);
        let svg = recall_over_runs(&rows);
        // The tuning line joins all three runs; the held-out one is two separate moves.
        assert!(svg.contains("<path d=\"M"), "{svg}");
        let dashed_paths = svg
            .lines()
            .filter(|l| l.starts_with("<path") && l.contains("stroke-dasharray"))
            .count();
        assert_eq!(dashed_paths, 3);
        let dashed_moves = svg
            .lines()
            .filter(|l| l.starts_with("<path") && l.contains("stroke-dasharray"))
            .map(|l| l.matches('M').count())
            .sum::<usize>();
        assert_eq!(dashed_moves, 6, "{svg}");
    }

    #[test]
    fn rank_movement_matches_snapshot() {
        let (before, after) = evals();
        let svg = rank_movement(&before, &after);
        // Biggest win first, then the unchanged and one-sided rows by id, the regression last.
        let order: Vec<usize> = [
            "quotas",
            "labels",
            "added",
            "caching",
            "dropped",
            "retention",
        ]
        .iter()
        .map(|id| {
            svg.find(&format!(">{id}</text>"))
                .unwrap_or_else(|| panic!("{id}: {svg}"))
        })
        .collect();
        assert!(order.windows(2).all(|w| w[0] < w[1]), "{order:?}");
        assert!(svg.contains("<title>quotas: 12 → 2</title>"), "{svg}");
        assert!(svg.contains("<title>labels: miss → 7</title>"), "{svg}");
        assert!(svg.contains("<title>retention: 2 → miss</title>"), "{svg}");
        assert!(
            svg.contains("<title>added: not evaluated → 4</title>"),
            "{svg}"
        );
        assert!(
            svg.contains("<title>dropped: 3 → not evaluated</title>"),
            "{svg}"
        );
        assert!(svg.contains(">miss</text>"), "{svg}");
        assert!(svg.contains("fill=\"#d03b3b\""), "{svg}");
        snapshot(RANK_MOVEMENT, &svg);

        let empty = rank_movement(
            &EvalSummary {
                queries: vec![],
                ..before.clone()
            },
            &EvalSummary {
                queries: vec![],
                ..after.clone()
            },
        );
        assert!(empty.contains(">no per-query rows</text>"), "{empty}");
    }

    #[test]
    fn rank_movement_collapses_unchanged_rows_past_forty() {
        let (mut before, mut after) = evals();
        for i in 0..50 {
            before.queries.push(query(&format!("same{i:02}"), 1.0));
            after.queries.push(query(&format!("same{i:02}"), 1.0));
        }
        let svg = rank_movement(&before, &after);
        assert!(svg.contains(">51 unchanged</text>"), "{svg}");
        assert!(!svg.contains(">same00</text>"), "{svg}");
        assert!(
            !svg.contains(">caching</text>"),
            "caching is unchanged too: {svg}"
        );
        assert!(svg.contains(">quotas</text>"), "{svg}");
        assert!(svg.contains(">retention</text>"), "{svg}");
        // The summary row sits between the wins and the rest.
        let summary = svg.find(">51 unchanged</text>").unwrap();
        assert!(svg.find(">quotas</text>").unwrap() < summary, "{svg}");
        assert!(summary < svg.find(">added</text>").unwrap(), "{svg}");
        assert!(svg.contains("height=\"180\""), "6 rows: {svg}");
    }

    #[test]
    fn recall_per_kind_matches_snapshot() {
        let (before, after) = evals();
        let svg = recall_per_kind(&after);
        assert!(
            svg.contains("<title>concept tuning: recall@5 0.700 (n = 5)</title>"),
            "{svg}"
        );
        assert!(
            svg.contains("<title>howto tuning: recall@5 0.900 (n = 10)</title>"),
            "{svg}"
        );
        assert!(svg.contains(">held-out</text>"), "{svg}");
        // The held-out split has a howto row but no concept row, so concept gets one bar.
        assert_eq!(
            svg.matches("<rect").count(),
            2 + 3,
            "two swatches, three bars: {svg}"
        );
        snapshot(RECALL_PER_KIND, &svg);

        // No kinds at all: the overall numbers as one group; no held-out, no second swatch.
        let plain = EvalSummary {
            tuning: Split {
                per_kind: BTreeMap::new(),
                ..before.tuning.clone()
            },
            holdout: None,
            ..before
        };
        let svg = recall_per_kind(&plain);
        assert!(
            svg.contains("<title>overall tuning: recall@5 0.800 (n = 40)</title>"),
            "{svg}"
        );
        assert!(!svg.contains("held-out"), "{svg}");
    }

    #[test]
    fn write_charts_writes_what_the_inputs_allow() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("nested/charts");
        let (before, after) = evals();
        let charts = write_charts(&out, None, None, None).unwrap();
        assert_eq!(charts, Charts::default());
        assert!(out.is_dir());

        let charts = write_charts(&out, Some(&before), None, None).unwrap();
        assert_eq!(charts.recall_per_kind, Some(out.join(RECALL_PER_KIND)));
        assert_eq!(charts.rank_movement, None);
        assert_eq!(charts.recall_over_runs, None);

        let charts = write_charts(&out, Some(&before), Some(&after), Some(&rows())).unwrap();
        let paths: Vec<&Path> = charts.paths().collect();
        assert_eq!(
            paths,
            [
                out.join(RECALL_PER_KIND),
                out.join(RANK_MOVEMENT),
                out.join(RECALL_OVER_RUNS)
            ]
        );
        for path in paths {
            let text = std::fs::read_to_string(path).unwrap();
            assert!(text.starts_with("<svg xmlns="), "{text}");
        }

        // A directory that cannot be created names itself.
        std::fs::write(dir.path().join("file"), "").unwrap();
        let err = write_charts(&dir.path().join("file/x"), None, None, None).unwrap_err();
        assert!(err.to_string().contains("file/x"), "{err}");
    }
}
