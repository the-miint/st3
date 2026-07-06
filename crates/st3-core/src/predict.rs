// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! End-to-end sink source-prediction driver.
//!
//! [`predict_sinks`] ties the pieces together for sink mode: collapse the source
//! samples by environment, precompute the estimator's source model once, then run
//! the collapsed-Gibbs sampler on each sink and collate the ensembles into a
//! [`SourceMixing`]. Each sink is seeded independently from the run seed via
//! [`crate::rng::rng_for_item`], so the result depends only on `(seed, sink)` —
//! never on iteration order (the serial loop here becomes a parallel one later
//! without changing output).
//!
//! Rarefaction is intentionally *not* performed here: callers rarefy the table
//! up front with [`crate::rarefy()`] if desired (matching the reference, which
//! treats rarefaction as a separate preprocessing step). Leave-one-out is a
//! separate driver (a later milestone).

use crate::collapse::collapse_sources;
use crate::collate::{SourceMixing, collate};
use crate::error::{Error, Result};
use crate::estimate::{GibbsEstimator, SinkModel, SinkVec};
use crate::metadata::SampleContext;
use crate::parallel::map_items;
use crate::params::GibbsParams;
use crate::rng::rng_for_item;
use crate::table::CountTable;

/// Estimate the source composition of every sink in `ctx`.
///
/// Sources are collapsed by `params.collapse`; each sink column of `table` is
/// then attributed over the collapsed environments plus `Unknown`. When
/// `params.contingency` is set, each sink's source × taxon assignment tally is
/// included in the result. Sinks are seeded by their position in
/// [`SampleContext::sink_indices`].
///
/// `jobs` sizes a per-call scoped worker pool: `0` uses all logical cores, `1`
/// runs serially, and `n` uses `n` threads. Because each sink is seeded from
/// `(seed, position)` alone and results are collated in sink order, the output is
/// byte-identical regardless of `jobs`.
///
/// # Examples
/// ```
/// use st3_core::{
///     CollapseMethod, CountTable, GibbsParams, Role, SampleContext, predict_sinks,
/// };
///
/// // Two source environments (envA carries f0, envB carries f1) and one sink
/// // that is mostly f0, so it should attribute mostly to envA.
/// let table = CountTable::from_coo(
///     vec!["f0".into(), "f1".into()],
///     vec!["a".into(), "b".into(), "sink".into()],
///     &[0, 1, 0, 1],
///     &[0, 1, 2, 2],
///     &[100.0, 100.0, 90.0, 10.0],
/// )?;
/// let ctx = SampleContext::new(
///     vec![Role::Source, Role::Source, Role::Sink],
///     vec![Some("envA".into()), Some("envB".into()), None],
/// )?;
/// // Tiny, fast, deterministic sampler settings.
/// let params = GibbsParams {
///     restarts: 4,
///     draws_per_restart: 2,
///     burnin: 5,
///     collapse: CollapseMethod::Sum,
///     ..GibbsParams::default()
/// };
/// let mixing = predict_sinks(&table, &ctx, &params, 42, 1)?;
///
/// // One row per sink; columns are the source environments then `Unknown`.
/// assert_eq!(mixing.env_names(), &["envA", "envB", "Unknown"]);
/// let row = mixing.mean_row(0);
/// assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9); // proportions sum to 1
/// assert!(row[0] > row[1]); // more envA than envB
/// # Ok::<(), st3_core::Error>(())
/// ```
///
/// # Errors
/// [`Error::EmptySink`] if a sink column has no sequences (the lowest-index empty
/// sink, independent of `jobs`); propagates [`collapse_sources`],
/// [`GibbsEstimator::prepare`] (parameter validation), and [`Error::ThreadPool`].
pub fn predict_sinks(
    table: &CountTable,
    ctx: &SampleContext,
    params: &GibbsParams,
    seed: u64,
    jobs: usize,
) -> Result<SourceMixing> {
    let sources = collapse_sources(table, ctx, params.collapse)?;
    let prep = GibbsEstimator::prepare(&sources, params)?;

    // Column labels: collapsed source envs (sorted) then Unknown (last).
    let mut env_names: Vec<String> = sources.env_names().to_vec();
    env_names.push("Unknown".to_string());

    let sink_cols = ctx.sink_indices();
    // Each sink is an independent work item seeded by its position, so the fan-out
    // is deterministic across thread counts (see `crate::parallel`).
    let pairs = map_items(jobs, sink_cols.len(), |item_index| {
        let s = sink_cols[item_index] as usize;
        if table.column_sum(s) == 0 {
            return Err(Error::EmptySink { sample_index: s });
        }
        let (rows, counts) = table.column(s);
        let sink = SinkVec::new(rows, counts);
        let mut rng = rng_for_item(seed, item_index as u64);
        let est = GibbsEstimator::estimate(&prep, &sink, params, &mut rng, params.contingency);
        Ok((est, table.sample_ids()[s].clone()))
    })?;
    let (estimates, sink_ids): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();

    Ok(collate(&estimates, sink_ids, env_names))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collapse::CollapseMethod;
    use crate::metadata::Role;

    /// Build a table + context from dense columns
    /// `(sample_id, role, env, counts)` (each `counts` has length `τ`).
    fn build(cols: &[(&str, Role, Option<&str>, &[u32])]) -> (CountTable, SampleContext) {
        let tau = cols[0].3.len();
        let mut rows = Vec::new();
        let mut sample_cols = Vec::new();
        let mut vals = Vec::new();
        for (c, (_, _, _, counts)) in cols.iter().enumerate() {
            for (t, &v) in counts.iter().enumerate() {
                if v != 0 {
                    rows.push(t as u32);
                    sample_cols.push(c as u32);
                    vals.push(f64::from(v));
                }
            }
        }
        let feature_ids = (0..tau).map(|i| format!("f{i}")).collect();
        let sample_ids: Vec<String> = cols.iter().map(|c| c.0.to_string()).collect();
        let table = CountTable::from_coo(feature_ids, sample_ids, &rows, &sample_cols, &vals)
            .expect("table builds");
        let roles = cols.iter().map(|c| c.1).collect();
        let envs = cols.iter().map(|c| c.2.map(str::to_string)).collect();
        let ctx = SampleContext::for_table(&table, roles, envs).expect("context builds");
        (table, ctx)
    }

    fn params(collapse: CollapseMethod, contingency: bool) -> GibbsParams {
        GibbsParams {
            alpha1: 0.001,
            alpha2: 0.1,
            beta: 10.0,
            restarts: 20,
            draws_per_restart: 2,
            burnin: 10,
            delay: 1,
            collapse,
            contingency,
        }
    }

    #[test]
    fn empty_sink_is_an_error() {
        let (table, ctx) = build(&[
            ("a", Role::Source, Some("envA"), &[10, 0]),
            ("b", Role::Source, Some("envB"), &[0, 10]),
            ("empty", Role::Sink, None, &[0, 0]),
        ]);
        // The empty sink is the third column (index 2).
        let err =
            predict_sinks(&table, &ctx, &params(CollapseMethod::Sum, false), 42, 1).unwrap_err();
        assert_eq!(err, Error::EmptySink { sample_index: 2 });
    }

    #[test]
    fn end_to_end_shape_and_rows_sum_to_one() {
        let (table, ctx) = build(&[
            ("a", Role::Source, Some("envA"), &[10, 0]),
            ("b", Role::Source, Some("envB"), &[0, 10]),
            ("s0", Role::Sink, None, &[8, 2]),
            ("s1", Role::Sink, None, &[3, 7]),
        ]);
        let sm = predict_sinks(&table, &ctx, &params(CollapseMethod::Sum, false), 7, 1).unwrap();
        assert_eq!(sm.n_sinks(), 2);
        assert_eq!(sm.n_envs(), 3); // envA, envB, Unknown
        assert_eq!(sm.env_names(), &["envA", "envB", "Unknown"]);
        assert_eq!(sm.sink_ids(), &["s0", "s1"]);
        assert!(sm.contingency().is_none());
        for i in 0..sm.n_sinks() {
            let row = sm.mean_row(i);
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "sink {i} sums to {sum}");
            assert!(row.iter().all(|&x| x >= 0.0));
        }
    }

    #[test]
    fn same_seed_is_deterministic() {
        let (table, ctx) = build(&[
            ("a", Role::Source, Some("envA"), &[10, 0]),
            ("b", Role::Source, Some("envB"), &[0, 10]),
            ("s0", Role::Sink, None, &[8, 2]),
        ]);
        let p = params(CollapseMethod::Sum, false);
        let a = predict_sinks(&table, &ctx, &p, 99, 1).unwrap();
        let b = predict_sinks(&table, &ctx, &p, 99, 1).unwrap();
        assert_eq!(a.means(), b.means());
        assert_eq!(a.stds(), b.stds());
    }

    #[test]
    fn output_is_identical_across_job_counts() {
        // Determinism guard (roadmap R8): the same seed must produce byte-identical
        // output at any thread count. Contingency on, to cover that path too.
        let (table, ctx) = build(&[
            ("a", Role::Source, Some("envA"), &[10, 0, 1]),
            ("b", Role::Source, Some("envB"), &[0, 10, 1]),
            ("c", Role::Source, Some("envC"), &[1, 1, 10]),
            ("s0", Role::Sink, None, &[8, 2, 1]),
            ("s1", Role::Sink, None, &[3, 7, 2]),
            ("s2", Role::Sink, None, &[1, 1, 9]),
            ("s3", Role::Sink, None, &[4, 4, 4]),
        ]);
        let p = params(CollapseMethod::Sum, true);
        let serial = predict_sinks(&table, &ctx, &p, 2024, 1).unwrap();
        for jobs in [0usize, 2, 8] {
            let parallel = predict_sinks(&table, &ctx, &p, 2024, jobs).unwrap();
            assert_eq!(parallel, serial, "jobs={jobs} diverged from serial");
        }
    }

    #[test]
    fn empty_sink_error_is_deterministic_across_jobs() {
        // Two empty sinks; the lowest-index one must always be reported.
        let (table, ctx) = build(&[
            ("a", Role::Source, Some("envA"), &[10, 0]),
            ("b", Role::Source, Some("envB"), &[0, 10]),
            ("s0", Role::Sink, None, &[5, 5]),
            ("empty1", Role::Sink, None, &[0, 0]),
            ("s2", Role::Sink, None, &[2, 8]),
            ("empty2", Role::Sink, None, &[0, 0]),
        ]);
        let p = params(CollapseMethod::Sum, false);
        for jobs in [1usize, 4] {
            let err = predict_sinks(&table, &ctx, &p, 7, jobs).unwrap_err();
            // empty1 is table column 3 (lowest-index empty sink).
            assert_eq!(err, Error::EmptySink { sample_index: 3 }, "jobs={jobs}");
        }
    }

    #[test]
    fn contingency_is_populated_when_requested() {
        let (table, ctx) = build(&[
            ("a", Role::Source, Some("envA"), &[10, 0]),
            ("b", Role::Source, Some("envB"), &[0, 10]),
            ("s0", Role::Sink, None, &[8, 2]),
        ]);
        let sm = predict_sinks(&table, &ctx, &params(CollapseMethod::Sum, true), 1, 1).unwrap();
        let tallies = sm.contingency().expect("contingency requested");
        assert_eq!(tallies.len(), 1);
        // Grand total of the tally equals the sink depth (8 + 2 = 10).
        let mass: f64 = tallies[0].triples().map(|(_, _, m)| m).sum();
        assert!((mass - 10.0).abs() < 1e-9, "tally mass = {mass}");
    }
}
