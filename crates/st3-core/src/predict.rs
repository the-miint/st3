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
/// # Errors
/// [`Error::EmptySink`] if a sink column has no sequences; propagates
/// [`collapse_sources`] and [`GibbsEstimator::prepare`] (parameter validation).
pub fn predict_sinks(
    table: &CountTable,
    ctx: &SampleContext,
    params: &GibbsParams,
    seed: u64,
) -> Result<SourceMixing> {
    let sources = collapse_sources(table, ctx, params.collapse)?;
    let prep = GibbsEstimator::prepare(&sources, params)?;

    // Column labels: collapsed source envs (sorted) then Unknown (last).
    let mut env_names: Vec<String> = sources.env_names().to_vec();
    env_names.push("Unknown".to_string());

    let sink_cols = ctx.sink_indices();
    let mut estimates = Vec::with_capacity(sink_cols.len());
    let mut sink_ids = Vec::with_capacity(sink_cols.len());
    for (item_index, &sink_col) in sink_cols.iter().enumerate() {
        let s = sink_col as usize;
        if table.column_sum(s) == 0 {
            return Err(Error::EmptySink { sample_index: s });
        }
        let (rows, counts) = table.column(s);
        let sink = SinkVec::new(rows, counts);
        let mut rng = rng_for_item(seed, item_index as u64);
        let est = GibbsEstimator::estimate(&prep, &sink, params, &mut rng, params.contingency);
        estimates.push(est);
        sink_ids.push(table.sample_ids()[s].clone());
    }

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
        let err = predict_sinks(&table, &ctx, &params(CollapseMethod::Sum, false), 42).unwrap_err();
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
        let sm = predict_sinks(&table, &ctx, &params(CollapseMethod::Sum, false), 7).unwrap();
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
        let a = predict_sinks(&table, &ctx, &p, 99).unwrap();
        let b = predict_sinks(&table, &ctx, &p, 99).unwrap();
        assert_eq!(a.means(), b.means());
        assert_eq!(a.stds(), b.stds());
    }

    #[test]
    fn contingency_is_populated_when_requested() {
        let (table, ctx) = build(&[
            ("a", Role::Source, Some("envA"), &[10, 0]),
            ("b", Role::Source, Some("envB"), &[0, 10]),
            ("s0", Role::Sink, None, &[8, 2]),
        ]);
        let sm = predict_sinks(&table, &ctx, &params(CollapseMethod::Sum, true), 1).unwrap();
        let tallies = sm.contingency().expect("contingency requested");
        assert_eq!(tallies.len(), 1);
        // Grand total of the tally equals the sink depth (8 + 2 = 10).
        let mass: f64 = tallies[0].triples().map(|(_, _, m)| m).sum();
        assert!((mass - 10.0).abs() < 1e-9, "tally mass = {mass}");
    }
}
