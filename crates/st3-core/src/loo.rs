// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Leave-one-out (LOO) source prediction driver.
//!
//! Sink mode ([`crate::predict_sinks`]) asks "where did each *sink* come from?".
//! LOO mode asks the complementary self-consistency question: does each *source*
//! sample get attributed back to its own environment? For every source sample in
//! turn, [`predict_loo`] holds that one sample out, re-collapses the *remaining*
//! source samples by environment, treats the held-out sample as a sink, and
//! estimates its source composition.
//!
//! Two granularity/scoping choices are load-bearing (see the algorithm
//! investigation, §8):
//!
//! * **Per sample, not per environment.** Each fold drops a single source
//!   *sample*, not a whole collapsed environment. Holding out the sole member of
//!   an environment makes that environment vanish from the fold's reduced
//!   sources (variable `V`); its column in the output is then exactly zero.
//! * **Columns are the source environments only.** The output frame is
//!   `sorted-unique(source envs) ++ ["Unknown"]` — never any sink-only
//!   environment. Each fold's reduced result (which may be missing a vanished
//!   env) is scattered *by environment name* into this fixed full frame, so rows
//!   still sum to one and a vanished env stays zero.
//!
//! Scope for this milestone is LOO **means** (validated against the committed
//! oracle). The standard deviations come for free from the shared
//! [`mod@crate::collate`] reduction and are populated for a uniform output contract,
//! but are not oracle-validated. Contingency is not produced in LOO mode: the
//! sampler is always run with `want_assign = false` and the result's
//! [`SourceMixing::contingency`] is `None` regardless of `params.contingency`.
//!
//! Determinism matches the rest of the crate: fold `k` is seeded by
//! [`crate::rng::rng_for_item`]`(seed, k)`, so the result depends only on
//! `(seed, held-out sample)` and never on iteration order or thread count.

use std::collections::{BTreeSet, HashMap};

use crate::collapse::collapse_subset;
use crate::collate::{SourceMixing, mean_std_over_ensemble};
use crate::error::{Error, Result};
use crate::estimate::{GibbsEstimator, SinkModel, SinkVec};
use crate::metadata::SampleContext;
use crate::params::GibbsParams;
use crate::rng::rng_for_item;
use crate::table::CountTable;

/// Leave-one-out source prediction over every source sample in `ctx`.
///
/// For each source sample (in ascending sample-index order, i.e.
/// [`SampleContext::source_indices`]) this holds the sample out, collapses the
/// remaining sources by `params.collapse`, and estimates the held-out sample's
/// source composition over the full frame `sorted-unique(source envs) ++
/// ["Unknown"]`. The returned [`SourceMixing`] reuses the sink-mode output type:
/// its `sink_ids` are the held-out **source** sample ids in fold order, its
/// `env_names` are the full frame, and its `contingency` is always `None`
/// (LOO does not produce assignment tallies).
///
/// # Errors
/// [`Error::EmptySink`] if a held-out source column has no sequences; propagates
/// [`collapse_subset`] ([`Error::NoSources`] when only a single source sample
/// exists in total, so nothing remains after holding it out) and
/// [`GibbsEstimator::prepare`] (parameter validation).
pub fn predict_loo(
    table: &CountTable,
    ctx: &SampleContext,
    params: &GibbsParams,
    seed: u64,
) -> Result<SourceMixing> {
    let source_indices = ctx.source_indices();
    let source_envs = ctx.source_envs();

    // Fixed full frame: sorted-unique SOURCE environments (a `BTreeSet` sorts in
    // the same byte order as collapse's `BTreeMap`), then `Unknown` last. Scoping
    // to source envs — never the whole metadata category — is the structural
    // guard against attributing to sink-only environments.
    let e: Vec<&str> = source_envs
        .iter()
        .copied()
        .collect::<BTreeSet<&str>>()
        .into_iter()
        .collect();
    let full_col: HashMap<&str, usize> = e.iter().enumerate().map(|(i, &name)| (name, i)).collect();
    let v_full = e.len() + 1;

    let mut env_names: Vec<String> = e.iter().map(|&s| s.to_string()).collect();
    env_names.push("Unknown".to_string());

    let n_folds = source_indices.len();
    let mut means = Vec::with_capacity(n_folds * v_full);
    let mut stds = Vec::with_capacity(n_folds * v_full);
    let mut sink_ids = Vec::with_capacity(n_folds);

    for (k, &h) in source_indices.iter().enumerate() {
        // Reduced sources: the remaining (index, env) pairs, dropping fold k.
        // Re-collapsed every fold because the source set changes each time. The
        // two parallel slices are filtered in one pass so the fold-exclusion
        // predicate cannot drift between them.
        let (reduced_idx, reduced_env): (Vec<u32>, Vec<&str>) = source_indices
            .iter()
            .zip(source_envs.iter())
            .enumerate()
            .filter(|&(j, _)| j != k)
            .map(|(_, (&idx, &env))| (idx, env))
            .unzip();
        let reduced_sources = collapse_subset(table, &reduced_idx, &reduced_env, params.collapse)?;

        // The held-out sample becomes the sink for this fold.
        let hs = h as usize;
        if table.column_sum(hs) == 0 {
            return Err(Error::EmptySink { sample_index: hs });
        }
        let (rows, counts) = table.column(hs);
        let sink = SinkVec::new(rows, counts);

        // Estimate over the reduced frame. Contingency is never requested in LOO.
        let prep = GibbsEstimator::prepare(&reduced_sources, params)?;
        let mut rng = rng_for_item(seed, k as u64);
        let est = GibbsEstimator::estimate(&prep, &sink, params, &mut rng, false);

        // Reduce the ensemble over the reduced frame (surviving known envs in
        // collapse order, Unknown last), then scatter into the fixed full frame.
        let v_reduced = reduced_sources.n_sources() + 1;
        let (mean_k, std_k) = mean_std_over_ensemble(est.ensemble(), v_reduced);
        let reduced_env_names = reduced_sources.env_names();
        means.extend_from_slice(&scatter_into_frame(
            reduced_env_names,
            &mean_k,
            &full_col,
            v_full,
        ));
        stds.extend_from_slice(&scatter_into_frame(
            reduced_env_names,
            &std_k,
            &full_col,
            v_full,
        ));
        sink_ids.push(table.sample_ids()[hs].clone());
    }

    Ok(SourceMixing::from_parts(
        sink_ids, env_names, means, stds, None,
    ))
}

/// Scatter a fold's reduced-frame row into the fixed full-frame row.
///
/// `reduced_envs` are the fold's surviving source environments in collapse
/// order; `reduced_row` is the reduced result, length `reduced_envs.len() + 1`
/// (the surviving known envs then `Unknown` last). `full_col` maps each full-frame
/// known environment name to its column index in `E` (the known columns, which
/// exclude `Unknown`), and `v_full = E.len() + 1`.
///
/// The result is a length-`v_full` row: each surviving env's value is placed at
/// its full-frame column, the `Unknown` value is placed last, and every other
/// column (including any environment that vanished from this fold) stays zero.
/// Because only zeros are introduced, the row sum is unchanged.
///
/// # Panics
/// Panics (in debug) if `reduced_row` is not length `reduced_envs.len() + 1`, or
/// (always) if a reduced env name is absent from `full_col`.
fn scatter_into_frame(
    reduced_envs: &[String],
    reduced_row: &[f64],
    full_col: &HashMap<&str, usize>,
    v_full: usize,
) -> Vec<f64> {
    debug_assert_eq!(
        reduced_row.len(),
        reduced_envs.len() + 1,
        "reduced_row must be the surviving known envs plus Unknown"
    );
    let mut row = vec![0.0; v_full];
    for (name, &value) in reduced_envs.iter().zip(reduced_row.iter()) {
        let col = full_col[name.as_str()];
        row[col] = value;
    }
    // The Unknown entry is the last of the reduced row; it always lands in the
    // last full-frame column.
    row[v_full - 1] = reduced_row[reduced_envs.len()];
    row
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collapse::CollapseMethod;
    use crate::metadata::Role;

    /// Build a table + context from dense columns `(sample_id, role, env, counts)`
    /// (each `counts` has length `τ`). Mirrors the `predict` test helper.
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

    fn params(collapse: CollapseMethod) -> GibbsParams {
        GibbsParams {
            alpha1: 0.001,
            alpha2: 0.1,
            beta: 10.0,
            restarts: 30,
            draws_per_restart: 2,
            burnin: 25,
            delay: 1,
            collapse,
            contingency: false,
        }
    }

    // (a) scatter_into_frame hand calc.
    #[test]
    fn scatter_places_by_name_and_zeros_the_rest() {
        // Full frame E = [envA, envB, envC], v_full = 4 (Unknown last).
        let full: HashMap<&str, usize> = [("envA", 0), ("envB", 1), ("envC", 2)]
            .into_iter()
            .collect();
        // Reduced fold kept only envB (+ Unknown): row = [envB=0.6, Unknown=0.4].
        let reduced_envs = vec!["envB".to_string()];
        let reduced_row = [0.6, 0.4];
        let row = scatter_into_frame(&reduced_envs, &reduced_row, &full, 4);
        let expected = [0.0, 0.6, 0.0, 0.4];
        assert_eq!(row.len(), 4);
        for (got, exp) in row.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12, "got {row:?}");
        }
    }

    #[test]
    fn scatter_full_frame_all_present() {
        // No vanished env: every known column filled, Unknown last.
        let full: HashMap<&str, usize> = [("a", 0), ("b", 1)].into_iter().collect();
        let reduced_envs = vec!["a".to_string(), "b".to_string()];
        let reduced_row = [0.5, 0.3, 0.2]; // a, b, Unknown
        let row = scatter_into_frame(&reduced_envs, &reduced_row, &full, 3);
        for (got, exp) in row.iter().zip([0.5, 0.3, 0.2].iter()) {
            assert!((got - exp).abs() < 1e-12, "got {row:?}");
        }
    }

    // (b) small hand-built table, 2 envs × 2 samples each, no vanishing env.
    #[test]
    fn each_held_out_sample_reidentifies_its_env() {
        let (table, ctx) = build(&[
            ("a1", Role::Source, Some("envA"), &[100, 100, 0, 0]),
            ("a2", Role::Source, Some("envA"), &[110, 90, 0, 0]),
            ("b1", Role::Source, Some("envB"), &[0, 0, 100, 100]),
            ("b2", Role::Source, Some("envB"), &[0, 0, 90, 110]),
        ]);
        let sm = predict_loo(&table, &ctx, &params(CollapseMethod::Sum), 42).unwrap();

        assert_eq!(sm.n_sinks(), 4);
        assert_eq!(sm.env_names(), &["envA", "envB", "Unknown"]);
        assert_eq!(sm.env_names().len(), 3);
        assert_eq!(sm.sink_ids(), &["a1", "a2", "b1", "b2"]);

        // envA samples are folds 0,1 (column 0); envB samples folds 2,3 (column 1).
        let own_col = [0usize, 0, 1, 1];
        for i in 0..sm.n_sinks() {
            let row = sm.mean_row(i);
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "row {i} sums to {sum}");
            let argmax = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            assert_eq!(argmax, own_col[i], "fold {i} argmax {argmax}, row {row:?}");
            assert!(
                row[own_col[i]] > 0.8,
                "fold {i} own-env share {} too low ({row:?})",
                row[own_col[i]]
            );
        }
    }

    // (c) missing-env: hold out env X's sole sample -> its column is exactly 0,
    // yet the full frame still lists X.
    #[test]
    fn sole_member_held_out_zeros_its_column() {
        let (table, ctx) = build(&[
            ("x1", Role::Source, Some("X"), &[100, 100, 0, 0]),
            ("y1", Role::Source, Some("Y"), &[0, 0, 100, 100]),
            ("y2", Role::Source, Some("Y"), &[0, 0, 90, 110]),
        ]);
        let sm = predict_loo(&table, &ctx, &params(CollapseMethod::Sum), 42).unwrap();

        // Full frame keeps X even though it vanishes when x1 is held out.
        assert_eq!(sm.env_names(), &["X", "Y", "Unknown"]);
        // Fold 0 holds out x1 (sole X); X column must be exactly 0.
        let row0 = sm.mean_row(0);
        assert_eq!(
            row0[0], 0.0,
            "vanished env column must be exact 0: {row0:?}"
        );
        assert!((row0.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    // (d) empty held-out source -> EmptySink.
    #[test]
    fn empty_held_out_source_is_an_error() {
        // Second source column (index 1) is all zero.
        let (table, ctx) = build(&[
            ("a", Role::Source, Some("envA"), &[10, 0]),
            ("z", Role::Source, Some("envB"), &[0, 0]),
            ("c", Role::Source, Some("envA"), &[5, 5]),
        ]);
        let err = predict_loo(&table, &ctx, &params(CollapseMethod::Sum), 42).unwrap_err();
        assert_eq!(err, Error::EmptySink { sample_index: 1 });
    }

    // (e) determinism: same seed -> identical means.
    #[test]
    fn same_seed_is_deterministic() {
        let (table, ctx) = build(&[
            ("a1", Role::Source, Some("envA"), &[10, 0]),
            ("a2", Role::Source, Some("envA"), &[8, 2]),
            ("b1", Role::Source, Some("envB"), &[0, 10]),
            ("b2", Role::Source, Some("envB"), &[2, 8]),
        ]);
        let p = params(CollapseMethod::Sum);
        let a = predict_loo(&table, &ctx, &p, 99).unwrap();
        let b = predict_loo(&table, &ctx, &p, 99).unwrap();
        assert_eq!(a.means(), b.means());
        assert_eq!(a.stds(), b.stds());
    }

    // Single source sample total -> nothing remains after holding it out.
    #[test]
    fn single_source_sample_yields_no_sources() {
        let (table, ctx) = build(&[
            ("only", Role::Source, Some("envA"), &[10, 5]),
            ("sink", Role::Sink, None, &[3, 3]),
        ]);
        let err = predict_loo(&table, &ctx, &params(CollapseMethod::Sum), 1).unwrap_err();
        assert_eq!(err, Error::NoSources);
    }

    // LOO ignores params.contingency and never emits a tally.
    #[test]
    fn loo_never_produces_contingency() {
        let (table, ctx) = build(&[
            ("a1", Role::Source, Some("envA"), &[10, 0]),
            ("a2", Role::Source, Some("envA"), &[8, 2]),
            ("b1", Role::Source, Some("envB"), &[0, 10]),
        ]);
        let mut p = params(CollapseMethod::Sum);
        p.contingency = true; // must be ignored
        let sm = predict_loo(&table, &ctx, &p, 5).unwrap();
        assert!(sm.contingency().is_none());
    }
}
