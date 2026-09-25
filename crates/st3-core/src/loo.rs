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
//!
//! Rarefaction, when requested, follows the reference's LOO order in
//! [`predict_loo_rarefied`]: there is no up-front collapse to subsample, so each
//! *source sample* is subsampled to the source depth before the folds, and every
//! fold collapses the already-rarefied remainder. The sink depth is ignored,
//! because sinks play no part in leave-one-out.

use std::collections::{BTreeSet, HashMap};

use crate::collapse::collapse_subset;
use crate::collate::{mean_std_over_ensemble, SourceMixing};
use crate::error::{Error, Result};
use crate::estimate::{GibbsEstimator, SinkModel, SinkVec};
use crate::metadata::SampleContext;
use crate::parallel::map_items;
use crate::params::GibbsParams;
use crate::rarefy::{active_depth, rarefy_per_sample, RarefyConfig};
use crate::rng::{rng_for_item, stage_seed, STAGE_RAREFY_SOURCES};
use crate::table::CountTable;

/// One fold's aligned result: the held-out sample's mean and std rows over the
/// fixed full frame, plus its id. Returned per fold and assembled in fold order.
struct FoldRow {
    mean_row: Vec<f64>,
    std_row: Vec<f64>,
    sink_id: String,
}

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
/// `jobs` sizes a per-call scoped worker pool: `0` uses all logical cores, `1`
/// runs serially, and `n` uses `n` threads. Each fold is seeded from `(seed, k)`
/// and rows are assembled in fold order, so the output is byte-identical
/// regardless of `jobs`.
///
/// # Examples
/// ```
/// use st3_core::{
///     CollapseMethod, CountTable, GibbsParams, Role, SampleContext, predict_loo,
/// };
///
/// // Two samples in each of two environments (envA on f0, envB on f1).
/// let table = CountTable::from_coo(
///     vec!["f0".into(), "f1".into()],
///     vec!["a1".into(), "a2".into(), "b1".into(), "b2".into()],
///     &[0, 0, 1, 1],
///     &[0, 1, 2, 3],
///     &[100.0, 100.0, 100.0, 100.0],
/// )?;
/// let ctx = SampleContext::new(
///     vec![Role::Source, Role::Source, Role::Source, Role::Source],
///     vec![
///         Some("envA".into()),
///         Some("envA".into()),
///         Some("envB".into()),
///         Some("envB".into()),
///     ],
/// )?;
/// let params = GibbsParams {
///     restarts: 4,
///     draws_per_restart: 2,
///     burnin: 5,
///     collapse: CollapseMethod::Sum,
///     ..GibbsParams::default()
/// };
/// let loo = predict_loo(&table, &ctx, &params, 42, 1)?;
///
/// // One row per held-out source sample; columns are the source envs then Unknown.
/// assert_eq!(loo.n_sinks(), 4);
/// assert_eq!(loo.env_names(), &["envA", "envB", "Unknown"]);
/// assert_eq!(loo.sink_ids(), &["a1", "a2", "b1", "b2"]);
/// assert!((loo.mean_row(0).iter().sum::<f64>() - 1.0).abs() < 1e-9);
/// assert!(loo.contingency().is_none()); // LOO never emits a tally
/// # Ok::<(), st3_core::Error>(())
/// ```
///
/// # Errors
/// [`Error::EmptySink`] if a held-out source column has no sequences; propagates
/// [`collapse_subset`] ([`Error::NoSources`] when only a single source sample
/// exists in total, so nothing remains after holding it out),
/// [`GibbsEstimator::prepare`] (parameter validation), and [`Error::ThreadPool`].
/// The lowest-index fold's error is returned, independent of `jobs`.
pub fn predict_loo(
    table: &CountTable,
    ctx: &SampleContext,
    params: &GibbsParams,
    seed: u64,
    jobs: usize,
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

    // Each fold is an independent work item: hold out source `k`, re-collapse the
    // rest, estimate, and scatter into the full frame. `map_items` runs the folds
    // over the scoped pool and returns them in fold order (see `crate::parallel`).
    let rows = map_items(jobs, source_indices.len(), |k| {
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
        let hs = source_indices[k] as usize;
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
        let mean_row = scatter_into_frame(reduced_env_names, &mean_k, &full_col, v_full);
        let std_row = scatter_into_frame(reduced_env_names, &std_k, &full_col, v_full);
        Ok(FoldRow {
            mean_row,
            std_row,
            sink_id: table.sample_ids()[hs].clone(),
        })
    })?;

    // Assemble the fold rows (in order) into the dense sink-major matrices.
    let n_folds = rows.len();
    let mut means = Vec::with_capacity(n_folds * v_full);
    let mut stds = Vec::with_capacity(n_folds * v_full);
    let mut sink_ids = Vec::with_capacity(n_folds);
    for row in rows {
        means.extend_from_slice(&row.mean_row);
        stds.extend_from_slice(&row.std_row);
        sink_ids.push(row.sink_id);
    }

    Ok(SourceMixing::from_parts(
        sink_ids, env_names, means, stds, None,
    ))
}

/// Leave-one-out source prediction, rarefying the source samples first as
/// `rarefy` directs.
///
/// When `rarefy.source_depth` is set, each source sample is subsampled to that
/// depth — per sample, because leave-one-out has no up-front collapse to
/// subsample, which is the reference's order — and [`predict_loo`] then runs on
/// the result. `rarefy.sink_depth` is ignored: sinks play no part in
/// leave-one-out. A depth of `None` (or `0`) makes this exactly [`predict_loo`].
/// A source sample shallower than the depth passes through unchanged, as
/// [`crate::rarefy()`] does.
///
/// The subsampling stage draws from its own stream derived from `seed`, distinct
/// from the sampler's, so the folds' randomness for a given seed is the same
/// with or without rarefaction; the output never depends on `jobs`.
///
/// # Errors
/// Those of [`predict_loo`], plus [`crate::rarefy()`]'s.
pub fn predict_loo_rarefied(
    table: &CountTable,
    ctx: &SampleContext,
    params: &GibbsParams,
    rarefy: &RarefyConfig,
    seed: u64,
    jobs: usize,
) -> Result<SourceMixing> {
    match active_depth(rarefy.source_depth) {
        Some(_) => {
            let depths = RarefyConfig {
                sink_depth: None,
                ..*rarefy
            }
            .depths_for(ctx);
            let rarefied = rarefy_per_sample(
                table,
                &depths,
                rarefy.with_replacement,
                stage_seed(seed, STAGE_RAREFY_SOURCES),
            )?;
            predict_loo(rarefied.table(), ctx, params, seed, jobs)
        }
        None => predict_loo(table, ctx, params, seed, jobs),
    }
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
        let sm = predict_loo(&table, &ctx, &params(CollapseMethod::Sum), 42, 1).unwrap();

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
        let sm = predict_loo(&table, &ctx, &params(CollapseMethod::Sum), 42, 1).unwrap();

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
        let err = predict_loo(&table, &ctx, &params(CollapseMethod::Sum), 42, 1).unwrap_err();
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
        let a = predict_loo(&table, &ctx, &p, 99, 1).unwrap();
        let b = predict_loo(&table, &ctx, &p, 99, 1).unwrap();
        assert_eq!(a.means(), b.means());
        assert_eq!(a.stds(), b.stds());
    }

    // Determinism guard (roadmap R8): identical output at any thread count.
    #[test]
    fn output_is_identical_across_job_counts() {
        let (table, ctx) = build(&[
            ("a1", Role::Source, Some("envA"), &[100, 100, 0, 0]),
            ("a2", Role::Source, Some("envA"), &[110, 90, 0, 0]),
            ("b1", Role::Source, Some("envB"), &[0, 0, 100, 100]),
            ("b2", Role::Source, Some("envB"), &[0, 0, 90, 110]),
            ("c1", Role::Source, Some("envC"), &[50, 0, 50, 0]),
        ]);
        let p = params(CollapseMethod::Sum);
        let serial = predict_loo(&table, &ctx, &p, 2024, 1).unwrap();
        for jobs in [0usize, 2, 8] {
            let parallel = predict_loo(&table, &ctx, &p, 2024, jobs).unwrap();
            assert_eq!(parallel, serial, "jobs={jobs} diverged from serial");
        }
    }

    // The lowest-index held-out empty source is reported regardless of jobs.
    #[test]
    fn empty_held_out_error_is_deterministic_across_jobs() {
        // Folds run over source_indices ascending: a(0), z1(1), c(2), z2(3).
        // z1 and z2 are empty source columns; the lowest fold (z1, index 1) wins.
        let (table, ctx) = build(&[
            ("a", Role::Source, Some("envA"), &[10, 0]),
            ("z1", Role::Source, Some("envB"), &[0, 0]),
            ("c", Role::Source, Some("envA"), &[5, 5]),
            ("z2", Role::Source, Some("envB"), &[0, 0]),
        ]);
        let p = params(CollapseMethod::Sum);
        for jobs in [1usize, 4] {
            let err = predict_loo(&table, &ctx, &p, 7, jobs).unwrap_err();
            assert_eq!(err, Error::EmptySink { sample_index: 1 }, "jobs={jobs}");
        }
    }

    // Single source sample total -> nothing remains after holding it out.
    #[test]
    fn single_source_sample_yields_no_sources() {
        let (table, ctx) = build(&[
            ("only", Role::Source, Some("envA"), &[10, 5]),
            ("sink", Role::Sink, None, &[3, 3]),
        ]);
        let err = predict_loo(&table, &ctx, &params(CollapseMethod::Sum), 1, 1).unwrap_err();
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
        let sm = predict_loo(&table, &ctx, &p, 5, 1).unwrap();
        assert!(sm.contingency().is_none());
    }

    // With no depth set the rarefied driver *is* `predict_loo`.
    #[test]
    fn rarefied_with_no_depths_equals_predict_loo() {
        let (table, ctx) = build(&[
            ("a1", Role::Source, Some("envA"), &[10, 0]),
            ("a2", Role::Source, Some("envA"), &[8, 2]),
            ("b1", Role::Source, Some("envB"), &[0, 10]),
        ]);
        let p = params(CollapseMethod::Sum);
        let plain = predict_loo(&table, &ctx, &p, 3, 1).unwrap();
        let rarefied =
            predict_loo_rarefied(&table, &ctx, &p, &RarefyConfig::default(), 3, 1).unwrap();
        assert_eq!(rarefied, plain);
    }

    // The ST2 leave-one-out order: there is no up-front collapse to subsample,
    // so each *source sample* is subsampled to the source depth and every fold
    // then collapses the remaining, already-rarefied samples. Equivalent to
    // rarefying the source columns per sample (with the source stage seed) and
    // running the plain driver on the result.
    #[test]
    fn source_depth_subsamples_each_source_sample_before_the_folds() {
        let (table, ctx) = build(&[
            ("a1", Role::Source, Some("envA"), &[100, 100, 0, 0]),
            ("a2", Role::Source, Some("envA"), &[110, 90, 0, 0]),
            ("b1", Role::Source, Some("envB"), &[0, 0, 100, 100]),
            ("b2", Role::Source, Some("envB"), &[0, 0, 90, 110]),
            ("s0", Role::Sink, None, &[50, 50, 50, 50]),
        ]);
        let p = params(CollapseMethod::Sum);
        let cfg = RarefyConfig {
            source_depth: Some(100),
            sink_depth: None,
            with_replacement: false,
        };
        let seed = 5;
        let got = predict_loo_rarefied(&table, &ctx, &p, &cfg, seed, 1).unwrap();

        let depths = cfg.depths_for(&ctx);
        let rarefied = rarefy_per_sample(
            &table,
            &depths,
            false,
            stage_seed(seed, STAGE_RAREFY_SOURCES),
        )
        .unwrap();
        for &s in ctx.source_indices() {
            assert_eq!(rarefied.table().column_sum(s as usize), 100, "source {s}");
        }
        let expected = predict_loo(rarefied.table(), &ctx, &p, seed, 1).unwrap();
        assert_eq!(got, expected);
    }
}
