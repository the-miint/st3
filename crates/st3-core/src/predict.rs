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
//! Rarefaction, when requested, follows SourceTracker2's sink-mode order in
//! [`predict_sinks_rarefied`]: sources are collapsed *first* and each collapsed
//! environment is then subsampled to the source depth, while sinks are
//! subsampled per sample. [`predict_sinks`] is the same driver with rarefaction
//! off. Leave-one-out is a separate driver ([`crate::predict_loo`]).

use crate::collapse::{collapse_sources, CollapsedSources};
use crate::collate::{collate, SourceMixing};
use crate::error::{Error, Result};
use crate::estimate::{GibbsEstimator, SinkModel, SinkVec};
use crate::metadata::SampleContext;
use crate::parallel::map_items;
use crate::params::GibbsParams;
use crate::rarefy::{active_depth, check_depth, rarefy_per_sample, Rarefied, RarefyConfig};
use crate::rng::{rng_for_item, stage_seed, STAGE_RAREFY_SINKS, STAGE_RAREFY_SOURCES};
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
/// byte-identical regardless of `jobs`. Equivalent to [`predict_sinks_rarefied`]
/// with rarefaction disabled.
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
/// sink, checked before any other work and so independent of `jobs`); propagates
/// [`collapse_sources`], [`GibbsEstimator::prepare`] (parameter validation), and
/// [`Error::ThreadPool`].
pub fn predict_sinks(
    table: &CountTable,
    ctx: &SampleContext,
    params: &GibbsParams,
    seed: u64,
    jobs: usize,
) -> Result<SourceMixing> {
    predict_sinks_rarefied(table, ctx, params, &RarefyConfig::default(), seed, jobs)
}

/// Estimate the source composition of every sink in `ctx`, rarefying first as
/// `rarefy` directs.
///
/// The order follows SourceTracker2's sink mode. Sources are collapsed by
/// `params.collapse` and then, when `rarefy.source_depth` is set, each
/// *collapsed environment* is subsampled to that depth — so every environment
/// enters the sampler with exactly `source_depth` sequences however many samples
/// it pools. Sinks are subsampled per sample to `rarefy.sink_depth`. A depth of
/// `None` (or `0`) leaves that side untouched; with both unset this is exactly
/// [`predict_sinks`]. An environment or sink shallower than its depth refuses
/// the run with [`Error::ShallowSamples`] before any sampling, as the reference
/// does; the error names the role, how many fall short, and the shallowest.
///
/// Each stochastic stage — sink subsampling, source subsampling, and the
/// sampler — draws from its own stream derived from `seed`, so the sampler's
/// randomness for a given seed is the same with or without rarefaction, and the
/// output depends on the seed alone, never on `jobs`.
///
/// # Examples
/// ```
/// use st3_core::{
///     CollapseMethod, CountTable, GibbsParams, RarefyConfig, Role, SampleContext,
///     predict_sinks_rarefied,
/// };
///
/// // envA pools two samples and envB has one; every column holds 100 sequences.
/// let table = CountTable::from_coo(
///     vec!["f0".into(), "f1".into()],
///     vec!["a1".into(), "a2".into(), "b".into(), "sink".into()],
///     &[0, 1, 0, 1, 1, 0, 1],
///     &[0, 0, 1, 1, 2, 3, 3],
///     &[90.0, 10.0, 80.0, 20.0, 100.0, 70.0, 30.0],
/// )?;
/// let ctx = SampleContext::new(
///     vec![Role::Source, Role::Source, Role::Source, Role::Sink],
///     vec![Some("envA".into()), Some("envA".into()), Some("envB".into()), None],
/// )?;
/// let params = GibbsParams {
///     restarts: 4,
///     draws_per_restart: 2,
///     burnin: 5,
///     collapse: CollapseMethod::Sum,
///     ..GibbsParams::default()
/// };
/// // Subsample each collapsed environment to 50 sequences and the sink to 40.
/// let rarefy = RarefyConfig {
///     source_depth: Some(50),
///     sink_depth: Some(40),
///     with_replacement: false,
/// };
/// let mixing = predict_sinks_rarefied(&table, &ctx, &params, &rarefy, 42, 1)?;
///
/// assert_eq!(mixing.env_names(), &["envA", "envB", "Unknown"]);
/// let row = mixing.mean_row(0);
/// assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
/// assert!(row[0] > row[1]); // the sink is mostly f0, like envA
/// # Ok::<(), st3_core::Error>(())
/// ```
///
/// # Errors
/// Those of [`predict_sinks`], plus [`Error::ShallowSamples`] and
/// [`crate::rarefy()`]'s.
pub fn predict_sinks_rarefied(
    table: &CountTable,
    ctx: &SampleContext,
    params: &GibbsParams,
    rarefy: &RarefyConfig,
    seed: u64,
    jobs: usize,
) -> Result<SourceMixing> {
    // Cheap preconditions first, so a run that cannot succeed fails before
    // collapse, subsampling, model precompute, or the worker pool spend anything
    // on it. The lowest-index empty sink is reported, as the serial loop would.
    if let Some(&s) = ctx
        .sink_indices()
        .iter()
        .find(|&&s| table.column_sum(s as usize) == 0)
    {
        return Err(Error::EmptySink {
            sample_index: s as usize,
        });
    }

    // Reference order: collapse, refuse a collapsed environment below the depth,
    // then subsample the collapsed environments.
    let sources = collapse_sources(table, ctx, params.collapse)?;
    let sources = match active_depth(rarefy.source_depth) {
        Some(depth) => {
            check_depth(
                sources.counts(),
                0..sources.n_sources(),
                depth,
                "collapsed source",
            )?;
            sources.rarefy(
                Some(depth),
                rarefy.with_replacement,
                stage_seed(seed, STAGE_RAREFY_SOURCES),
            )?
        }
        None => sources,
    };

    // Sinks are subsampled per sample; the source columns of `table` are left
    // alone (their depth is `None` here) because the collapsed copy above is what
    // the sampler sees.
    let sinks: Option<Rarefied> = match active_depth(rarefy.sink_depth) {
        Some(depth) => {
            check_depth(
                table,
                ctx.sink_indices().iter().map(|&s| s as usize),
                depth,
                "sink",
            )?;
            let depths = RarefyConfig {
                source_depth: None,
                ..*rarefy
            }
            .depths_for(ctx);
            Some(rarefy_per_sample(
                table,
                &depths,
                rarefy.with_replacement,
                stage_seed(seed, STAGE_RAREFY_SINKS),
            )?)
        }
        None => None,
    };
    let sink_table = sinks.as_ref().map_or(table, Rarefied::table);

    estimate_sinks(&sources, sink_table, ctx, params, seed, jobs)
}

/// Estimate every sink of `table` against already-collapsed (and possibly
/// rarefied) `sources`: the shared body of the two sink drivers. Every sink
/// column must be non-empty (the caller has checked).
fn estimate_sinks(
    sources: &CollapsedSources,
    table: &CountTable,
    ctx: &SampleContext,
    params: &GibbsParams,
    seed: u64,
    jobs: usize,
) -> Result<SourceMixing> {
    let prep = GibbsEstimator::prepare(sources, params)?;

    // Column labels: collapsed source envs (sorted) then Unknown (last).
    let mut env_names: Vec<String> = sources.env_names().to_vec();
    env_names.push("Unknown".to_string());

    let sink_cols = ctx.sink_indices();
    // Each sink is an independent work item seeded by its position, so the fan-out
    // is deterministic across thread counts (see `crate::parallel`).
    let pairs = map_items(jobs, sink_cols.len(), |item_index| {
        let s = sink_cols[item_index] as usize;
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

    fn rarefy_cfg(source_depth: Option<u32>, sink_depth: Option<u32>) -> RarefyConfig {
        RarefyConfig {
            source_depth,
            sink_depth,
            with_replacement: false,
        }
    }

    /// Densify environment `env` of `sources` to a length-`tau` count vector.
    fn dense_env(sources: &CollapsedSources, env: usize, tau: usize) -> Vec<u32> {
        let (rows, counts) = sources.column(env);
        let mut d = vec![0u32; tau];
        for (&r, &c) in rows.iter().zip(counts.iter()) {
            d[r as usize] = c;
        }
        d
    }

    // With no depth set the rarefied driver *is* `predict_sinks`, so a caller
    // can route every run through it without changing unrarefied results.
    #[test]
    fn rarefied_with_no_depths_equals_predict_sinks() {
        let (table, ctx) = build(&[
            ("a", Role::Source, Some("envA"), &[10, 0]),
            ("b", Role::Source, Some("envB"), &[0, 10]),
            ("s0", Role::Sink, None, &[8, 2]),
        ]);
        let p = params(CollapseMethod::Sum, true);
        let plain = predict_sinks(&table, &ctx, &p, 3, 1).unwrap();
        let rarefied =
            predict_sinks_rarefied(&table, &ctx, &p, &RarefyConfig::default(), 3, 1).unwrap();
        assert_eq!(rarefied, plain);
    }

    // Sinks are subsampled per sample before estimation. The contingency
    // tally's mass equals the sink's sequence count that the sampler saw, so a
    // 10-sequence sink rarefied to 6 must tally exactly 6.
    #[test]
    fn sink_depth_subsamples_each_sink_before_estimation() {
        let (table, ctx) = build(&[
            ("a", Role::Source, Some("envA"), &[10, 0]),
            ("b", Role::Source, Some("envB"), &[0, 10]),
            ("s0", Role::Sink, None, &[8, 2]),
            ("s1", Role::Sink, None, &[3, 7]),
        ]);
        let p = params(CollapseMethod::Sum, true);
        let sm =
            predict_sinks_rarefied(&table, &ctx, &p, &rarefy_cfg(None, Some(6)), 1, 1).unwrap();
        for (i, tally) in sm.contingency().unwrap().iter().enumerate() {
            let mass: f64 = tally.triples().map(|(_, _, m)| m).sum();
            assert!((mass - 6.0).abs() < 1e-9, "sink {i} tally mass = {mass}");
        }
    }

    // The ST2 sink-mode order: collapse first, then subsample each *collapsed
    // environment* to the source depth. Here envA pools two 6-sequence samples
    // (12 under Sum) and the depth is 10: every member is shallower than the
    // depth but the environment is not. Per-sample-first rarefaction would
    // leave both members untouched and hand the sampler a 12-count environment;
    // the reference order hands it exactly 10. Proved by equivalence with
    // `predict_sinks` on a table whose source columns *are* the collapsed-then-
    // rarefied environments (a single member collapses to itself under Sum).
    #[test]
    fn source_depth_applies_to_the_collapsed_environment_not_its_members() {
        let sink: &[u32] = &[5, 3, 2];
        let (table, ctx) = build(&[
            ("a1", Role::Source, Some("envA"), &[4, 2, 0]),
            ("a2", Role::Source, Some("envA"), &[2, 4, 0]),
            ("b", Role::Source, Some("envB"), &[0, 0, 10]),
            ("s0", Role::Sink, None, sink),
        ]);
        let p = params(CollapseMethod::Sum, false);
        let seed = 11;
        let got =
            predict_sinks_rarefied(&table, &ctx, &p, &rarefy_cfg(Some(10), None), seed, 1).unwrap();

        // The sampler's input under the reference order.
        let sources = collapse_sources(&table, &ctx, CollapseMethod::Sum)
            .unwrap()
            .rarefy(Some(10), false, stage_seed(seed, STAGE_RAREFY_SOURCES))
            .unwrap();
        assert_eq!(
            sources.counts().column_sum(0),
            10,
            "envA rarefied to the depth"
        );
        assert_eq!(
            sources.counts().column_sum(1),
            10,
            "envB rarefied to the depth"
        );
        let env_a = dense_env(&sources, 0, 3);
        let env_b = dense_env(&sources, 1, 3);
        let (table2, ctx2) = build(&[
            ("envA", Role::Source, Some("envA"), &env_a),
            ("envB", Role::Source, Some("envB"), &env_b),
            ("s0", Role::Sink, None, sink),
        ]);
        let expected = predict_sinks(&table2, &ctx2, &p, seed, 1).unwrap();
        assert_eq!(got, expected);
    }

    // Determinism guard extends to the rarefied path: both subsampling stages
    // and the sampler are seeded per item, so `jobs` cannot change the output.
    #[test]
    fn rarefied_output_is_identical_across_job_counts() {
        let (table, ctx) = build(&[
            ("a1", Role::Source, Some("envA"), &[10, 0, 1]),
            ("a2", Role::Source, Some("envA"), &[8, 2, 1]),
            ("b", Role::Source, Some("envB"), &[0, 10, 1]),
            ("c", Role::Source, Some("envC"), &[1, 1, 10]),
            ("s0", Role::Sink, None, &[8, 2, 1]),
            ("s1", Role::Sink, None, &[3, 7, 2]),
            ("s2", Role::Sink, None, &[1, 1, 9]),
        ]);
        let p = params(CollapseMethod::Mean, true);
        let cfg = rarefy_cfg(Some(8), Some(8));
        let serial = predict_sinks_rarefied(&table, &ctx, &p, &cfg, 2024, 1).unwrap();
        for jobs in [0usize, 2, 8] {
            let parallel = predict_sinks_rarefied(&table, &ctx, &p, &cfg, 2024, jobs).unwrap();
            assert_eq!(parallel, serial, "jobs={jobs} diverged from serial");
        }
    }

    // The reference refuses to run when a *collapsed* environment cannot reach
    // the source depth. Under mean collapse two members that each hold exactly
    // the depth can still collapse to nothing: here each has one sequence on
    // ten disjoint taxa, so every per-taxon mean floors to zero. Per-sample
    // checking would have passed both members; checking the environment, as
    // SourceTracker2 does, rejects the run before any sampling.
    #[test]
    fn shallow_collapsed_source_is_rejected_before_sampling() {
        let a1: Vec<u32> = (0..20).map(|t| u32::from(t < 10)).collect();
        let a2: Vec<u32> = (0..20).map(|t| u32::from(t >= 10)).collect();
        let b: Vec<u32> = (0..20).map(|t| if t < 10 { 2 } else { 0 }).collect();
        let sink = [1u32; 20];
        let (table, ctx) = build(&[
            ("a1", Role::Source, Some("envA"), &a1[..]),
            ("a2", Role::Source, Some("envA"), &a2[..]),
            ("b", Role::Source, Some("envB"), &b[..]),
            ("s0", Role::Sink, None, &sink[..]),
        ]);
        let p = params(CollapseMethod::Mean, false);
        let err = predict_sinks_rarefied(&table, &ctx, &p, &rarefy_cfg(Some(10), None), 1, 1)
            .unwrap_err();
        assert_eq!(
            err,
            Error::ShallowSamples {
                what: "collapsed source",
                depth: 10,
                count: 1,
                shallowest: 0,
            }
        );
    }

    // Sinks below the sink depth are refused up front too, counting every
    // shallow sink and naming the shallowest, exactly as the reference does.
    #[test]
    fn shallow_sinks_are_rejected_before_sampling() {
        let (table, ctx) = build(&[
            ("a", Role::Source, Some("envA"), &[10, 0]),
            ("b", Role::Source, Some("envB"), &[0, 10]),
            ("s0", Role::Sink, None, &[8, 2]), // 10
            ("s1", Role::Sink, None, &[3, 1]), // 4
            ("s2", Role::Sink, None, &[2, 5]), // 7
        ]);
        let p = params(CollapseMethod::Sum, false);
        let err =
            predict_sinks_rarefied(&table, &ctx, &p, &rarefy_cfg(None, Some(8)), 1, 1).unwrap_err();
        assert_eq!(
            err,
            Error::ShallowSamples {
                what: "sink",
                depth: 8,
                count: 2,
                shallowest: 4,
            }
        );
    }

    // Cheap input preconditions run before any expensive work. An empty sink is
    // reported before the source model is prepared (where the sampler
    // parameters are validated), and therefore before collapse, subsampling,
    // or the worker pool spend anything on a run that cannot succeed. With
    // `jobs > 1` the alternative is other sinks running full chains first.
    #[test]
    fn empty_sink_is_reported_before_the_model_is_prepared() {
        let (table, ctx) = build(&[
            ("a", Role::Source, Some("envA"), &[10, 0]),
            ("b", Role::Source, Some("envB"), &[0, 10]),
            ("s0", Role::Sink, None, &[5, 5]),
            ("empty", Role::Sink, None, &[0, 0]),
        ]);
        let mut p = params(CollapseMethod::Sum, false);
        p.restarts = 0; // invalid; rejected when the model is prepared
        let err = predict_sinks(&table, &ctx, &p, 1, 1).unwrap_err();
        assert_eq!(err, Error::EmptySink { sample_index: 3 });
    }
}
