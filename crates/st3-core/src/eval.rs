// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Statistical-equivalence harness: synthetic mixtures with known truth + scoring.
//!
//! Where the fixture-backed tests check st3 against committed R/Python oracles,
//! this module checks it against **ground truth generated in-process**: two source
//! distributions `P`, `Q` are drawn from a Dirichlet, synthetic sinks are built as
//! known mixtures `m·P + (1−m)·Q`, and the estimator must recover `m`. Following
//! the authors' validation recipe (algorithm investigation §12 / Supp Fig 7,
//! design §16), recovery quality is scored by R² of recovered-vs-true mixing and
//! related to the Jensen–Shannon divergence between `P` and `Q`: well-separated
//! sources (high JSD) are recovered accurately, and accuracy degrades gracefully
//! as the sources become indistinguishable (JSD → 0).
//!
//! The pure scoring functions ([`rmse`], [`r_squared`], [`jensen_shannon_divergence`])
//! and the generator ([`simulate_two_source`]) are exposed independently so the
//! deferred α-tuning milestone can reuse them as its scoring substrate (design
//! §14): generate one scenario, then score [`predict_sinks`] output under varying
//! α. Everything is seeded through [`crate::rng::rng_for_item`], so a scenario and
//! its score depend only on the seed.
//!
//! Metric conventions (chosen here — the corpus names the metrics but not their
//! formulas): **R²** is the squared Pearson correlation (∈ [0, 1], robust to the
//! systematic Unknown-mass offset, → 0 as recovery decorrelates from truth), and
//! **JSD** is in bits (base-2, ∈ [0, 1]).

use rand::RngExt;
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand_distr::multi::Dirichlet;

use crate::collate::SourceMixing;
use crate::error::{Error, Result};
use crate::metadata::{Role, SampleContext};
use crate::params::GibbsParams;
use crate::predict::predict_sinks;
use crate::rng::{ItemRng, rng_for_item};
use crate::table::CountTable;

/// Environment label of the source [`Simulation::truth`] measures the fraction of.
const SOURCE_A: &str = "sourceA";
/// Environment label of the complementary source.
const SOURCE_B: &str = "sourceB";

/// Work-item index for the scenario-generation RNG stream. Deliberately outside
/// the range [`predict_sinks`] seeds its per-sink chains over (`0..n_sinks`), so
/// data generation and estimation draw from independent streams for the same run
/// `seed` (see [`crate::rng::rng_for_item`]).
const GEN_ITEM: u64 = u64::MAX;

/// Root-mean-square error between two equal-length series.
///
/// `sqrt(mean_i (a_i − b_i)²)`.
///
/// # Panics
/// Panics (in debug) if `a` and `b` differ in length.
pub(crate) fn rmse(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "rmse operands must be equal length");
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let sse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum();
    (sse / n as f64).sqrt()
}

/// Coefficient of determination as the **squared Pearson correlation** between
/// `predicted` and `truth` (∈ [0, 1]).
///
/// This is the "R² of estimated vs true" a scatter plot implies: scale- and
/// bias-invariant, and → 0 as the two decorrelate. If either series has
/// effectively zero variance (`< 1e-12`), the correlation is undefined and this
/// returns `0.0` (uninformative) rather than a NaN.
///
/// # Panics
/// Panics (in debug) if `predicted` and `truth` differ in length.
pub(crate) fn r_squared(predicted: &[f64], truth: &[f64]) -> f64 {
    debug_assert_eq!(
        predicted.len(),
        truth.len(),
        "r_squared operands must be equal length"
    );
    let n = predicted.len();
    if n == 0 {
        return 0.0;
    }
    let inv_n = 1.0 / n as f64;
    let mean_p: f64 = predicted.iter().sum::<f64>() * inv_n;
    let mean_t: f64 = truth.iter().sum::<f64>() * inv_n;

    let mut cov = 0.0;
    let mut var_p = 0.0;
    let mut var_t = 0.0;
    for (&p, &t) in predicted.iter().zip(truth.iter()) {
        let dp = p - mean_p;
        let dt = t - mean_t;
        cov += dp * dt;
        var_p += dp * dp;
        var_t += dt * dt;
    }
    // Undefined correlation when either series is (near-)constant.
    if var_p < 1e-12 || var_t < 1e-12 {
        return 0.0;
    }
    let r = cov / (var_p * var_t).sqrt();
    r * r
}

/// Jensen–Shannon divergence between two probability vectors, in **bits**.
///
/// `JSD(p, q) = ½ Σ p_i log₂(p_i / m_i) + ½ Σ q_i log₂(q_i / m_i)` with
/// `m = (p + q) / 2` and the convention `0 · log 0 = 0`. For probability vectors
/// the result lies in `[0, 1]`: `0` when `p == q`, `1` when their supports are
/// disjoint.
///
/// # Panics
/// Panics (in debug) if `p` and `q` differ in length.
pub(crate) fn jensen_shannon_divergence(p: &[f64], q: &[f64]) -> f64 {
    debug_assert_eq!(p.len(), q.len(), "jsd operands must be equal length");
    // Half of each component's contribution to KL(·‖M); 0·log0 ≡ 0.
    let half_kl = |x: f64, m: f64| if x > 0.0 { x * (x / m).log2() } else { 0.0 };
    let mut acc = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let mi = 0.5 * (pi + qi);
        if mi > 0.0 {
            acc += 0.5 * (half_kl(pi, mi) + half_kl(qi, mi));
        }
    }
    // Guard tiny negative round-off so the result stays in [0, 1].
    acc.max(0.0)
}

/// Parameters for a two-source Dirichlet mixture simulation.
///
/// Two source distributions are drawn from a symmetric `Dirichlet(concentration)`
/// over `n_taxa` taxa; a small `concentration` makes them spiky and well-separated
/// (high JSD), a large one makes them near-uniform and similar (low JSD).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimConfig {
    /// Number of taxa (features); the Dirichlet dimension. Must be ≥ 2.
    pub n_taxa: usize,
    /// Source samples drawn per environment (each becomes a source column).
    pub samples_per_source: usize,
    /// Sequences per source sample (multinomial depth). Must be ≥ 1.
    pub seqs_per_sample: u32,
    /// Number of synthetic sinks (mixture trials). Must be ≥ 1.
    pub n_trials: usize,
    /// Sequences per synthetic sink (multinomial depth). Must be ≥ 1.
    pub sink_depth: u32,
    /// Symmetric Dirichlet concentration. Must be finite and > 0.
    pub concentration: f64,
}

impl SimConfig {
    /// Validate the field ranges.
    ///
    /// # Errors
    /// [`Error::InvalidParam`] if `n_taxa < 2`, any count is zero, or
    /// `concentration` is not finite and positive.
    fn validate(&self) -> Result<()> {
        let bad = |name, reason| Err(Error::InvalidParam { name, reason });
        if self.n_taxa < 2 {
            return bad("n_taxa", "must be at least 2");
        }
        if self.samples_per_source == 0 {
            return bad("samples_per_source", "must be positive");
        }
        if self.seqs_per_sample == 0 {
            return bad("seqs_per_sample", "must be positive");
        }
        if self.n_trials == 0 {
            return bad("n_trials", "must be positive");
        }
        if self.sink_depth == 0 {
            return bad("sink_depth", "must be positive");
        }
        if !(self.concentration.is_finite() && self.concentration > 0.0) {
            return bad("concentration", "must be finite and positive");
        }
        Ok(())
    }
}

/// A generated two-source scenario: the count table, its sample context, the true
/// mixing weight of each sink, and the JSD between the two source distributions.
#[derive(Debug, Clone, PartialEq)]
pub struct Simulation {
    table: CountTable,
    ctx: SampleContext,
    truth: Vec<f64>,
    jsd: f64,
}

impl Simulation {
    /// The synthetic count table (source columns then sink columns).
    pub fn table(&self) -> &CountTable {
        &self.table
    }

    /// The source/sink split aligned to [`Self::table`].
    pub fn context(&self) -> &SampleContext {
        &self.ctx
    }

    /// True mixing weight (fraction from source `A`) of each sink, in sink order.
    pub fn truth(&self) -> &[f64] {
        &self.truth
    }

    /// Jensen–Shannon divergence (bits) between the two source distributions.
    pub fn jsd(&self) -> f64 {
        self.jsd
    }

    /// The environment name of source `A` (the one [`Self::truth`] measures).
    pub fn source_a_env(&self) -> &str {
        SOURCE_A
    }
}

/// Recovery quality of a simulation run: its JSD and the R²/RMSE of recovered vs
/// true mixing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecoveryScore {
    /// JSD (bits) between the two source distributions of the scenario.
    pub jsd: f64,
    /// Squared Pearson correlation of recovered vs true mixing (∈ [0, 1]).
    pub r2: f64,
    /// RMSE of recovered vs true mixing.
    pub rmse: f64,
}

/// Generate a two-source Dirichlet mixture scenario with known mixing weights.
///
/// Draws `P, Q ~ Dirichlet(concentration)` over `n_taxa`, builds
/// `samples_per_source` source columns from each (multinomial, `seqs_per_sample`
/// deep, environments `sourceA`/`sourceB`), then `n_trials` sink columns, each a
/// mixture `round(m·sink_depth)` reads from `P` plus the remainder from `Q` for a
/// fresh `m ~ Uniform(0, 1)`. All randomness derives from `seed`.
///
/// # Examples
/// ```
/// use st3_core::{SimConfig, simulate_two_source};
///
/// let cfg = SimConfig {
///     n_taxa: 8,
///     samples_per_source: 2,
///     seqs_per_sample: 100,
///     n_trials: 4,
///     sink_depth: 100,
///     concentration: 0.1, // small concentration -> well-separated sources
/// };
/// let sim = simulate_two_source(&cfg, 42)?;
///
/// // 2 * samples_per_source source columns, then n_trials sink columns.
/// assert_eq!(sim.table().n_samples(), 2 * 2 + 4);
/// assert_eq!(sim.truth().len(), 4);
/// assert!(sim.truth().iter().all(|&m| (0.0..=1.0).contains(&m)));
/// assert!((0.0..=1.0).contains(&sim.jsd()));
/// # Ok::<(), st3_core::Error>(())
/// ```
///
/// # Errors
/// [`Error::InvalidParam`] if any [`SimConfig`] field is out of range; propagates
/// [`CountTable::from_coo`] and [`SampleContext::for_table`] construction errors.
pub fn simulate_two_source(cfg: &SimConfig, seed: u64) -> Result<Simulation> {
    cfg.validate()?;

    // The generation stream uses the reserved item index (see `GEN_ITEM`), so it
    // is independent of the estimator's per-sink streams `0..n_sinks` even when the
    // same run `seed` drives both.
    let mut rng = rng_for_item(seed, GEN_ITEM);

    // Two symmetric-Dirichlet source distributions over the taxa.
    let dir =
        Dirichlet::new(&vec![cfg.concentration; cfg.n_taxa]).map_err(|_| Error::InvalidParam {
            name: "concentration",
            reason: "does not yield a valid Dirichlet distribution",
        })?;
    let p_probs: Vec<f64> = dir.sample(&mut rng);
    let q_probs: Vec<f64> = dir.sample(&mut rng);
    let jsd = jensen_shannon_divergence(&p_probs, &q_probs);

    let wi_p = WeightedIndex::new(p_probs.iter().copied()).map_err(|_| Error::InvalidParam {
        name: "concentration",
        reason: "source distribution P has no positive mass",
    })?;
    let wi_q = WeightedIndex::new(q_probs.iter().copied()).map_err(|_| Error::InvalidParam {
        name: "concentration",
        reason: "source distribution Q has no positive mass",
    })?;

    let feature_ids: Vec<String> = (0..cfg.n_taxa).map(|t| format!("t{t}")).collect();
    let mut sample_ids: Vec<String> = Vec::new();
    let mut roles: Vec<Role> = Vec::new();
    let mut envs: Vec<Option<String>> = Vec::new();
    let mut rows: Vec<u32> = Vec::new();
    let mut cols: Vec<u32> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    let mut dense = vec![0u32; cfg.n_taxa];
    let mut col_idx: u32 = 0;

    // One dense column -> COO nonzeros for the current sample.
    let emit =
        |dense: &[u32], col: u32, rows: &mut Vec<u32>, cols: &mut Vec<u32>, vals: &mut Vec<f64>| {
            for (t, &c) in dense.iter().enumerate() {
                if c > 0 {
                    rows.push(t as u32);
                    cols.push(col);
                    vals.push(f64::from(c));
                }
            }
        };

    // Source columns: `samples_per_source` from P (env sourceA), then from Q.
    for (env, tag, wi) in [(SOURCE_A, "srcA", &wi_p), (SOURCE_B, "srcB", &wi_q)] {
        for i in 0..cfg.samples_per_source {
            dense.fill(0);
            multinomial_into(&mut rng, wi, cfg.seqs_per_sample, &mut dense);
            emit(&dense, col_idx, &mut rows, &mut cols, &mut vals);
            sample_ids.push(format!("{tag}_{i}"));
            roles.push(Role::Source);
            envs.push(Some(env.to_string()));
            col_idx += 1;
        }
    }

    // Sink columns: known mixtures m·P + (1−m)·Q at depth `sink_depth`.
    let mut truth = Vec::with_capacity(cfg.n_trials);
    for t in 0..cfg.n_trials {
        let m: f64 = rng.random::<f64>();
        truth.push(m);
        let n_p = (m * f64::from(cfg.sink_depth)).round() as u32;
        let n_q = cfg.sink_depth - n_p; // m ∈ [0,1) ⇒ n_p ≤ sink_depth, no underflow
        dense.fill(0);
        multinomial_into(&mut rng, &wi_p, n_p, &mut dense);
        multinomial_into(&mut rng, &wi_q, n_q, &mut dense);
        emit(&dense, col_idx, &mut rows, &mut cols, &mut vals);
        sample_ids.push(format!("sink_{t}"));
        roles.push(Role::Sink);
        envs.push(None);
        col_idx += 1;
    }

    let table = CountTable::from_coo(feature_ids, sample_ids, &rows, &cols, &vals)?;
    let ctx = SampleContext::for_table(&table, roles, envs)?;
    Ok(Simulation {
        table,
        ctx,
        truth,
        jsd,
    })
}

/// Tally `depth` categorical draws from `wi` into the dense per-taxon `out`.
fn multinomial_into(rng: &mut ItemRng, wi: &WeightedIndex<f64>, depth: u32, out: &mut [u32]) {
    for _ in 0..depth {
        out[wi.sample(rng)] += 1;
    }
}

/// Score recovered mixing against a scenario's ground truth.
///
/// For each sink, the recovered source-`A` fraction is `est_A / (est_A + est_B)`
/// (renormalized over the two known environments so the systematic Unknown mass
/// does not bias the comparison; if that denominator is ~0 the sink contributes a
/// neutral `0.5`). Returns the scenario JSD plus the R² and RMSE of the recovered
/// fractions against [`Simulation::truth`].
///
/// # Panics
/// Panics if `mixing` does not carry both source environments of `sim`, or its
/// sink count differs from `sim.truth().len()`.
pub(crate) fn score_recovery(sim: &Simulation, mixing: &SourceMixing) -> RecoveryScore {
    assert_eq!(
        mixing.n_sinks(),
        sim.truth().len(),
        "mixing sink count must match the simulation trial count"
    );
    let col_a = mixing
        .env_names()
        .iter()
        .position(|e| e == sim.source_a_env())
        .expect("source A environment present in mixing");
    let col_b = mixing
        .env_names()
        .iter()
        .position(|e| e == SOURCE_B)
        .expect("source B environment present in mixing");

    let predicted: Vec<f64> = (0..mixing.n_sinks())
        .map(|i| {
            let row = mixing.mean_row(i);
            let (a, b) = (row[col_a], row[col_b]);
            let known = a + b;
            // Renormalize over the two known envs; a neutral 0.5 if all mass fled
            // to Unknown (uninformative for this sink).
            if known < 1e-9 { 0.5 } else { a / known }
        })
        .collect();

    RecoveryScore {
        jsd: sim.jsd(),
        r2: r_squared(&predicted, sim.truth()),
        rmse: rmse(&predicted, sim.truth()),
    }
}

/// Generate a scenario, run [`predict_sinks`], and score recovery — the
/// end-to-end convenience the equivalence suite and the future α-tuner drive.
///
/// # Errors
/// Propagates [`simulate_two_source`] and [`predict_sinks`].
pub fn run_two_source_recovery(
    cfg: &SimConfig,
    params: &GibbsParams,
    seed: u64,
) -> Result<RecoveryScore> {
    let sim = simulate_two_source(cfg, seed)?;
    // Each scenario runs serially; a tuner parallelizes across the scenario grid
    // instead, which avoids nested pools and keeps per-scenario cost predictable.
    let mixing = predict_sinks(sim.table(), sim.context(), params, seed, 1)?;
    Ok(score_recovery(&sim, &mixing))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collapse::CollapseMethod;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "expected {b}, got {a}");
    }

    // (a) rmse hand calc.
    #[test]
    fn rmse_hand_calc() {
        approx(rmse(&[0.0, 1.0], &[0.0, 0.0]), 0.5f64.sqrt(), 1e-12);
        approx(rmse(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0, 1e-12);
    }

    // (b) r_squared: perfect, anti-correlated, constant, and a rational mid.
    #[test]
    fn r_squared_bounds_and_mid() {
        // predicted == truth (non-constant) -> 1.
        approx(r_squared(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 1.0, 1e-12);
        // perfectly anti-correlated -> 1 (squared).
        approx(r_squared(&[0.0, 1.0, 2.0], &[2.0, 1.0, 0.0]), 1.0, 1e-12);
        // constant predicted -> 0 (zero-variance guard).
        approx(r_squared(&[0.5, 0.5, 0.5], &[0.0, 1.0, 2.0]), 0.0, 1e-12);
        // mid hand calc: r^2 = 4.5^2 / (5.0 * 4.75) = 20.25 / 23.75.
        approx(
            r_squared(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0, 1.0, 3.0]),
            20.25 / 23.75,
            1e-12,
        );
    }

    // (c) jensen_shannon_divergence: endpoints exact, symmetry, mid bracket.
    #[test]
    fn jsd_endpoints_symmetry_and_mid() {
        // Identical -> 0.
        approx(
            jensen_shannon_divergence(&[0.5, 0.5], &[0.5, 0.5]),
            0.0,
            1e-12,
        );
        // Disjoint support -> 1 bit.
        approx(
            jensen_shannon_divergence(&[1.0, 0.0], &[0.0, 1.0]),
            1.0,
            1e-12,
        );
        // Symmetric, and a mid value in a tight bracket.
        let p = [0.5, 0.5];
        let q = [0.25, 0.75];
        let d = jensen_shannon_divergence(&p, &q);
        approx(d, jensen_shannon_divergence(&q, &p), 1e-12);
        assert!((0.04..0.06).contains(&d), "mid JSD out of bracket: {d}");
    }

    fn tiny_cfg() -> SimConfig {
        SimConfig {
            n_taxa: 8,
            samples_per_source: 2,
            seqs_per_sample: 100,
            n_trials: 5,
            sink_depth: 100,
            concentration: 0.2,
        }
    }

    // (d) simulate_two_source determinism.
    #[test]
    fn simulate_is_deterministic() {
        let a = simulate_two_source(&tiny_cfg(), 42).unwrap();
        let b = simulate_two_source(&tiny_cfg(), 42).unwrap();
        assert_eq!(a.truth(), b.truth());
        assert_eq!(a.jsd(), b.jsd());
        assert_eq!(a.table(), b.table());
    }

    // (e) simulate_two_source shape and ranges.
    #[test]
    fn simulate_shape_and_ranges() {
        let cfg = tiny_cfg();
        let sim = simulate_two_source(&cfg, 7).unwrap();
        // 2 * samples_per_source sources + n_trials sinks.
        assert_eq!(
            sim.table().n_samples(),
            2 * cfg.samples_per_source + cfg.n_trials
        );
        assert_eq!(
            sim.context().source_indices().len(),
            2 * cfg.samples_per_source
        );
        assert_eq!(sim.context().sink_indices().len(), cfg.n_trials);
        assert_eq!(sim.truth().len(), cfg.n_trials);
        assert!(sim.truth().iter().all(|&m| (0.0..=1.0).contains(&m)));
        assert!((0.0..=1.0).contains(&sim.jsd()));
        // Every sink column is non-empty.
        for &s in sim.context().sink_indices() {
            assert!(sim.table().column_sum(s as usize) > 0);
        }
    }

    // Invalid configs are rejected.
    #[test]
    fn invalid_config_is_an_error() {
        let mut cfg = tiny_cfg();
        cfg.n_taxa = 1;
        assert!(matches!(
            simulate_two_source(&cfg, 1),
            Err(Error::InvalidParam { .. })
        ));
        let mut cfg = tiny_cfg();
        cfg.concentration = 0.0;
        assert!(matches!(
            simulate_two_source(&cfg, 1),
            Err(Error::InvalidParam { .. })
        ));
    }

    // run_two_source_recovery is reproducible and returns sane fields.
    #[test]
    fn recovery_runs_and_is_reproducible() {
        let cfg = tiny_cfg();
        let params = GibbsParams {
            restarts: 8,
            draws_per_restart: 2,
            burnin: 8,
            collapse: CollapseMethod::Sum,
            ..GibbsParams::default()
        };
        let a = run_two_source_recovery(&cfg, &params, 3).unwrap();
        let b = run_two_source_recovery(&cfg, &params, 3).unwrap();
        assert_eq!(a, b);
        assert!((0.0..=1.0).contains(&a.r2));
        assert!(a.rmse >= 0.0);
        assert!((0.0..=1.0).contains(&a.jsd));
    }
}
