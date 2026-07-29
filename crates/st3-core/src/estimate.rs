// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! The pluggable sink-estimation seam and its collapsed-Gibbs implementation.
//!
//! [`SinkModel`] is the estimator seam: `prepare` precomputes the
//! depth-independent, sink-agnostic source model once per run (validating the
//! parameters at the same time), and `estimate` turns one sink into an *ensemble
//! of proportion vectors* — one vector per retained Gibbs draw, each of length
//! `V` (known environments in collapse order, then Unknown last) and summing to
//! one. Collation (mean/std across the ensemble, a later milestone) is
//! sampler-agnostic and never changes when the estimator is swapped.
//!
//! [`GibbsEstimator`] is the collapsed-Gibbs sampler. Its `Prepared` type is the
//! [`ConditionalProbability`] known-source term. `estimate` applies the
//! depth-dependent scaling for the sink, then grows `restarts` independent Markov
//! chains: each sink sequence is randomly assigned an environment, then
//! repeatedly withdrawn and redrawn from the conditional distribution; the
//! environment counts are snapshotted at the thinned draw schedule.
//!
//! The randomness comes from a caller-supplied [`ItemRng`]; the driver seeds it
//! per sink via [`crate::rng::rng_for_item`], so a sink's result depends only on
//! the run seed and the sink, never on iteration order or thread count. The
//! stream is Xoshiro256++, so it does not bit-match the Python reference's NumPy
//! stream; equivalence is statistical (means converge) and the golden test pins
//! *our* stream.

use rand::seq::SliceRandom;
use rand::RngExt;

use crate::collapse::CollapsedSources;
use crate::cp::{fill_jp, ConditionalProbability, JpTerms};
use crate::error::Result;
use crate::params::GibbsParams;
use crate::rng::ItemRng;
use crate::table::{Count, FeatureIdx};

/// A single sink as a sparse count vector on the shared feature axis.
///
/// `rows`/`counts` are the sink's nonzero `(feature, count)` entries (a column
/// of the working table), and `depth` is their total — the number of sequences
/// the sampler assigns.
pub struct SinkVec<'a> {
    rows: &'a [FeatureIdx],
    counts: &'a [Count],
    depth: usize,
}

impl<'a> SinkVec<'a> {
    /// Wrap a sink's nonzero `(feature, count)` slices, computing its depth.
    ///
    /// `rows` and `counts` are parallel and index the same `τ`-taxon axis as the
    /// collapsed sources.
    #[must_use]
    pub fn new(rows: &'a [FeatureIdx], counts: &'a [Count]) -> Self {
        let depth = counts.iter().map(|&c| c as usize).sum();
        Self {
            rows,
            counts,
            depth,
        }
    }

    /// The sink's sequencing depth (sum of its counts).
    pub fn depth(&self) -> usize {
        self.depth
    }
}

/// The result of estimating one sink: an ensemble of proportion vectors and,
/// optionally, the mean source × taxon assignment tally.
///
/// Each ensemble vector has length `V` (known environments in collapse order,
/// then Unknown) and sums to one. The ensemble has one vector per retained draw
/// (`restarts · draws_per_restart`).
#[derive(Debug, Clone, PartialEq)]
pub struct SinkEstimate {
    ensemble: Vec<Vec<f64>>,
    assignments: Option<CooTally>,
}

impl SinkEstimate {
    /// The proportion vectors, one per retained draw.
    pub fn ensemble(&self) -> &[Vec<f64>] {
        &self.ensemble
    }

    /// The mean assignment tally, if it was requested.
    pub fn assignments(&self) -> Option<&CooTally> {
        self.assignments.as_ref()
    }

    /// Assemble an estimate directly from its parts (test-only).
    ///
    /// Lets sibling modules' unit tests build a `SinkEstimate` with a chosen
    /// ensemble without running the sampler.
    #[cfg(test)]
    pub(crate) fn from_parts(ensemble: Vec<Vec<f64>>, assignments: Option<CooTally>) -> Self {
        Self {
            ensemble,
            assignments,
        }
    }
}

/// A dense source × taxon assignment tally, averaged over draws.
///
/// Row index `v` spans all `V` environments **including the trailing Unknown**
/// (`n_sources` = `V`); column index `t` spans the `τ` taxa. `mean_counts` is
/// source-major (`v * n_features + t`) and holds the mean number of sink
/// sequences of taxon `t` attributed to environment `v` per draw, so the whole
/// table sums to the sink depth. It is dense internally but emitted sparsely via
/// [`CooTally::triples`]; the COO/streaming form is finalized in collation.
#[derive(Debug, Clone, PartialEq)]
pub struct CooTally {
    n_sources: usize,
    n_features: usize,
    mean_counts: Vec<f64>,
}

impl CooTally {
    /// Number of environment rows, including the trailing Unknown (`V`).
    pub fn n_sources(&self) -> usize {
        self.n_sources
    }

    /// Number of taxa (`τ`).
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Nonzero `(environment, taxon, mean_count)` triples, row-major.
    pub fn triples(&self) -> impl Iterator<Item = (u32, u32, f64)> + '_ {
        let tau = self.n_features;
        self.mean_counts
            .iter()
            .enumerate()
            .filter(|&(_, &m)| m != 0.0)
            .map(move |(i, &m)| ((i / tau) as u32, (i % tau) as u32, m))
    }
}

/// The estimator seam: precompute a source model once, then estimate each sink.
///
/// Deliberately split from the design sketch so that validation happens exactly
/// once, in `prepare` (which returns [`Result`]), leaving `estimate` infallible
/// and cheap to call per sink. The RNG is the concrete [`ItemRng`] rather than a
/// generic, so the hot loop is monomorphized with no dynamic dispatch.
pub trait SinkModel {
    /// The precomputed, sink-independent model shared across sinks. `Sync` so a
    /// future parallel driver can share one instance read-only across threads.
    type Prepared: Sync;

    /// Validate `p` and precompute the sink-independent model from `sources`.
    ///
    /// # Errors
    /// Propagates [`GibbsParams::validate`].
    fn prepare(sources: &CollapsedSources, p: &GibbsParams) -> Result<Self::Prepared>;

    /// Estimate one `sink`, drawing all randomness from `rng`.
    ///
    /// Infallible: `p` was validated in `prepare`, and a `D >= 1` sink is a
    /// documented precondition (the driver filters empty sinks). When
    /// `want_assign` is set, the returned estimate carries the mean assignment
    /// tally; otherwise it is `None` and no per-sequence tally is accumulated.
    fn estimate(
        prep: &Self::Prepared,
        sink: &SinkVec<'_>,
        p: &GibbsParams,
        rng: &mut ItemRng,
        want_assign: bool,
    ) -> SinkEstimate;
}

/// The collapsed-Gibbs source-attribution estimator.
pub struct GibbsEstimator;

impl SinkModel for GibbsEstimator {
    type Prepared = ConditionalProbability;

    fn prepare(sources: &CollapsedSources, p: &GibbsParams) -> Result<Self::Prepared> {
        p.validate()?;
        Ok(ConditionalProbability::precompute(sources, p.alpha1))
    }

    fn estimate(
        prep: &Self::Prepared,
        sink: &SinkVec<'_>,
        p: &GibbsParams,
        rng: &mut ItemRng,
        want_assign: bool,
    ) -> SinkEstimate {
        let n_known = prep.n_known();
        let v = prep.v();
        let tau = prep.n_features();
        let depth = sink.depth();
        debug_assert!(depth >= 1, "estimate requires a non-empty sink (D >= 1)");
        debug_assert!(
            sink.rows.iter().all(|&r| (r as usize) < tau),
            "sink feature index out of range of the source taxon axis"
        );

        // Depth-dependent quantities (Eqs. 3–5). The depth scaling of α2 is
        // intentional (see the algorithm investigation), not a defect.
        let d = depth as f64;
        let terms = JpTerms {
            beta: p.beta,
            alpha2_n: p.alpha2 * d,
            alpha2_n_tau: p.alpha2 * d * tau as f64,
            denominator_p_v: d - 1.0 + p.beta * v as f64,
        };

        // Per-sink scaled known-source term; never mutate the shared table.
        let inv_denom = 1.0 / terms.denominator_p_v;
        let known_source_cp: Vec<f64> = prep.known_p_tv().iter().map(|&x| x * inv_denom).collect();

        // Bookkeeping vector: feature f repeated sink_count[f] times, length D.
        let taxon_sequence: Vec<usize> = sink
            .rows
            .iter()
            .zip(sink.counts.iter())
            .flat_map(|(&f, &c)| std::iter::repeat_n(f as usize, c as usize))
            .collect();
        debug_assert_eq!(taxon_sequence.len(), depth);

        let total_draws = p.restarts as usize * p.draws_per_restart as usize;
        let total_passes = p.burnin + (p.draws_per_restart - 1) * p.delay + 1;
        let unknown_idx = v - 1;

        // Scratch allocated once, reused across restarts.
        let mut seq_env: Vec<usize> = vec![0; depth];
        let mut envcounts: Vec<u32> = vec![0; v];
        let mut unknown_vector: Vec<u32> = vec![0; tau];
        let mut order: Vec<usize> = (0..depth).collect();
        let mut jp: Vec<f64> = vec![0.0; v];

        let mut ensemble: Vec<Vec<f64>> = Vec::with_capacity(total_draws);
        let mut tally: Option<Vec<f64>> = want_assign.then(|| vec![0.0; v * tau]);
        let inv_d = 1.0 / d;

        for _restart in 0..p.restarts {
            // Uniform-random initial environment for every sequence.
            envcounts.fill(0);
            unknown_vector.fill(0);
            let mut unknown_sum: u32 = 0;
            for (slot, &t) in seq_env.iter_mut().zip(taxon_sequence.iter()) {
                let e = rng.random_range(0..v);
                *slot = e;
                envcounts[e] += 1;
                if e == unknown_idx {
                    unknown_vector[t] += 1;
                    unknown_sum += 1;
                }
            }

            for rep in 1..=total_passes {
                // Random visitation order so no sequence is systematically
                // updated against a more-converged model than its peers.
                order.shuffle(rng);
                for &seq_index in order.iter() {
                    let e = seq_env[seq_index];
                    let t = taxon_sequence[seq_index];

                    // Withdraw this sequence, leaving the leave-one-out counts.
                    envcounts[e] -= 1;
                    if e == unknown_idx {
                        unknown_vector[t] -= 1;
                        unknown_sum -= 1;
                    }

                    let cp_row = &known_source_cp[t * n_known..t * n_known + n_known];
                    let total = fill_jp(
                        cp_row,
                        &envcounts,
                        f64::from(unknown_vector[t]),
                        f64::from(unknown_sum),
                        &terms,
                        &mut jp,
                    );

                    // Reassign by inverse-CDF draw, then re-add.
                    let new_e = draw_env(&jp, total, rng);
                    seq_env[seq_index] = new_e;
                    envcounts[new_e] += 1;
                    if new_e == unknown_idx {
                        unknown_vector[t] += 1;
                        unknown_sum += 1;
                    }
                }

                if rep > p.burnin && (rep - (p.burnin + 1)) % p.delay == 0 {
                    ensemble.push(envcounts.iter().map(|&n| f64::from(n) * inv_d).collect());
                    if let Some(tally) = tally.as_mut() {
                        for (&t, &e) in taxon_sequence.iter().zip(seq_env.iter()) {
                            tally[e * tau + t] += 1.0;
                        }
                    }
                }
            }
        }

        let assignments = tally.map(|mut counts| {
            let inv = 1.0 / total_draws as f64;
            for c in counts.iter_mut() {
                *c *= inv;
            }
            CooTally {
                n_sources: v,
                n_features: tau,
                mean_counts: counts,
            }
        });

        SinkEstimate {
            ensemble,
            assignments,
        }
    }
}

/// Draw an environment index from unnormalized weights `jp` (summing to `total`)
/// by inverse CDF: pick `x` uniformly in `[0, total)` and return the first index
/// whose running cumulative sum reaches `x` (equivalent to `searchsorted`,
/// `side='left'`).
fn draw_env(jp: &[f64], total: f64, rng: &mut ItemRng) -> usize {
    let x = rng.random::<f64>() * total;
    let mut acc = 0.0;
    for (i, &w) in jp.iter().enumerate() {
        acc += w;
        if acc >= x {
            return i;
        }
    }
    // Floating-point slack can leave `acc` a hair below `x`; the last bin (the
    // Unknown environment) absorbs it.
    jp.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collapse::{collapse_subset, CollapseMethod};
    use crate::rng::rng_for_item;
    use crate::table::CountTable;

    /// Build [`CollapsedSources`] with one column per environment from dense
    /// per-environment count vectors (each `cols[e]` has length `τ`).
    fn sources_from_envs(cols: &[&[u32]]) -> CollapsedSources {
        let tau = cols[0].len();
        let mut rows = Vec::new();
        let mut sample_cols = Vec::new();
        let mut vals = Vec::new();
        for (e, col) in cols.iter().enumerate() {
            assert_eq!(col.len(), tau, "ragged env columns");
            for (t, &val) in col.iter().enumerate() {
                if val != 0 {
                    rows.push(t as u32);
                    sample_cols.push(e as u32);
                    vals.push(f64::from(val));
                }
            }
        }
        let feature_ids = (0..tau).map(|i| format!("f{i}")).collect();
        let sample_ids: Vec<String> = (0..cols.len()).map(|i| format!("e{i}")).collect();
        let table = CountTable::from_coo(feature_ids, sample_ids, &rows, &sample_cols, &vals)
            .expect("source table builds");
        let indices: Vec<u32> = (0..cols.len() as u32).collect();
        let env_owned: Vec<String> = (0..cols.len()).map(|i| format!("e{i}")).collect();
        let envs: Vec<&str> = env_owned.iter().map(String::as_str).collect();
        collapse_subset(&table, &indices, &envs, CollapseMethod::Sum).expect("collapse")
    }

    /// A dense sink `(rows, counts)` for [`SinkVec::new`] from a dense vector.
    fn sink_parts(dense: &[u32]) -> (Vec<FeatureIdx>, Vec<Count>) {
        let mut rows = Vec::new();
        let mut counts = Vec::new();
        for (t, &c) in dense.iter().enumerate() {
            if c != 0 {
                rows.push(t as u32);
                counts.push(c);
            }
        }
        (rows, counts)
    }

    fn params(restarts: u32, draws: u32, burnin: u32, delay: u32) -> GibbsParams {
        GibbsParams {
            alpha1: 0.5,
            alpha2: 0.5,
            beta: 1.0,
            restarts,
            draws_per_restart: draws,
            burnin,
            delay,
            collapse: CollapseMethod::Sum,
            contingency: false,
        }
    }

    #[test]
    fn ensemble_vectors_are_valid_proportions() {
        let sources = sources_from_envs(&[&[3, 1], &[1, 3]]);
        let p = params(4, 3, 5, 2);
        let prep = GibbsEstimator::prepare(&sources, &p).unwrap();
        let (rows, counts) = sink_parts(&[2, 1]);
        let sink = SinkVec::new(&rows, &counts);
        let mut rng = rng_for_item(7, 0);
        let est = GibbsEstimator::estimate(&prep, &sink, &p, &mut rng, false);

        assert_eq!(est.ensemble().len(), 12); // restarts · draws
        for vec in est.ensemble() {
            assert_eq!(vec.len(), 3); // V
            let sum: f64 = vec.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "draw sums to {sum}");
            assert!(vec.iter().all(|&x| x >= 0.0), "negative proportion");
        }
    }

    #[test]
    fn same_seed_same_ensemble_different_seed_differs() {
        let sources = sources_from_envs(&[&[3, 1], &[1, 3]]);
        let p = params(3, 2, 4, 1);
        let prep = GibbsEstimator::prepare(&sources, &p).unwrap();
        let (rows, counts) = sink_parts(&[2, 2]);
        let sink = SinkVec::new(&rows, &counts);

        let a = GibbsEstimator::estimate(&prep, &sink, &p, &mut rng_for_item(42, 0), false);
        let b = GibbsEstimator::estimate(&prep, &sink, &p, &mut rng_for_item(42, 0), false);
        assert_eq!(a.ensemble(), b.ensemble(), "same seed must be identical");

        let c = GibbsEstimator::estimate(&prep, &sink, &p, &mut rng_for_item(43, 0), false);
        assert_ne!(a.ensemble(), c.ensemble(), "different seed should differ");
    }

    #[test]
    fn golden_single_pass_replay() {
        // A pinned snapshot of OUR sampler (Xoshiro256++) — not the NumPy
        // reference. restarts=1, draws=1, burnin=1, delay=1 ⇒ 2 passes, 1 draw.
        let sources = sources_from_envs(&[&[3, 1], &[1, 3]]);
        let p = params(1, 1, 1, 1);
        let prep = GibbsEstimator::prepare(&sources, &p).unwrap();
        let (rows, counts) = sink_parts(&[2, 1]); // D = 3
        let sink = SinkVec::new(&rows, &counts);
        let mut rng = rng_for_item(42, 0);
        let est = GibbsEstimator::estimate(&prep, &sink, &p, &mut rng, false);

        assert_eq!(est.ensemble().len(), 1);
        let draw = &est.ensemble()[0];
        let expected = [GOLDEN_0, GOLDEN_1, GOLDEN_2];
        for (got, exp) in draw.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-12, "golden drift: {draw:?}");
        }
    }

    #[test]
    fn two_disjoint_sources_separate() {
        // env0 owns taxa {0,1}; env1 owns taxa {2,3}. A sink of only env0's taxa
        // must attribute overwhelmingly to env0. The sink is deep enough that the
        // observed counts dominate the β prior that otherwise props up Unknown.
        let sources = sources_from_envs(&[&[100, 100, 0, 0], &[0, 0, 100, 100]]);
        let mut p = params(100, 1, 100, 1);
        p.alpha1 = 0.001;
        p.alpha2 = 0.1;
        p.beta = 10.0;
        let prep = GibbsEstimator::prepare(&sources, &p).unwrap();
        let (rows, counts) = sink_parts(&[100, 100, 0, 0]);
        let sink = SinkVec::new(&rows, &counts);
        let mut rng = rng_for_item(2024, 0);
        let est = GibbsEstimator::estimate(&prep, &sink, &p, &mut rng, false);

        let mean = mean_ensemble(&est, 3);
        assert!(mean[0] > 0.8, "env0 mean = {}", mean[0]);
        assert!(mean[1] < 0.05, "env1 mean = {}", mean[1]);
    }

    #[test]
    fn novel_taxon_goes_to_unknown() {
        // Sources cover taxa {0,1}; the sink is entirely taxon 2, which no source
        // has, so it must land in Unknown (the last environment).
        let sources = sources_from_envs(&[&[100, 100, 0], &[100, 100, 0]]);
        let mut p = params(100, 1, 100, 1);
        p.alpha1 = 0.001;
        p.alpha2 = 0.1;
        p.beta = 10.0;
        let prep = GibbsEstimator::prepare(&sources, &p).unwrap();
        let (rows, counts) = sink_parts(&[0, 0, 100]);
        let sink = SinkVec::new(&rows, &counts);
        let mut rng = rng_for_item(99, 0);
        let est = GibbsEstimator::estimate(&prep, &sink, &p, &mut rng, false);

        let mean = mean_ensemble(&est, 3);
        assert!(mean[2] > 0.8, "Unknown mean = {}", mean[2]);
    }

    #[test]
    fn single_source_v_equals_two() {
        let sources = sources_from_envs(&[&[10, 5]]);
        let p = params(5, 2, 3, 1);
        let prep = GibbsEstimator::prepare(&sources, &p).unwrap();
        assert_eq!(prep.v(), 2);
        let (rows, counts) = sink_parts(&[3, 2]);
        let sink = SinkVec::new(&rows, &counts);
        let mut rng = rng_for_item(1, 0);
        let est = GibbsEstimator::estimate(&prep, &sink, &p, &mut rng, false);
        for vec in est.ensemble() {
            assert_eq!(vec.len(), 2);
            assert!((vec.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn assignment_tally_shape_and_mass() {
        let sources = sources_from_envs(&[&[3, 1], &[1, 3]]);
        let p = params(4, 2, 5, 1);
        let prep = GibbsEstimator::prepare(&sources, &p).unwrap();
        let (rows, counts) = sink_parts(&[2, 3]); // D = 5
        let sink = SinkVec::new(&rows, &counts);

        // want_assign = false ⇒ no tally.
        let none = GibbsEstimator::estimate(&prep, &sink, &p, &mut rng_for_item(5, 0), false);
        assert!(none.assignments().is_none());

        // want_assign = true ⇒ V×τ tally whose mass is the sink depth.
        let some = GibbsEstimator::estimate(&prep, &sink, &p, &mut rng_for_item(5, 0), true);
        let tally = some.assignments().expect("tally present");
        assert_eq!(tally.n_sources(), 3); // V
        assert_eq!(tally.n_features(), 2); // τ
        let mass: f64 = tally.mean_counts.iter().sum();
        assert!((mass - 5.0).abs() < 1e-9, "tally mass = {mass}");
        // triples() enumerates the nonzero cells with the same total mass.
        let triple_mass: f64 = tally.triples().map(|(_, _, m)| m).sum();
        assert!((triple_mass - mass).abs() < 1e-12);
    }

    fn mean_ensemble(est: &SinkEstimate, v: usize) -> Vec<f64> {
        let mut acc = vec![0.0; v];
        for vec in est.ensemble() {
            for (a, &x) in acc.iter_mut().zip(vec.iter()) {
                *a += x;
            }
        }
        let n = est.ensemble().len() as f64;
        acc.iter().map(|&s| s / n).collect()
    }

    // Golden snapshot of our sampler; captured on first run (see the test).
    // envcounts = [2, 0, 1] over depth 3 ⇒ [2/3, 0, 1/3].
    const GOLDEN_0: f64 = 2.0 / 3.0;
    const GOLDEN_1: f64 = 0.0;
    const GOLDEN_2: f64 = 1.0 / 3.0;
}
