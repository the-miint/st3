// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Collation of per-sink estimator output into the run's mixing results.
//!
//! Collation is sampler-agnostic: it consumes the ensemble of proportion vectors
//! each [`SinkEstimate`] carries (for collapsed-Gibbs, one vector per retained
//! draw; for a future point estimator, a size-1 ensemble) and reduces it to the
//! per-sink mixing **mean** and **standard deviation** (Eq. 7), plus the optional
//! source × taxon assignment tally. It never changes when the estimator is
//! swapped.
//!
//! The standard deviation is the **per-draw** sd — the spread of the ensemble's
//! `n_v / D` proportion vectors, `sd_v = sqrt(mean_d (π_v^(d) − mean_v)²)`
//! (population, `ddof = 0`). This is deliberately *not* the Python reference's
//! `proportions_std`, which divides each count by the grand total `N_draws · D`
//! rather than the per-draw depth `D` and so understates the true sd by a factor
//! of `N_draws`. The mean is unaffected by that bug and matches both references.

use crate::estimate::{CooTally, SinkEstimate};

/// Per-column mean and population (`ddof = 0`) standard deviation over an
/// ensemble of `v`-length proportion vectors.
///
/// The mean is `(1/N) Σ_d draw[d]`; the sd is `sqrt((1/N) Σ_d (draw[d] − mean)²)`.
/// A single-draw ensemble yields an all-zero sd (no observable spread) rather
/// than a NaN.
///
/// # Panics
/// Panics (in debug) if the ensemble is empty or any draw is not length `v`.
pub(crate) fn mean_std_over_ensemble(ensemble: &[Vec<f64>], v: usize) -> (Vec<f64>, Vec<f64>) {
    let n = ensemble.len();
    debug_assert!(n >= 1, "ensemble must be non-empty");
    let inv_n = 1.0 / n as f64;

    let mut mean = vec![0.0f64; v];
    for draw in ensemble {
        debug_assert_eq!(draw.len(), v, "ensemble vector length mismatch");
        for (m, &x) in mean.iter_mut().zip(draw.iter()) {
            *m += x;
        }
    }
    for m in mean.iter_mut() {
        *m *= inv_n;
    }

    let mut var = vec![0.0f64; v];
    for draw in ensemble {
        for ((s, &x), &m) in var.iter_mut().zip(draw.iter()).zip(mean.iter()) {
            let d = x - m;
            *s += d * d;
        }
    }
    let std = var.iter().map(|&s| (s * inv_n).sqrt()).collect();
    (mean, std)
}

/// The run's mixing results: per-sink source proportions, their standard
/// deviations, and optionally the per-sink source × taxon assignment tables.
///
/// `means` and `stds` are dense and **sink-major flat** (length `n_sinks · V`,
/// row `i` at `i * V`), where `V = env_names.len()`; `env_names` are the source
/// environments in collapse order followed by `Unknown` (always last). When
/// present, `contingency[i]` is the [`CooTally`] for sink `i`, aligned to
/// `sink_ids`.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceMixing {
    sink_ids: Vec<String>,
    env_names: Vec<String>,
    means: Vec<f64>,
    stds: Vec<f64>,
    contingency: Option<Vec<CooTally>>,
}

impl SourceMixing {
    /// Sink identifiers, in row order.
    pub fn sink_ids(&self) -> &[String] {
        &self.sink_ids
    }

    /// Environment (column) names: source envs in collapse order, then `Unknown`.
    pub fn env_names(&self) -> &[String] {
        &self.env_names
    }

    /// Number of sinks (rows).
    pub fn n_sinks(&self) -> usize {
        self.sink_ids.len()
    }

    /// Number of environments including `Unknown` (`V`, the column count).
    pub fn n_envs(&self) -> usize {
        self.env_names.len()
    }

    /// The full sink-major mean matrix (`n_sinks · V`, row `i` at `i * V`).
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// The full sink-major std matrix (`n_sinks · V`, row `i` at `i * V`).
    pub fn stds(&self) -> &[f64] {
        &self.stds
    }

    /// Mean proportions for sink `i` (length `V`).
    ///
    /// # Panics
    /// Panics if `i >= self.n_sinks()`.
    pub fn mean_row(&self, i: usize) -> &[f64] {
        let v = self.n_envs();
        &self.means[i * v..i * v + v]
    }

    /// Standard deviations for sink `i` (length `V`).
    ///
    /// # Panics
    /// Panics if `i >= self.n_sinks()`.
    pub fn std_row(&self, i: usize) -> &[f64] {
        let v = self.n_envs();
        &self.stds[i * v..i * v + v]
    }

    /// Per-sink assignment tallies (aligned to [`Self::sink_ids`]), if requested.
    pub fn contingency(&self) -> Option<&[CooTally]> {
        self.contingency.as_deref()
    }

    /// Assemble a [`SourceMixing`] from already-collated parts.
    ///
    /// [`collate`] is the usual builder, but the leave-one-out driver
    /// ([`crate::predict_loo`]) constructs each row itself — scattering a
    /// per-fold reduced-frame result into the fixed full frame — and then hands
    /// the finished matrices here. `means` and `stds` must be dense and
    /// **sink-major flat**, length `sink_ids.len() * env_names.len()` (row `i` at
    /// `i * V`, where `V = env_names.len()`); `env_names` are the source
    /// environments in collapse order then `Unknown`. `contingency`, when `Some`,
    /// is aligned to `sink_ids`.
    ///
    /// # Panics
    /// Panics (in debug) if `means` or `stds` is not length
    /// `sink_ids.len() * env_names.len()`.
    pub(crate) fn from_parts(
        sink_ids: Vec<String>,
        env_names: Vec<String>,
        means: Vec<f64>,
        stds: Vec<f64>,
        contingency: Option<Vec<CooTally>>,
    ) -> SourceMixing {
        let expected = sink_ids.len() * env_names.len();
        debug_assert_eq!(means.len(), expected, "means must be dense sink-major");
        debug_assert_eq!(stds.len(), expected, "stds must be dense sink-major");
        SourceMixing {
            sink_ids,
            env_names,
            means,
            stds,
            contingency,
        }
    }
}

/// Collate per-sink estimates into a [`SourceMixing`].
///
/// `estimates`, `sink_ids` are parallel (sink `i`'s estimate and id). `env_names`
/// are the `V` column labels (source envs then `Unknown`). Each estimate's
/// ensemble vectors must have length `V`. The contingency is gathered
/// all-or-nothing: `Some` only when every estimate carries an assignment tally
/// (i.e. the run requested it), else `None`.
///
/// # Panics
/// Panics (in debug) if `estimates` and `sink_ids` differ in length, or an
/// ensemble vector is not length `env_names.len()`.
pub fn collate(
    estimates: &[SinkEstimate],
    sink_ids: Vec<String>,
    env_names: Vec<String>,
) -> SourceMixing {
    debug_assert_eq!(
        estimates.len(),
        sink_ids.len(),
        "estimates and sink_ids must be parallel"
    );
    let v = env_names.len();

    let mut means = Vec::with_capacity(estimates.len() * v);
    let mut stds = Vec::with_capacity(estimates.len() * v);
    for est in estimates {
        let (mean, std) = mean_std_over_ensemble(est.ensemble(), v);
        means.extend_from_slice(&mean);
        stds.extend_from_slice(&std);
    }

    // `Option<Vec<_>>: FromIterator<Option<_>>` yields `Some` only if every
    // element is `Some` — exactly the all-or-nothing contingency gather.
    let contingency = if estimates.is_empty() {
        None
    } else {
        estimates.iter().map(|e| e.assignments().cloned()).collect()
    };

    SourceMixing {
        sink_ids,
        env_names,
        means,
        stds,
        contingency,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "expected {b}, got {a}");
    }

    #[test]
    fn mean_std_hand_calc() {
        // Two draws over V=2. mean = [0.75, 0.25].
        // pop var col0 = ((.5-.75)^2 + (1-.75)^2)/2 = (.0625+.0625)/2 = .0625 -> sd .25.
        let ensemble = vec![vec![0.5, 0.5], vec![1.0, 0.0]];
        let (mean, std) = mean_std_over_ensemble(&ensemble, 2);
        approx(mean[0], 0.75);
        approx(mean[1], 0.25);
        approx(std[0], 0.25);
        approx(std[1], 0.25);
    }

    #[test]
    fn std_is_per_draw_not_shrunk_by_num_draws() {
        // Guard against reintroducing the Python x N_draws std bug (design §16):
        // the kernel's sd must equal an independent brute-force per-draw sd
        // (bit-exact), and must NOT equal that value divided by N_draws.
        let ensemble = vec![
            vec![0.6, 0.4],
            vec![0.2, 0.8],
            vec![0.5, 0.5],
            vec![0.9, 0.1],
        ];
        let v = 2;
        let n = ensemble.len() as f64;
        let (mean, std) = mean_std_over_ensemble(&ensemble, v);

        // Independent brute-force population sd.
        let mut brute = vec![0.0f64; v];
        for k in 0..v {
            let mu: f64 = ensemble.iter().map(|d| d[k]).sum::<f64>() / n;
            let var: f64 = ensemble.iter().map(|d| (d[k] - mu).powi(2)).sum::<f64>() / n;
            brute[k] = var.sqrt();
        }
        for k in 0..v {
            assert_eq!(std[k], brute[k], "sd must match brute force exactly");
            // The buggy value would be brute[k] / n; ensure we are not that.
            assert!(
                (std[k] - brute[k] / n).abs() > 1e-9,
                "sd looks shrunk by N_draws (the Python bug)"
            );
        }
        approx(mean[0], (0.6 + 0.2 + 0.5 + 0.9) / 4.0);
    }

    #[test]
    fn single_draw_has_zero_std() {
        let (mean, std) = mean_std_over_ensemble(&[vec![0.3, 0.7]], 2);
        approx(mean[0], 0.3);
        approx(std[0], 0.0);
        approx(std[1], 0.0);
    }

    #[test]
    fn collate_assembles_matrix_without_contingency() {
        // Two sinks, V=2, one draw each -> means equal the draws, stds zero.
        let estimates = vec![
            SinkEstimate::from_parts(vec![vec![0.8, 0.2]], None),
            SinkEstimate::from_parts(vec![vec![0.1, 0.9], vec![0.3, 0.7]], None),
        ];
        let sm = collate(
            &estimates,
            vec!["s0".into(), "s1".into()],
            vec!["envA".into(), "Unknown".into()],
        );
        assert_eq!(sm.n_sinks(), 2);
        assert_eq!(sm.n_envs(), 2);
        assert_eq!(sm.means().len(), 4);
        assert_eq!(sm.stds().len(), 4);
        assert_eq!(sm.sink_ids(), &["s0".to_string(), "s1".to_string()]);
        assert_eq!(sm.env_names(), &["envA".to_string(), "Unknown".to_string()]);
        // Row slicing.
        approx(sm.mean_row(0)[0], 0.8);
        approx(sm.mean_row(1)[1], 0.8); // (0.9 + 0.7) / 2
        approx(sm.std_row(0)[0], 0.0); // single draw
        approx(sm.std_row(1)[0], 0.1); // sd of [0.1, 0.3] = 0.1
        // No tallies requested.
        assert!(sm.contingency().is_none());
    }
}
