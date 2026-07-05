// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! The collapsed-Gibbs conditional-probability precompute and its inner kernel.
//!
//! [`ConditionalProbability`] holds the part of the source-attribution model that
//! does not depend on a sink's depth: the known-source taxon fit
//! `known_p_tv[v][t] = (m_{t,v} + α1) / (m_·v + τ·α1)` (Eq. 2), one value per
//! (known environment `v`, taxon `t`). It is computed once per run and shared,
//! read-only, across every sink.
//!
//! The table is stored **taxon-major** — all `V-1` known-environment values for a
//! taxon are contiguous — because the sampler's hot loop, for each withdrawn
//! sequence, reads exactly the row for that sequence's taxon across all
//! environments. Taxon-major keeps that read a single cache-friendly slice.
//!
//! The depth-dependent scaling (`known_source_cp = known_p_tv / denominator_p_v`,
//! the depth-scaled `α2` terms) is *not* done here; it belongs to the estimator,
//! which knows a sink's depth. `fill_jp` is the per-sequence kernel that
//! combines the scaled known-source row with the current Unknown-environment
//! state to produce the unnormalized draw weights.

use crate::collapse::CollapsedSources;

/// Depth-independent conditional-probability model precomputed from the
/// collapsed sources.
///
/// See the module documentation for the storage layout and what is deliberately
/// left to the estimator.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionalProbability {
    /// Known-source taxon fit, taxon-major: length `n_features * n_known`,
    /// indexed `t * n_known + v`. Row `t` spans the `n_known` known environments.
    known_p_tv: Vec<f64>,
    /// Number of known source environments (`V - 1`).
    n_known: usize,
    /// Number of features (`τ`).
    n_features: usize,
}

impl ConditionalProbability {
    /// Precompute the depth-independent known-source term (Eq. 2) from collapsed
    /// sources under prior `alpha1`.
    ///
    /// `sources` supplies the `V-1` known environments (one collapsed column
    /// each) on a shared feature axis of `τ` taxa; the Unknown environment is
    /// materialized later by the sampler and is not stored here.
    #[must_use]
    pub fn precompute(sources: &CollapsedSources, alpha1: f64) -> Self {
        let n_known = sources.n_sources();
        let n_features = sources.feature_ids().len();
        let mut known_p_tv = vec![0.0f64; n_features * n_known];

        for v in 0..n_known {
            let (rows, counts) = sources.column(v);
            let m_dot_v: u64 = counts.iter().map(|&c| u64::from(c)).sum();
            let denom = m_dot_v as f64 + n_features as f64 * alpha1;
            // Absent taxa (count 0) all share the same smoothed value; write it
            // to every taxon's slot for this environment first (stride n_known),
            // then overwrite the taxa that are actually present.
            let base = alpha1 / denom;
            for slot in known_p_tv[v..].iter_mut().step_by(n_known) {
                *slot = base;
            }
            for (&r, &c) in rows.iter().zip(counts.iter()) {
                known_p_tv[r as usize * n_known + v] = (f64::from(c) + alpha1) / denom;
            }
        }

        Self {
            known_p_tv,
            n_known,
            n_features,
        }
    }

    /// Number of known source environments (`V - 1`).
    pub fn n_known(&self) -> usize {
        self.n_known
    }

    /// Total number of environments including Unknown (`V`).
    pub fn v(&self) -> usize {
        self.n_known + 1
    }

    /// Number of features (`τ`).
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// The known-source row for taxon `t`: its fit across all `n_known`
    /// environments, in environment order.
    ///
    /// # Panics
    /// Panics if `t >= self.n_features()`.
    pub fn known_p_tv_row(&self, t: usize) -> &[f64] {
        let start = t * self.n_known;
        &self.known_p_tv[start..start + self.n_known]
    }

    /// The whole taxon-major known-source table (`t * n_known + v`).
    pub(crate) fn known_p_tv(&self) -> &[f64] {
        &self.known_p_tv
    }
}

/// Depth-scaled model constants that stay fixed across every `fill_jp` call for
/// a single sink.
///
/// The estimator computes these once from the sink depth `D`, then passes them by
/// reference into the hot loop. `beta` is the test-sequence prior; `alpha2_n` is
/// `α2·D`; `alpha2_n_tau` is `α2·D·τ`; `denominator_p_v` is `D − 1 + β·V`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct JpTerms {
    pub beta: f64,
    pub alpha2_n: f64,
    pub alpha2_n_tau: f64,
    pub denominator_p_v: f64,
}

/// Fill the unnormalized per-sequence draw weights `jp` and return their sum.
///
/// This is the sampler's innermost computation, run once per sequence per pass.
/// `cp_row` is the depth-scaled known-source row for the withdrawn sequence's
/// taxon (`known_p_tv_row(t) / denominator_p_v`, length `V-1`). `n_vnoti` are the
/// leave-one-out environment counts after withdrawal (length `V`, Unknown last).
/// The Unknown-environment inputs `m_xiv` (count of this taxon currently in
/// Unknown) and `m_v` (total sequences in Unknown) are post-withdrawal; `terms`
/// carries the depth-scaled constants.
///
/// For each known environment `v`: `jp[v] = cp_row[v] · (n_vnoti[v] + β)`. For the
/// Unknown environment (last): `jp[V-1] = ((m_xiv + α2·D)·(n_vnoti[V-1] + β)) /
/// ((m_v + α2·D·τ)·denominator_p_v)`. The weights are not normalized; the caller
/// draws by inverse CDF over the returned sum.
///
/// # Panics
/// Panics (in debug) if `jp` is not one longer than `cp_row`.
pub(crate) fn fill_jp(
    cp_row: &[f64],
    n_vnoti: &[u32],
    m_xiv: f64,
    m_v: f64,
    terms: &JpTerms,
    jp: &mut [f64],
) -> f64 {
    let n_known = cp_row.len();
    debug_assert_eq!(jp.len(), n_known + 1, "jp must be one longer than cp_row");

    let mut total = 0.0;
    for ((slot, &cp), &n) in jp.iter_mut().zip(cp_row.iter()).zip(n_vnoti.iter()) {
        let w = cp * (f64::from(n) + terms.beta);
        *slot = w;
        total += w;
    }

    let unknown = ((m_xiv + terms.alpha2_n) * (f64::from(n_vnoti[n_known]) + terms.beta))
        / ((m_v + terms.alpha2_n_tau) * terms.denominator_p_v);
    jp[n_known] = unknown;
    total + unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collapse::{CollapseMethod, collapse_subset};
    use crate::table::CountTable;

    /// Build [`CollapsedSources`] with one column per environment straight from
    /// dense per-environment count vectors (each `cols[e]` has length `τ`).
    ///
    /// One source sample per environment collapsed by `Sum` is the identity, so
    /// the collapsed columns are exactly the supplied vectors, in `e0, e1, …`
    /// order (already lexicographic).
    fn sources_from_envs(cols: &[&[u32]]) -> CollapsedSources {
        let tau = cols[0].len();
        let mut rows = Vec::new();
        let mut sample_cols = Vec::new();
        let mut vals = Vec::new();
        for (e, col) in cols.iter().enumerate() {
            assert_eq!(col.len(), tau, "ragged env columns");
            for (t, &v) in col.iter().enumerate() {
                if v != 0 {
                    rows.push(t as u32);
                    sample_cols.push(e as u32);
                    vals.push(f64::from(v));
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

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "expected {b}, got {a}");
    }

    #[test]
    fn known_p_tv_matches_hand_calc() {
        // env0 = {t0:8, t1:2} (m_·0 = 10), env1 = {t0:1, t1:3} (m_·1 = 4), α1 = 1.
        let sources = sources_from_envs(&[&[8, 2], &[1, 3]]);
        let cp = ConditionalProbability::precompute(&sources, 1.0);
        assert_eq!(cp.n_known(), 2);
        assert_eq!(cp.v(), 3);
        assert_eq!(cp.n_features(), 2);

        // Row t0 across envs: (8+1)/(10+2)=0.75, (1+1)/(4+2)=1/3.
        let row0 = cp.known_p_tv_row(0);
        approx(row0[0], 0.75);
        approx(row0[1], 1.0 / 3.0);
        // Row t1 across envs: (2+1)/12=0.25, (3+1)/6=2/3.
        let row1 = cp.known_p_tv_row(1);
        approx(row1[0], 0.25);
        approx(row1[1], 2.0 / 3.0);
    }

    #[test]
    fn fill_jp_matches_hand_calc() {
        // Continue the hand calc: D = 10, β = 1, V = 3 ⇒ denominator_p_v = 12;
        // α2 = 1 ⇒ alpha2_n = 10, alpha2_n_tau = 20. Scale row t0 by 1/12.
        let sources = sources_from_envs(&[&[8, 2], &[1, 3]]);
        let cp = ConditionalProbability::precompute(&sources, 1.0);
        let denominator_p_v = 12.0;
        let cp_row_t0: Vec<f64> = cp
            .known_p_tv_row(0)
            .iter()
            .map(|&x| x / denominator_p_v)
            .collect();
        approx(cp_row_t0[0], 0.0625); // 0.75/12
        approx(cp_row_t0[1], 1.0 / 36.0); // (1/3)/12

        let terms = JpTerms {
            beta: 1.0,
            alpha2_n: 10.0,
            alpha2_n_tau: 20.0,
            denominator_p_v,
        };
        let mut jp = [0.0f64; 3];
        let total = fill_jp(
            &cp_row_t0,
            &[3, 4, 2],
            2.0, /* m_xiv */
            5.0, /* m_v */
            &terms,
            &mut jp,
        );
        // jp[0] = 0.0625·4 = 0.25; jp[1] = (1/36)·5 = 5/36;
        // jp[2] = ((2+10)·(2+1)) / ((5+20)·12) = 36/300 = 0.12.
        approx(jp[0], 0.25);
        approx(jp[1], 5.0 / 36.0);
        approx(jp[2], 0.12);
        // total = 0.25 + 5/36 + 0.12 = 229/450.
        approx(total, 229.0 / 450.0);
    }

    #[test]
    fn absent_taxon_uses_smoothed_prior() {
        // env0 = {t0:5} (t1 absent), single env. α1 = 0.5, m_·0 = 5, τ = 2.
        let sources = sources_from_envs(&[&[5, 0]]);
        let cp = ConditionalProbability::precompute(&sources, 0.5);
        let denom = 5.0 + 2.0 * 0.5; // 6
        approx(cp.known_p_tv_row(0)[0], (5.0 + 0.5) / denom);
        approx(cp.known_p_tv_row(1)[0], 0.5 / denom); // absent taxon
    }
}
