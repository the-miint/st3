// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! In-kernel rarefaction (random subsampling to a common depth).
//!
//! Subsampling every sample to a common depth stops deep samples from
//! dominating purely by sequence count. Two modes are offered: without
//! replacement (reservoir sampling of `depth` sequences from the sample) and
//! with replacement (multinomial draws). Each sample (column) is subsampled
//! with its own PRNG derived from the run seed via [`crate::rng::rng_for_item`],
//! so output depends on the seed alone — never on iteration order or job count.
//!
//! Samples too shallow to reach the target depth are passed through unchanged
//! and flagged; the caller decides policy. The feature axis is preserved
//! exactly: features that subsample to zero remain in the table's `feature_ids`
//! as structural zeros, so sources and sinks keep an identical taxon axis.

use rand::Rng;
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand::seq::IteratorRandom;

use crate::error::{Error, Result};
use crate::metadata::SampleContext;
use crate::rng::rng_for_item;
use crate::table::{Count, CountTable, FeatureIdx};

/// Outcome of rarefying a single sample (column).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleStatus {
    /// The column was subsampled to the target depth.
    Rarefied,
    /// The column total was below the target depth; passed through unchanged.
    TooShallow,
    /// Rarefaction was disabled for this column (depth `None` or `0`); unchanged.
    Passthrough,
}

/// A rarefied table plus the per-sample outcome, on the input's exact axes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rarefied {
    table: CountTable,
    status: Vec<SampleStatus>,
    with_replacement: bool,
}

impl Rarefied {
    /// The rarefied table: identical feature axis, sample axis, and sample order
    /// as the input. Features that subsampled to zero remain in `feature_ids`.
    pub fn table(&self) -> &CountTable {
        &self.table
    }

    /// Per-sample outcome, parallel to `table().sample_ids()`.
    pub fn status(&self) -> &[SampleStatus] {
        &self.status
    }

    /// Whether sampling was with replacement.
    pub fn with_replacement(&self) -> bool {
        self.with_replacement
    }

    /// Sample indices flagged [`SampleStatus::TooShallow`], ascending — the
    /// samples a caller must drop from results or reject.
    pub fn too_shallow(&self) -> Vec<u32> {
        self.status
            .iter()
            .enumerate()
            .filter(|(_, s)| **s == SampleStatus::TooShallow)
            .map(|(i, _)| i as u32)
            .collect()
    }
}

/// Rarefaction configuration.
///
/// Depths are `Option<u32>`; `None` (or `Some(0)`) disables rarefaction for that
/// role. The role split is applied by the caller, which unpacks this into a
/// per-sample depth vector for [`rarefy_per_sample`].
///
/// The [`Default`] disables both depths and samples without replacement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RarefyConfig {
    /// Target depth for source samples (`None`/`0` disables).
    pub source_depth: Option<u32>,
    /// Target depth for sink samples (`None`/`0` disables).
    pub sink_depth: Option<u32>,
    /// Sample with replacement (multinomial) rather than without (reservoir).
    pub with_replacement: bool,
}

impl RarefyConfig {
    /// Build the per-sample depth vector for [`rarefy_per_sample`], applying
    /// `source_depth` to source samples and `sink_depth` to sinks, aligned to
    /// `ctx`'s sample axis.
    #[must_use]
    pub fn depths_for(&self, ctx: &SampleContext) -> Vec<Option<u32>> {
        let n = ctx.source_indices().len() + ctx.sink_indices().len();
        let mut depths = vec![None; n];
        for &i in ctx.source_indices() {
            depths[i as usize] = self.source_depth;
        }
        for &i in ctx.sink_indices() {
            depths[i as usize] = self.sink_depth;
        }
        depths
    }
}

/// Rarefy every column of `table` to the same `depth`.
///
/// `None` or `Some(0)` disables rarefaction (every column
/// [`SampleStatus::Passthrough`], output equals input). Columns whose total is
/// below `depth` are [`SampleStatus::TooShallow`] and passed through unchanged;
/// the rest are [`SampleStatus::Rarefied`] to exactly `depth`. The result
/// depends only on `seed`.
///
/// # Errors
/// [`Error::CountOverflow`] only in the (effectively unreachable) with-replacement
/// weight-overflow case.
pub fn rarefy(
    table: &CountTable,
    depth: Option<u32>,
    with_replacement: bool,
    seed: u64,
) -> Result<Rarefied> {
    rarefy_impl(table, |_s| depth, with_replacement, seed)
}

/// Rarefy each column to its own depth (`depths[s]` for sample `s`).
///
/// The general primitive used to apply distinct source/sink depths in one
/// deterministic pass. Column `s` is always seeded `rng_for_item(seed, s)`
/// regardless of its depth, so per-column results are order- and jobs-independent.
///
/// # Examples
/// ```
/// use st3_core::{CountTable, SampleStatus, rarefy_per_sample};
///
/// let table = CountTable::from_coo(
///     vec!["f0".into(), "f1".into()],
///     vec!["deep".into(), "shallow".into()],
///     &[0, 1, 0],
///     &[0, 0, 1],
///     &[60.0, 40.0, 5.0], // deep sums to 100, shallow sums to 5
/// )?;
/// // Rarefy each sample to depth 50, without replacement, seeded.
/// let rarefied = rarefy_per_sample(&table, &[Some(50), Some(50)], false, 42)?;
///
/// // The deep sample is subsampled to exactly 50; the shallow one cannot reach
/// // the target depth and is passed through unchanged and flagged.
/// assert_eq!(rarefied.table().column_sum(0), 50);
/// assert_eq!(rarefied.status()[0], SampleStatus::Rarefied);
/// assert_eq!(rarefied.status()[1], SampleStatus::TooShallow);
/// assert_eq!(rarefied.table().column_sum(1), 5);
/// # Ok::<(), st3_core::Error>(())
/// ```
///
/// # Errors
/// [`Error::LengthMismatch`] if `depths.len() != table.n_samples()`, plus the
/// errors of [`rarefy`].
pub fn rarefy_per_sample(
    table: &CountTable,
    depths: &[Option<u32>],
    with_replacement: bool,
    seed: u64,
) -> Result<Rarefied> {
    if depths.len() != table.n_samples() {
        return Err(Error::LengthMismatch {
            what: "rarefaction depths",
            expected: table.n_samples(),
            actual: depths.len(),
        });
    }
    rarefy_impl(table, |s| depths[s], with_replacement, seed)
}

/// Shared driver: rarefy each column to `depth_of(sample_index)`.
fn rarefy_impl(
    table: &CountTable,
    depth_of: impl Fn(usize) -> Option<u32>,
    with_replacement: bool,
    seed: u64,
) -> Result<Rarefied> {
    let n_features = table.n_features();
    let n_samples = table.n_samples();
    let mut col_ptr = vec![0usize; n_samples + 1];
    let mut row_idx: Vec<FeatureIdx> = Vec::new();
    let mut values: Vec<Count> = Vec::new();
    let mut status = Vec::with_capacity(n_samples);
    let mut scratch = vec![0u32; n_features];

    for s in 0..n_samples {
        let (rows, counts) = table.column(s);
        match depth_of(s).filter(|&d| d > 0) {
            None => {
                row_idx.extend_from_slice(rows);
                values.extend_from_slice(counts);
                status.push(SampleStatus::Passthrough);
            }
            Some(depth) => {
                let total: u64 = counts.iter().map(|&c| u64::from(c)).sum();
                if total < u64::from(depth) {
                    row_idx.extend_from_slice(rows);
                    values.extend_from_slice(counts);
                    status.push(SampleStatus::TooShallow);
                } else {
                    scratch.fill(0);
                    let mut rng = rng_for_item(seed, s as u64);
                    if with_replacement {
                        rarefy_with_replacement(rows, counts, depth, &mut scratch, &mut rng)?;
                    } else {
                        rarefy_without_replacement(rows, counts, depth, &mut scratch, &mut rng);
                    }
                    for (f, &c) in scratch.iter().enumerate() {
                        if c != 0 {
                            row_idx.push(f as FeatureIdx);
                            values.push(c);
                        }
                    }
                    status.push(SampleStatus::Rarefied);
                }
            }
        }
        col_ptr[s + 1] = row_idx.len();
    }

    let table = CountTable::from_parts(
        table.feature_ids().to_vec(),
        table.sample_ids().to_vec(),
        col_ptr,
        row_idx,
        values,
    );
    Ok(Rarefied {
        table,
        status,
        with_replacement,
    })
}

/// Reservoir-sample `depth` sequences without replacement into `scratch`.
///
/// `scratch` has length `n_features` and must be all-zero on entry. The lazy
/// expansion yields each feature `count` times in fixed (ascending) CSC order,
/// so the result is reproducible for a given `rng`. Caller guarantees
/// `depth <= sum(counts)`.
fn rarefy_without_replacement<R: Rng + ?Sized>(
    rows: &[FeatureIdx],
    counts: &[Count],
    depth: u32,
    scratch: &mut [u32],
    rng: &mut R,
) {
    let expanded = rows
        .iter()
        .zip(counts.iter())
        .flat_map(|(&f, &c)| std::iter::repeat_n(f, c as usize));
    for f in expanded.sample(rng, depth as usize) {
        scratch[f as usize] += 1;
    }
}

/// Draw `depth` sequences with replacement (multinomial) into `scratch`.
///
/// Weights are the column's counts as `u64` (exact; avoids cumulative overflow
/// when a column total exceeds `u32::MAX`).
fn rarefy_with_replacement<R: Rng + ?Sized>(
    rows: &[FeatureIdx],
    counts: &[Count],
    depth: u32,
    scratch: &mut [u32],
    rng: &mut R,
) -> Result<()> {
    let dist = WeightedIndex::new(counts.iter().map(|&c| u64::from(c))).map_err(|_| {
        Error::CountOverflow {
            context: "rarefy with replacement",
            value: counts.iter().map(|&c| u64::from(c)).sum(),
        }
    })?;
    for _ in 0..depth {
        let k = dist.sample(rng);
        scratch[rows[k] as usize] += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(scratch: &[u32]) -> u32 {
        scratch.iter().sum()
    }

    #[test]
    fn wor_sum_equals_depth() {
        let (rows, counts) = (vec![0u32, 1, 2], vec![5u32, 3, 2]); // total 10
        let mut scratch = vec![0u32; 3];
        let mut rng = rng_for_item(1, 0);
        rarefy_without_replacement(&rows, &counts, 4, &mut scratch, &mut rng);
        assert_eq!(sum(&scratch), 4);
    }

    #[test]
    fn wor_never_exceeds_original() {
        let (rows, counts) = (vec![0u32, 1, 2], vec![5u32, 3, 2]);
        for seed in 0..20 {
            let mut scratch = vec![0u32; 3];
            let mut rng = rng_for_item(seed, 0);
            rarefy_without_replacement(&rows, &counts, 6, &mut scratch, &mut rng);
            for (f, &c) in scratch.iter().enumerate() {
                assert!(c <= counts[f], "feature {f}: {c} > {}", counts[f]);
            }
        }
    }

    #[test]
    fn wor_exact_at_full_depth() {
        let (rows, counts) = (vec![0u32, 1, 2], vec![5u32, 3, 2]); // total 10
        for seed in 0..10 {
            let mut scratch = vec![0u32; 3];
            let mut rng = rng_for_item(seed, 0);
            rarefy_without_replacement(&rows, &counts, 10, &mut scratch, &mut rng);
            assert_eq!(scratch, vec![5, 3, 2]); // all sequences chosen
        }
    }

    #[test]
    fn wr_sum_equals_depth() {
        let (rows, counts) = (vec![0u32, 1], vec![3u32, 1]);
        let mut scratch = vec![0u32; 2];
        let mut rng = rng_for_item(1, 0);
        rarefy_with_replacement(&rows, &counts, 50, &mut scratch, &mut rng).unwrap();
        assert_eq!(sum(&scratch), 50);
    }

    #[test]
    fn wr_can_exceed_original() {
        // 10 draws into 2 bins of original count 1 ⇒ some bin > 1 (pigeonhole).
        let (rows, counts) = (vec![0u32, 1], vec![1u32, 1]);
        let mut scratch = vec![0u32; 2];
        let mut rng = rng_for_item(7, 0);
        rarefy_with_replacement(&rows, &counts, 10, &mut scratch, &mut rng).unwrap();
        assert!(scratch.iter().any(|&c| c > 1));
    }

    #[test]
    fn wr_single_feature_gets_all_mass() {
        let (rows, counts) = (vec![3u32], vec![7u32]);
        let mut scratch = vec![0u32; 4];
        let mut rng = rng_for_item(2, 0);
        rarefy_with_replacement(&rows, &counts, 5, &mut scratch, &mut rng).unwrap();
        assert_eq!(scratch, vec![0, 0, 0, 5]);
    }

    #[test]
    fn kernel_same_seed_same_output() {
        let (rows, counts) = (vec![0u32, 1, 2], vec![5u32, 3, 2]);
        let mut a = vec![0u32; 3];
        let mut b = vec![0u32; 3];
        rarefy_without_replacement(&rows, &counts, 4, &mut a, &mut rng_for_item(9, 3));
        rarefy_without_replacement(&rows, &counts, 4, &mut b, &mut rng_for_item(9, 3));
        assert_eq!(a, b);
    }

    /// A small table with one env-less column layout for driver tests.
    fn table_3cols() -> CountTable {
        // 4 features x 3 samples.
        CountTable::from_coo(
            (0..4).map(|i| format!("f{i}")).collect(),
            (0..3).map(|i| format!("s{i}")).collect(),
            &[0, 1, 2, 3, 0, 2, 1, 3],
            &[0, 0, 0, 0, 1, 1, 2, 2],
            &[10.0, 20.0, 30.0, 40.0, 15.0, 25.0, 35.0, 45.0],
        )
        .unwrap()
    }

    #[test]
    fn driver_column_depends_only_on_seed_and_index() {
        let t = table_3cols();
        let seed = 123;
        let depth = 20;
        let out = rarefy(&t, Some(depth), false, seed).unwrap();
        // Recompute each column in isolation and compare to the driver output.
        for s in 0..t.n_samples() {
            let (rows, counts) = t.column(s);
            let mut scratch = vec![0u32; t.n_features()];
            let mut rng = rng_for_item(seed, s as u64);
            rarefy_without_replacement(rows, counts, depth, &mut scratch, &mut rng);
            let expected: Vec<(u32, u32)> = scratch
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c != 0)
                .map(|(f, &c)| (f as u32, c))
                .collect();
            let (orows, ocounts) = out.table().column(s);
            let got: Vec<(u32, u32)> = orows.iter().copied().zip(ocounts.iter().copied()).collect();
            assert_eq!(got, expected, "column {s}");
        }
    }

    #[test]
    fn driver_order_independent() {
        // Forward vs reversed processing must yield identical per-column output;
        // proved by comparing the driver (forward) to isolated per-column recompute
        // in reverse — same result, since each column is independent.
        let t = table_3cols();
        let out = rarefy(&t, Some(15), true, 55).unwrap();
        for s in (0..t.n_samples()).rev() {
            let (rows, counts) = t.column(s);
            let mut scratch = vec![0u32; t.n_features()];
            let mut rng = rng_for_item(55, s as u64);
            rarefy_with_replacement(rows, counts, 15, &mut scratch, &mut rng).unwrap();
            let (orows, ocounts) = out.table().column(s);
            let mut dense = vec![0u32; t.n_features()];
            for (&f, &c) in orows.iter().zip(ocounts.iter()) {
                dense[f as usize] = c;
            }
            assert_eq!(dense, scratch, "column {s}");
        }
    }

    #[test]
    fn per_sample_length_mismatch() {
        let t = table_3cols();
        let e = rarefy_per_sample(&t, &[Some(5), Some(5)], false, 0).unwrap_err();
        assert!(matches!(e, Error::LengthMismatch { .. }));
    }

    #[test]
    fn config_depths_for_maps_roles_to_samples() {
        use crate::metadata::Role;
        // samples: source, sink, source, sink  (indices 0..3)
        let ctx = SampleContext::new(
            vec![Role::Source, Role::Sink, Role::Source, Role::Sink],
            vec![Some("a".into()), None, Some("b".into()), None],
        )
        .unwrap();
        let cfg = RarefyConfig {
            source_depth: Some(500),
            sink_depth: Some(1000),
            with_replacement: false,
        };
        assert_eq!(
            cfg.depths_for(&ctx),
            vec![Some(500), Some(1000), Some(500), Some(1000)]
        );
        // Disabled source depth propagates as None to source samples only.
        let cfg = RarefyConfig {
            source_depth: None,
            sink_depth: Some(1000),
            with_replacement: false,
        };
        assert_eq!(
            cfg.depths_for(&ctx),
            vec![None, Some(1000), None, Some(1000)]
        );
    }
}
