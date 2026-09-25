// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Collapse of source samples into one representative source per environment.
//!
//! Two aggregation modes are supported: [`CollapseMethod::Sum`] (the R
//! reference) sums an environment's source samples per taxon; and
//! [`CollapseMethod::Mean`] (the Python reference) takes the per-taxon mean and
//! floors it to an integer. Collapsed environments are ordered by
//! lexicographically sorted name, and the collapsed table reuses the input's
//! feature axis so sources and sinks share an identical taxon axis by
//! construction.

use std::collections::BTreeMap;

use crate::error::{Axis, Error, Result};
use crate::metadata::SampleContext;
use crate::rarefy::rarefy;
use crate::table::{Count, CountTable, FeatureIdx};

/// How source samples in one environment are aggregated per taxon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollapseMethod {
    /// Sum counts across the environment's source samples (R reference).
    Sum,
    /// Per-taxon mean across the environment's samples, floored to an integer
    /// (Python `sourcetracker2`).
    Mean,
}

/// Collapsed sources: one column per environment (lexicographically sorted),
/// on the same feature axis as the input table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollapsedSources {
    table: CountTable,
    method: CollapseMethod,
}

impl CollapsedSources {
    /// Environment names, in column order (lexicographically sorted).
    pub fn env_names(&self) -> &[String] {
        self.table.sample_ids()
    }

    /// Feature identifiers — identical to the input table's feature axis.
    pub fn feature_ids(&self) -> &[String] {
        self.table.feature_ids()
    }

    /// Number of source environments (excludes the data-driven Unknown, which
    /// is materialized later by the estimator).
    pub fn n_sources(&self) -> usize {
        self.table.n_samples()
    }

    /// The aggregation method used.
    pub fn method(&self) -> CollapseMethod {
        self.method
    }

    /// The underlying collapsed count table.
    pub fn counts(&self) -> &CountTable {
        &self.table
    }

    /// Nonzero `(feature_idx, count)` of environment `env`, ascending.
    ///
    /// # Panics
    /// Panics if `env >= self.n_sources()`.
    pub fn column(&self, env: usize) -> (&[FeatureIdx], &[Count]) {
        self.table.column(env)
    }

    /// Subsample every environment column to `depth`, as SourceTracker2 does in
    /// sink mode: collapse first, then rarefy the *collapsed* environments.
    ///
    /// `None` or `Some(0)` disables subsampling (the output equals the input). An
    /// environment whose total is below `depth` is passed through unchanged,
    /// exactly as [`crate::rarefy()`] does; a caller wanting the reference's
    /// fail-fast policy calls [`crate::check_depth`] first, as the rarefied
    /// drivers do. The environment order, the feature axis, and the collapse
    /// method are preserved.
    ///
    /// # Errors
    /// Those of [`crate::rarefy()`].
    pub fn rarefy(
        &self,
        depth: Option<u32>,
        with_replacement: bool,
        seed: u64,
    ) -> Result<CollapsedSources> {
        let rarefied = rarefy(&self.table, depth, with_replacement, seed)?;
        Ok(CollapsedSources {
            table: rarefied.into_table(),
            method: self.method,
        })
    }
}

/// Collapse the sources described by `ctx` on `table`.
///
/// # Errors
/// Propagates every error of [`collapse_subset`].
pub fn collapse_sources(
    table: &CountTable,
    ctx: &SampleContext,
    method: CollapseMethod,
) -> Result<CollapsedSources> {
    collapse_subset(table, ctx.source_indices(), &ctx.source_envs(), method)
}

/// Collapse an arbitrary subset of source columns, grouped by `envs`.
///
/// `source_indices` and `envs` are parallel: `source_indices[k]` is a column of
/// `table` whose environment is `envs[k]`. This subset form is what leave-one-out
/// uses (drop one source index and re-collapse).
///
/// # Errors
/// [`Error::LengthMismatch`] if the two slices differ in length;
/// [`Error::NoSources`] if empty; [`Error::IndexOutOfBounds`] for a sample index
/// past the table; [`Error::CountOverflow`] if a summed count exceeds [`Count`].
pub fn collapse_subset(
    table: &CountTable,
    source_indices: &[u32],
    envs: &[&str],
    method: CollapseMethod,
) -> Result<CollapsedSources> {
    if source_indices.len() != envs.len() {
        return Err(Error::LengthMismatch {
            what: "envs",
            expected: source_indices.len(),
            actual: envs.len(),
        });
    }
    if source_indices.is_empty() {
        return Err(Error::NoSources);
    }

    // BTreeMap keys iterate in lexicographic (byte) order — the same order as
    // Python `sorted()` and R `sort()` for the ASCII environment names in use —
    // so the output columns are sorted with no explicit sort or hash-order risk.
    let mut members: BTreeMap<&str, Vec<u32>> = BTreeMap::new();
    for (&idx, &env) in source_indices.iter().zip(envs.iter()) {
        if idx as usize >= table.n_samples() {
            return Err(Error::IndexOutOfBounds {
                axis: Axis::Sample,
                index: idx,
                len: table.n_samples(),
            });
        }
        members.entry(env).or_default().push(idx);
    }

    let n_features = table.n_features();
    let overflow_context = match method {
        CollapseMethod::Sum => "sum collapse",
        CollapseMethod::Mean => "mean collapse",
    };
    let env_names: Vec<String> = members.keys().map(|k| (*k).to_owned()).collect();
    let mut col_ptr = vec![0usize; env_names.len() + 1];
    let mut row_idx: Vec<FeatureIdx> = Vec::new();
    let mut values: Vec<Count> = Vec::new();
    let mut scratch = vec![0u64; n_features];

    for (c, member_indices) in members.values().enumerate() {
        scratch.fill(0);
        for &m in member_indices {
            let (rows, counts) = table.column(m as usize);
            for (&r, &count) in rows.iter().zip(counts.iter()) {
                scratch[r as usize] += count as u64;
            }
        }

        let n_members = member_indices.len() as u64;
        for (f, &acc) in scratch.iter().enumerate() {
            let value = match method {
                CollapseMethod::Sum => acc,
                CollapseMethod::Mean => acc / n_members, // integer division = floor(mean)
            };
            if value == 0 {
                continue;
            }
            let stored = Count::try_from(value).map_err(|_| Error::CountOverflow {
                context: overflow_context,
                value,
            })?;
            row_idx.push(f as FeatureIdx);
            values.push(stored);
        }
        col_ptr[c + 1] = row_idx.len();
    }

    let table = CountTable::from_parts(
        table.feature_ids().to_vec(),
        env_names,
        col_ptr,
        row_idx,
        values,
    );
    Ok(CollapsedSources { table, method })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small table from dense columns (each inner vec is a sample).
    fn table_from_columns(n_features: usize, columns: &[&[u32]]) -> CountTable {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for (c, column) in columns.iter().enumerate() {
            for (r, &v) in column.iter().enumerate() {
                if v != 0 {
                    rows.push(r as u32);
                    cols.push(c as u32);
                    vals.push(v as f64);
                }
            }
        }
        let feature_ids = (0..n_features).map(|i| format!("f{i}")).collect();
        let sample_ids = (0..columns.len()).map(|i| format!("s{i}")).collect();
        CountTable::from_coo(feature_ids, sample_ids, &rows, &cols, &vals).unwrap()
    }

    #[test]
    fn collapse_sum_small() {
        // two envs: A = {s0, s1}, B = {s2}
        let t = table_from_columns(2, &[&[1, 2], &[3, 4], &[5, 6]]);
        let cs = collapse_subset(&t, &[0, 1, 2], &["A", "A", "B"], CollapseMethod::Sum).unwrap();
        assert_eq!(cs.env_names(), &["A".to_string(), "B".to_string()]);
        assert_eq!(cs.column(0), (&[0u32, 1][..], &[4u32, 6][..])); // 1+3, 2+4
        assert_eq!(cs.column(1), (&[0u32, 1][..], &[5u32, 6][..]));
    }

    #[test]
    fn collapse_mean_floors() {
        // env with counts {4, 5} on a single feature -> floor(4.5) = 4
        let t = table_from_columns(1, &[&[4], &[5]]);
        let cs = collapse_subset(&t, &[0, 1], &["e", "e"], CollapseMethod::Mean).unwrap();
        assert_eq!(cs.column(0), (&[0u32][..], &[4u32][..]));

        // counts {100, 90, 95} -> floor(285/3) = 95
        let t = table_from_columns(1, &[&[100], &[90], &[95]]);
        let cs = collapse_subset(&t, &[0, 1, 2], &["e", "e", "e"], CollapseMethod::Mean).unwrap();
        assert_eq!(cs.column(0), (&[0u32][..], &[95u32][..]));
    }

    #[test]
    fn collapse_env_order_lexicographic() {
        // The reference `collapse_source_data` docstring example.
        // samples s1,s2,s3 with envs "3.0","0.4","3.0"; s4 excluded.
        let t = table_from_columns(
            4,
            &[
                &[10, 50, 10, 70], // s1  env 3.0
                &[0, 25, 10, 5],   // s2  env 0.4
                &[0, 25, 10, 5],   // s3  env 3.0
                &[1, 0, 0, 0],     // s4  (not in the subset)
            ],
        );
        let cs =
            collapse_subset(&t, &[0, 1, 2], &["3.0", "0.4", "3.0"], CollapseMethod::Sum).unwrap();
        // "0.4" sorts before "3.0" (byte order).
        assert_eq!(cs.env_names(), &["0.4".to_string(), "3.0".to_string()]);
        assert_eq!(cs.column(0), (&[1u32, 2, 3][..], &[25u32, 10, 5][..])); // s2
        assert_eq!(
            cs.column(1),
            (&[0u32, 1, 2, 3][..], &[10u32, 75, 20, 75][..])
        ); // s1+s3
    }

    #[test]
    fn collapse_shares_feature_axis() {
        let t = table_from_columns(3, &[&[1, 0, 2], &[0, 3, 0]]);
        let cs = collapse_subset(&t, &[0, 1], &["A", "B"], CollapseMethod::Sum).unwrap();
        assert_eq!(cs.feature_ids(), t.feature_ids());
    }

    #[test]
    fn collapse_sum_overflow() {
        // 3e9 + 2e9 = 5e9 > u32::MAX
        let t = table_from_columns(1, &[&[3_000_000_000], &[2_000_000_000]]);
        let e = collapse_subset(&t, &[0, 1], &["e", "e"], CollapseMethod::Sum).unwrap_err();
        assert_eq!(
            e,
            Error::CountOverflow {
                context: "sum collapse",
                value: 5_000_000_000
            }
        );
    }

    #[test]
    fn err_collapse_len_mismatch() {
        let t = table_from_columns(1, &[&[1], &[2]]);
        let e = collapse_subset(&t, &[0, 1], &["a"], CollapseMethod::Sum).unwrap_err();
        assert!(matches!(e, Error::LengthMismatch { .. }));
    }

    #[test]
    fn collapse_subset_loo() {
        // Full sources: A={s0,s1}, B={s2}. Hold out the sole B member.
        let t = table_from_columns(2, &[&[1, 1], &[2, 2], &[9, 9]]);
        let full = collapse_subset(&t, &[0, 1, 2], &["A", "A", "B"], CollapseMethod::Sum).unwrap();
        assert_eq!(full.n_sources(), 2);

        let loo = collapse_subset(&t, &[0, 1], &["A", "A"], CollapseMethod::Sum).unwrap();
        assert_eq!(loo.n_sources(), 1);
        assert_eq!(loo.env_names(), &["A".to_string()]);
    }

    // SourceTracker2 sink mode subsamples the *collapsed* environments, so the
    // collapsed table must be rarefiable in place: every environment column
    // lands on exactly `depth`, with names, feature axis, and method preserved.
    #[test]
    fn rarefy_subsamples_each_environment_to_depth() {
        // A = {s0, s1} sums to 12; B = {s2} to 9. Depth 8 is below both.
        let t = table_from_columns(2, &[&[3, 3], &[3, 3], &[5, 4]]);
        let cs = collapse_subset(&t, &[0, 1, 2], &["A", "A", "B"], CollapseMethod::Sum).unwrap();
        let r = cs.rarefy(Some(8), false, 7).unwrap();
        assert_eq!(r.env_names(), cs.env_names());
        assert_eq!(r.feature_ids(), cs.feature_ids());
        assert_eq!(r.method(), CollapseMethod::Sum);
        for env in 0..r.n_sources() {
            assert_eq!(r.counts().column_sum(env), 8, "env {env}");
        }
    }

    // An environment below the depth passes through unchanged, exactly like
    // `rarefy` on a plain table; the policy for it belongs to the caller.
    #[test]
    fn rarefy_passes_a_shallow_environment_through() {
        let t = table_from_columns(2, &[&[3, 3], &[3, 3], &[5, 4]]);
        let cs = collapse_subset(&t, &[0, 1, 2], &["A", "A", "B"], CollapseMethod::Sum).unwrap();
        // Depth 10: A (12) is subsampled, B (9) is not.
        let r = cs.rarefy(Some(10), false, 7).unwrap();
        assert_eq!(r.counts().column_sum(0), 10);
        assert_eq!(r.column(1), cs.column(1));
    }

    #[test]
    fn rarefy_disabled_is_the_identity() {
        let t = table_from_columns(2, &[&[3, 3], &[5, 4]]);
        let cs = collapse_subset(&t, &[0, 1], &["A", "B"], CollapseMethod::Mean).unwrap();
        assert_eq!(cs.rarefy(None, false, 1).unwrap(), cs);
        assert_eq!(cs.rarefy(Some(0), true, 1).unwrap(), cs);
    }
}
