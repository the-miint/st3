// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! The sample-major sparse count table and its COO ingest.
//!
//! Counts are stored in compressed-sparse-column (CSC) form: columns are
//! samples, rows are features. Every downstream step is per-sample (splitting
//! by role, summing an environment's columns, pulling a sink's taxon vector,
//! subsampling a sample), so sample-major storage keeps each of those a
//! contiguous slice. Conversion from COO happens exactly once, at ingest, which
//! is also the single point where floating-point counts are floored to
//! integers.

use std::collections::HashSet;

use crate::error::{Axis, Error, Result};

/// Stored count type.
///
/// A single sample's sequencing depth is far below `u32::MAX` (~4.3e9), so
/// `u32` is ample per entry. Sum-collapse accumulates in `u64` and narrows back
/// to this type, reporting [`Error::CountOverflow`] rather than wrapping. Widen
/// this alias to `u64` if a real workload ever needs it.
pub type Count = u32;

/// Feature (row) index type for the CSC arrays.
pub type FeatureIdx = u32;

/// A sample-major CSC integer count table with a single shared feature axis.
///
/// Columns are samples, rows are features. Structural zeros are not stored;
/// within a column, `row_idx` is strictly ascending. All stored values are
/// positive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CountTable {
    feature_ids: Vec<String>,
    sample_ids: Vec<String>,
    col_ptr: Vec<usize>,
    row_idx: Vec<FeatureIdx>,
    values: Vec<Count>,
}

impl CountTable {
    /// Build a CSC table from COO triples `(row = feature, col = sample, value)`.
    ///
    /// Floating-point values are floored to integers here — the single
    /// float-to-integer boundary in the pipeline. Explicit zeros (and values
    /// that floor to zero) are dropped rather than stored.
    ///
    /// # Examples
    /// ```
    /// use st3_core::CountTable;
    ///
    /// // Two features, two samples. Within a column rows are stored ascending;
    /// // sample 1 gets features 0 and 1 with counts 2 and 3.
    /// let table = CountTable::from_coo(
    ///     vec!["f0".into(), "f1".into()],
    ///     vec!["s0".into(), "s1".into()],
    ///     &[0, 1, 0],       // feature (row) indices
    ///     &[0, 1, 1],       // sample (col) indices
    ///     &[5.0, 3.0, 2.0], // counts
    /// )?;
    /// assert_eq!(table.n_features(), 2);
    /// assert_eq!(table.n_samples(), 2);
    /// assert_eq!(table.column(0), (&[0u32][..], &[5u32][..]));
    /// assert_eq!(table.column_sum(1), 5); // 3 + 2
    /// # Ok::<(), st3_core::Error>(())
    /// ```
    ///
    /// # Errors
    /// Returns an [`Error`] for an empty axis, mismatched input lengths,
    /// duplicate ids, non-finite or negative values, out-of-bounds coordinates,
    /// a value that overflows [`Count`], or a repeated `(row, col)` coordinate.
    pub fn from_coo(
        feature_ids: Vec<String>,
        sample_ids: Vec<String>,
        rows: &[u32],
        cols: &[u32],
        values: &[f64],
    ) -> Result<Self> {
        let n_features = feature_ids.len();
        let n_samples = sample_ids.len();
        if n_features == 0 {
            return Err(Error::EmptyTable {
                axis: Axis::Feature,
            });
        }
        if n_samples == 0 {
            return Err(Error::EmptyTable { axis: Axis::Sample });
        }

        let n = rows.len();
        if cols.len() != n {
            return Err(Error::LengthMismatch {
                what: "coo column indices",
                expected: n,
                actual: cols.len(),
            });
        }
        if values.len() != n {
            return Err(Error::LengthMismatch {
                what: "coo values",
                expected: n,
                actual: values.len(),
            });
        }

        if let Some(id) = first_duplicate(&feature_ids) {
            return Err(Error::DuplicateId {
                axis: Axis::Feature,
                id: id.clone(),
            });
        }
        if let Some(id) = first_duplicate(&sample_ids) {
            return Err(Error::DuplicateId {
                axis: Axis::Sample,
                id: id.clone(),
            });
        }

        // (col, row, count), collected in input order then sorted for CSC.
        let mut entries: Vec<(u32, FeatureIdx, Count)> = Vec::with_capacity(n);
        for ((&row, &col), &value) in rows.iter().zip(cols.iter()).zip(values.iter()) {
            if !value.is_finite() {
                return Err(Error::NonFiniteValue { row, col });
            }
            let floored = value.floor();
            if floored < 0.0 {
                return Err(Error::NegativeCount { row, col, value });
            }
            if row as usize >= n_features {
                return Err(Error::IndexOutOfBounds {
                    axis: Axis::Feature,
                    index: row,
                    len: n_features,
                });
            }
            if col as usize >= n_samples {
                return Err(Error::IndexOutOfBounds {
                    axis: Axis::Sample,
                    index: col,
                    len: n_samples,
                });
            }
            if floored > Count::MAX as f64 {
                return Err(Error::CountOverflow {
                    context: "coo ingest",
                    value: floored as u64,
                });
            }
            if floored == 0.0 {
                continue; // structural zero
            }
            entries.push((col, row, floored as Count));
        }

        entries.sort_unstable_by_key(|&(col, row, _)| (col, row));

        // Reject repeated coordinates (they would otherwise silently merge).
        for pair in entries.windows(2) {
            let (c0, r0, _) = pair[0];
            let (c1, r1, _) = pair[1];
            if c0 == c1 && r0 == r1 {
                return Err(Error::DuplicateCoordinate { row: r0, col: c0 });
            }
        }

        // col_ptr via counting sort layout: shift counts by one, then prefix-sum.
        let mut col_ptr = vec![0usize; n_samples + 1];
        for &(col, _, _) in &entries {
            col_ptr[col as usize + 1] += 1;
        }
        let mut acc = 0usize;
        for slot in col_ptr.iter_mut() {
            acc += *slot;
            *slot = acc;
        }

        let row_idx = entries.iter().map(|&(_, row, _)| row).collect();
        let values = entries.iter().map(|&(_, _, count)| count).collect();

        Ok(Self {
            feature_ids,
            sample_ids,
            col_ptr,
            row_idx,
            values,
        })
    }

    /// Number of features (rows).
    pub fn n_features(&self) -> usize {
        self.feature_ids.len()
    }

    /// Number of samples (columns).
    pub fn n_samples(&self) -> usize {
        self.sample_ids.len()
    }

    /// Feature identifiers, in row order.
    pub fn feature_ids(&self) -> &[String] {
        &self.feature_ids
    }

    /// Sample identifiers, in column order.
    pub fn sample_ids(&self) -> &[String] {
        &self.sample_ids
    }

    /// Number of stored (nonzero) entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Nonzero `(feature_idx, count)` of sample `s`, ascending by feature.
    ///
    /// # Panics
    /// Panics if `s >= self.n_samples()`.
    pub fn column(&self, s: usize) -> (&[FeatureIdx], &[Count]) {
        let start = self.col_ptr[s];
        let end = self.col_ptr[s + 1];
        (&self.row_idx[start..end], &self.values[start..end])
    }

    /// Total counts in sample `s` (an empty column sums to 0).
    ///
    /// # Panics
    /// Panics if `s >= self.n_samples()`.
    pub fn column_sum(&self, s: usize) -> u64 {
        let (_, counts) = self.column(s);
        counts.iter().map(|&c| c as u64).sum()
    }

    /// Assemble a table directly from validated CSC parts (crate-internal).
    ///
    /// Used by collapse, which produces canonical CSC and reuses an existing,
    /// already-unique feature axis. The caller guarantees the invariants.
    pub(crate) fn from_parts(
        feature_ids: Vec<String>,
        sample_ids: Vec<String>,
        col_ptr: Vec<usize>,
        row_idx: Vec<FeatureIdx>,
        values: Vec<Count>,
    ) -> Self {
        Self {
            feature_ids,
            sample_ids,
            col_ptr,
            row_idx,
            values,
        }
    }
}

/// Return the first identifier that appears more than once, in scan order.
fn first_duplicate(ids: &[String]) -> Option<&String> {
    let mut seen = HashSet::with_capacity(ids.len());
    ids.iter().find(|id| !seen.insert(id.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(prefix: &str, n: usize) -> Vec<String> {
        (0..n).map(|i| format!("{prefix}{i}")).collect()
    }

    #[test]
    fn from_coo_builds_expected_csc() {
        let t = CountTable::from_coo(
            ids("f", 2),
            ids("s", 2),
            &[0, 1, 0],
            &[0, 1, 1],
            &[5.0, 3.0, 2.0],
        )
        .unwrap();
        assert_eq!(t.n_features(), 2);
        assert_eq!(t.n_samples(), 2);
        assert_eq!(t.nnz(), 3);
        assert_eq!(t.column(0), (&[0u32][..], &[5u32][..]));
        assert_eq!(t.column(1), (&[0u32, 1][..], &[2u32, 3][..]));
    }

    #[test]
    fn from_coo_floors_floats() {
        // 2.9 -> 2, 3.0 -> 3, 0.5 -> 0 (dropped).
        let t = CountTable::from_coo(
            ids("f", 2),
            ids("s", 2),
            &[0, 1, 0],
            &[0, 0, 1],
            &[2.9, 3.0, 0.5],
        )
        .unwrap();
        assert_eq!(t.column(0), (&[0u32, 1][..], &[2u32, 3][..]));
        assert_eq!(t.column(1), (&[][..], &[][..]));
        assert_eq!(t.nnz(), 2);
    }

    #[test]
    fn from_coo_drops_structural_zeros() {
        let t = CountTable::from_coo(ids("f", 1), ids("s", 1), &[0], &[0], &[0.0]).unwrap();
        assert_eq!(t.nnz(), 0);
        assert_eq!(t.column(0), (&[][..], &[][..]));
    }

    #[test]
    fn from_coo_sorts_rows_within_column() {
        // Rows supplied descending; CSC must store them ascending.
        let t = CountTable::from_coo(
            ids("f", 3),
            ids("s", 1),
            &[2, 0, 1],
            &[0, 0, 0],
            &[1.0, 2.0, 3.0],
        )
        .unwrap();
        assert_eq!(t.column(0), (&[0u32, 1, 2][..], &[2u32, 3, 1][..]));
    }

    #[test]
    fn column_and_column_sum() {
        let t = CountTable::from_coo(
            ids("f", 2),
            ids("s", 3),
            &[0, 1, 0],
            &[0, 0, 2],
            &[4.0, 6.0, 7.0],
        )
        .unwrap();
        assert_eq!(t.column_sum(0), 10);
        assert_eq!(t.column_sum(1), 0); // empty column
        assert_eq!(t.column_sum(2), 7);
    }

    #[test]
    fn err_empty_features() {
        let e = CountTable::from_coo(vec![], ids("s", 1), &[], &[], &[]).unwrap_err();
        assert_eq!(
            e,
            Error::EmptyTable {
                axis: Axis::Feature
            }
        );
    }

    #[test]
    fn err_empty_samples() {
        let e = CountTable::from_coo(ids("f", 1), vec![], &[], &[], &[]).unwrap_err();
        assert_eq!(e, Error::EmptyTable { axis: Axis::Sample });
    }

    #[test]
    fn err_coo_length_mismatch() {
        let e =
            CountTable::from_coo(ids("f", 1), ids("s", 1), &[0, 0], &[0], &[1.0, 1.0]).unwrap_err();
        assert!(matches!(e, Error::LengthMismatch { .. }));
    }

    #[test]
    fn err_duplicate_feature_id() {
        let e = CountTable::from_coo(
            vec!["f0".into(), "f0".into()],
            ids("s", 1),
            &[0],
            &[0],
            &[1.0],
        )
        .unwrap_err();
        assert_eq!(
            e,
            Error::DuplicateId {
                axis: Axis::Feature,
                id: "f0".into()
            }
        );
    }

    #[test]
    fn err_duplicate_sample_id() {
        let e = CountTable::from_coo(
            ids("f", 1),
            vec!["s0".into(), "s0".into()],
            &[0],
            &[0],
            &[1.0],
        )
        .unwrap_err();
        assert_eq!(
            e,
            Error::DuplicateId {
                axis: Axis::Sample,
                id: "s0".into()
            }
        );
    }

    #[test]
    fn err_feature_index_oob() {
        let e = CountTable::from_coo(ids("f", 1), ids("s", 1), &[5], &[0], &[1.0]).unwrap_err();
        assert_eq!(
            e,
            Error::IndexOutOfBounds {
                axis: Axis::Feature,
                index: 5,
                len: 1
            }
        );
    }

    #[test]
    fn err_sample_index_oob() {
        let e = CountTable::from_coo(ids("f", 1), ids("s", 1), &[0], &[5], &[1.0]).unwrap_err();
        assert_eq!(
            e,
            Error::IndexOutOfBounds {
                axis: Axis::Sample,
                index: 5,
                len: 1
            }
        );
    }

    #[test]
    fn err_negative_count() {
        let e = CountTable::from_coo(ids("f", 1), ids("s", 1), &[0], &[0], &[-0.5]).unwrap_err();
        assert_eq!(
            e,
            Error::NegativeCount {
                row: 0,
                col: 0,
                value: -0.5
            }
        );
    }

    #[test]
    fn err_nan_value() {
        let e =
            CountTable::from_coo(ids("f", 1), ids("s", 1), &[0], &[0], &[f64::NAN]).unwrap_err();
        assert_eq!(e, Error::NonFiniteValue { row: 0, col: 0 });
    }

    #[test]
    fn err_inf_value() {
        let e = CountTable::from_coo(ids("f", 1), ids("s", 1), &[0], &[0], &[f64::INFINITY])
            .unwrap_err();
        assert_eq!(e, Error::NonFiniteValue { row: 0, col: 0 });
    }

    #[test]
    fn err_duplicate_coordinate() {
        let e = CountTable::from_coo(ids("f", 1), ids("s", 1), &[0, 0], &[0, 0], &[1.0, 2.0])
            .unwrap_err();
        assert_eq!(e, Error::DuplicateCoordinate { row: 0, col: 0 });
    }

    #[test]
    fn err_count_overflow() {
        let e = CountTable::from_coo(ids("f", 1), ids("s", 1), &[0], &[0], &[5e9]).unwrap_err();
        assert_eq!(
            e,
            Error::CountOverflow {
                context: "coo ingest",
                value: 5_000_000_000
            }
        );
    }
}
