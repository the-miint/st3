// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Error type for preprocessing.
//!
//! Preprocessing validates its inputs eagerly and returns a descriptive
//! [`Error`] rather than panicking, so callers (including the future FFI layer)
//! can surface a precise reason. This module is a leaf: it has no dependencies
//! on other modules in the crate.

use std::fmt;

/// The axis an identifier or index refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// The feature (taxon) axis — rows of the count table.
    Feature,
    /// The sample axis — columns of the count table.
    Sample,
}

impl Axis {
    /// A lowercase noun for use in error messages.
    fn noun(self) -> &'static str {
        match self {
            Axis::Feature => "feature",
            Axis::Sample => "sample",
        }
    }
}

/// A `Result` whose error is a preprocessing [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Every eager validation failure produced by preprocessing.
///
/// `PartialEq` (but not `Eq`) is derived so tests can compare against an
/// expected variant; `NegativeCount` carries an `f64`, which has no total
/// equality.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The table has zero features or zero samples.
    EmptyTable {
        /// Which axis was empty.
        axis: Axis,
    },
    /// Two parallel inputs had different lengths.
    LengthMismatch {
        /// A short label for the inputs that disagreed.
        what: &'static str,
        /// The expected length.
        expected: usize,
        /// The actual length received.
        actual: usize,
    },
    /// An identifier appeared more than once on an axis.
    DuplicateId {
        /// The axis the duplicated id belongs to.
        axis: Axis,
        /// The offending identifier.
        id: String,
    },
    /// A COO coordinate index was greater than or equal to the axis length.
    IndexOutOfBounds {
        /// The axis the out-of-range index refers to.
        axis: Axis,
        /// The offending index.
        index: u32,
        /// The axis length it violated.
        len: usize,
    },
    /// A COO value floored to a negative count.
    NegativeCount {
        /// The feature (row) coordinate.
        row: u32,
        /// The sample (column) coordinate.
        col: u32,
        /// The original (pre-floor) value.
        value: f64,
    },
    /// A COO value was NaN or infinite.
    NonFiniteValue {
        /// The feature (row) coordinate.
        row: u32,
        /// The sample (column) coordinate.
        col: u32,
    },
    /// The same `(row, col)` coordinate appeared twice in the COO input.
    DuplicateCoordinate {
        /// The feature (row) coordinate.
        row: u32,
        /// The sample (column) coordinate.
        col: u32,
    },
    /// A count exceeded the storable range during ingest or sum-collapse.
    CountOverflow {
        /// Where the overflow occurred (e.g. `"coo ingest"`).
        context: &'static str,
        /// The value that did not fit.
        value: u64,
    },
    /// No sample was labeled as a source.
    NoSources,
    /// A source sample lacked a non-empty environment label.
    MissingEnv {
        /// Index (into the sample axis) of the source without an environment.
        sample_index: usize,
    },
    /// A Gibbs parameter was outside its valid range.
    InvalidParam {
        /// The parameter name.
        name: &'static str,
        /// Why it was rejected.
        reason: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::EmptyTable { axis } => {
                write!(f, "the count table has no {} entries.", axis.noun())
            }
            Error::LengthMismatch {
                what,
                expected,
                actual,
            } => write!(f, "{what} has length {actual} but expected {expected}."),
            Error::DuplicateId { axis, id } => {
                write!(f, "the {} id {id:?} appears more than once.", axis.noun())
            }
            Error::IndexOutOfBounds { axis, index, len } => write!(
                f,
                "the {} index {index} is out of bounds for length {len}.",
                axis.noun()
            ),
            Error::NegativeCount { row, col, value } => write!(
                f,
                "the count at (feature {row}, sample {col}) is negative ({value})."
            ),
            Error::NonFiniteValue { row, col } => write!(
                f,
                "the value at (feature {row}, sample {col}) is not finite."
            ),
            Error::DuplicateCoordinate { row, col } => write!(
                f,
                "the coordinate (feature {row}, sample {col}) appears more than once."
            ),
            Error::CountOverflow { context, value } => write!(
                f,
                "the count {value} in {context} exceeds the storable range."
            ),
            Error::NoSources => write!(f, "no sample was labeled as a source."),
            Error::MissingEnv { sample_index } => write!(
                f,
                "the source sample at index {sample_index} has no environment label."
            ),
            Error::InvalidParam { name, reason } => {
                write!(f, "the parameter {name} is invalid: {reason}.")
            }
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_nonempty_and_is_std_error() {
        let variants = [
            Error::EmptyTable {
                axis: Axis::Feature,
            },
            Error::LengthMismatch {
                what: "envs",
                expected: 2,
                actual: 3,
            },
            Error::DuplicateId {
                axis: Axis::Sample,
                id: "s0".into(),
            },
            Error::IndexOutOfBounds {
                axis: Axis::Feature,
                index: 5,
                len: 2,
            },
            Error::NegativeCount {
                row: 0,
                col: 0,
                value: -1.0,
            },
            Error::NonFiniteValue { row: 0, col: 0 },
            Error::DuplicateCoordinate { row: 0, col: 0 },
            Error::CountOverflow {
                context: "coo ingest",
                value: 5,
            },
            Error::NoSources,
            Error::MissingEnv { sample_index: 0 },
            Error::InvalidParam {
                name: "beta",
                reason: "must be finite",
            },
        ];
        for e in &variants {
            assert!(!e.to_string().trim().is_empty());
            let _: &dyn std::error::Error = e; // trait impl check
        }
    }
}
