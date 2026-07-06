// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Error type for the Arrow boundary.
//!
//! Import validates its Arrow inputs eagerly — before any core computation —
//! and returns a descriptive [`Error`] rather than panicking, so a caller
//! (ultimately the C ABI in a later milestone) can surface a precise reason. The
//! boundary's own failures (a missing or wrongly typed column, an unexpected
//! null, an unknown role, an index that will not fit core's `u32` coordinate)
//! are distinct variants; failures raised inside the core library or by Arrow
//! itself are wrapped as [`Error::Core`] / [`Error::Arrow`].

use std::fmt;

use arrow::error::ArrowError;

/// A `Result` whose error is an Arrow-boundary [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// A failure importing an Arrow COO batch / metadata table, or exporting a
/// result batch.
///
/// `PartialEq` is intentionally not derived: the wrapped [`ArrowError`] does not
/// implement it. Tests match on the variant (destructuring fields) rather than
/// comparing whole values.
#[derive(Debug)]
pub enum Error {
    /// A required column was absent from the struct array or record batch.
    MissingColumn {
        /// The name of the column that was expected but not found.
        name: String,
    },
    /// A column was present but had an unsupported Arrow data type.
    WrongType {
        /// The column name.
        name: String,
        /// A short description of the accepted type(s).
        expected: &'static str,
        /// The actual Arrow data type, formatted for the message.
        got: String,
    },
    /// A cell that must carry a value was null.
    UnexpectedNull {
        /// The column the null appeared in.
        column: String,
        /// The row index of the null cell.
        row: usize,
    },
    /// A `role` string was neither `"source"` nor `"sink"`.
    UnknownRole {
        /// The offending role string.
        value: String,
        /// The row index it appeared at.
        row: usize,
    },
    /// A COO coordinate did not fit core's `u32` index range (`0..=u32::MAX`).
    ///
    /// Arrow accepts `Int64` `row`/`col` columns, so a negative or too-large
    /// value is reported here rather than being silently truncated on the cast.
    IndexOutOfRange {
        /// The column the out-of-range value came from (`"row"` or `"col"`).
        column: String,
        /// The row index of the offending value.
        row: usize,
        /// The offending value.
        value: i64,
    },
    /// A core preprocessing error (validation, flooring, bounds).
    Core(st3_core::Error),
    /// An underlying Arrow error (e.g. assembling a record batch).
    Arrow(ArrowError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::MissingColumn { name } => write!(f, "missing required column '{name}'"),
            Error::WrongType {
                name,
                expected,
                got,
            } => write!(f, "column '{name}' must be {expected}, got {got}"),
            Error::UnexpectedNull { column, row } => {
                write!(f, "unexpected null in column '{column}' at row {row}")
            }
            Error::UnknownRole { value, row } => write!(
                f,
                "unknown role '{value}' at row {row} (expected 'source' or 'sink')"
            ),
            Error::IndexOutOfRange { column, row, value } => write!(
                f,
                "index {value} in column '{column}' at row {row} is out of range for u32"
            ),
            Error::Core(e) => write!(f, "core error: {e}"),
            Error::Arrow(e) => write!(f, "arrow error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Core(e) => Some(e),
            Error::Arrow(e) => Some(e),
            _ => None,
        }
    }
}

impl From<st3_core::Error> for Error {
    fn from(e: st3_core::Error) -> Self {
        Error::Core(e)
    }
}

impl From<ArrowError> for Error {
    fn from(e: ArrowError) -> Self {
        Error::Arrow(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_covers_boundary_variants() {
        let cases = [
            (
                Error::MissingColumn { name: "row".into() },
                "missing required column 'row'",
            ),
            (
                Error::WrongType {
                    name: "val".into(),
                    expected: "Int32, Int64 or Float64",
                    got: "Utf8".into(),
                },
                "column 'val' must be Int32, Int64 or Float64, got Utf8",
            ),
            (
                Error::UnexpectedNull {
                    column: "row".into(),
                    row: 3,
                },
                "unexpected null in column 'row' at row 3",
            ),
            (
                Error::UnknownRole {
                    value: "donor".into(),
                    row: 1,
                },
                "unknown role 'donor' at row 1 (expected 'source' or 'sink')",
            ),
            (
                Error::IndexOutOfRange {
                    column: "col".into(),
                    row: 0,
                    value: -1,
                },
                "index -1 in column 'col' at row 0 is out of range for u32",
            ),
        ];
        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected);
        }
    }

    #[test]
    fn wraps_core_and_arrow_errors_via_from() {
        let core_err = st3_core::Error::NoSources;
        let wrapped: Error = core_err.into();
        assert!(matches!(wrapped, Error::Core(_)));

        let arrow_err = ArrowError::SchemaError("boom".into());
        let wrapped: Error = arrow_err.into();
        assert!(matches!(wrapped, Error::Arrow(_)));
    }

    #[test]
    fn source_exposes_wrapped_error() {
        use std::error::Error as _;
        let wrapped = Error::Core(st3_core::Error::NoSources);
        assert!(wrapped.source().is_some());
        let boundary = Error::MissingColumn { name: "x".into() };
        assert!(boundary.source().is_none());
    }
}
