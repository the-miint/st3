// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Typed-column normalization for imported Arrow arrays.
//!
//! Callers may hand us either integer width and any string flavor. Each of the
//! three small enums here probes an array's concrete type once via
//! `downcast_ref` and then exposes a single uniform accessor, so the rest of the
//! boundary reads values without caring which physical type arrived. Null and
//! (for indices) `u32`-range checks happen here, eagerly, producing the
//! descriptive [`Error`] variants — no silent truncation, no deferred panic.
//!
//! This mirrors the enum-normalization idiom common to Arrow ingest code; the
//! implementations are written fresh for the exact type sets SourceTracker3
//! accepts (design §6): `row`/`col` ∈ {Int32, Int64}; `val` ∈ {Int32, Int64,
//! Float64}; ids and metadata strings ∈ {Utf8, LargeUtf8}.

use arrow::array::{Array, Float64Array, Int32Array, Int64Array, LargeStringArray, StringArray};

use crate::error::{Error, Result};

/// An integer index column (`row`/`col`), normalized across Int32/Int64.
#[derive(Debug)]
pub(crate) enum IndexColumn<'a> {
    I32(&'a Int32Array),
    I64(&'a Int64Array),
}

impl<'a> IndexColumn<'a> {
    /// Downcast `array` to a supported integer array, or fail with
    /// [`Error::WrongType`].
    pub(crate) fn from_array(array: &'a dyn Array, name: &str) -> Result<Self> {
        if let Some(a) = array.as_any().downcast_ref::<Int32Array>() {
            Ok(IndexColumn::I32(a))
        } else if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
            Ok(IndexColumn::I64(a))
        } else {
            Err(Error::WrongType {
                name: name.to_string(),
                expected: "Int32 or Int64",
                got: format!("{:?}", array.data_type()),
            })
        }
    }

    fn len(&self) -> usize {
        match self {
            IndexColumn::I32(a) => a.len(),
            IndexColumn::I64(a) => a.len(),
        }
    }

    fn is_null(&self, i: usize) -> bool {
        match self {
            IndexColumn::I32(a) => a.is_null(i),
            IndexColumn::I64(a) => a.is_null(i),
        }
    }

    fn value(&self, i: usize) -> i64 {
        match self {
            IndexColumn::I32(a) => a.value(i) as i64,
            IndexColumn::I64(a) => a.value(i),
        }
    }

    /// Materialize the whole column as `Vec<u32>`, null-checking every cell and
    /// rejecting any value outside `0..=u32::MAX`.
    ///
    /// # Errors
    /// [`Error::UnexpectedNull`] for a null cell; [`Error::IndexOutOfRange`] for
    /// a negative or too-large value.
    pub(crate) fn to_u32_vec(&self, name: &str) -> Result<Vec<u32>> {
        let n = self.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            if self.is_null(i) {
                return Err(Error::UnexpectedNull {
                    column: name.to_string(),
                    row: i,
                });
            }
            let v = self.value(i);
            if v < 0 || v > u32::MAX as i64 {
                return Err(Error::IndexOutOfRange {
                    column: name.to_string(),
                    row: i,
                    value: v,
                });
            }
            out.push(v as u32);
        }
        Ok(out)
    }
}

/// A count column (`val`), normalized across Int32/Int64/Float64 to `f64`.
///
/// Float values are passed through unrounded: the single float-to-integer
/// boundary is core's `from_coo`, which floors on ingest, so this layer never
/// re-floors.
#[derive(Debug)]
pub(crate) enum ValueColumn<'a> {
    I32(&'a Int32Array),
    I64(&'a Int64Array),
    F64(&'a Float64Array),
}

impl<'a> ValueColumn<'a> {
    /// Downcast `array` to a supported count array, or fail with
    /// [`Error::WrongType`].
    pub(crate) fn from_array(array: &'a dyn Array, name: &str) -> Result<Self> {
        if let Some(a) = array.as_any().downcast_ref::<Int32Array>() {
            Ok(ValueColumn::I32(a))
        } else if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
            Ok(ValueColumn::I64(a))
        } else if let Some(a) = array.as_any().downcast_ref::<Float64Array>() {
            Ok(ValueColumn::F64(a))
        } else {
            Err(Error::WrongType {
                name: name.to_string(),
                expected: "Int32, Int64 or Float64",
                got: format!("{:?}", array.data_type()),
            })
        }
    }

    fn len(&self) -> usize {
        match self {
            ValueColumn::I32(a) => a.len(),
            ValueColumn::I64(a) => a.len(),
            ValueColumn::F64(a) => a.len(),
        }
    }

    fn is_null(&self, i: usize) -> bool {
        match self {
            ValueColumn::I32(a) => a.is_null(i),
            ValueColumn::I64(a) => a.is_null(i),
            ValueColumn::F64(a) => a.is_null(i),
        }
    }

    fn value(&self, i: usize) -> f64 {
        match self {
            ValueColumn::I32(a) => a.value(i) as f64,
            ValueColumn::I64(a) => a.value(i) as f64,
            ValueColumn::F64(a) => a.value(i),
        }
    }

    /// Materialize the whole column as `Vec<f64>`, null-checking every cell.
    ///
    /// # Errors
    /// [`Error::UnexpectedNull`] for a null cell.
    pub(crate) fn to_f64_vec(&self, name: &str) -> Result<Vec<f64>> {
        let n = self.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            if self.is_null(i) {
                return Err(Error::UnexpectedNull {
                    column: name.to_string(),
                    row: i,
                });
            }
            out.push(self.value(i));
        }
        Ok(out)
    }
}

/// A string column (ids / metadata), normalized across Utf8/LargeUtf8.
///
/// Nullability is left to the caller: `sample_id`/`feature_ids`/`role` are
/// materialized non-null via [`Self::to_string_vec`], while the nullable `env`
/// column is read cell-by-cell with [`Self::is_null`] / [`Self::value`].
#[derive(Debug)]
pub(crate) enum StrColumn<'a> {
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
}

impl<'a> StrColumn<'a> {
    /// Downcast `array` to a supported string array, or fail with
    /// [`Error::WrongType`].
    pub(crate) fn from_array(array: &'a dyn Array, name: &str) -> Result<Self> {
        if let Some(a) = array.as_any().downcast_ref::<StringArray>() {
            Ok(StrColumn::Utf8(a))
        } else if let Some(a) = array.as_any().downcast_ref::<LargeStringArray>() {
            Ok(StrColumn::LargeUtf8(a))
        } else {
            Err(Error::WrongType {
                name: name.to_string(),
                expected: "Utf8 or LargeUtf8",
                got: format!("{:?}", array.data_type()),
            })
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            StrColumn::Utf8(a) => a.len(),
            StrColumn::LargeUtf8(a) => a.len(),
        }
    }

    pub(crate) fn is_null(&self, i: usize) -> bool {
        match self {
            StrColumn::Utf8(a) => a.is_null(i),
            StrColumn::LargeUtf8(a) => a.is_null(i),
        }
    }

    pub(crate) fn value(&self, i: usize) -> &str {
        match self {
            StrColumn::Utf8(a) => a.value(i),
            StrColumn::LargeUtf8(a) => a.value(i),
        }
    }

    /// Materialize the whole column as `Vec<String>`, null-checking every cell.
    ///
    /// # Errors
    /// [`Error::UnexpectedNull`] for a null cell.
    pub(crate) fn to_string_vec(&self, name: &str) -> Result<Vec<String>> {
        let n = self.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            if self.is_null(i) {
                return Err(Error::UnexpectedNull {
                    column: name.to_string(),
                    row: i,
                });
            }
            out.push(self.value(i).to_string());
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_column_int32_and_int64_to_u32() {
        let a = Int32Array::from(vec![0, 1, 2]);
        let col = IndexColumn::from_array(&a, "row").unwrap();
        assert_eq!(col.to_u32_vec("row").unwrap(), vec![0u32, 1, 2]);

        let b = Int64Array::from(vec![7i64, 8, 9]);
        let col = IndexColumn::from_array(&b, "col").unwrap();
        assert_eq!(col.to_u32_vec("col").unwrap(), vec![7u32, 8, 9]);
    }

    #[test]
    fn index_column_negative_is_out_of_range() {
        let a = Int64Array::from(vec![0i64, -1]);
        let col = IndexColumn::from_array(&a, "row").unwrap();
        let e = col.to_u32_vec("row").unwrap_err();
        assert!(matches!(
            e,
            Error::IndexOutOfRange {
                row: 1,
                value: -1,
                ..
            }
        ));
    }

    #[test]
    fn index_column_overflow_is_out_of_range() {
        let too_big = u32::MAX as i64 + 1;
        let a = Int64Array::from(vec![too_big]);
        let col = IndexColumn::from_array(&a, "col").unwrap();
        let e = col.to_u32_vec("col").unwrap_err();
        assert!(matches!(
            e,
            Error::IndexOutOfRange {
                row: 0,
                column,
                value,
            } if column == "col" && value == too_big
        ));
    }

    #[test]
    fn index_column_wrong_type() {
        let a = Float64Array::from(vec![1.0]);
        let e = IndexColumn::from_array(&a, "row").unwrap_err();
        assert!(matches!(e, Error::WrongType { name, .. } if name == "row"));
    }

    #[test]
    fn index_column_null_is_unexpected() {
        let a = Int32Array::from(vec![Some(0), None, Some(2)]);
        let col = IndexColumn::from_array(&a, "row").unwrap();
        let e = col.to_u32_vec("row").unwrap_err();
        assert!(matches!(e, Error::UnexpectedNull { row: 1, .. }));
    }

    #[test]
    fn value_column_all_widths_to_f64() {
        let i32a = Int32Array::from(vec![1, 2]);
        assert_eq!(
            ValueColumn::from_array(&i32a, "val")
                .unwrap()
                .to_f64_vec("val")
                .unwrap(),
            vec![1.0, 2.0]
        );
        let i64a = Int64Array::from(vec![3i64, 4]);
        assert_eq!(
            ValueColumn::from_array(&i64a, "val")
                .unwrap()
                .to_f64_vec("val")
                .unwrap(),
            vec![3.0, 4.0]
        );
        let f64a = Float64Array::from(vec![2.9, 3.0]);
        assert_eq!(
            ValueColumn::from_array(&f64a, "val")
                .unwrap()
                .to_f64_vec("val")
                .unwrap(),
            vec![2.9, 3.0]
        );
    }

    #[test]
    fn value_column_wrong_type() {
        let a = StringArray::from(vec!["x"]);
        let e = ValueColumn::from_array(&a, "val").unwrap_err();
        assert!(matches!(e, Error::WrongType { name, .. } if name == "val"));
    }

    #[test]
    fn value_column_null_is_unexpected() {
        let a = Float64Array::from(vec![Some(1.0), None]);
        let col = ValueColumn::from_array(&a, "val").unwrap();
        let e = col.to_f64_vec("val").unwrap_err();
        assert!(matches!(e, Error::UnexpectedNull { row: 1, .. }));
    }

    #[test]
    fn str_column_utf8_and_large_utf8() {
        let a = StringArray::from(vec!["x", "y"]);
        let col = StrColumn::from_array(&a, "ids").unwrap();
        assert_eq!(col.value(0), "x");
        assert_eq!(col.to_string_vec("ids").unwrap(), vec!["x", "y"]);

        let b = LargeStringArray::from(vec!["p", "q"]);
        let col = StrColumn::from_array(&b, "ids").unwrap();
        assert_eq!(col.value(1), "q");
        assert_eq!(col.to_string_vec("ids").unwrap(), vec!["p", "q"]);
    }

    #[test]
    fn str_column_wrong_type() {
        let a = Int32Array::from(vec![1]);
        let e = StrColumn::from_array(&a, "ids").unwrap_err();
        assert!(matches!(e, Error::WrongType { name, .. } if name == "ids"));
    }

    #[test]
    fn str_column_null_is_unexpected_when_materialized() {
        let a = StringArray::from(vec![Some("a"), None]);
        let col = StrColumn::from_array(&a, "sample_id").unwrap();
        assert!(col.is_null(1));
        let e = col.to_string_vec("sample_id").unwrap_err();
        assert!(matches!(e, Error::UnexpectedNull { row: 1, .. }));
    }
}
