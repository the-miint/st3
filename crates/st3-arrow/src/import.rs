// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! COO + metadata import into core types.
//!
//! [`import`] is the one entry point: a COO `StructArray` (`row`/`col`/`val`
//! children), a feature-id array, and a per-sample metadata `RecordBatch`
//! (`sample_id`/`role`/`env`, design decision D14) become a validated
//! [`CountTable`] and its [`SampleContext`]. All boundary validation — missing
//! columns, wrong types, unexpected nulls, unknown roles, out-of-range indices —
//! happens here, before any core call; length, duplicate, bounds, and flooring
//! checks are core's ([`CountTable::from_coo`]) and surface as [`Error::Core`].

use arrow::array::{Array, RecordBatch, StructArray};
use st3_core::{CountTable, Role, SampleContext};

use crate::error::{Error, Result};
use crate::normalize::{IndexColumn, StrColumn, ValueColumn};

/// Label for the COO struct as a whole (used in the struct-null error message).
const COO_STRUCT: &str = "coo";
/// Field name of the COO feature (taxon) index child.
const COO_ROW: &str = "row";
/// Field name of the COO sample index child.
const COO_COL: &str = "col";
/// Field name of the COO count child.
const COO_VAL: &str = "val";

/// Column name for the feature-id array (used only in error messages).
const FEATURE_IDS: &str = "feature_ids";
/// Metadata column: sample identifier.
const META_SAMPLE_ID: &str = "sample_id";
/// Metadata column: per-sample role (`"source"` / `"sink"`).
const META_ROLE: &str = "role";
/// Metadata column: per-sample environment (nullable).
const META_ENV: &str = "env";

/// Import a COO batch, feature ids, and per-sample metadata into core types.
///
/// `coo` is a `StructArray` with `row` (feature index), `col` (sample index),
/// and `val` (count) children; `feature_ids` is a string array in row order;
/// `metadata` is a `RecordBatch` with `sample_id` (string), `role`
/// (`"source"`/`"sink"`), and `env` (nullable string) columns in sample-index
/// order. Integer widths and string flavors are normalized (design §6). Float
/// `val` is floored by [`CountTable::from_coo`]; this layer does not re-floor.
///
/// # Errors
/// [`Error::MissingColumn`] for an absent required column; [`Error::WrongType`]
/// for an unsupported column type; [`Error::UnexpectedNull`] for a null in a
/// required cell; [`Error::UnknownRole`] for a role other than source/sink;
/// [`Error::IndexOutOfRange`] for a `row`/`col` value outside `0..=u32::MAX`;
/// and [`Error::Core`] for any core validation failure (length mismatch,
/// duplicate id/coordinate, out-of-bounds coordinate, non-finite/negative/
/// overflowing value, empty axis, no sources).
pub fn import(
    coo: &StructArray,
    feature_ids: &dyn Array,
    metadata: &RecordBatch,
) -> Result<(CountTable, SampleContext)> {
    // A struct-level null entry keeps stale child values (Arrow does not zero a
    // child under a struct null), which the per-child null checks below cannot
    // see. Reject any null COO entry up front so such an entry never imports as a
    // phantom count.
    if coo.null_count() > 0 {
        let row = (0..coo.len()).find(|&i| coo.is_null(i)).unwrap_or(0);
        return Err(Error::UnexpectedNull {
            column: COO_STRUCT.to_string(),
            row,
        });
    }

    // All Arrow-boundary validation happens here, before any core call.
    // 1. COO children -> normalized parallel arrays (null/range-checked here).
    let rows = IndexColumn::from_array(coo_child(coo, COO_ROW)?, COO_ROW)?.to_u32_vec(COO_ROW)?;
    let cols = IndexColumn::from_array(coo_child(coo, COO_COL)?, COO_COL)?.to_u32_vec(COO_COL)?;
    let vals = ValueColumn::from_array(coo_child(coo, COO_VAL)?, COO_VAL)?.to_f64_vec(COO_VAL)?;

    // 2. Axis labels.
    let feature_ids =
        StrColumn::from_array(feature_ids, FEATURE_IDS)?.to_string_vec(FEATURE_IDS)?;
    let sample_ids =
        StrColumn::from_array(batch_column(metadata, META_SAMPLE_ID)?, META_SAMPLE_ID)?
            .to_string_vec(META_SAMPLE_ID)?;

    // 3. Per-sample roles and environments (also boundary-validated up front).
    let roles = parse_roles(batch_column(metadata, META_ROLE)?)?;
    let envs = parse_envs(batch_column(metadata, META_ENV)?)?;

    // 4. Core: build the table (validation + the single float->int floor), then
    //    the source/sink split (which cross-checks role/env lengths vs the table).
    let table = CountTable::from_coo(feature_ids, sample_ids, &rows, &cols, &vals)?;
    let ctx = SampleContext::for_table(&table, roles, envs)?;

    Ok((table, ctx))
}

/// Fetch a required COO child by field name.
fn coo_child<'a>(coo: &'a StructArray, name: &str) -> Result<&'a dyn Array> {
    coo.column_by_name(name)
        .map(|a| a.as_ref())
        .ok_or_else(|| Error::MissingColumn {
            name: name.to_string(),
        })
}

/// Fetch a required record-batch column by name.
fn batch_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a dyn Array> {
    batch
        .column_by_name(name)
        .map(|a| a.as_ref())
        .ok_or_else(|| Error::MissingColumn {
            name: name.to_string(),
        })
}

/// Parse the (non-null) `role` column into [`Role`]s.
fn parse_roles(array: &dyn Array) -> Result<Vec<Role>> {
    let col = StrColumn::from_array(array, META_ROLE)?;
    let n = col.len();
    let mut roles = Vec::with_capacity(n);
    for i in 0..n {
        if col.is_null(i) {
            return Err(Error::UnexpectedNull {
                column: META_ROLE.to_string(),
                row: i,
            });
        }
        let role = match col.value(i) {
            "source" => Role::Source,
            "sink" => Role::Sink,
            other => {
                return Err(Error::UnknownRole {
                    value: other.to_string(),
                    row: i,
                });
            }
        };
        roles.push(role);
    }
    Ok(roles)
}

/// Parse the (nullable) `env` column; a null becomes `None`.
fn parse_envs(array: &dyn Array) -> Result<Vec<Option<String>>> {
    let col = StrColumn::from_array(array, META_ENV)?;
    let n = col.len();
    let mut envs = Vec::with_capacity(n);
    for i in 0..n {
        if col.is_null(i) {
            envs.push(None);
        } else {
            envs.push(Some(col.value(i).to_string()));
        }
    }
    Ok(envs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Int32Array, Int64Array, StringArray};
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::{DataType, Field, Fields, Schema};

    /// A COO struct array with Int32 `row`/`col` and Int64 `val` children.
    fn coo_i32_i64(rows: Vec<i32>, cols: Vec<i32>, vals: Vec<i64>) -> StructArray {
        StructArray::from(vec![
            (
                Arc::new(Field::new(COO_ROW, DataType::Int32, false)),
                Arc::new(Int32Array::from(rows)) as ArrayRef,
            ),
            (
                Arc::new(Field::new(COO_COL, DataType::Int32, false)),
                Arc::new(Int32Array::from(cols)) as ArrayRef,
            ),
            (
                Arc::new(Field::new(COO_VAL, DataType::Int64, false)),
                Arc::new(Int64Array::from(vals)) as ArrayRef,
            ),
        ])
    }

    /// A metadata record batch with the D14 schema.
    fn meta(sample_ids: Vec<&str>, roles: Vec<&str>, envs: Vec<Option<&str>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(META_SAMPLE_ID, DataType::Utf8, false),
            Field::new(META_ROLE, DataType::Utf8, false),
            Field::new(META_ENV, DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(sample_ids)) as ArrayRef,
                Arc::new(StringArray::from(roles)) as ArrayRef,
                Arc::new(StringArray::from(envs)) as ArrayRef,
            ],
        )
        .unwrap()
    }

    fn feature_ids(ids: Vec<&str>) -> ArrayRef {
        Arc::new(StringArray::from(ids)) as ArrayRef
    }

    /// The canonical small dataset used by the happy-path round-trip.
    ///
    /// 2 features, 3 samples: s0 (source, envA), s1 (source, envB), s2 (sink).
    fn sample_dataset() -> (StructArray, ArrayRef, RecordBatch) {
        let coo = coo_i32_i64(vec![0, 1, 0, 1], vec![0, 0, 1, 2], vec![5, 3, 2, 7]);
        let fids = feature_ids(vec!["f0", "f1"]);
        let md = meta(
            vec!["s0", "s1", "s2"],
            vec!["source", "source", "sink"],
            vec![Some("envA"), Some("envB"), None],
        );
        (coo, fids, md)
    }

    #[test]
    fn import_matches_direct_core_construction() {
        let (coo, fids, md) = sample_dataset();
        let (table, ctx) = import(&coo, fids.as_ref(), &md).unwrap();

        let expected_table = CountTable::from_coo(
            vec!["f0".into(), "f1".into()],
            vec!["s0".into(), "s1".into(), "s2".into()],
            &[0, 1, 0, 1],
            &[0, 0, 1, 2],
            &[5.0, 3.0, 2.0, 7.0],
        )
        .unwrap();
        assert_eq!(table, expected_table);

        let expected_ctx = SampleContext::new(
            vec![Role::Source, Role::Source, Role::Sink],
            vec![Some("envA".into()), Some("envB".into()), None],
        )
        .unwrap();
        assert_eq!(ctx, expected_ctx);
    }

    #[test]
    fn import_accepts_large_utf8_ids() {
        use arrow::array::LargeStringArray;
        let (coo, _, md) = sample_dataset();
        let fids: ArrayRef = Arc::new(LargeStringArray::from(vec!["f0", "f1"]));
        let (table, _) = import(&coo, fids.as_ref(), &md).unwrap();
        assert_eq!(table.feature_ids(), &["f0".to_string(), "f1".to_string()]);
    }

    #[test]
    fn import_floors_float_val_via_core() {
        // val 2.9 -> 2, 0.5 -> dropped (floors to 0); core owns the floor.
        let coo = StructArray::from(vec![
            (
                Arc::new(Field::new(COO_ROW, DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![0, 1, 0])) as ArrayRef,
            ),
            (
                Arc::new(Field::new(COO_COL, DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![0, 0, 1])) as ArrayRef,
            ),
            (
                Arc::new(Field::new(COO_VAL, DataType::Float64, false)),
                Arc::new(arrow::array::Float64Array::from(vec![2.9, 3.0, 0.5])) as ArrayRef,
            ),
        ]);
        let fids = feature_ids(vec!["f0", "f1"]);
        let md = meta(
            vec!["s0", "s1"],
            vec!["source", "sink"],
            vec![Some("envA"), None],
        );
        let (table, _) = import(&coo, fids.as_ref(), &md).unwrap();
        assert_eq!(table.column(0), (&[0u32, 1][..], &[2u32, 3][..]));
        assert_eq!(table.column(1), (&[][..], &[][..]));
    }

    #[test]
    fn err_missing_coo_child() {
        // A struct array with row/col but no val.
        let coo = StructArray::from(vec![
            (
                Arc::new(Field::new(COO_ROW, DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![0])) as ArrayRef,
            ),
            (
                Arc::new(Field::new(COO_COL, DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![0])) as ArrayRef,
            ),
        ]);
        let fids = feature_ids(vec!["f0"]);
        let md = meta(vec!["s0"], vec!["source"], vec![Some("envA")]);
        let e = import(&coo, fids.as_ref(), &md).unwrap_err();
        assert!(matches!(e, Error::MissingColumn { name } if name == "val"));
    }

    #[test]
    fn err_val_wrong_type() {
        let coo = StructArray::from(vec![
            (
                Arc::new(Field::new(COO_ROW, DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![0])) as ArrayRef,
            ),
            (
                Arc::new(Field::new(COO_COL, DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![0])) as ArrayRef,
            ),
            (
                Arc::new(Field::new(COO_VAL, DataType::Utf8, false)),
                Arc::new(StringArray::from(vec!["nope"])) as ArrayRef,
            ),
        ]);
        let fids = feature_ids(vec!["f0"]);
        let md = meta(vec!["s0"], vec!["source"], vec![Some("envA")]);
        let e = import(&coo, fids.as_ref(), &md).unwrap_err();
        assert!(matches!(e, Error::WrongType { name, .. } if name == "val"));
    }

    #[test]
    fn err_null_in_row() {
        let coo = StructArray::from(vec![
            (
                Arc::new(Field::new(COO_ROW, DataType::Int32, true)),
                Arc::new(Int32Array::from(vec![Some(0), None])) as ArrayRef,
            ),
            (
                Arc::new(Field::new(COO_COL, DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![0, 1])) as ArrayRef,
            ),
            (
                Arc::new(Field::new(COO_VAL, DataType::Int64, false)),
                Arc::new(Int64Array::from(vec![1i64, 2])) as ArrayRef,
            ),
        ]);
        let fids = feature_ids(vec!["f0"]);
        let md = meta(
            vec!["s0", "s1"],
            vec!["source", "sink"],
            vec![Some("envA"), None],
        );
        let e = import(&coo, fids.as_ref(), &md).unwrap_err();
        assert!(matches!(e, Error::UnexpectedNull { row: 1, column } if column == "row"));
    }

    #[test]
    fn err_missing_role_column() {
        let (coo, fids, _) = sample_dataset();
        let schema = Arc::new(Schema::new(vec![
            Field::new(META_SAMPLE_ID, DataType::Utf8, false),
            Field::new(META_ENV, DataType::Utf8, true),
        ]));
        let md = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["s0", "s1", "s2"])) as ArrayRef,
                Arc::new(StringArray::from(vec![Some("envA"), Some("envB"), None])) as ArrayRef,
            ],
        )
        .unwrap();
        let e = import(&coo, fids.as_ref(), &md).unwrap_err();
        assert!(matches!(e, Error::MissingColumn { name } if name == "role"));
    }

    #[test]
    fn err_unknown_role() {
        let (coo, fids, _) = sample_dataset();
        let md = meta(
            vec!["s0", "s1", "s2"],
            vec!["source", "donor", "sink"],
            vec![Some("envA"), Some("envB"), None],
        );
        let e = import(&coo, fids.as_ref(), &md).unwrap_err();
        assert!(matches!(e, Error::UnknownRole { value, row: 1 } if value == "donor"));
    }

    #[test]
    fn err_index_out_of_range_surfaces_before_core() {
        // A col index that does not fit u32 must be caught at the boundary.
        let too_big = u32::MAX as i64 + 1;
        let coo = StructArray::from(vec![
            (
                Arc::new(Field::new(COO_ROW, DataType::Int64, false)),
                Arc::new(Int64Array::from(vec![0i64])) as ArrayRef,
            ),
            (
                Arc::new(Field::new(COO_COL, DataType::Int64, false)),
                Arc::new(Int64Array::from(vec![too_big])) as ArrayRef,
            ),
            (
                Arc::new(Field::new(COO_VAL, DataType::Int64, false)),
                Arc::new(Int64Array::from(vec![1i64])) as ArrayRef,
            ),
        ]);
        let fids = feature_ids(vec!["f0"]);
        let md = meta(vec!["s0"], vec!["source"], vec![Some("envA")]);
        let e = import(&coo, fids.as_ref(), &md).unwrap_err();
        assert!(matches!(e, Error::IndexOutOfRange { column, .. } if column == "col"));
    }

    #[test]
    fn err_struct_level_null_entry() {
        // Entry 1 is struct-level null; its children hold stale non-null values.
        // Without the up-front guard those would import as a phantom count.
        let fields: Fields = vec![
            Arc::new(Field::new(COO_ROW, DataType::Int32, false)),
            Arc::new(Field::new(COO_COL, DataType::Int32, false)),
            Arc::new(Field::new(COO_VAL, DataType::Int64, false)),
        ]
        .into();
        let coo = StructArray::try_new(
            fields,
            vec![
                Arc::new(Int32Array::from(vec![0, 0])) as ArrayRef,
                Arc::new(Int32Array::from(vec![0, 0])) as ArrayRef,
                Arc::new(Int64Array::from(vec![5i64, 9])) as ArrayRef,
            ],
            Some(NullBuffer::from(vec![true, false])),
        )
        .unwrap();
        let fids = feature_ids(vec!["f0"]);
        let md = meta(
            vec!["s0", "s1"],
            vec!["source", "sink"],
            vec![Some("envA"), None],
        );
        let e = import(&coo, fids.as_ref(), &md).unwrap_err();
        assert!(matches!(e, Error::UnexpectedNull { row: 1, column } if column == COO_STRUCT));
    }

    #[test]
    fn err_core_failure_wraps() {
        // A col index within u32 but beyond the sample axis -> core IndexOutOfBounds.
        let coo = coo_i32_i64(vec![0], vec![9], vec![1]);
        let fids = feature_ids(vec!["f0"]);
        let md = meta(vec!["s0"], vec!["source"], vec![Some("envA")]);
        let e = import(&coo, fids.as_ref(), &md).unwrap_err();
        assert!(matches!(e, Error::Core(_)));
    }
}
