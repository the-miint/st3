// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Arrow C Data Interface marshaling — the one place with `unsafe`.
//!
//! This module bridges the raw C Data Interface (`FFI_ArrowArray` /
//! `FFI_ArrowSchema` / `FFI_ArrowArrayStream`) and the safe arrow-rs conversions
//! in [`st3_arrow`]. Import moves a borrowed FFI array into a safe `ArrayRef`;
//! export writes a computed `RecordBatch` (or a streamed reader) back out into
//! caller-provided FFI slots. All pointer validation and the resulting `unsafe`
//! reads/writes live here so the entry points in `lib.rs` stay declarative.
//!
//! ## Ownership at import
//! Following the C Data Interface, an imported **array** is *consumed*: this
//! reads it by value and the resulting `ArrayRef` owns it, releasing it on drop.
//! The caller must relinquish the array and not release it again. An imported
//! **schema** is only *borrowed*; the caller retains and releases it.

use arrow::array::{Array, ArrayRef, RecordBatch, StructArray, make_array};
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi, to_ffi};
use arrow::ffi_stream::FFI_ArrowArrayStream;
use st3_arrow::ContingencyReader;
use st3_core::{CountTable, SampleContext};

use crate::handle::require_ptr;
use crate::last_error::set_last_error;
use crate::panic::PanicGuardReader;
use crate::status::{St3Status, status_of_arrow};

/// Convert an already-owned FFI array (with its still-borrowed schema) into a
/// safe [`ArrayRef`].
///
/// The array is consumed by value: on success the returned `ArrayRef` owns it, and
/// on failure `from_ffi` drops (releases) it. Either way it is released exactly
/// once. The schema is only borrowed.
///
/// # Safety
/// `ffi_array` must be a valid C Data Interface array and `schema` a valid,
/// aligned pointer to its schema.
unsafe fn ffi_to_array(
    ffi_array: FFI_ArrowArray,
    schema: *const FFI_ArrowSchema,
) -> Result<ArrayRef, St3Status> {
    let ffi_schema = unsafe { &*schema };
    let data = unsafe { from_ffi(ffi_array, ffi_schema) }.map_err(|e| {
        set_last_error(format!("failed to import Arrow array: {e}"));
        St3Status::ErrInvalidInput
    })?;
    Ok(make_array(data))
}

/// Downcast an imported array to a [`StructArray`], or fail with a message.
fn as_struct<'a>(array: &'a ArrayRef, name: &str) -> Result<&'a StructArray, St3Status> {
    array.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
        set_last_error(format!(
            "{name} must be a struct array, got {:?}",
            array.data_type()
        ));
        St3Status::ErrInvalidInput
    })
}

/// Import the three FFI pairs (COO, feature ids, metadata) into core types.
///
/// Validates every pointer, imports each array (consuming it — see the module
/// docs), downcasts the COO and metadata to struct arrays, and delegates the
/// domain validation to [`st3_arrow::import`].
///
/// # Safety
/// Each non-null pointer must reference a valid C Data Interface array/schema.
/// Once all six pointers are validated, all three arrays are *consumed* — they are
/// moved out of the caller's memory up front (before any conversion), so every one
/// is released exactly once whether the import succeeds or fails. The caller must
/// relinquish the arrays and not release them again.
pub(crate) unsafe fn import_dataset(
    coo: *const FFI_ArrowArray,
    coo_schema: *const FFI_ArrowSchema,
    feature_ids: *const FFI_ArrowArray,
    feature_ids_schema: *const FFI_ArrowSchema,
    metadata: *const FFI_ArrowArray,
    metadata_schema: *const FFI_ArrowSchema,
) -> Result<(CountTable, SampleContext), St3Status> {
    require_ptr(coo, "coo")?;
    require_ptr(coo_schema, "coo_schema")?;
    require_ptr(feature_ids, "feature_ids")?;
    require_ptr(feature_ids_schema, "feature_ids_schema")?;
    require_ptr(metadata, "metadata")?;
    require_ptr(metadata_schema, "metadata_schema")?;

    // Phase 1 — take ownership of all three arrays up front. From here on every
    // one is an owned Rust value, so any early return below drops (releases) the
    // ones not yet converted. This makes consumption deterministic: exactly once
    // each, regardless of where a later conversion fails.
    // SAFETY: the pointers were just validated non-null and aligned; the caller
    // guarantees they reference valid C Data Interface arrays.
    let coo_ffi = unsafe { std::ptr::read(coo) };
    let feature_ids_ffi = unsafe { std::ptr::read(feature_ids) };
    let metadata_ffi = unsafe { std::ptr::read(metadata) };

    // Phase 2 — convert each (borrowing its schema). A failure here drops the
    // still-owned arrays above, releasing them.
    // SAFETY: each FFI array is owned and valid; each schema pointer is valid.
    let coo_arr = unsafe { ffi_to_array(coo_ffi, coo_schema) }?;
    let feature_ids_arr = unsafe { ffi_to_array(feature_ids_ffi, feature_ids_schema) }?;
    let metadata_arr = unsafe { ffi_to_array(metadata_ffi, metadata_schema) }?;

    let coo_struct = as_struct(&coo_arr, "coo")?;
    let metadata_struct = as_struct(&metadata_arr, "metadata")?;
    let metadata_batch = RecordBatch::from(metadata_struct);

    st3_arrow::import(coo_struct, feature_ids_arr.as_ref(), &metadata_batch).map_err(|e| {
        set_last_error(&e);
        status_of_arrow(&e)
    })
}

/// Export a computed `RecordBatch` into caller-provided FFI slots.
///
/// The batch is converted to a `StructArray` and its data written out as an
/// `(FFI_ArrowArray, FFI_ArrowSchema)` pair. On success the caller owns both and
/// must release them.
///
/// # Safety
/// `out_array` and `out_schema` must be valid, writable, aligned pointers to
/// *uninitialized* `FFI_ArrowArray` / `FFI_ArrowSchema` slots (their prior
/// contents, if any, are overwritten without being dropped).
pub(crate) unsafe fn export_batch(
    batch: RecordBatch,
    out_array: *mut FFI_ArrowArray,
    out_schema: *mut FFI_ArrowSchema,
) -> Result<(), St3Status> {
    let data = StructArray::from(batch).into_data();
    let (ffi_array, ffi_schema) = to_ffi(&data).map_err(|e| {
        set_last_error(format!("failed to export Arrow array: {e}"));
        St3Status::ErrInvalidInput
    })?;
    // SAFETY: the slots are valid and writable; `ptr::write` moves the values in
    // without dropping the prior (uninitialized) contents.
    unsafe {
        std::ptr::write(out_array, ffi_array);
        std::ptr::write(out_schema, ffi_schema);
    }
    Ok(())
}

/// Export a per-sink contingency reader as an FFI array stream into a caller slot.
///
/// The reader is wrapped in a [`PanicGuardReader`] so a panic in its lazily
/// driven `next` cannot unwind across the C boundary. On success the caller owns
/// the stream and must release it.
///
/// # Safety
/// `out_stream` must be a valid, writable, aligned pointer to an *uninitialized*
/// `FFI_ArrowArrayStream` slot.
pub(crate) unsafe fn export_stream(
    reader: ContingencyReader,
    out_stream: *mut FFI_ArrowArrayStream,
) {
    let stream = FFI_ArrowArrayStream::new(Box::new(PanicGuardReader::new(reader)));
    // SAFETY: the slot is valid and writable; `ptr::write` does not drop prior
    // (uninitialized) contents.
    unsafe { std::ptr::write(out_stream, stream) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow::array::{ArrayRef as AArrayRef, Int32Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field};
    use arrow::ffi::to_ffi;

    /// A COO struct array (Int32 row/col, Int64 val) for the canonical dataset.
    fn coo_struct() -> StructArray {
        StructArray::from(vec![
            (
                Arc::new(Field::new("row", DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![0, 1, 0, 1])) as AArrayRef,
            ),
            (
                Arc::new(Field::new("col", DataType::Int32, false)),
                Arc::new(Int32Array::from(vec![0, 0, 1, 2])) as AArrayRef,
            ),
            (
                Arc::new(Field::new("val", DataType::Int64, false)),
                Arc::new(Int64Array::from(vec![5, 3, 2, 7])) as AArrayRef,
            ),
        ])
    }

    /// The metadata struct array with the D14 schema.
    fn metadata_struct() -> StructArray {
        StructArray::from(vec![
            (
                Arc::new(Field::new("sample_id", DataType::Utf8, false)),
                Arc::new(StringArray::from(vec!["s0", "s1", "s2"])) as AArrayRef,
            ),
            (
                Arc::new(Field::new("role", DataType::Utf8, false)),
                Arc::new(StringArray::from(vec!["source", "source", "sink"])) as AArrayRef,
            ),
            (
                Arc::new(Field::new("env", DataType::Utf8, true)),
                Arc::new(StringArray::from(vec![Some("envA"), Some("envB"), None])) as AArrayRef,
            ),
        ])
    }

    fn feature_ids_array() -> StringArray {
        StringArray::from(vec!["f0", "f1"])
    }

    /// Turn a struct/array into an owned FFI pair via `to_ffi`.
    fn to_ffi_pair(array: &dyn Array) -> (FFI_ArrowArray, FFI_ArrowSchema) {
        let data = array.to_data();
        to_ffi(&data).expect("to_ffi succeeds")
    }

    /// The core-direct reference import from the same logical data.
    fn core_direct() -> (CountTable, SampleContext) {
        let coo = coo_struct();
        let fids = feature_ids_array();
        let md = RecordBatch::from(&metadata_struct());
        st3_arrow::import(&coo, &fids, &md).unwrap()
    }

    #[test]
    fn import_dataset_matches_core_direct() {
        let (coo_a, coo_s) = to_ffi_pair(&coo_struct());
        let (fid_a, fid_s) = to_ffi_pair(&feature_ids_array());
        let (md_a, md_s) = to_ffi_pair(&metadata_struct());

        // SAFETY: all six pointers are valid; the three arrays are consumed, so
        // they are forgotten below to relinquish ownership to the import.
        let result = unsafe { import_dataset(&coo_a, &coo_s, &fid_a, &fid_s, &md_a, &md_s) };
        std::mem::forget(coo_a);
        std::mem::forget(fid_a);
        std::mem::forget(md_a);

        let (table, ctx) = result.expect("import succeeds");
        let (exp_table, exp_ctx) = core_direct();
        assert_eq!(table, exp_table);
        assert_eq!(ctx, exp_ctx);
    }

    #[test]
    fn null_coo_pointer_is_invalid_input() {
        // A null `coo` fails `require_ptr` before any array is imported, so no
        // array is consumed and everything drops normally at end of scope.
        let (coo_a, coo_s) = to_ffi_pair(&coo_struct());
        let (fid_a, fid_s) = to_ffi_pair(&feature_ids_array());
        let (md_a, md_s) = to_ffi_pair(&metadata_struct());
        // SAFETY: `coo` is null; require_ptr rejects it before consuming anything.
        let result =
            unsafe { import_dataset(std::ptr::null(), &coo_s, &fid_a, &fid_s, &md_a, &md_s) };
        assert_eq!(result.unwrap_err(), St3Status::ErrInvalidInput);
        // Keep the built arrays alive until here so the null path is the only
        // reason for failure; they were not consumed.
        drop((coo_a, fid_a, md_a));
    }

    #[test]
    fn non_struct_coo_is_invalid_input() {
        // A plain string array is not a valid COO struct.
        let bad_coo = feature_ids_array();
        let (coo_a, coo_s) = to_ffi_pair(&bad_coo);
        let (fid_a, fid_s) = to_ffi_pair(&feature_ids_array());
        let (md_a, md_s) = to_ffi_pair(&metadata_struct());
        // SAFETY: all pointers valid; all three arrays consumed -> forgotten.
        let result = unsafe { import_dataset(&coo_a, &coo_s, &fid_a, &fid_s, &md_a, &md_s) };
        std::mem::forget(coo_a);
        std::mem::forget(fid_a);
        std::mem::forget(md_a);
        assert_eq!(result.unwrap_err(), St3Status::ErrInvalidInput);
    }

    #[test]
    fn missing_role_column_is_invalid_input_with_message() {
        crate::last_error::clear_last_error();
        // Metadata without a `role` column.
        let bad_md = StructArray::from(vec![
            (
                Arc::new(Field::new("sample_id", DataType::Utf8, false)),
                Arc::new(StringArray::from(vec!["s0", "s1", "s2"])) as AArrayRef,
            ),
            (
                Arc::new(Field::new("env", DataType::Utf8, true)),
                Arc::new(StringArray::from(vec![Some("envA"), Some("envB"), None])) as AArrayRef,
            ),
        ]);
        let (coo_a, coo_s) = to_ffi_pair(&coo_struct());
        let (fid_a, fid_s) = to_ffi_pair(&feature_ids_array());
        let (md_a, md_s) = to_ffi_pair(&bad_md);
        // SAFETY: all pointers valid; all three arrays consumed -> forgotten.
        let result = unsafe { import_dataset(&coo_a, &coo_s, &fid_a, &fid_s, &md_a, &md_s) };
        std::mem::forget(coo_a);
        std::mem::forget(fid_a);
        std::mem::forget(md_a);

        assert_eq!(result.unwrap_err(), St3Status::ErrInvalidInput);
        let ptr = crate::st3_last_error();
        assert!(!ptr.is_null(), "a descriptive last error must be set");
        let msg = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_str().unwrap();
        assert!(msg.contains("role"), "message was {msg:?}");
    }
}
