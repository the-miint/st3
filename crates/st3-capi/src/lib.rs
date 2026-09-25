// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! SourceTracker3 C ABI (cdylib + staticlib) over the Arrow C Data Interface.
//!
//! This crate exposes the parallel [`st3_core`] estimator to C/C++ consumers,
//! marshaling data across the Apache Arrow C Data Interface (design §4/§6/§13).
//! The dependency direction stays one-way — capi → arrow → core — so the core
//! never learns about Arrow or C.
//!
//! The surface is deliberately small and reentrant: opaque handles created and
//! destroyed by the library, a versioned [`St3Config`], a coarse [`St3Status`]
//! return plus a thread-local last-error string ([`st3_last_error`]), and an
//! unwind guard on every entry (including the lazily driven contingency stream)
//! so a panic can never cross the C boundary.

#![deny(missing_docs)]

mod config;
mod ffi_arrow;
mod handle;
mod last_error;
mod panic;
mod status;

use arrow::array::RecordBatch;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow::ffi_stream::FFI_ArrowArrayStream;
use st3_core::{predict_loo_rarefied, predict_sinks_rarefied, SourceMixing};

use crate::handle::require_ptr;
use crate::last_error::{clear_last_error, set_last_error};
use crate::panic::ffi_guard_result;
use crate::status::{status_of_arrow, status_of_core};

pub use config::{St3Collapse, St3Config, St3EstimatorKind, ST3_CONFIG_V1};
pub use handle::{st3_result_free, st3_table_free, St3Result, St3Table};
pub use last_error::st3_last_error;
pub use status::St3Status;

/// Returns the SourceTracker3 C ABI version.
///
/// The version is bumped only on a breaking change to the ABI (struct layout,
/// enum discriminants, or function signatures); additive changes keep it stable.
/// It currently returns `0` (the v1 surface has not shipped a breaking change).
/// Returns a compile-time constant and so needs no unwind guard.
#[unsafe(no_mangle)]
pub extern "C" fn st3_abi_version() -> u32 {
    0
}

/// Import a COO count table, feature ids, and per-sample metadata into a table
/// handle, over the Arrow C Data Interface.
///
/// Each input is a borrowed `(ArrowArray, ArrowSchema)` pair: `coo` is a struct
/// array with `row` (feature index), `col` (sample index), and `val` (count)
/// children; `feature_ids` is a string array in feature order; and `metadata` is
/// a struct array with `sample_id`, `role` (`"source"`/`"sink"`), and `env`
/// (nullable) columns in sample order. On success a new handle is
/// written to `*out` and must later be freed with `st3_table_free`. On failure
/// `*out` is set to null and the reason is available from `st3_last_error`.
///
/// # Safety
/// Every non-null array/schema pointer must reference a valid C Data Interface
/// structure, and `out` must be a valid, writable `*mut *mut St3Table`. When all
/// pointers are valid, each of the three arrays is *consumed* (the library
/// releases it); the caller must relinquish them and not release them again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn st3_table_from_arrow(
    coo: *const FFI_ArrowArray,
    coo_schema: *const FFI_ArrowSchema,
    feature_ids: *const FFI_ArrowArray,
    feature_ids_schema: *const FFI_ArrowSchema,
    metadata: *const FFI_ArrowArray,
    metadata_schema: *const FFI_ArrowSchema,
    out: *mut *mut St3Table,
) -> St3Status {
    clear_last_error();
    ffi_guard_result(|| {
        require_ptr(out, "out")?;
        // SAFETY: `out` was just validated non-null and aligned. Clear it so a
        // caller sees a defined null handle on any failure below.
        unsafe { *out = std::ptr::null_mut() };

        // SAFETY: the array/schema pointers are validated inside `import_dataset`;
        // the caller guarantees any non-null pointer references a valid structure.
        let (table, ctx) = unsafe {
            ffi_arrow::import_dataset(
                coo,
                coo_schema,
                feature_ids,
                feature_ids_schema,
                metadata,
                metadata_schema,
            )
        }?;
        let handle = Box::new(St3Table::new(table, ctx));
        // SAFETY: `out` validated above; we write exactly one owning pointer.
        unsafe { *out = Box::into_raw(handle) };
        Ok(())
    })
}

/// Run the configured estimator over an imported table (the safe inner body of
/// [`st3_run`]).
///
/// Lowers the config to core inputs and runs the rarefaction-aware sink or
/// leave-one-out driver, which applies the reference's rarefaction order itself.
fn run_inner(handle: &St3Table, config: &St3Config) -> Result<SourceMixing, St3Status> {
    let plan = config.to_core()?;
    let ds = handle.dataset();
    let mixing = if plan.loo {
        predict_loo_rarefied(
            &ds.table,
            &ds.ctx,
            &plan.params,
            &plan.rarefy,
            plan.seed,
            plan.jobs,
        )
    } else {
        predict_sinks_rarefied(
            &ds.table,
            &ds.ctx,
            &plan.params,
            &plan.rarefy,
            plan.seed,
            plan.jobs,
        )
    };
    mixing.map_err(|e| {
        set_last_error(&e);
        status_of_core(&e)
    })
}

/// Run source attribution over an imported table, producing a result handle.
///
/// `table` is a handle from `st3_table_from_arrow`; `config` is a versioned
/// `St3Config`. With `config.loo` set the run performs leave-one-out source
/// prediction, otherwise sink prediction. Rarefaction follows SourceTracker2:
/// in sink mode the sources are collapsed and each collapsed environment is then
/// subsampled to `source_rarefaction_depth`, while each sink is subsampled to
/// `sink_rarefaction_depth`; in leave-one-out mode each source sample is
/// subsampled to `source_rarefaction_depth` and the sink depth is ignored. A
/// depth of `0` disables that side. On success a new result handle is written to
/// `*out` and must later be freed with `st3_result_free`. On failure `*out` is
/// set to null and the reason is available from `st3_last_error`.
///
/// # Safety
/// `table` must be a live handle from `st3_table_from_arrow`, `config` a valid
/// pointer to an initialized `St3Config`, and `out` a valid, writable
/// `*mut *mut St3Result`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn st3_run(
    table: *const St3Table,
    config: *const St3Config,
    out: *mut *mut St3Result,
) -> St3Status {
    clear_last_error();
    ffi_guard_result(|| {
        require_ptr(out, "out")?;
        // SAFETY: `out` was just validated non-null and aligned. Clear it so a
        // caller sees a defined null handle on any failure below.
        unsafe { *out = std::ptr::null_mut() };

        require_ptr(table, "table")?;
        require_ptr(config, "config")?;
        // Validate the version/size prefix (reading only 8 bytes) before forming a
        // full `&St3Config`, so a differently-sized caller allocation is rejected
        // rather than over-read.
        // SAFETY: `config` validated non-null and aligned above.
        unsafe { config::validate_prefix(config) }?;

        // SAFETY: `table` validated non-null and aligned; the prefix check confirms
        // `config` points to a full St3Config. The caller guarantees both stay live
        // for the call.
        let handle = unsafe { &*table };
        let config = unsafe { &*config };

        let mixing = run_inner(handle, config)?;
        // SAFETY: `out` validated above; we write exactly one owning pointer.
        unsafe { *out = Box::into_raw(Box::new(St3Result::new(mixing))) };
        Ok(())
    })
}

/// Shared body of the dense exporters: validate the pointers, build the batch
/// with `build`, and marshal it into the caller's FFI slots.
fn export_dense(
    result: *const St3Result,
    out: *mut FFI_ArrowArray,
    out_schema: *mut FFI_ArrowSchema,
    build: fn(&SourceMixing) -> st3_arrow::Result<RecordBatch>,
) -> Result<(), St3Status> {
    require_ptr(out, "out")?;
    require_ptr(out_schema, "out_schema")?;
    // SAFETY: both validated non-null and aligned. Initialize the slots to a
    // released/empty state so they are defined even if a later step fails; the
    // slots are (per the contract) uninitialized, so nothing is dropped.
    unsafe {
        std::ptr::write(out, FFI_ArrowArray::empty());
        std::ptr::write(out_schema, FFI_ArrowSchema::empty());
    }
    require_ptr(result, "result")?;
    // SAFETY: `result` validated non-null and aligned; the caller guarantees it
    // is a live handle for the duration of the call.
    let mixing = unsafe { &*result }.mixing();
    let batch = build(mixing).map_err(|e| {
        set_last_error(&e);
        status_of_arrow(&e)
    })?;
    // SAFETY: `out`/`out_schema` validated above and point to writable slots.
    unsafe { ffi_arrow::export_batch(batch, out, out_schema) }
}

/// Export a result's mixing **means** as a dense Arrow record batch.
///
/// Writes an `(ArrowArray, ArrowSchema)` pair — a `sink_id` column plus one
/// `Float64` column per environment (`Unknown` last), one row per sink — into the
/// caller-provided slots. On success the caller owns and must release both.
///
/// # Safety
/// `result` must be a live handle from `st3_run`; `out` and `out_schema` must
/// be valid, writable, aligned pointers to uninitialized FFI slots.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn st3_result_means(
    result: *const St3Result,
    out: *mut FFI_ArrowArray,
    out_schema: *mut FFI_ArrowSchema,
) -> St3Status {
    clear_last_error();
    ffi_guard_result(|| export_dense(result, out, out_schema, st3_arrow::means_batch))
}

/// Export a result's mixing **standard deviations** as a dense Arrow record
/// batch, same shape as `st3_result_means`.
///
/// The values are the per-draw standard deviations, never the Python
/// reference's `×N_draws` quantity.
///
/// # Safety
/// As `st3_result_means`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn st3_result_stds(
    result: *const St3Result,
    out: *mut FFI_ArrowArray,
    out_schema: *mut FFI_ArrowSchema,
) -> St3Status {
    clear_last_error();
    ffi_guard_result(|| export_dense(result, out, out_schema, st3_arrow::stds_batch))
}

/// Export a result's per-sink source × taxon assignment tally as an Arrow array
/// stream.
///
/// Yields one flat-COO record batch per sink over the schema
/// `[sink, source, feature, value]`. Requires the run to have been
/// configured with `contingency = true`; otherwise returns
/// `ST3_STATUS_ERR_INVALID_INPUT`. On success the caller owns the stream and must
/// release it.
///
/// # Safety
/// `result` must be a live handle from `st3_run`; `out_stream` must be a valid,
/// writable, aligned pointer to an uninitialized `FFI_ArrowArrayStream` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn st3_result_contingency_stream(
    result: *const St3Result,
    out_stream: *mut FFI_ArrowArrayStream,
) -> St3Status {
    clear_last_error();
    ffi_guard_result(|| {
        require_ptr(out_stream, "out_stream")?;
        // SAFETY: validated non-null and aligned. Initialize to a released/empty
        // state so the slot is defined even if a later step fails.
        unsafe { std::ptr::write(out_stream, FFI_ArrowArrayStream::empty()) };
        require_ptr(result, "result")?;
        // SAFETY: `result` validated non-null and aligned; the caller guarantees
        // it is a live handle for the duration of the call.
        let mixing = unsafe { &*result }.mixing();
        match st3_arrow::contingency_reader(mixing) {
            Some(reader) => {
                // SAFETY: `out_stream` validated above and points to a writable slot.
                unsafe { ffi_arrow::export_stream(reader, out_stream) };
                Ok(())
            }
            None => {
                set_last_error(
                    "result has no contingency; set config.contingency = true before running",
                );
                Err(St3Status::ErrInvalidInput)
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    use st3_core::{
        predict_loo, predict_sinks, CollapseMethod, CountTable, GibbsParams, Role, SampleContext,
    };

    const SEED: u64 = 42;

    #[test]
    fn abi_version_is_zero() {
        assert_eq!(st3_abi_version(), 0);
    }

    /// Light, deterministic sampler params (fast; behaviour is seed-fixed).
    fn light_gibbs(collapse: CollapseMethod, contingency: bool) -> GibbsParams {
        GibbsParams {
            alpha1: 0.001,
            alpha2: 0.1,
            beta: 10.0,
            restarts: 6,
            draws_per_restart: 2,
            burnin: 5,
            delay: 1,
            collapse,
            contingency,
        }
    }

    /// Build an [`St3Config`] that lowers back to exactly `p` (no field drift).
    fn config_from_params(
        p: &GibbsParams,
        seed: u64,
        jobs: i32,
        loo: bool,
        source_depth: i32,
        sink_depth: i32,
        with_replacement: bool,
    ) -> St3Config {
        let collapse = match p.collapse {
            CollapseMethod::Mean => St3Collapse::Mean as i32,
            CollapseMethod::Sum => St3Collapse::Sum as i32,
        };
        St3Config {
            struct_version: ST3_CONFIG_V1,
            struct_size: size_of::<St3Config>() as u32,
            seed,
            jobs,
            source_rarefaction_depth: source_depth,
            sink_rarefaction_depth: sink_depth,
            with_replacement: with_replacement as u8,
            collapse,
            loo: loo as u8,
            contingency: p.contingency as u8,
            estimator: St3EstimatorKind::GibbsCollapsed as i32,
            alpha1: p.alpha1,
            alpha2: p.alpha2,
            beta: p.beta,
            restarts: p.restarts,
            draws_per_restart: p.draws_per_restart,
            burnin: p.burnin,
            delay: p.delay,
        }
    }

    /// A 3-env dataset: sources a/b/c, sinks s0/s1/s2, deep enough to rarefy.
    fn dataset() -> (CountTable, SampleContext) {
        let table = CountTable::from_coo(
            vec!["f0".into(), "f1".into(), "f2".into()],
            vec![
                "a".into(),
                "b".into(),
                "c".into(),
                "s0".into(),
                "s1".into(),
                "s2".into(),
            ],
            // (row, col, val) triples flattened below.
            &[0, 2, 0, 1, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            &[0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            &[
                80.0, 20.0, 90.0, 10.0, 95.0, 5.0, 60.0, 30.0, 10.0, 20.0, 20.0, 60.0, 40.0, 40.0,
                40.0,
            ],
        )
        .unwrap();
        let ctx = SampleContext::new(
            vec![
                Role::Source,
                Role::Source,
                Role::Source,
                Role::Sink,
                Role::Sink,
                Role::Sink,
            ],
            vec![
                Some("envA".into()),
                Some("envB".into()),
                Some("envC".into()),
                None,
                None,
                None,
            ],
        )
        .unwrap();
        (table, ctx)
    }

    /// Run `st3_run` through the C entry and return the owned result mixing.
    fn run_via_ffi(table: &CountTable, ctx: &SampleContext, config: &St3Config) -> SourceMixing {
        let handle = Box::into_raw(Box::new(St3Table::new(table.clone(), ctx.clone())));
        let mut out: *mut St3Result = std::ptr::null_mut();
        // SAFETY: `handle` is a live handle; `config`/`out` are valid pointers.
        let status = unsafe { st3_run(handle, config, &mut out) };
        assert_eq!(status, St3Status::Ok, "st3_run failed");
        assert!(!out.is_null());
        // SAFETY: `out` is the handle just produced by `st3_run`.
        let mixing = unsafe { &*out }.mixing().clone();
        // SAFETY: both handles came from `Box::into_raw` above and are freed once.
        unsafe {
            st3_result_free(out);
            st3_table_free(handle);
        }
        mixing
    }

    #[test]
    fn run_sink_matches_core_direct() {
        let (table, ctx) = dataset();
        let params = light_gibbs(CollapseMethod::Sum, false);
        let expected = predict_sinks(&table, &ctx, &params, SEED, 1).unwrap();
        let config = config_from_params(&params, SEED, 1, false, 0, 0, false);
        let got = run_via_ffi(&table, &ctx, &config);
        assert_eq!(got, expected);
    }

    #[test]
    fn run_sink_matches_core_direct_with_contingency() {
        let (table, ctx) = dataset();
        let params = light_gibbs(CollapseMethod::Sum, true);
        let expected = predict_sinks(&table, &ctx, &params, SEED, 1).unwrap();
        let config = config_from_params(&params, SEED, 1, false, 0, 0, false);
        let got = run_via_ffi(&table, &ctx, &config);
        assert_eq!(got, expected);
        assert!(got.contingency().is_some());
    }

    #[test]
    fn run_loo_matches_core_direct() {
        let (table, ctx) = dataset();
        let params = light_gibbs(CollapseMethod::Sum, false);
        let expected = predict_loo(&table, &ctx, &params, SEED, 1).unwrap();
        let config = config_from_params(&params, SEED, 1, true, 0, 0, false);
        let got = run_via_ffi(&table, &ctx, &config);
        assert_eq!(got, expected);
    }

    #[test]
    fn run_is_deterministic_across_job_counts() {
        let (table, ctx) = dataset();
        let params = light_gibbs(CollapseMethod::Sum, true);
        let serial = predict_sinks(&table, &ctx, &params, SEED, 1).unwrap();
        for jobs in [0i32, 2] {
            let config = config_from_params(&params, SEED, jobs, false, 0, 0, false);
            let got = run_via_ffi(&table, &ctx, &config);
            assert_eq!(got, serial, "jobs={jobs} diverged from serial");
        }
    }

    // The ABI's rarefaction depths lower onto the core's rarefaction-aware sink
    // driver (SourceTracker2's order: collapse, then subsample each collapsed
    // environment to the source depth; sinks per sample), not onto a hand-rolled
    // per-sample pass over the raw table.
    #[test]
    fn run_rarefaction_matches_core_direct_and_preserves_axis() {
        let (table, ctx) = dataset();
        let params = light_gibbs(CollapseMethod::Sum, false);
        // Rarefy sinks to depth 50, sources to 60 (both below their totals).
        let rarefy = st3_core::RarefyConfig {
            source_depth: Some(60),
            sink_depth: Some(50),
            with_replacement: false,
        };
        let expected = predict_sinks_rarefied(&table, &ctx, &params, &rarefy, SEED, 1).unwrap();

        let config = config_from_params(&params, SEED, 1, false, 60, 50, false);
        let got = run_via_ffi(&table, &ctx, &config);
        assert_eq!(got, expected);
        // Sample axis preserved: still three sinks.
        assert_eq!(got.n_sinks(), 3);
    }

    // Leave-one-out lowers onto the core's rarefaction-aware LOO driver (per
    // source sample; the sink depth plays no part).
    #[test]
    fn run_loo_rarefaction_matches_core_direct() {
        let (table, ctx) = dataset();
        let params = light_gibbs(CollapseMethod::Sum, false);
        let rarefy = st3_core::RarefyConfig {
            source_depth: Some(60),
            sink_depth: Some(50),
            with_replacement: true,
        };
        let expected = predict_loo_rarefied(&table, &ctx, &params, &rarefy, SEED, 1).unwrap();

        let config = config_from_params(&params, SEED, 1, true, 60, 50, true);
        let got = run_via_ffi(&table, &ctx, &config);
        assert_eq!(got, expected);
    }

    #[test]
    fn run_rejects_null_table() {
        let (table, ctx) = dataset();
        let params = light_gibbs(CollapseMethod::Sum, false);
        let config = config_from_params(&params, SEED, 1, false, 0, 0, false);
        let mut out: *mut St3Result = std::ptr::null_mut();
        clear_last_error();
        // SAFETY: table is null; the entry rejects it before any deref.
        let status = unsafe { st3_run(std::ptr::null(), &config, &mut out) };
        assert_eq!(status, St3Status::ErrInvalidInput);
        assert!(out.is_null());
        assert!(!st3_last_error().is_null());
        drop((table, ctx));
    }

    /// Wrap a mixing in a heap result handle for the exporter tests.
    fn result_handle(mixing: SourceMixing) -> *mut St3Result {
        Box::into_raw(Box::new(St3Result::new(mixing)))
    }

    /// Re-import an exported dense pair (consuming the array, borrowing the schema).
    fn reimport_batch(array: FFI_ArrowArray, schema: &FFI_ArrowSchema) -> RecordBatch {
        use arrow::array::{make_array, StructArray};
        // SAFETY: `array`/`schema` were produced by this library's own exporter.
        let data = unsafe { arrow::ffi::from_ffi(array, schema) }.unwrap();
        let arr = make_array(data);
        RecordBatch::from(arr.as_any().downcast_ref::<StructArray>().unwrap())
    }

    #[test]
    fn result_means_roundtrips_to_means_batch() {
        let (table, ctx) = dataset();
        let params = light_gibbs(CollapseMethod::Sum, false);
        let mixing = predict_sinks(&table, &ctx, &params, SEED, 1).unwrap();
        let expected = st3_arrow::means_batch(&mixing).unwrap();

        let handle = result_handle(mixing);
        let mut out_array = FFI_ArrowArray::empty();
        let mut out_schema = FFI_ArrowSchema::empty();
        // SAFETY: live handle; valid, writable, uninitialized FFI slots.
        let status = unsafe { st3_result_means(handle, &mut out_array, &mut out_schema) };
        assert_eq!(status, St3Status::Ok);

        let reimported = reimport_batch(out_array, &out_schema);
        assert_eq!(reimported, expected);

        // SAFETY: `handle` came from `Box::into_raw` above and is freed once.
        unsafe { st3_result_free(handle) };
    }

    #[test]
    fn result_stds_roundtrips_to_stds_batch() {
        let (table, ctx) = dataset();
        let params = light_gibbs(CollapseMethod::Sum, false);
        let mixing = predict_sinks(&table, &ctx, &params, SEED, 1).unwrap();
        let expected = st3_arrow::stds_batch(&mixing).unwrap();

        let handle = result_handle(mixing);
        let mut out_array = FFI_ArrowArray::empty();
        let mut out_schema = FFI_ArrowSchema::empty();
        // SAFETY: live handle; valid, writable, uninitialized FFI slots.
        let status = unsafe { st3_result_stds(handle, &mut out_array, &mut out_schema) };
        assert_eq!(status, St3Status::Ok);

        let reimported = reimport_batch(out_array, &out_schema);
        assert_eq!(reimported, expected);

        // SAFETY: `handle` came from `Box::into_raw` above and is freed once.
        unsafe { st3_result_free(handle) };
    }

    #[test]
    fn contingency_stream_roundtrips_to_reader() {
        use arrow::ffi_stream::ArrowArrayStreamReader;

        let (table, ctx) = dataset();
        let params = light_gibbs(CollapseMethod::Sum, true);
        let mixing = predict_sinks(&table, &ctx, &params, SEED, 1).unwrap();
        let expected: Vec<RecordBatch> = st3_arrow::contingency_reader(&mixing)
            .unwrap()
            .map(|b| b.unwrap())
            .collect();

        let handle = result_handle(mixing);
        let mut out_stream = FFI_ArrowArrayStream::empty();
        // SAFETY: live handle; valid, writable, uninitialized stream slot.
        let status = unsafe { st3_result_contingency_stream(handle, &mut out_stream) };
        assert_eq!(status, St3Status::Ok);

        // SAFETY: `out_stream` was just populated by the exporter; `from_raw`
        // moves it out and nulls the source, so `out_stream` drops safely after.
        let reader = unsafe { ArrowArrayStreamReader::from_raw(&mut out_stream) }.unwrap();
        let got: Vec<RecordBatch> = reader.map(|b| b.unwrap()).collect();
        assert_eq!(got, expected);
        // One contingency batch per sink; the dataset has three sinks.
        assert_eq!(got.len(), 3);

        // SAFETY: `handle` came from `Box::into_raw` above and is freed once.
        unsafe { st3_result_free(handle) };
    }

    #[test]
    fn contingency_stream_without_contingency_is_invalid_input() {
        let (table, ctx) = dataset();
        // contingency = false, so the result carries no tallies.
        let params = light_gibbs(CollapseMethod::Sum, false);
        let mixing = predict_sinks(&table, &ctx, &params, SEED, 1).unwrap();

        let handle = result_handle(mixing);
        let mut out_stream = FFI_ArrowArrayStream::empty();
        clear_last_error();
        // SAFETY: live handle; valid stream slot.
        let status = unsafe { st3_result_contingency_stream(handle, &mut out_stream) };
        assert_eq!(status, St3Status::ErrInvalidInput);
        assert!(!st3_last_error().is_null());

        // SAFETY: `handle` came from `Box::into_raw` above and is freed once.
        unsafe { st3_result_free(handle) };
    }
}
