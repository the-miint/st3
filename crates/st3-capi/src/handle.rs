// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Opaque handles and their destructors.
//!
//! The library hands C two opaque pointers: [`St3Table`] (an imported count
//! table plus its sample context) and [`St3Result`] (a completed run's mixing
//! results). Their internals never cross the boundary — C sees only a pointer.
//! Each is heap-owned via `Box::into_raw` at creation and reclaimed by the
//! matching `*_free` function, which is null-tolerant and unwind-guarded.
//!
//! Handles hold plain `Box`-owned data, not `Arc`: `st3_run` borrows `&table` /
//! `&ctx` for the duration of one call and there is no cross-handle sharing to
//! amortize, so shared ownership would add cost without benefit (design §15,
//! "share by borrow, not Arc").

use st3_core::{CountTable, SampleContext, SourceMixing};

use crate::last_error::set_last_error;
use crate::panic::ffi_guard_void;
use crate::status::St3Status;

/// The imported dataset behind an [`St3Table`]: a count table and the source /
/// sink split over its sample axis.
pub(crate) struct Dataset {
    /// The sparse count table.
    pub table: CountTable,
    /// The source / sink role and environment split, aligned to `table`.
    pub ctx: SampleContext,
}

/// Opaque handle to an imported count table and its sample context.
///
/// Created by `st3_table_from_arrow` and destroyed by `st3_table_free`. It is
/// only ever passed by pointer.
pub struct St3Table(pub(crate) Dataset);

/// Opaque handle to a completed run's mixing results.
///
/// Created by `st3_run` and destroyed by `st3_result_free`.
pub struct St3Result(pub(crate) SourceMixing);

impl St3Table {
    /// Wrap an imported dataset in a handle.
    pub(crate) fn new(table: CountTable, ctx: SampleContext) -> Self {
        St3Table(Dataset { table, ctx })
    }

    /// Borrow the imported dataset.
    pub(crate) fn dataset(&self) -> &Dataset {
        &self.0
    }
}

impl St3Result {
    /// Wrap a run result in a handle.
    pub(crate) fn new(mixing: SourceMixing) -> Self {
        St3Result(mixing)
    }

    /// Borrow the mixing result.
    pub(crate) fn mixing(&self) -> &SourceMixing {
        &self.0
    }
}

/// Whether `ptr` is non-null and correctly aligned for `T`.
///
/// A debug sanity check for pointers crossing the boundary: it catches an
/// accidental null or an obviously garbled (misaligned) pointer. It cannot detect
/// use-after-free or a wild-but-aligned pointer — those are inherent limits of a
/// C FFI.
#[inline]
pub(crate) fn is_nonnull_aligned<T>(ptr: *const T) -> bool {
    !ptr.is_null() && (ptr as usize) % std::mem::align_of::<T>() == 0
}

/// Require a boundary pointer to be non-null and aligned, recording a
/// `"<name> pointer is null or misaligned"` last error and returning
/// [`St3Status::ErrInvalidInput`] otherwise. The single validation used by every
/// entry point, so the message stays uniform.
pub(crate) fn require_ptr<T>(ptr: *const T, name: &str) -> Result<(), St3Status> {
    if is_nonnull_aligned(ptr) {
        Ok(())
    } else {
        set_last_error(format!("{name} pointer is null or misaligned"));
        Err(St3Status::ErrInvalidInput)
    }
}

/// Free a table handle previously returned by `st3_table_from_arrow`.
///
/// A null pointer is a no-op.
///
/// # Safety
/// `handle` must be null or a pointer returned by `st3_table_from_arrow` that
/// has not already been freed. Passing any other pointer, or freeing twice, is
/// undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn st3_table_free(handle: *mut St3Table) {
    ffi_guard_void(|| {
        if !handle.is_null() {
            // SAFETY: a non-null `handle` was produced by `Box::into_raw` in
            // `st3_table_from_arrow` and, per the contract, has not been freed.
            drop(unsafe { Box::from_raw(handle) });
        }
    });
}

/// Free a result handle previously returned by `st3_run`.
///
/// A null pointer is a no-op.
///
/// # Safety
/// `handle` must be null or a pointer returned by `st3_run` that has not already
/// been freed. Passing any other pointer, or freeing twice, is undefined
/// behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn st3_result_free(handle: *mut St3Result) {
    ffi_guard_void(|| {
        if !handle.is_null() {
            // SAFETY: a non-null `handle` was produced by `Box::into_raw` in
            // `st3_run` and, per the contract, has not been freed.
            drop(unsafe { Box::from_raw(handle) });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use st3_core::{Role, SampleContext};

    /// A minimal dataset for exercising the handle lifecycle.
    fn tiny_dataset() -> (CountTable, SampleContext) {
        let table = CountTable::from_coo(
            vec!["f0".into(), "f1".into()],
            vec!["s0".into(), "s1".into()],
            &[0, 1],
            &[0, 1],
            &[5.0, 3.0],
        )
        .unwrap();
        let ctx = SampleContext::new(
            vec![Role::Source, Role::Sink],
            vec![Some("envA".into()), None],
        )
        .unwrap();
        (table, ctx)
    }

    #[test]
    fn table_free_null_is_a_no_op() {
        // SAFETY: a null pointer is explicitly a no-op.
        unsafe { st3_table_free(std::ptr::null_mut()) };
    }

    #[test]
    fn result_free_null_is_a_no_op() {
        // SAFETY: a null pointer is explicitly a no-op.
        unsafe { st3_result_free(std::ptr::null_mut()) };
    }

    #[test]
    fn table_roundtrip_create_and_free() {
        let (table, ctx) = tiny_dataset();
        let handle = Box::into_raw(Box::new(St3Table::new(table, ctx)));
        assert!(is_nonnull_aligned(handle));
        // Borrow through the handle before freeing.
        let ds = unsafe { &*handle }.dataset();
        assert_eq!(ds.table.n_samples(), 2);
        assert_eq!(ds.ctx.source_indices(), &[0]);
        // SAFETY: `handle` came from `Box::into_raw` above and is freed once.
        unsafe { st3_table_free(handle) };
    }

    #[test]
    fn is_nonnull_aligned_rejects_null_and_misaligned() {
        assert!(!is_nonnull_aligned(std::ptr::null::<u64>()));
        // A live but deliberately misaligned u64 pointer: one byte into an
        // 8-aligned buffer, so its address is never a multiple of 8.
        let aligned = [0u64; 2];
        let misaligned = unsafe { aligned.as_ptr().cast::<u8>().add(1) }.cast::<u64>();
        assert!(!is_nonnull_aligned(misaligned));
        // A real, aligned allocation is accepted.
        let boxed = Box::new(7u64);
        let ptr: *const u64 = &*boxed;
        assert!(is_nonnull_aligned(ptr));
    }
}
