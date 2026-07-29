// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Panic isolation at the C boundary.
//!
//! An `extern "C"` function that unwinds is undefined behavior, so every entry
//! runs its body through [`ffi_guard`], which catches any panic, records the
//! message as the thread's last error, and returns [`St3Status::ErrPanic`].
//! [`PanicGuardReader`] extends the same protection to the lazily driven Arrow
//! stream: the consumer pulls batches by calling the reader's `next` through the
//! C Data Interface, and that call must not unwind either.

use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};

use arrow::array::{RecordBatch, RecordBatchReader};
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;

use crate::last_error::set_last_error;
use crate::status::St3Status;

/// Extract a human-readable message from a caught panic payload.
fn panic_message(payload: &(dyn Any + Send)) -> &str {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.as_str()
    } else {
        "unknown panic"
    }
}

/// Run a C ABI body, converting any panic into [`St3Status::ErrPanic`] plus a
/// last-error message. Every entry funnels through this, so a panic never
/// unwinds across the C boundary.
pub(crate) fn ffi_guard(body: impl FnOnce() -> St3Status) -> St3Status {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(status) => status,
        Err(payload) => {
            set_last_error(format!("panic: {}", panic_message(payload.as_ref())));
            St3Status::ErrPanic
        }
    }
}

/// Run a C ABI body that uses `?` (returning `Result<(), St3Status>`) under an
/// unwind guard, collapsing the result into the flat [`St3Status`] the ABI
/// returns. `Ok(())` becomes [`St3Status::Ok`]; `Err(status)` and any caught panic
/// become that nonzero status.
pub(crate) fn ffi_guard_result(body: impl FnOnce() -> Result<(), St3Status>) -> St3Status {
    ffi_guard(move || match body() {
        Ok(()) => St3Status::Ok,
        Err(status) => status,
    })
}

/// Run a void-returning C ABI body (a resource free) under an unwind guard.
///
/// A free cannot report a status, so a panic — never expected while dropping the
/// library's own plain-data handles — is caught and discarded rather than
/// allowed to unwind across the C boundary.
pub(crate) fn ffi_guard_void(body: impl FnOnce()) {
    let _ = catch_unwind(AssertUnwindSafe(body));
}

/// A [`RecordBatchReader`] wrapper that isolates panics in the inner reader's
/// `next`.
///
/// The exported contingency stream is driven lazily by the C consumer: each
/// `get_next` call reaches back into Rust and runs this `next` (arrow's exported
/// stream does not itself guard the callback). Wrapping that call in
/// [`catch_unwind`] converts a panic into an [`ArrowError::ExternalError`] the C
/// Data Interface can report, rather than an unwind across the boundary (which
/// would abort the process).
///
/// `next` is the only fallible callback and the only one guarded. `size_hint`
/// forwards trivially, and `schema` is likewise unguarded because the only reader
/// this wraps in practice — `st3_arrow::ContingencyReader` — returns its schema by
/// cloning a cached `SchemaRef` (an `Arc` bump), which cannot panic; a future
/// reader with a fallible `schema` would need the same treatment there.
pub(crate) struct PanicGuardReader<R>(R);

impl<R: RecordBatchReader> PanicGuardReader<R> {
    /// Wrap `inner` so its `next` is panic-isolated.
    pub(crate) fn new(inner: R) -> Self {
        Self(inner)
    }
}

impl<R: RecordBatchReader> Iterator for PanicGuardReader<R> {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        match catch_unwind(AssertUnwindSafe(|| self.0.next())) {
            Ok(item) => item,
            Err(payload) => {
                let msg = format!(
                    "panic in contingency reader: {}",
                    panic_message(payload.as_ref())
                );
                Some(Err(ArrowError::ExternalError(msg.into())))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<R: RecordBatchReader> RecordBatchReader for PanicGuardReader<R> {
    fn schema(&self) -> SchemaRef {
        self.0.schema()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::last_error::{clear_last_error, st3_last_error};

    #[test]
    fn ok_body_passes_status_through() {
        let status = ffi_guard(|| St3Status::Ok);
        assert_eq!(status, St3Status::Ok);
    }

    #[test]
    fn panicking_body_becomes_err_panic_with_message() {
        clear_last_error();
        let status = ffi_guard(|| panic!("kaboom"));
        assert_eq!(status, St3Status::ErrPanic);

        let ptr = st3_last_error();
        assert!(!ptr.is_null());
        let msg = unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_str()
            .unwrap()
            .to_string();
        assert!(msg.contains("panic"), "message was {msg:?}");
        assert!(msg.contains("kaboom"), "message was {msg:?}");
    }

    /// A reader whose `next` panics on the first pull, used to prove isolation.
    struct PanickingReader {
        schema: SchemaRef,
    }

    impl Iterator for PanickingReader {
        type Item = Result<RecordBatch, ArrowError>;
        fn next(&mut self) -> Option<Self::Item> {
            panic!("reader boom");
        }
    }

    impl RecordBatchReader for PanickingReader {
        fn schema(&self) -> SchemaRef {
            self.schema.clone()
        }
    }

    #[test]
    fn reader_panic_becomes_external_error_not_unwind() {
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let schema: SchemaRef =
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let mut guarded = PanicGuardReader::new(PanickingReader {
            schema: schema.clone(),
        });
        // schema forwards.
        assert_eq!(guarded.schema(), schema);
        // The inner panic is caught and surfaced as an Arrow error.
        match guarded.next() {
            Some(Err(ArrowError::ExternalError(e))) => {
                assert!(e.to_string().contains("reader boom"), "got {e}");
            }
            other => panic!("expected an external error, got {other:?}"),
        }
    }
}
