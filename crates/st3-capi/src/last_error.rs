// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Thread-local last-error string.
//!
//! Each entry returns only a coarse [`St3Status`](crate::St3Status); the precise
//! reason is recorded here, per thread, so the boundary stays reentrant with no
//! process-global mutable state. Every entry clears the slot on the way in and
//! sets it on failure; [`st3_last_error`] hands the caller a borrowed pointer to
//! the current thread's message (or null).

use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::fmt::Display;

thread_local! {
    /// The current thread's last error, or `None` if the last call succeeded.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Record `msg` as this thread's last error.
///
/// Interior NUL bytes are escaped (a C string cannot carry them), so no message
/// is ever silently truncated or dropped.
pub(crate) fn set_last_error(msg: impl Display) {
    let sanitized = msg.to_string().replace('\0', "\\0");
    // `CString::new` can only fail on an interior NUL, which was just escaped.
    let cstr = CString::new(sanitized).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(cstr));
}

/// Clear this thread's last error. Called at the start of every entry.
pub(crate) fn clear_last_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

/// Return the current thread's last-error message, or null if none is set.
///
/// The pointer borrows a thread-local buffer owned by the library. It stays
/// valid until the next SourceTracker3 call on the same thread; the caller must
/// not free it or use it from another thread.
#[unsafe(no_mangle)]
pub extern "C" fn st3_last_error() -> *const c_char {
    // A thread-local borrow-and-read cannot panic here (no reentrant borrow), so
    // this needs no unwind guard. The returned pointer outlives the borrow: the
    // `CString` is owned by the thread-local, not by the dropped `Ref`.
    LAST_ERROR.with(|e| match &*e.borrow() {
        Some(c) => c.as_ptr(),
        None => std::ptr::null(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_after_clear() {
        set_last_error("something");
        clear_last_error();
        assert!(st3_last_error().is_null());
    }

    #[test]
    fn returns_set_message() {
        clear_last_error();
        set_last_error("boom 42");
        let ptr = st3_last_error();
        assert!(!ptr.is_null());
        let msg = unsafe { std::ffi::CStr::from_ptr(ptr) };
        assert_eq!(msg.to_str().unwrap(), "boom 42");
    }

    #[test]
    fn interior_nul_is_escaped_not_dropped() {
        clear_last_error();
        set_last_error("a\0b");
        let ptr = st3_last_error();
        let msg = unsafe { std::ffi::CStr::from_ptr(ptr) };
        assert_eq!(msg.to_str().unwrap(), "a\\0b");
    }
}
