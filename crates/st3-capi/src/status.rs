// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Status codes returned by every C ABI entry, and the mapping onto them.
//!
//! Every `extern "C"` entry returns an [`St3Status`]: [`St3Status::Ok`] (zero) on
//! success, a nonzero code otherwise. The code is a coarse category; the precise,
//! human-readable reason is available from [`st3_last_error`](crate::st3_last_error)
//! until the next call on the same thread.

use st3_arrow::Error as ArrowError;
use st3_core::Error as CoreError;

/// Result status of a C ABI call. `Ok` is zero; every failure is nonzero.
///
/// The discriminants are ABI-stable: new variants are appended, never reordered.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum St3Status {
    /// The call succeeded.
    Ok = 0,
    /// An input was malformed: an invalid configuration, a null or misaligned
    /// pointer, a bad Arrow schema/type/null, an unknown role, an out-of-range
    /// index, or any other core validation failure. The specifics are in the
    /// last-error string.
    ErrInvalidInput,
    /// A sample was too shallow for the requested rarefaction depth: in sink
    /// mode a collapsed source environment or a sink, in leave-one-out mode a
    /// source sample. The run is refused before any sampling, as SourceTracker2
    /// does; the last-error string names the role, how many samples fall
    /// short, and the shallowest total.
    ErrShallowSample,
    /// No sample was labeled as a source.
    ErrNoSources,
    /// Reserved. An allocation failed. Rust aborts on allocation failure rather
    /// than returning, so this is defined for ABI stability but not yet produced.
    ErrOom,
    /// A panic was caught at the boundary (a library bug). See the last-error
    /// string for the panic message.
    ErrPanic,
}

/// Map a core error onto a status.
///
/// [`CoreError::NoSources`] is distinguished as [`St3Status::ErrNoSources`] and
/// [`CoreError::ShallowSamples`] as [`St3Status::ErrShallowSample`]; every other
/// core failure is an [`St3Status::ErrInvalidInput`] whose specifics are carried
/// in the last-error string.
pub(crate) fn status_of_core(err: &CoreError) -> St3Status {
    match err {
        CoreError::NoSources => St3Status::ErrNoSources,
        CoreError::ShallowSamples { .. } => St3Status::ErrShallowSample,
        _ => St3Status::ErrInvalidInput,
    }
}

/// Map an Arrow-boundary error onto a status.
///
/// A wrapped core error ([`ArrowError::Core`]) delegates to [`status_of_core`];
/// every boundary or Arrow failure is an [`St3Status::ErrInvalidInput`].
pub(crate) fn status_of_arrow(err: &ArrowError) -> St3Status {
    match err {
        ArrowError::Core(e) => status_of_core(e),
        _ => St3Status::ErrInvalidInput,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_no_sources_maps_to_err_no_sources() {
        assert_eq!(
            status_of_core(&CoreError::NoSources),
            St3Status::ErrNoSources
        );
    }

    #[test]
    fn core_shallow_samples_maps_to_err_shallow_sample() {
        let e = CoreError::ShallowSamples {
            what: "sink",
            depth: 1000,
            count: 1,
            shallowest: 200,
        };
        assert_eq!(status_of_core(&e), St3Status::ErrShallowSample);
    }

    #[test]
    fn other_core_errors_map_to_invalid_input() {
        let cases = [
            CoreError::EmptySink { sample_index: 0 },
            CoreError::InvalidParam {
                name: "beta",
                reason: "must be finite",
            },
            CoreError::ThreadPool {
                reason: "no threads".into(),
            },
        ];
        for e in cases {
            assert_eq!(status_of_core(&e), St3Status::ErrInvalidInput);
        }
    }

    #[test]
    fn arrow_core_wrapper_delegates_to_core_mapping() {
        let wrapped = ArrowError::Core(CoreError::NoSources);
        assert_eq!(status_of_arrow(&wrapped), St3Status::ErrNoSources);
    }

    #[test]
    fn arrow_boundary_errors_map_to_invalid_input() {
        let cases = [
            ArrowError::MissingColumn {
                name: "role".into(),
            },
            ArrowError::UnknownRole {
                value: "donor".into(),
                row: 1,
            },
            ArrowError::Core(CoreError::EmptySink { sample_index: 2 }),
        ];
        for e in cases {
            assert_eq!(status_of_arrow(&e), St3Status::ErrInvalidInput);
        }
    }

    #[test]
    fn ok_is_zero() {
        assert_eq!(St3Status::Ok as i32, 0);
    }
}
