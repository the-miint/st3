// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! SourceTracker3 core library (algorithm; no FFI, no Arrow, no I/O).
//!
//! M2 provides the deterministic preprocessing front-end: a sparse
//! [`CountTable`], the source/sink split ([`SampleContext`]), collapse of
//! sources by environment ([`collapse_sources`]), and the sampler
//! configuration ([`GibbsParams`]).

pub mod collapse;
pub mod error;
pub mod metadata;
pub mod params;
pub mod table;

pub use collapse::{CollapseMethod, CollapsedSources, collapse_sources, collapse_subset};
pub use error::{Axis, Error, Result};
pub use metadata::{Role, SampleContext};
pub use params::GibbsParams;
pub use table::{Count, CountTable, FeatureIdx};

/// Returns the crate name. Retained as a stable anchor used by sibling crates
/// and the benchmark harness.
pub fn name() -> &'static str {
    "st3-core"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_smoke() {
        assert_eq!(name(), "st3-core");
    }
}
