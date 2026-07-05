// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! SourceTracker3 core library (algorithm; no FFI, no Arrow, no I/O).
//!
//! The deterministic preprocessing front-end provides a sparse [`CountTable`],
//! the source/sink split ([`SampleContext`]), collapse of sources by
//! environment ([`collapse_sources`]), seeded rarefaction ([`rarefy()`]), and the
//! sampler configuration ([`GibbsParams`]).
//!
//! The estimator turns collapsed sources plus a sink into an ensemble of source
//! proportion vectors: [`ConditionalProbability`] precomputes the
//! depth-independent known-source term, and the pluggable [`SinkModel`] seam —
//! implemented by [`GibbsEstimator`] — runs the collapsed-Gibbs sampler per
//! sink, returning a [`SinkEstimate`].
//!
//! [`predict_sinks`] is the end-to-end sink-mode driver: collapse, estimate each
//! sink, and [`collate()`] the ensembles into a [`SourceMixing`] of mixing means,
//! per-draw standard deviations (Eq. 7), and optional source × taxon tallies.

pub mod collapse;
pub mod collate;
pub mod cp;
pub mod error;
pub mod estimate;
pub mod metadata;
pub mod params;
pub mod predict;
pub mod rarefy;
pub mod rng;
pub mod table;

pub use collapse::{CollapseMethod, CollapsedSources, collapse_sources, collapse_subset};
pub use collate::{SourceMixing, collate};
pub use cp::ConditionalProbability;
pub use error::{Axis, Error, Result};
pub use estimate::{CooTally, GibbsEstimator, SinkEstimate, SinkModel, SinkVec};
pub use metadata::{Role, SampleContext};
pub use params::GibbsParams;
pub use predict::predict_sinks;
pub use rarefy::{Rarefied, RarefyConfig, SampleStatus, rarefy, rarefy_per_sample};
pub use rng::{ItemRng, rng_for_item};
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
