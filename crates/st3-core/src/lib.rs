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
//! sink, and collate the ensembles into a [`SourceMixing`] of mixing means,
//! per-draw standard deviations (Eq. 7), and optional source × taxon tallies.
//!
//! [`predict_loo`] is the complementary leave-one-out driver: for each source
//! sample in turn it holds that sample out, re-collapses the remaining sources,
//! and estimates the held-out sample's composition over the source-scoped frame —
//! a per-sample self-consistency check reusing the same [`SourceMixing`] output.
//!
//! [`simulate_two_source`] and [`run_two_source_recovery`] are the
//! statistical-equivalence harness: they generate two-source Dirichlet mixtures
//! with known mixing weights and score recovered-vs-true proportions (R² related
//! to the Jensen–Shannon divergence between sources), turning "verifiable
//! statistical equivalence" into an enforced test.
//!
//! # Example
//!
//! Build a table from COO triples, label each sample, and estimate a sink's
//! source composition end to end:
//!
//! ```
//! use st3_core::{
//!     CollapseMethod, CountTable, GibbsParams, Role, SampleContext, predict_sinks,
//! };
//!
//! // Two source environments (envA on f0, envB on f1) and one sink dominated by f0.
//! let table = CountTable::from_coo(
//!     vec!["f0".into(), "f1".into()],
//!     vec!["a".into(), "b".into(), "sink".into()],
//!     &[0, 1, 0, 1],
//!     &[0, 1, 2, 2],
//!     &[100.0, 100.0, 90.0, 10.0],
//! )?;
//! let ctx = SampleContext::new(
//!     vec![Role::Source, Role::Source, Role::Sink],
//!     vec![Some("envA".into()), Some("envB".into()), None],
//! )?;
//! // Tiny, fast, deterministic sampler settings for the example.
//! let params = GibbsParams {
//!     restarts: 4,
//!     draws_per_restart: 2,
//!     burnin: 5,
//!     collapse: CollapseMethod::Sum,
//!     ..GibbsParams::default()
//! };
//! let mixing = predict_sinks(&table, &ctx, &params, 42, 1)?;
//!
//! // One row per sink; columns are the source environments then `Unknown`.
//! assert_eq!(mixing.env_names(), &["envA", "envB", "Unknown"]);
//! let row = mixing.mean_row(0);
//! assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
//! # Ok::<(), st3_core::Error>(())
//! ```

#![deny(missing_docs)]

// Modules are private; the crate's public surface is exactly the curated set of
// re-exports below (one path per item, mirroring st3-arrow and st3-capi).
mod collapse;
mod collate;
mod cp;
mod error;
mod estimate;
mod eval;
mod loo;
mod metadata;
mod parallel;
mod params;
mod predict;
mod rarefy;
mod rng;
mod table;

pub use collapse::{collapse_sources, collapse_subset, CollapseMethod, CollapsedSources};
pub use collate::SourceMixing;
pub use cp::ConditionalProbability;
pub use error::{Axis, Error, Result};
pub use estimate::{CooTally, GibbsEstimator, SinkEstimate, SinkModel, SinkVec};
pub use eval::{
    run_two_source_recovery, simulate_two_source, RecoveryScore, SimConfig, Simulation,
};
pub use loo::predict_loo;
pub use metadata::{Role, SampleContext};
pub use params::GibbsParams;
pub use predict::predict_sinks;
pub use rarefy::{rarefy, rarefy_per_sample, Rarefied, RarefyConfig, SampleStatus};
pub use rng::{rng_for_item, ItemRng};
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
