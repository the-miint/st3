// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! SourceTracker3 Arrow boundary (COO <-> core conversion).
//!
//! This crate is the language-neutral data boundary between Apache Arrow and the
//! pure-Rust [`st3_core`] library. The dependency direction is one-way — Arrow
//! knows about core, core never learns about Arrow (design §4).
//!
//! - [`import`] turns a COO `StructArray` (`row`/`col`/`val`), a feature-id
//!   array, and a per-sample metadata `RecordBatch` (`sample_id`/`role`/`env`)
//!   into a validated [`st3_core::CountTable`] and [`st3_core::SampleContext`].
//!   Integer widths and string flavors are normalized on ingest; boundary
//!   validation is eager and descriptive ([`Error`]).
//! - [`means_batch`] / [`stds_batch`] export a [`st3_core::SourceMixing`] as
//!   dense `RecordBatch`es (a `sink_id` column plus one `Float64` column per
//!   environment).
//! - [`contingency_reader`] streams the per-sink source × taxon assignments as a
//!   [`ContingencyReader`] — one flat-COO `RecordBatch` per sink over the uniform
//!   schema `[sink, source, feature, value]`.
//!
//! Only safe arrow-rs types are used here; the Arrow C Data Interface marshaling
//! lives in the C ABI crate (a later milestone), driving these conversions.

mod error;
mod export;
mod import;
mod normalize;

pub use error::{Error, Result};
pub use export::{ContingencyReader, contingency_reader, means_batch, stds_batch};
pub use import::import;

/// Identifies this crate and the core it is built against. Stable anchor proving
/// the Arrow crate links the core crate.
pub fn backend_name() -> String {
    format!("st3-arrow over {}", st3_core::name())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrow_uses_core() {
        assert_eq!(backend_name(), "st3-arrow over st3-core");
    }
}
