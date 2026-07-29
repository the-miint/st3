// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! End-to-end Arrow boundary round-trip on the committed fixtures.
//!
//! For each fixture, the same data is taken down two paths: core-direct
//! (`CountTable::from_coo` + `SampleContext::for_table`) and arrow
//! (`import` of a COO `StructArray` + ids + metadata batch). The imported types
//! must equal the direct ones, and after `predict_sinks` the exported dense
//! means/stds batches must equal the batches from the core-direct result — i.e.
//! arrow→core→arrow reproduces core-direct exactly. `jobs = 1` pins a serial
//! run (the conversion is pure, so there is no RNG axis to sweep here).

mod common;

use common::{arrow_inputs, core_direct, fixtures_dir, load_matrix, load_metadata};
use st3_arrow::{import, means_batch, stds_batch};
use st3_core::{predict_sinks, CollapseMethod, GibbsParams};

/// Light, deterministic params — the round-trip checks equality of two runs of
/// the same deterministic pipeline, so heavy sampling would only slow the gate.
fn params() -> GibbsParams {
    GibbsParams {
        alpha1: 0.001,
        alpha2: 0.1,
        beta: 10.0,
        restarts: 8,
        draws_per_restart: 2,
        burnin: 10,
        delay: 1,
        collapse: CollapseMethod::Sum,
        contingency: false,
    }
}

fn roundtrip_matches_core_direct(fixture: &str) {
    let dir = fixtures_dir().join(fixture);
    let matrix = load_matrix(&dir.join("table.tsv"));
    let metadata = load_metadata(&dir.join("metadata.tsv"));

    // Core-direct reference.
    let (table_d, ctx_d) = core_direct(&matrix, &metadata);

    // Arrow path: build Arrow inputs from the same fixture, then import.
    let (coo, feature_ids, meta_batch) = arrow_inputs(&matrix, &metadata);
    let (table_a, ctx_a) = import(&coo, feature_ids.as_ref(), &meta_batch).expect("import");

    // Import reproduces the core types exactly.
    assert_eq!(table_a, table_d, "{fixture}: imported table != core-direct");
    assert_eq!(ctx_a, ctx_d, "{fixture}: imported context != core-direct");

    // Run the same deterministic estimation on each and export.
    let p = params();
    let mixing_d = predict_sinks(&table_d, &ctx_d, &p, 42, 1).expect("core-direct predict");
    let mixing_a = predict_sinks(&table_a, &ctx_a, &p, 42, 1).expect("arrow predict");
    assert_eq!(mixing_a, mixing_d, "{fixture}: mixing differs after import");

    let means_d = means_batch(&mixing_d).expect("means core-direct");
    let means_a = means_batch(&mixing_a).expect("means arrow");
    assert_eq!(means_a, means_d, "{fixture}: exported means batch differs");

    let stds_d = stds_batch(&mixing_d).expect("stds core-direct");
    let stds_a = stds_batch(&mixing_a).expect("stds arrow");
    assert_eq!(stds_a, stds_d, "{fixture}: exported stds batch differs");

    // Sanity on the exported shape: one row per sink, sink_id + one col per env.
    assert_eq!(means_a.num_rows(), mixing_a.n_sinks());
    assert_eq!(means_a.num_columns(), 1 + mixing_a.n_envs());
}

#[test]
fn tiny_test_roundtrip() {
    roundtrip_matches_core_direct("tiny_test");
}

#[test]
fn synthetic_small_roundtrip() {
    roundtrip_matches_core_direct("synthetic_small");
}
