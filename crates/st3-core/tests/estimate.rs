// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! End-to-end estimator check against a committed reference oracle.
//!
//! Runs the collapsed-Gibbs estimator on the `synthetic_small` fixture with the
//! pinned reference parameters (mean collapse, no rarefaction) and compares the
//! per-sink mean source proportions to `expected_sink_mean.tsv`. Our sampler uses
//! a Xoshiro256++ stream that does not bit-match the reference's NumPy stream, so
//! this is a *statistical* equivalence check: with 100 restarts × 10 draws the
//! mean proportions converge to the oracle within an absolute tolerance of 0.02.

mod common;

use common::{
    build_context, build_table, fixtures_dir, load_matrix, load_metadata, reference_params,
};
use st3_core::{
    collapse_sources, rng_for_item, CollapseMethod, GibbsEstimator, SinkEstimate, SinkModel,
    SinkVec,
};

/// Mean of the ensemble's proportion vectors (length `v`).
fn mean_ensemble(est: &SinkEstimate, v: usize) -> Vec<f64> {
    let mut acc = vec![0.0; v];
    for vec in est.ensemble() {
        for (a, &x) in acc.iter_mut().zip(vec.iter()) {
            *a += x;
        }
    }
    let n = est.ensemble().len() as f64;
    acc.iter().map(|&s| s / n).collect()
}

#[test]
fn synthetic_small_sink_means_match_oracle() {
    let dir = fixtures_dir().join("synthetic_small");
    let table = build_table(&load_matrix(&dir.join("table.tsv")));
    let ctx = build_context(&table, &load_metadata(&dir.join("metadata.tsv")));
    let expected = load_matrix(&dir.join("expected_sink_mean.tsv"));

    let params = reference_params(CollapseMethod::Mean, false);
    let sources = collapse_sources(&table, &ctx, params.collapse).expect("collapse");
    let prep = GibbsEstimator::prepare(&sources, &params).expect("prepare");

    // Estimator column order: collapsed source envs (sorted) then Unknown.
    let mut env_labels: Vec<String> = sources.env_names().to_vec();
    env_labels.push("Unknown".to_string());
    let v = env_labels.len();
    assert_eq!(v, expected.col_labels.len(), "column count");

    // The oracle's `sink_id` order need not match the table's sink column order;
    // map each expected row to its sink column by id.
    for (item_index, &sink_col) in ctx.sink_indices().iter().enumerate() {
        let sink_id = &table.sample_ids()[sink_col as usize];
        let (rows, counts) = table.column(sink_col as usize);
        let sink = SinkVec::new(rows, counts);
        let mut rng = rng_for_item(42, item_index as u64);
        let est = GibbsEstimator::estimate(&prep, &sink, &params, &mut rng, false);
        let got = mean_ensemble(&est, v);

        let exp_row = expected
            .row_labels
            .iter()
            .position(|r| r == sink_id)
            .unwrap_or_else(|| panic!("oracle has no row for sink {sink_id}"));

        for (col, label) in expected.col_labels.iter().enumerate() {
            let my_col = env_labels
                .iter()
                .position(|l| l == label)
                .unwrap_or_else(|| panic!("estimator has no env column {label}"));
            let expected_p = expected.data[exp_row][col];
            let got_p = got[my_col];
            assert!(
                (got_p - expected_p).abs() < 0.02,
                "sink {sink_id} env {label}: got {got_p:.5}, oracle {expected_p:.5} (Δ {:.5})",
                (got_p - expected_p).abs()
            );
        }

        // Sanity: a proportion vector sums to one.
        let sum: f64 = got.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sink {sink_id} sums to {sum}");
    }
}
