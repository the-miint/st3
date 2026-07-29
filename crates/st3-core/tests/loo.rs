// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Leave-one-out source-prediction checks against the committed reference oracles.
//!
//! Runs [`predict_loo`] on both fixtures with the pinned reference parameters and
//! compares each held-out source sample's mixing means to the committed
//! `expected_loo_sum.tsv`. Our Xoshiro256++ stream does not bit-match the
//! reference's, so equivalence is statistical (absolute tolerance 0.02). Two
//! structural invariants are checked exactly (RNG-independent): the missing-env
//! column is zero when a sole-member environment is held out, and the output
//! columns are the *source* environments only (never sink-only envs).

mod common;

use common::{
    build_context, build_table, fixtures_dir, load_matrix, load_metadata, reference_params, Matrix,
};
use st3_core::{predict_loo, CollapseMethod};

/// Look up the oracle value for `source_id` × `env_label`, joining by label so
/// the file's row/column order need not match the driver's fold order.
fn oracle_cell(m: &Matrix, source_id: &str, env_label: &str) -> f64 {
    let r = m
        .row_labels
        .iter()
        .position(|r| r == source_id)
        .unwrap_or_else(|| panic!("oracle has no row for source {source_id}"));
    let c = m
        .col_labels
        .iter()
        .position(|c| c == env_label)
        .unwrap_or_else(|| panic!("oracle has no column {env_label}"));
    m.data[r][c]
}

// (f) + (g): LOO means match the committed sum-collapse oracle, and every row
// sums to one, on both fixtures.
//
// Tolerance is statistical, and per-fixture. `synthetic_small` has three
// well-separated sources, so each held-out sample re-identifies its own env
// sharply and 0.02 holds. `tiny_test`'s environments overlap heavily — even the
// oracle shows a held-out sewage/seawater sample spread almost uniformly across
// all envs — so the mixing proportions are high-variance. Both the oracle and
// our estimate are finite-sample Gibbs means (100 restarts, and Gibbs draws are
// correlated within a restart, so the effective sample size is ~ the restart
// count), each carrying ~0.02 Monte-Carlo noise; comparing two independent noisy
// estimates, the worst cell compounds just past 0.02. Convergence was verified
// out-of-test: raising restarts to 800 drops tiny_test's worst cell from 0.021
// to 0.014 (no systematic bias — the residual is the oracle's own noise), and a
// five-seed sweep at 100 restarts stays in 0.021–0.027. 0.03 keeps a real check
// with margin for that noise.
#[test]
fn loo_means_match_oracle_on_both_fixtures() {
    for (fixture, tol) in [("synthetic_small", 0.02), ("tiny_test", 0.03)] {
        let dir = fixtures_dir().join(fixture);
        let table = build_table(&load_matrix(&dir.join("table.tsv")));
        let ctx = build_context(&table, &load_metadata(&dir.join("metadata.tsv")));

        let sm = predict_loo(
            &table,
            &ctx,
            &reference_params(CollapseMethod::Sum, false),
            42,
            1,
        )
        .expect("predict_loo");
        let expected = load_matrix(&dir.join("expected_loo_sum.tsv"));

        for i in 0..sm.n_sinks() {
            let source_id = &sm.sink_ids()[i];
            let mean = sm.mean_row(i);
            let sum: f64 = mean.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "{fixture} {source_id} row sums to {sum}"
            );
            for (k, label) in sm.env_names().iter().enumerate() {
                let exp = oracle_cell(&expected, source_id, label);
                assert!(
                    (mean[k] - exp).abs() < tol,
                    "{fixture} {source_id}/{label}: got {:.5}, oracle {:.5} (tol {tol})",
                    mean[k],
                    exp
                );
            }
        }
    }
}

// (g) missing-env invariant: tiny_test s7 is the sole drainwater source, so
// holding it out makes drainwater vanish — that column must be exactly 0.
#[test]
fn tiny_test_sole_drainwater_column_is_exact_zero() {
    let dir = fixtures_dir().join("tiny_test");
    let table = build_table(&load_matrix(&dir.join("table.tsv")));
    let ctx = build_context(&table, &load_metadata(&dir.join("metadata.tsv")));

    let sm = predict_loo(
        &table,
        &ctx,
        &reference_params(CollapseMethod::Sum, false),
        42,
        1,
    )
    .expect("predict_loo");

    let s7 = sm
        .sink_ids()
        .iter()
        .position(|s| s == "s7")
        .expect("s7 held out");
    let drainwater = sm
        .env_names()
        .iter()
        .position(|e| e == "drainwater")
        .expect("drainwater column present");
    let row = sm.mean_row(s7);
    assert!(
        row[drainwater].abs() < 1e-12,
        "s7 drainwater must be exactly 0, got {}",
        row[drainwater]
    );
}

// (h) source-scoped column guard: the output columns are the source
// environments plus Unknown — NOT the sink pond envs (guards the be_sparse leak
// where LOO columns spanned the whole metadata category).
#[test]
fn tiny_test_columns_are_source_scoped() {
    let dir = fixtures_dir().join("tiny_test");
    let table = build_table(&load_matrix(&dir.join("table.tsv")));
    let ctx = build_context(&table, &load_metadata(&dir.join("metadata.tsv")));

    let sm = predict_loo(
        &table,
        &ctx,
        &reference_params(CollapseMethod::Sum, false),
        42,
        1,
    )
    .expect("predict_loo");

    assert_eq!(
        sm.env_names(),
        &["drainwater", "seawater", "sewage", "Unknown"],
        "LOO columns must be exactly the source envs + Unknown"
    );
    // The sink pond environments must not leak into the column frame.
    for pond in ["pond1", "pond2", "pond3", "pond4", "pond5"] {
        assert!(
            !sm.env_names().iter().any(|e| e == pond),
            "sink env {pond} leaked into LOO columns"
        );
    }
    assert!(sm.contingency().is_none());
}

// Determinism guard (roadmap R8) on real data: LOO output is byte-identical
// across thread counts on both fixtures.
#[test]
fn loo_output_is_identical_across_job_counts() {
    for fixture in ["synthetic_small", "tiny_test"] {
        let dir = fixtures_dir().join(fixture);
        let table = build_table(&load_matrix(&dir.join("table.tsv")));
        let ctx = build_context(&table, &load_metadata(&dir.join("metadata.tsv")));
        // Determinism is independent of sampler depth, so use light params to keep
        // this cross-product (2 fixtures × 4 job counts, folds per fixture) fast.
        let mut params = reference_params(CollapseMethod::Sum, false);
        params.restarts = 8;
        params.draws_per_restart = 2;
        params.burnin = 10;
        let serial = predict_loo(&table, &ctx, &params, 42, 1).expect("predict_loo");
        for jobs in [0usize, 2, 8] {
            let parallel = predict_loo(&table, &ctx, &params, 42, jobs).expect("predict_loo");
            assert_eq!(parallel, serial, "{fixture} jobs={jobs}");
        }
    }
}
