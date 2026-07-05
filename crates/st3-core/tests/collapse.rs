// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Collapse against the committed fixtures with hand-computed oracles.
//!
//! The committed `expected_*.tsv` files are estimator *proportions*, not
//! collapsed counts, so collapse counts are verified against hand-computed
//! oracles here; environment names/order are cross-checked against the fixture
//! headers (minus the trailing `Unknown` column, which the estimator adds).

mod common;

use common::{build_context, build_table, fixtures_dir, load_matrix, load_metadata};
use st3_core::{CollapseMethod, CollapsedSources, collapse_sources};

/// Densify one collapsed environment column to length `n_features`.
fn dense(cs: &CollapsedSources, env: usize, n_features: usize) -> Vec<u64> {
    let (rows, counts) = cs.column(env);
    let mut d = vec![0u64; n_features];
    for (&r, &c) in rows.iter().zip(counts.iter()) {
        d[r as usize] = c as u64;
    }
    d
}

#[test]
fn synthetic_small_sum_collapse() {
    let dir = fixtures_dir().join("synthetic_small");
    let table = build_table(&load_matrix(&dir.join("table.tsv")));
    let ctx = build_context(&table, &load_metadata(&dir.join("metadata.tsv")));
    let cs = collapse_sources(&table, &ctx, CollapseMethod::Sum).unwrap();

    assert_eq!(cs.env_names(), &["envA", "envB", "envC"]);
    assert_eq!(dense(&cs, 0, 6), [190, 210, 0, 0, 0, 0]); // envA = a1+a2
    assert_eq!(dense(&cs, 1, 6), [0, 0, 210, 190, 0, 0]); // envB = b1+b2
    assert_eq!(dense(&cs, 2, 6), [0, 0, 0, 0, 190, 210]); // envC = c1+c2
}

#[test]
fn synthetic_small_mean_collapse() {
    let dir = fixtures_dir().join("synthetic_small");
    let table = build_table(&load_matrix(&dir.join("table.tsv")));
    let ctx = build_context(&table, &load_metadata(&dir.join("metadata.tsv")));
    let cs = collapse_sources(&table, &ctx, CollapseMethod::Mean).unwrap();

    // Means are integral here (e.g. mean(100,90)=95), isolating divisor from floor.
    assert_eq!(dense(&cs, 0, 6), [95, 105, 0, 0, 0, 0]);
    assert_eq!(dense(&cs, 1, 6), [0, 0, 105, 95, 0, 0]);
    assert_eq!(dense(&cs, 2, 6), [0, 0, 0, 0, 95, 105]);
}

#[test]
fn tiny_test_sum_collapse() {
    let dir = fixtures_dir().join("tiny_test");
    let table = build_table(&load_matrix(&dir.join("table.tsv")));
    let ctx = build_context(&table, &load_metadata(&dir.join("metadata.tsv")));
    let cs = collapse_sources(&table, &ctx, CollapseMethod::Sum).unwrap();

    assert_eq!(cs.env_names(), &["drainwater", "seawater", "sewage"]);
    // value(o_i, s_j) = 10*i + j. drainwater = s7; seawater = s4+s5; sewage = s8+s9.
    for i in 0..20u64 {
        assert_eq!(dense(&cs, 0, 20)[i as usize], 10 * i + 7, "drainwater o{i}");
        assert_eq!(dense(&cs, 1, 20)[i as usize], 20 * i + 9, "seawater o{i}");
        assert_eq!(dense(&cs, 2, 20)[i as usize], 20 * i + 17, "sewage o{i}");
    }
}

#[test]
fn tiny_test_mean_collapse() {
    let dir = fixtures_dir().join("tiny_test");
    let table = build_table(&load_matrix(&dir.join("table.tsv")));
    let ctx = build_context(&table, &load_metadata(&dir.join("metadata.tsv")));
    let cs = collapse_sources(&table, &ctx, CollapseMethod::Mean).unwrap();

    // Floor bites: seawater=floor((20i+9)/2)=10i+4; sewage=floor((20i+17)/2)=10i+8.
    for i in 0..20u64 {
        assert_eq!(dense(&cs, 0, 20)[i as usize], 10 * i + 7, "drainwater o{i}");
        assert_eq!(dense(&cs, 1, 20)[i as usize], 10 * i + 4, "seawater o{i}");
        assert_eq!(dense(&cs, 2, 20)[i as usize], 10 * i + 8, "sewage o{i}");
    }
}

#[test]
fn env_names_match_fixture_headers() {
    for set in ["tiny_test", "synthetic_small"] {
        let dir = fixtures_dir().join(set);
        let table = build_table(&load_matrix(&dir.join("table.tsv")));
        let ctx = build_context(&table, &load_metadata(&dir.join("metadata.tsv")));
        let cs = collapse_sources(&table, &ctx, CollapseMethod::Sum).unwrap();

        // Fixture columns are [sorted envs..., "Unknown"]; drop the trailing Unknown.
        let headers = load_matrix(&dir.join("expected_sink_sum.tsv")).col_labels;
        let expected_envs = &headers[..headers.len() - 1];
        assert_eq!(cs.env_names(), expected_envs, "{set} env order");

        // Collapse reuses the input feature axis.
        assert_eq!(cs.feature_ids(), table.feature_ids(), "{set} feature axis");
    }
}
