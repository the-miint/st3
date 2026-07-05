// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! End-to-end sink-prediction checks against the committed reference oracles.
//!
//! Runs [`predict_sinks`] on both fixtures with the pinned reference parameters
//! and compares the per-sink mixing means and standard deviations to the
//! committed matrices. Our Xoshiro256++ stream does not bit-match the reference's
//! NumPy/R streams, so mean and std equivalence is statistical (absolute
//! tolerance 0.02). The contingency table is additionally checked against exact
//! first-principles invariants that hold regardless of the RNG.

mod common;

use std::collections::{HashMap, HashSet};

use common::{
    Matrix, build_context, build_table, fixtures_dir, load_matrix, load_metadata, reference_params,
};
use st3_core::{CollapseMethod, CountTable, SampleContext, SourceMixing, predict_sinks};

/// Run the full sum-collapse contingency pipeline on `fixture` at the pinned
/// reference parameters (seed 42), returning the table, context, and result.
fn contingency_run(fixture: &str) -> (CountTable, SampleContext, SourceMixing) {
    let dir = fixtures_dir().join(fixture);
    let table = build_table(&load_matrix(&dir.join("table.tsv")));
    let ctx = build_context(&table, &load_metadata(&dir.join("metadata.tsv")));
    let sm = predict_sinks(
        &table,
        &ctx,
        &reference_params(CollapseMethod::Sum, true),
        42,
    )
    .expect("predict");
    (table, ctx, sm)
}

/// Load a committed contingency oracle as `(sink_id, source, feature) -> mean_count`.
fn load_contingency_oracle(fixture: &str) -> HashMap<(String, String, String), f64> {
    let path = fixtures_dir()
        .join(fixture)
        .join("expected_contingency_sum.tsv");
    let text = std::fs::read_to_string(&path).expect("read contingency oracle");
    let mut map = HashMap::new();
    for line in text.lines().skip(1).filter(|l| !l.is_empty()) {
        let mut f = line.split('\t');
        let sink_id = f.next().unwrap().to_string();
        let source = f.next().unwrap().to_string();
        let feature = f.next().unwrap().to_string();
        let value: f64 = f.next().unwrap().parse().unwrap();
        map.insert((sink_id, source, feature), value);
    }
    map
}

/// Look up the oracle value for `sink_id` × `env_label` in a loaded matrix,
/// joining by label so file row/column order need not match the estimator's.
fn oracle_cell(m: &Matrix, sink_id: &str, env_label: &str) -> f64 {
    let r = m
        .row_labels
        .iter()
        .position(|r| r == sink_id)
        .unwrap_or_else(|| panic!("oracle has no row for sink {sink_id}"));
    let c = m
        .col_labels
        .iter()
        .position(|c| c == env_label)
        .unwrap_or_else(|| panic!("oracle has no column {env_label}"));
    m.data[r][c]
}

/// Drive `predict_sinks` on `fixture` and compare per-sink means (and, if given,
/// stds) to the named oracle files within `tol`.
fn assert_matches_oracle(
    fixture: &str,
    collapse: CollapseMethod,
    mean_file: &str,
    sd_file: Option<&str>,
) {
    let dir = fixtures_dir().join(fixture);
    let table = build_table(&load_matrix(&dir.join("table.tsv")));
    let ctx = build_context(&table, &load_metadata(&dir.join("metadata.tsv")));

    let sm = predict_sinks(&table, &ctx, &reference_params(collapse, false), 42).expect("predict");

    let expected_mean = load_matrix(&dir.join(mean_file));
    let expected_sd = sd_file.map(|f| load_matrix(&dir.join(f)));

    for i in 0..sm.n_sinks() {
        let sink_id = &sm.sink_ids()[i];
        let mean = sm.mean_row(i);
        let std = sm.std_row(i);
        let sum: f64 = mean.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "sink {sink_id} mean sums to {sum}"
        );

        for (k, label) in sm.env_names().iter().enumerate() {
            let exp_m = oracle_cell(&expected_mean, sink_id, label);
            assert!(
                (mean[k] - exp_m).abs() < 0.02,
                "{fixture} {sink_id}/{label} mean: got {:.5}, oracle {:.5}",
                mean[k],
                exp_m
            );
            if let Some(sd) = &expected_sd {
                let exp_s = oracle_cell(sd, sink_id, label);
                assert!(
                    (std[k] - exp_s).abs() < 0.02,
                    "{fixture} {sink_id}/{label} std: got {:.5}, oracle {:.5}",
                    std[k],
                    exp_s
                );
            }
        }
    }
}

#[test]
fn synthetic_small_sum_means_and_stds() {
    assert_matches_oracle(
        "synthetic_small",
        CollapseMethod::Sum,
        "expected_sink_sum.tsv",
        Some("expected_sink_sum_sd.tsv"),
    );
}

#[test]
fn synthetic_small_mean_collapse_means() {
    assert_matches_oracle(
        "synthetic_small",
        CollapseMethod::Mean,
        "expected_sink_mean.tsv",
        None,
    );
}

#[test]
fn tiny_test_sum_means_and_stds() {
    assert_matches_oracle(
        "tiny_test",
        CollapseMethod::Sum,
        "expected_sink_sum.tsv",
        Some("expected_sink_sum_sd.tsv"),
    );
}

#[test]
fn contingency_off_yields_no_tally() {
    let dir = fixtures_dir().join("synthetic_small");
    let table = build_table(&load_matrix(&dir.join("table.tsv")));
    let ctx = build_context(&table, &load_metadata(&dir.join("metadata.tsv")));
    let sm = predict_sinks(
        &table,
        &ctx,
        &reference_params(CollapseMethod::Sum, false),
        42,
    )
    .unwrap();
    assert!(sm.contingency().is_none());
}

#[test]
fn contingency_satisfies_first_principles_invariants() {
    // With contingency on, each sink's source × taxon tally must satisfy exact
    // invariants independent of the RNG (checked on BOTH fixtures):
    //   - column t (over sources) sums to the sink's count of taxon t;
    //   - row v (over taxa) / D equals that source's mixing mean;
    //   - the whole table sums to the sink depth D.
    for fixture in ["synthetic_small", "tiny_test"] {
        let (table, ctx, sm) = contingency_run(fixture);
        let tallies = sm.contingency().expect("contingency requested");
        assert_eq!(tallies.len(), sm.n_sinks());

        for (i, &sink_col) in ctx.sink_indices().iter().enumerate() {
            let tally = &tallies[i];
            let v = tally.n_sources();
            let tau = tally.n_features();
            assert_eq!(v, sm.n_envs());

            // Densify the tally: dense[src * tau + feat].
            let mut dense = vec![0.0f64; v * tau];
            for (src, feat, m) in tally.triples() {
                dense[src as usize * tau + feat as usize] = m;
            }

            // The sink's own taxon counts.
            let (rows, counts) = table.column(sink_col as usize);
            let mut sink_dense = vec![0.0f64; tau];
            for (&r, &c) in rows.iter().zip(counts.iter()) {
                sink_dense[r as usize] = f64::from(c);
            }
            let depth: f64 = sink_dense.iter().sum();

            // Invariant 1: per-taxon column sum == sink count of that taxon.
            for t in 0..tau {
                let col_sum: f64 = (0..v).map(|s| dense[s * tau + t]).sum();
                assert!(
                    (col_sum - sink_dense[t]).abs() < 1e-9,
                    "{fixture} sink {i} taxon {t}: column sum {col_sum} != sink count {}",
                    sink_dense[t]
                );
            }

            // Invariant 2: per-source row sum / D == mixing mean of that source.
            let mean = sm.mean_row(i);
            for s in 0..v {
                let row_sum: f64 = (0..tau).map(|t| dense[s * tau + t]).sum();
                assert!(
                    (row_sum / depth - mean[s]).abs() < 1e-9,
                    "{fixture} sink {i} source {s}: row_sum/D {} != mixing mean {}",
                    row_sum / depth,
                    mean[s]
                );
            }

            // Invariant 3: grand total == D.
            let grand: f64 = dense.iter().sum();
            assert!(
                (grand - depth).abs() < 1e-9,
                "{fixture} sink {i} grand total {grand} != D {depth}"
            );
        }
    }
}

#[test]
fn contingency_matches_reference_oracle() {
    // Cross-check the per-sink source x taxon mean assignments against the
    // committed reference oracle (long COO: sink_id, source, feature, mean_count).
    // Our RNG differs from the reference's, so we compare D-normalized cells at an
    // absolute tolerance, iterating the UNION of our cells and the oracle's — so
    // mass we place where the oracle has none (and vice versa) is caught, not just
    // cells the oracle happens to list. The exact marginals are covered by the
    // invariant test; here we bound intra-cell placement error.
    //
    // Tolerance: cells are fractions of the whole sink (most < 0.02), so a per-cell
    // bound loose enough for a proportion would be toothless here. The observed max
    // divergence between our Xoshiro stream and the reference's is well under 0.01
    // on both fixtures; 0.01 keeps real coverage while tolerating Monte-Carlo noise.
    const TOL: f64 = 0.01;

    for fixture in ["synthetic_small", "tiny_test"] {
        let (table, ctx, sm) = contingency_run(fixture);
        let tallies = sm.contingency().expect("contingency requested");
        let feature_ids = table.feature_ids();

        // Our cells, D-normalized. Depth per sink from the table column.
        let mut got: HashMap<(String, String, String), f64> = HashMap::new();
        for (i, &sink_col) in ctx.sink_indices().iter().enumerate() {
            let sink_id = sm.sink_ids()[i].clone();
            let depth = table.column_sum(sink_col as usize) as f64;
            for (src, feat, m) in tallies[i].triples() {
                let key = (
                    sink_id.clone(),
                    sm.env_names()[src as usize].clone(),
                    feature_ids[feat as usize].clone(),
                );
                got.insert(key, m / depth);
            }
        }
        let depth_of: HashMap<String, f64> = ctx
            .sink_indices()
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                (
                    sm.sink_ids()[i].clone(),
                    table.column_sum(c as usize) as f64,
                )
            })
            .collect();

        // Oracle cells, D-normalized to the same scale.
        let oracle_raw = load_contingency_oracle(fixture);
        let oracle: HashMap<(String, String, String), f64> = oracle_raw
            .into_iter()
            .map(|(k, v)| {
                let d = depth_of[&k.0];
                (k, v / d)
            })
            .collect();

        // Compare over the UNION of keys (missing side = 0.0).
        let keys: HashSet<&(String, String, String)> = got.keys().chain(oracle.keys()).collect();
        assert!(
            !keys.is_empty(),
            "{fixture}: no contingency cells to compare"
        );
        let mut max_diff = 0.0f64;
        for key in keys {
            let g = got.get(key).copied().unwrap_or(0.0);
            let o = oracle.get(key).copied().unwrap_or(0.0);
            let diff = (g - o).abs();
            max_diff = max_diff.max(diff);
            assert!(
                diff < TOL,
                "{fixture} {}/{}/{}: got {g:.5}, oracle {o:.5} (Δ {diff:.5})",
                key.0,
                key.1,
                key.2
            );
        }
        // Coverage: every sink is represented on both sides.
        for sink_id in sm.sink_ids() {
            assert!(
                oracle.keys().any(|k| &k.0 == sink_id),
                "{fixture}: oracle missing sink {sink_id}"
            );
        }
        eprintln!("{fixture}: {} cells, max Δ = {max_diff:.5}", got.len());
    }
}
