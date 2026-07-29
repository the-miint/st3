// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Rarefaction against the committed fixtures plus first-principles statistical
//! checks. No external subsampling library is used as an oracle; correctness is
//! established by exact invariants (sum, bounds, axis preservation, determinism)
//! and by convergence of the sampling means to their analytic expectations.

mod common;

use common::{build_table, fixtures_dir, load_matrix};
use st3_core::{rarefy, rarefy_per_sample, CountTable, SampleStatus};

fn tiny_table() -> CountTable {
    build_table(&load_matrix(&fixtures_dir().join("tiny_test/table.tsv")))
}

/// Densify one column of a table to length `n_features`.
fn dense_column(t: &CountTable, s: usize) -> Vec<u32> {
    let (rows, counts) = t.column(s);
    let mut d = vec![0u32; t.n_features()];
    for (&f, &c) in rows.iter().zip(counts.iter()) {
        d[f as usize] = c;
    }
    d
}

#[test]
fn columns_sum_to_depth_both_modes() {
    let t = tiny_table();
    for with_replacement in [false, true] {
        let out = rarefy(&t, Some(1000), with_replacement, 42).unwrap();
        assert!(out.status().iter().all(|&s| s == SampleStatus::Rarefied));
        for s in 0..out.table().n_samples() {
            assert_eq!(
                out.table().column_sum(s),
                1000,
                "sample {s} mode {with_replacement}"
            );
        }
    }
}

#[test]
fn feature_and_sample_axes_preserved() {
    let t = tiny_table();
    let out = rarefy(&t, Some(1000), false, 42).unwrap();
    assert_eq!(out.table().feature_ids(), t.feature_ids());
    assert_eq!(out.table().sample_ids(), t.sample_ids());
    assert_eq!(out.table().n_features(), t.n_features());
    assert_eq!(out.table().n_samples(), t.n_samples());
}

#[test]
fn without_replacement_never_exceeds_input() {
    let t = tiny_table();
    let out = rarefy(&t, Some(1000), false, 7).unwrap();
    for s in 0..t.n_samples() {
        let orig = dense_column(&t, s);
        let got = dense_column(out.table(), s);
        for (f, (&g, &o)) in got.iter().zip(orig.iter()).enumerate() {
            assert!(g <= o, "sample {s} feature {f}: {g} > {o}");
        }
    }
}

#[test]
fn too_shallow_mix() {
    // tiny_test column totals are 1900 + 20*j. depth 2000 leaves s0..s4 shallow.
    let t = tiny_table();
    let out = rarefy(&t, Some(2000), false, 42).unwrap();
    assert_eq!(out.too_shallow(), vec![0, 1, 2, 3, 4]);
    // Shallow columns are passed through unchanged.
    for s in 0..5 {
        assert_eq!(out.status()[s], SampleStatus::TooShallow);
        assert_eq!(dense_column(out.table(), s), dense_column(&t, s));
    }
    // s5 total == depth (2000), s6..s9 deeper: all rarefied to exactly 2000.
    for s in 5..10 {
        assert_eq!(out.status()[s], SampleStatus::Rarefied);
        assert_eq!(out.table().column_sum(s), 2000);
    }
}

#[test]
fn passthrough_equals_input() {
    let t = tiny_table();
    let out = rarefy(&t, None, false, 42).unwrap();
    assert!(out.status().iter().all(|&s| s == SampleStatus::Passthrough));
    assert_eq!(out.table(), &t);
}

#[test]
fn depth_zero_is_passthrough() {
    let t = tiny_table();
    let out = rarefy(&t, Some(0), true, 42).unwrap();
    assert!(out.status().iter().all(|&s| s == SampleStatus::Passthrough));
    assert_eq!(out.table(), &t);
}

#[test]
fn determinism_same_seed() {
    let t = tiny_table();
    for with_replacement in [false, true] {
        let a = rarefy(&t, Some(1000), with_replacement, 99).unwrap();
        let b = rarefy(&t, Some(1000), with_replacement, 99).unwrap();
        assert_eq!(a, b);
    }
}

#[test]
fn different_seeds_generally_differ() {
    let t = tiny_table();
    let a = rarefy(&t, Some(1000), false, 1).unwrap();
    let b = rarefy(&t, Some(1000), false, 2).unwrap();
    assert_ne!(a.table(), b.table());
}

#[test]
fn per_sample_heterogeneous_depths() {
    let t = tiny_table();
    // Disable half, rarefy the rest to 1500 (all totals >= 1900 so none shallow).
    let depths: Vec<Option<u32>> = (0..t.n_samples())
        .map(|s| if s % 2 == 0 { None } else { Some(1500) })
        .collect();
    let out = rarefy_per_sample(&t, &depths, false, 3).unwrap();
    for s in 0..t.n_samples() {
        if s % 2 == 0 {
            assert_eq!(out.status()[s], SampleStatus::Passthrough);
            assert_eq!(out.table().column_sum(s), t.column_sum(s));
        } else {
            assert_eq!(out.status()[s], SampleStatus::Rarefied);
            assert_eq!(out.table().column_sum(s), 1500);
        }
    }
}

/// A single-sample table with a known proportion (feature 0 = 0.75).
fn proportion_table() -> CountTable {
    CountTable::from_coo(
        vec!["f0".into(), "f1".into()],
        vec!["s0".into()],
        &[0, 1],
        &[0, 0],
        &[300.0, 100.0],
    )
    .unwrap()
}

#[test]
fn with_replacement_converges_to_proportion() {
    // Multinomial mean of feature 0 = depth * 0.75. Average over many seeds.
    let t = proportion_table();
    let depth = 200u32;
    let trials = 400u64;
    let mut total_f0 = 0u64;
    for seed in 0..trials {
        let out = rarefy(&t, Some(depth), true, seed).unwrap();
        total_f0 += u64::from(dense_column(out.table(), 0)[0]);
    }
    let est = total_f0 as f64 / (trials * u64::from(depth)) as f64;
    assert!(
        (est - 0.75).abs() < 0.02,
        "with-replacement proportion {est}"
    );
}

#[test]
fn without_replacement_converges_to_proportion() {
    // Hypergeometric mean of feature 0 = depth * 300/400 = 0.75 * depth.
    let t = proportion_table();
    let depth = 200u32;
    let trials = 400u64;
    let mut total_f0 = 0u64;
    for seed in 0..trials {
        let out = rarefy(&t, Some(depth), false, seed).unwrap();
        total_f0 += u64::from(dense_column(out.table(), 0)[0]);
    }
    let est = total_f0 as f64 / (trials * u64::from(depth)) as f64;
    assert!(
        (est - 0.75).abs() < 0.02,
        "without-replacement proportion {est}"
    );
}
