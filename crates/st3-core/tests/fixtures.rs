// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Smoke test for the committed reference fixtures.
//!
//! No estimator exists yet, so this milestone only asserts that every fixture
//! loads with the expected shape and that proportion rows sum to one. Numeric
//! equivalence against st3 output arrives with the estimator (later milestones).

mod common;

use common::{Matrix, fixtures_dir, load_matrix, load_metadata};
use std::path::Path;

/// Proportion matrices are renormalized per row, so each row sums to one.
fn assert_rows_sum_to_one(m: &Matrix, what: &str) {
    for r in 0..m.nrows() {
        let s = m.row_sum(r);
        assert!(
            (s - 1.0).abs() < 1e-3,
            "{what}: row {} ({}) sums to {s}, expected ~1.0",
            r,
            m.row_labels[r]
        );
    }
}

fn assert_non_empty_provenance(dir: &Path) {
    let prov = std::fs::read_to_string(dir.join("PROVENANCE.md"))
        .unwrap_or_else(|e| panic!("{}: {e}", dir.join("PROVENANCE.md").display()));
    assert!(
        !prov.trim().is_empty(),
        "{}: PROVENANCE.md is empty",
        dir.display()
    );
}

#[test]
fn tiny_test_fixtures_load() {
    let dir = fixtures_dir().join("tiny_test");
    let envs = ["drainwater", "seawater", "sewage", "Unknown"];

    // Input table: 20 features x 10 samples.
    let table = load_matrix(&dir.join("table.tsv"));
    assert_eq!(table.nrows(), 20, "tiny_test table features");
    assert_eq!(table.ncols(), 10, "tiny_test table samples");

    // Metadata: 10 samples, 5 source + 5 sink.
    let md = load_metadata(&dir.join("metadata.tsv"));
    assert_eq!(md.rows.len(), 10, "tiny_test metadata rows");
    assert_eq!(md.count_where("source_sink", "source"), 5, "sources");
    assert_eq!(md.count_where("source_sink", "sink"), 5, "sinks");

    // Primary oracle (R, sum-collapse): sink means 5 sinks x 4 classes.
    let sink = load_matrix(&dir.join("expected_sink_sum.tsv"));
    assert_eq!((sink.nrows(), sink.ncols()), (5, 4), "sink sum shape");
    assert_eq!(sink.col_labels, envs, "sink sum columns");
    assert_rows_sum_to_one(&sink, "expected_sink_sum");

    // Primary oracle stds: same shape (no row-sum constraint).
    let sd = load_matrix(&dir.join("expected_sink_sum_sd.tsv"));
    assert_eq!((sd.nrows(), sd.ncols()), (5, 4), "sink sum sd shape");
    assert_eq!(sd.col_labels, envs, "sink sum sd columns");

    // Primary oracle LOO: 5 held-out source samples x 4 classes.
    let loo = load_matrix(&dir.join("expected_loo_sum.tsv"));
    assert_eq!((loo.nrows(), loo.ncols()), (5, 4), "loo sum shape");
    assert_eq!(loo.col_labels, envs, "loo sum columns");
    assert_rows_sum_to_one(&loo, "expected_loo_sum");

    // Secondary oracle (Python, mean-collapse): sink means, same shape.
    let mean = load_matrix(&dir.join("expected_sink_mean.tsv"));
    assert_eq!((mean.nrows(), mean.ncols()), (5, 4), "sink mean shape");
    assert_eq!(mean.col_labels, envs, "sink mean columns");
    assert_rows_sum_to_one(&mean, "expected_sink_mean");

    assert_non_empty_provenance(&dir);
}

#[test]
fn synthetic_small_fixtures_load() {
    let dir = fixtures_dir().join("synthetic_small");
    let envs = ["envA", "envB", "envC", "Unknown"];

    // Input table: 6 features x 9 samples (6 source + 3 sink).
    let table = load_matrix(&dir.join("table.tsv"));
    assert_eq!(table.nrows(), 6, "synthetic table features");
    assert_eq!(table.ncols(), 9, "synthetic table samples");

    let md = load_metadata(&dir.join("metadata.tsv"));
    assert_eq!(md.rows.len(), 9, "synthetic metadata rows");
    assert_eq!(md.count_where("source_sink", "source"), 6, "sources");
    assert_eq!(md.count_where("source_sink", "sink"), 3, "sinks");

    // Primary oracle: 3 sinks x 4 classes.
    let sink = load_matrix(&dir.join("expected_sink_sum.tsv"));
    assert_eq!((sink.nrows(), sink.ncols()), (3, 4), "sink sum shape");
    assert_eq!(sink.col_labels, envs, "sink sum columns");
    assert_rows_sum_to_one(&sink, "expected_sink_sum");

    let sd = load_matrix(&dir.join("expected_sink_sum_sd.tsv"));
    assert_eq!((sd.nrows(), sd.ncols()), (3, 4), "sink sum sd shape");
    assert_eq!(sd.col_labels, envs, "sink sum sd columns");

    // LOO: 6 held-out source samples x 4 classes.
    let loo = load_matrix(&dir.join("expected_loo_sum.tsv"));
    assert_eq!((loo.nrows(), loo.ncols()), (6, 4), "loo sum shape");
    assert_eq!(loo.col_labels, envs, "loo sum columns");
    assert_rows_sum_to_one(&loo, "expected_loo_sum");

    let mean = load_matrix(&dir.join("expected_sink_mean.tsv"));
    assert_eq!((mean.nrows(), mean.ncols()), (3, 4), "sink mean shape");
    assert_eq!(mean.col_labels, envs, "sink mean columns");
    assert_rows_sum_to_one(&mean, "expected_sink_mean");

    assert_non_empty_provenance(&dir);
}
