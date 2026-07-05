// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Shared, dependency-free helpers for fixture-backed integration tests.
//!
//! A tiny std-only TSV reader used by the reference-fixture tests and reused by
//! later milestones. It intentionally has no crate dependencies: fixtures are
//! plain LF-terminated TSV, so parsing them needs nothing beyond `std`.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use st3_core::{CountTable, Role, SampleContext};

/// Absolute path to the committed `fixtures/` directory (repository root),
/// resolved relative to this crate's manifest so tests are location-independent.
pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// A labeled numeric matrix parsed from a TSV file.
///
/// The first line is a header whose first cell is a corner label (e.g.
/// `feature_id`) and whose remaining cells are the column labels. Each
/// subsequent line begins with a row label followed by one `f64` per column.
pub struct Matrix {
    /// Row labels, in file order.
    pub row_labels: Vec<String>,
    /// Column labels, in file order (header cells after the corner label).
    pub col_labels: Vec<String>,
    /// Row-major values; `data[r][c]` aligns with `row_labels[r]`/`col_labels[c]`.
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    /// Number of data rows.
    pub fn nrows(&self) -> usize {
        self.row_labels.len()
    }

    /// Number of value columns.
    pub fn ncols(&self) -> usize {
        self.col_labels.len()
    }

    /// Sum of row `r`.
    pub fn row_sum(&self, r: usize) -> f64 {
        self.data[r].iter().sum()
    }
}

/// Parse a labeled numeric matrix from a TSV file, panicking with a
/// path-qualified message on any I/O or parse error (test-only helper).
pub fn load_matrix(path: &Path) -> Matrix {
    let text = read_to_string(path);
    let mut lines = text.lines().filter(|l| !l.is_empty());

    let header = lines
        .next()
        .unwrap_or_else(|| panic!("{}: empty file", path.display()));
    let col_labels: Vec<String> = header.split('\t').skip(1).map(str::to_owned).collect();
    assert!(
        !col_labels.is_empty(),
        "{}: header has no column labels",
        path.display()
    );

    let mut row_labels = Vec::new();
    let mut data = Vec::new();
    for (n, line) in lines.enumerate() {
        let mut cells = line.split('\t');
        let label = cells
            .next()
            .unwrap_or_else(|| panic!("{}: row {} is empty", path.display(), n + 1));
        let values: Vec<f64> = cells
            .map(|c| {
                c.parse::<f64>().unwrap_or_else(|_| {
                    panic!(
                        "{}: row {} has non-numeric cell {:?}",
                        path.display(),
                        n + 1,
                        c
                    )
                })
            })
            .collect();
        assert_eq!(
            values.len(),
            col_labels.len(),
            "{}: row {} ({}) has {} values, expected {}",
            path.display(),
            n + 1,
            label,
            values.len(),
            col_labels.len()
        );
        row_labels.push(label.to_owned());
        data.push(values);
    }

    Matrix {
        row_labels,
        col_labels,
        data,
    }
}

/// A parsed metadata table: a header row plus string-valued rows (ids included
/// as the first cell of each row).
pub struct Metadata {
    /// Column names, in file order (includes the id column).
    pub columns: Vec<String>,
    /// One entry per data row; each is `columns.len()` string cells.
    pub rows: Vec<Vec<String>>,
}

impl Metadata {
    /// Values of the named column across all rows, in file order.
    ///
    /// Panics if the column is absent.
    pub fn column(&self, name: &str) -> Vec<&str> {
        let idx = self
            .columns
            .iter()
            .position(|c| c == name)
            .unwrap_or_else(|| panic!("metadata has no column {name:?}"));
        self.rows.iter().map(|r| r[idx].as_str()).collect()
    }

    /// Count of rows whose `column` equals `value`.
    pub fn count_where(&self, column: &str, value: &str) -> usize {
        self.column(column).iter().filter(|v| **v == value).count()
    }
}

/// Parse a metadata TSV (header + string rows), panicking on error (test-only).
pub fn load_metadata(path: &Path) -> Metadata {
    let text = read_to_string(path);
    let mut lines = text.lines().filter(|l| !l.is_empty());

    let header = lines
        .next()
        .unwrap_or_else(|| panic!("{}: empty file", path.display()));
    let columns: Vec<String> = header.split('\t').map(str::to_owned).collect();

    let mut rows = Vec::new();
    for (n, line) in lines.enumerate() {
        let cells: Vec<String> = line.split('\t').map(str::to_owned).collect();
        assert_eq!(
            cells.len(),
            columns.len(),
            "{}: row {} has {} cells, expected {}",
            path.display(),
            n + 1,
            cells.len(),
            columns.len()
        );
        rows.push(cells);
    }

    Metadata { columns, rows }
}

fn read_to_string(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// Build a [`CountTable`] from a loaded dense [`Matrix`] (features × samples).
pub fn build_table(m: &Matrix) -> CountTable {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for (r, row) in m.data.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            if v != 0.0 {
                rows.push(r as u32);
                cols.push(c as u32);
                vals.push(v);
            }
        }
    }
    CountTable::from_coo(
        m.row_labels.clone(),
        m.col_labels.clone(),
        &rows,
        &cols,
        &vals,
    )
    .expect("fixture table builds")
}

/// Build a [`SampleContext`] aligned to `table`'s sample axis from loaded
/// metadata (columns `sample_id`, `source_sink`, `env`), matching by id.
pub fn build_context(table: &CountTable, md: &Metadata) -> SampleContext {
    let ids = md.column("sample_id");
    let source_sink = md.column("source_sink");
    let env = md.column("env");
    let mut roles = Vec::with_capacity(table.n_samples());
    let mut envs = Vec::with_capacity(table.n_samples());
    for sid in table.sample_ids() {
        let idx = ids
            .iter()
            .position(|x| x == sid)
            .unwrap_or_else(|| panic!("sample {sid} absent from metadata"));
        let role = match source_sink[idx] {
            "source" => Role::Source,
            "sink" => Role::Sink,
            other => panic!("unexpected source_sink value {other:?}"),
        };
        roles.push(role);
        envs.push(Some(env[idx].to_owned()));
    }
    SampleContext::for_table(table, roles, envs).expect("context builds")
}
