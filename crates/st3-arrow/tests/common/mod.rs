// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Shared, dependency-free helpers for the Arrow round-trip integration test.
//!
//! A tiny std-only TSV reader (mirroring st3-core's fixture loader) plus two
//! parallel builders: one that constructs core types directly from a fixture,
//! and one that constructs the equivalent Arrow inputs. The two must agree — the
//! round-trip test asserts arrow→core→arrow reproduces core-direct.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int32Array, RecordBatch, StringArray, StructArray};
use arrow::datatypes::{DataType, Field, Schema};
use st3_core::{CountTable, Role, SampleContext};

/// Absolute path to the committed `fixtures/` directory, resolved relative to
/// this crate's manifest so tests are location-independent.
pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// A labeled numeric matrix parsed from a TSV file (features × samples).
pub struct Matrix {
    /// Row (feature) labels, in file order.
    pub row_labels: Vec<String>,
    /// Column (sample) labels, in file order.
    pub col_labels: Vec<String>,
    /// Row-major values; `data[r][c]` aligns with `row_labels[r]`/`col_labels[c]`.
    pub data: Vec<Vec<f64>>,
}

/// Parse a labeled numeric matrix from a TSV file (test-only; panics on error).
pub fn load_matrix(path: &Path) -> Matrix {
    let text = read_to_string(path);
    let mut lines = text.lines().filter(|l| !l.is_empty());
    let header = lines
        .next()
        .unwrap_or_else(|| panic!("{}: empty file", path.display()));
    let col_labels: Vec<String> = header.split('\t').skip(1).map(str::to_owned).collect();

    let mut row_labels = Vec::new();
    let mut data = Vec::new();
    for line in lines {
        let mut cells = line.split('\t');
        let label = cells.next().expect("row label");
        let values: Vec<f64> = cells
            .map(|c| c.parse::<f64>().expect("numeric cell"))
            .collect();
        assert_eq!(
            values.len(),
            col_labels.len(),
            "ragged row in {}",
            path.display()
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

/// A parsed metadata table (header + string rows; the id is the first cell).
pub struct Metadata {
    /// Column names, in file order.
    pub columns: Vec<String>,
    /// One entry per data row; each is `columns.len()` string cells.
    pub rows: Vec<Vec<String>>,
}

impl Metadata {
    /// Values of the named column across all rows, in file order.
    pub fn column(&self, name: &str) -> Vec<&str> {
        let idx = self
            .columns
            .iter()
            .position(|c| c == name)
            .unwrap_or_else(|| panic!("metadata has no column {name:?}"));
        self.rows.iter().map(|r| r[idx].as_str()).collect()
    }
}

/// Parse a metadata TSV (header + string rows), test-only (panics on error).
pub fn load_metadata(path: &Path) -> Metadata {
    let text = read_to_string(path);
    let mut lines = text.lines().filter(|l| !l.is_empty());
    let header = lines
        .next()
        .unwrap_or_else(|| panic!("{}: empty file", path.display()));
    let columns: Vec<String> = header.split('\t').map(str::to_owned).collect();
    let mut rows = Vec::new();
    for line in lines {
        let cells: Vec<String> = line.split('\t').map(str::to_owned).collect();
        assert_eq!(cells.len(), columns.len(), "ragged metadata row");
        rows.push(cells);
    }
    Metadata { columns, rows }
}

fn read_to_string(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// Nonzero COO triples `(row, col, value)` from the matrix, in row-major order.
fn coo_triples(m: &Matrix) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
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
    (rows, cols, vals)
}

/// Per-sample roles and environments, aligned to the table's column order
/// (`m.col_labels`) by matching sample ids into the metadata. Sinks keep their
/// (unused) environment value, exactly as the core-direct path does.
fn roles_and_envs(m: &Matrix, md: &Metadata) -> (Vec<Role>, Vec<Option<String>>) {
    let ids = md.column("sample_id");
    let source_sink = md.column("source_sink");
    let env = md.column("env");
    let mut roles = Vec::with_capacity(m.col_labels.len());
    let mut envs = Vec::with_capacity(m.col_labels.len());
    for sid in &m.col_labels {
        let idx = ids
            .iter()
            .position(|x| x == sid)
            .unwrap_or_else(|| panic!("sample {sid} absent from metadata"));
        roles.push(match source_sink[idx] {
            "source" => Role::Source,
            "sink" => Role::Sink,
            other => panic!("unexpected source_sink value {other:?}"),
        });
        envs.push(Some(env[idx].to_owned()));
    }
    (roles, envs)
}

/// Build core types directly from a fixture (the reference path).
pub fn core_direct(m: &Matrix, md: &Metadata) -> (CountTable, SampleContext) {
    let (rows, cols, vals) = coo_triples(m);
    let table = CountTable::from_coo(
        m.row_labels.clone(),
        m.col_labels.clone(),
        &rows,
        &cols,
        &vals,
    )
    .expect("fixture table builds");
    let (roles, envs) = roles_and_envs(m, md);
    let ctx = SampleContext::for_table(&table, roles, envs).expect("context builds");
    (table, ctx)
}

/// Build the equivalent Arrow inputs from the same fixture: a COO `StructArray`
/// (Int32 `row`/`col`, Float64 `val`), a feature-id array, and a metadata
/// `RecordBatch` (`sample_id`/`role`/`env`) aligned to the table's column order.
pub fn arrow_inputs(m: &Matrix, md: &Metadata) -> (StructArray, ArrayRef, RecordBatch) {
    let (rows, cols, vals) = coo_triples(m);
    let coo = StructArray::from(vec![
        (
            Arc::new(Field::new("row", DataType::Int32, false)),
            Arc::new(Int32Array::from(
                rows.iter().map(|&r| r as i32).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            Arc::new(Field::new("col", DataType::Int32, false)),
            Arc::new(Int32Array::from(
                cols.iter().map(|&c| c as i32).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            Arc::new(Field::new("val", DataType::Float64, false)),
            Arc::new(Float64Array::from(vals)) as ArrayRef,
        ),
    ]);

    let feature_ids: ArrayRef = Arc::new(StringArray::from(
        m.row_labels.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    ));

    let (roles, envs) = roles_and_envs(m, md);
    let role_strs: Vec<&str> = roles
        .iter()
        .map(|r| match r {
            Role::Source => "source",
            Role::Sink => "sink",
        })
        .collect();
    let env_strs: Vec<Option<&str>> = envs.iter().map(|e| e.as_deref()).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("sample_id", DataType::Utf8, false),
        Field::new("role", DataType::Utf8, false),
        Field::new("env", DataType::Utf8, true),
    ]));
    let metadata = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(
                m.col_labels.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(role_strs)) as ArrayRef,
            Arc::new(StringArray::from(env_strs)) as ArrayRef,
        ],
    )
    .expect("metadata batch builds");

    (coo, feature_ids, metadata)
}
