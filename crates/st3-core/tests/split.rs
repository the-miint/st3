// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Source/sink split against the committed fixtures.

mod common;

use common::{build_context, build_table, fixtures_dir, load_matrix, load_metadata};

#[test]
fn synthetic_small_split() {
    let dir = fixtures_dir().join("synthetic_small");
    let table = build_table(&load_matrix(&dir.join("table.tsv")));
    let ctx = build_context(&table, &load_metadata(&dir.join("metadata.tsv")));
    assert_eq!(ctx.source_indices().len(), 6, "6 source samples");
    assert_eq!(ctx.sink_indices().len(), 3, "3 sink samples");
}

#[test]
fn tiny_test_split() {
    let dir = fixtures_dir().join("tiny_test");
    let table = build_table(&load_matrix(&dir.join("table.tsv")));
    let ctx = build_context(&table, &load_metadata(&dir.join("metadata.tsv")));
    assert_eq!(ctx.source_indices().len(), 5, "5 source samples");
    assert_eq!(ctx.sink_indices().len(), 5, "5 sink samples");
}
