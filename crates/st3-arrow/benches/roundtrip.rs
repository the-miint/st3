// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Criterion benchmarks for the Arrow boundary: COO+metadata import and the
//! dense/streaming export path.
//!
//! A deterministic scenario is generated with [`st3_core::simulate_two_source`],
//! turned into the three Arrow inputs `import` expects (a COO `StructArray`, a
//! feature-id array, and a `sample_id`/`role`/`env` metadata `RecordBatch`), and
//! then run once through the core to produce a [`st3_core::SourceMixing`] with a
//! contingency table. The benches time, separately: `import`, the dense
//! `means_batch`/`stds_batch` exports, and draining the streaming
//! `contingency_reader`. Inputs are built once, outside `b.iter`.

use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int32Array, RecordBatch, StringArray, StructArray};
use arrow::datatypes::{DataType, Field, Schema};
use criterion::{Criterion, criterion_group, criterion_main};
use st3_arrow::{contingency_reader, import, means_batch, stds_batch};
use st3_core::{
    CollapseMethod, GibbsParams, Role, SimConfig, Simulation, predict_sinks, simulate_two_source,
};

/// Fixed seed for generation and estimation.
const SEED: u64 = 42;

/// A deterministic two-source scenario sized for the boundary benches.
fn make_sim() -> Simulation {
    let cfg = SimConfig {
        n_taxa: 600,
        samples_per_source: 3,
        seqs_per_sample: 600,
        n_trials: 12,
        sink_depth: 600,
        concentration: 0.1,
    };
    simulate_two_source(&cfg, SEED).expect("benchmark simulation is valid")
}

/// Light sampler params with the contingency tally on (so the exported mixing
/// carries a per-sink assignment table to drain).
fn params() -> GibbsParams {
    GibbsParams {
        alpha1: 0.001,
        alpha2: 0.1,
        beta: 10.0,
        restarts: 10,
        draws_per_restart: 4,
        burnin: 10,
        delay: 1,
        collapse: CollapseMethod::Sum,
        contingency: true,
    }
}

/// Build the three Arrow `import` inputs from a simulation's core table+context.
fn arrow_inputs(sim: &Simulation) -> (StructArray, ArrayRef, RecordBatch) {
    let table = sim.table();
    let ctx = sim.context();

    // COO nonzeros, column-major (import does not require row-major order).
    let mut rows: Vec<i32> = Vec::new();
    let mut cols: Vec<i32> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    for s in 0..table.n_samples() {
        let (feats, counts) = table.column(s);
        for (&f, &c) in feats.iter().zip(counts.iter()) {
            rows.push(f as i32);
            cols.push(s as i32);
            vals.push(f64::from(c));
        }
    }
    let coo = StructArray::from(vec![
        (
            Arc::new(Field::new("row", DataType::Int32, false)),
            Arc::new(Int32Array::from(rows)) as ArrayRef,
        ),
        (
            Arc::new(Field::new("col", DataType::Int32, false)),
            Arc::new(Int32Array::from(cols)) as ArrayRef,
        ),
        (
            Arc::new(Field::new("val", DataType::Float64, false)),
            Arc::new(Float64Array::from(vals)) as ArrayRef,
        ),
    ]);

    let feature_ids: ArrayRef = Arc::new(StringArray::from(
        table
            .feature_ids()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
    ));

    let sample_ids: Vec<&str> = table.sample_ids().iter().map(String::as_str).collect();
    let roles: Vec<&str> = (0..table.n_samples())
        .map(|i| match ctx.role(i) {
            Role::Source => "source",
            Role::Sink => "sink",
        })
        .collect();
    let envs: Vec<Option<&str>> = (0..table.n_samples()).map(|i| ctx.env(i)).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("sample_id", DataType::Utf8, false),
        Field::new("role", DataType::Utf8, false),
        Field::new("env", DataType::Utf8, true),
    ]));
    let metadata = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(sample_ids)) as ArrayRef,
            Arc::new(StringArray::from(roles)) as ArrayRef,
            Arc::new(StringArray::from(envs)) as ArrayRef,
        ],
    )
    .expect("metadata batch builds");

    (coo, feature_ids, metadata)
}

fn bench_roundtrip(c: &mut Criterion) {
    let sim = make_sim();
    let (coo, feature_ids, metadata) = arrow_inputs(&sim);

    let mut group = c.benchmark_group("arrow");

    // Import: COO + metadata -> core CountTable + SampleContext.
    group.bench_function("import", |b| {
        b.iter(|| {
            black_box(
                import(black_box(&coo), feature_ids.as_ref(), black_box(&metadata))
                    .expect("import"),
            )
        });
    });

    // Produce a mixing (with contingency) once for the export benches.
    let mixing = predict_sinks(sim.table(), sim.context(), &params(), SEED, 1)
        .expect("predict_sinks for export benches");

    group.bench_function("means_batch", |b| {
        b.iter(|| black_box(means_batch(black_box(&mixing)).expect("means_batch")));
    });
    group.bench_function("stds_batch", |b| {
        b.iter(|| black_box(stds_batch(black_box(&mixing)).expect("stds_batch")));
    });
    group.bench_function("contingency_drain", |b| {
        b.iter(|| {
            let reader = contingency_reader(black_box(&mixing)).expect("contingency present");
            let batches: Vec<_> = reader.map(|r| r.expect("contingency batch")).collect();
            black_box(batches)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_roundtrip);
criterion_main!(benches);
