// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Smoke benchmark: proves the criterion harness compiles and runs. Replaced by
//! real benchmarks in the performance milestone.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn smoke(c: &mut Criterion) {
    c.bench_function("noop", |b| b.iter(|| black_box(st3_core::name())));
}

criterion_group!(benches, smoke);
criterion_main!(benches);
