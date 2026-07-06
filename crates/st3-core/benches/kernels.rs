// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Criterion benchmarks over the real hot paths of `st3-core`.
//!
//! Benches see only the crate's **public** API (they are separate compilation
//! units, so the `tests/` loaders are out of reach), so every workload is built
//! from [`st3_core::simulate_two_source`] — a public, deterministic, sized
//! generator. Each input is constructed once, outside `b.iter`, and the timed
//! closure wraps it in `black_box` so the optimizer cannot hoist the work out.
//!
//! Groups (ids are stable, so the opt-in wall-clock guard in
//! `tests/perf_guard.rs` can key off them):
//! * `predict_sinks/{small,medium}` — the collapsed-Gibbs sampler end to end,
//!   contingency off, at two representative sizes;
//! * `predict_sinks_contingency/{off,on}` — a dedicated size with the per-sink
//!   assignment tally off vs on, exposing the skip-storage win (design §15.4);
//! * `predict_loo` — the leave-one-out driver (one fold per source sample);
//! * `rarefy_per_sample` — seeded subsampling of every column;
//! * `predict_sinks_parallel/{serial,all_cores}` — the scoped-pool scaling of a
//!   many-sink workload at `jobs = 1` vs `jobs = 0`.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use st3_core::{
    CollapseMethod, GibbsParams, SimConfig, Simulation, predict_loo, predict_sinks,
    rarefy_per_sample, simulate_two_source,
};

/// A fixed seed for both generation and estimation across every bench.
const SEED: u64 = 42;

/// Light-but-representative sampler parameters. Restarts/draws/burnin are kept
/// modest so `cargo bench` stays in the tens-of-seconds range while still
/// exercising the full restart/burnin/thinned-draw schedule.
fn sampler_params(contingency: bool) -> GibbsParams {
    GibbsParams {
        alpha1: 0.001,
        alpha2: 0.1,
        beta: 10.0,
        restarts: 20,
        draws_per_restart: 5,
        burnin: 20,
        delay: 1,
        collapse: CollapseMethod::Sum,
        contingency,
    }
}

/// Build a deterministic two-source scenario of the given shape.
fn make_sim(
    n_taxa: usize,
    samples_per_source: usize,
    seqs_per_sample: u32,
    n_trials: usize,
    sink_depth: u32,
) -> Simulation {
    let cfg = SimConfig {
        n_taxa,
        samples_per_source,
        seqs_per_sample,
        n_trials,
        sink_depth,
        concentration: 0.1,
    };
    simulate_two_source(&cfg, SEED).expect("benchmark simulation is valid")
}

/// `predict_sinks` at two sizes, contingency off.
fn bench_predict_sinks(c: &mut Criterion) {
    let params = sampler_params(false);
    let sizes = [
        ("small", make_sim(200, 1, 400, 6, 400)),
        ("medium", make_sim(1000, 3, 800, 10, 800)),
    ];
    let mut group = c.benchmark_group("predict_sinks");
    for (label, sim) in &sizes {
        group.bench_with_input(BenchmarkId::from_parameter(label), sim, |b, sim| {
            b.iter(|| {
                black_box(
                    predict_sinks(sim.table(), sim.context(), &params, SEED, 1)
                        .expect("predict_sinks"),
                )
            });
        });
    }
    group.finish();
}

/// The contingency (per-sink assignment tally) off/on cost. Its own dedicated
/// size (distinct from `predict_sinks/{small,medium}`) so the off case is not a
/// redundant re-timing of `predict_sinks/medium`; the off↔on delta on one fixed
/// workload isolates the tally's cost.
fn bench_contingency(c: &mut Criterion) {
    let sim = make_sim(700, 3, 700, 8, 700);
    let mut group = c.benchmark_group("predict_sinks_contingency");
    for (label, contingency) in [("off", false), ("on", true)] {
        let params = sampler_params(contingency);
        group.bench_with_input(BenchmarkId::from_parameter(label), &params, |b, params| {
            b.iter(|| {
                black_box(
                    predict_sinks(sim.table(), sim.context(), params, SEED, 1)
                        .expect("predict_sinks"),
                )
            });
        });
    }
    group.finish();
}

/// Leave-one-out: one fold per source sample (here 2 × 5 = 10 folds).
fn bench_predict_loo(c: &mut Criterion) {
    let sim = make_sim(400, 5, 500, 2, 500);
    let params = sampler_params(false);
    c.bench_function("predict_loo", |b| {
        b.iter(|| {
            black_box(
                predict_loo(sim.table(), sim.context(), &params, SEED, 1).expect("predict_loo"),
            )
        });
    });
}

/// Seeded rarefaction of every column to a common depth. Sources (2000 deep) and
/// sinks (2000 deep) all subsample down to 1000, so every column is `Rarefied`.
fn bench_rarefy(c: &mut Criterion) {
    let sim = make_sim(500, 8, 2000, 8, 2000);
    let table = sim.table();
    let depths = vec![Some(1000u32); table.n_samples()];
    c.bench_function("rarefy_per_sample", |b| {
        b.iter(|| {
            black_box(rarefy_per_sample(table, black_box(&depths), false, SEED).expect("rarefy"))
        });
    });
}

/// Parallel scaling of a many-sink workload: serial (`jobs = 1`) vs all logical
/// cores (`jobs = 0`). Output is byte-identical across both by construction; this
/// only measures the scoped-pool crossover.
fn bench_parallel(c: &mut Criterion) {
    let sim = make_sim(500, 3, 600, 32, 600);
    let params = sampler_params(false);
    let mut group = c.benchmark_group("predict_sinks_parallel");
    for (label, jobs) in [("serial", 1usize), ("all_cores", 0usize)] {
        group.bench_with_input(BenchmarkId::from_parameter(label), &jobs, |b, &jobs| {
            b.iter(|| {
                black_box(
                    predict_sinks(sim.table(), sim.context(), &params, SEED, jobs)
                        .expect("predict_sinks"),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_predict_sinks,
    bench_contingency,
    bench_predict_loo,
    bench_rarefy,
    bench_parallel
);
criterion_main!(benches);
