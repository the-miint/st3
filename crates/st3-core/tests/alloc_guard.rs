// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Machine-independent allocation guard for the sampler's per-pass loop.
//!
//! Design §15 lists "zero per-pass allocation" as a structural performance win:
//! the estimator allocates all per-sink scratch (`seq_env`, `envcounts`,
//! `unknown_vector`, `order`, `jp`) once, before the restart/pass loops, and the
//! per-pass loop itself allocates nothing (`estimate.rs`). This test encodes that
//! invariant deterministically, so it belongs in the default gate (`make test`)
//! as the real CI regression net — unlike the wall-clock guard, it needs no
//! reference machine.
//!
//! The probe is a **counting global allocator** (local to this integration-test
//! binary) plus a burnin-invariance argument: two `predict_sinks` runs that
//! differ *only* in `burnin` must record the **identical** number of
//! allocations. Extra burnin passes execute the per-pass loop more times but
//! record no draws, so if that loop is allocation-free the two counts match
//! exactly; the only allocations are the fixed scratch plus the
//! `restarts · draws` ensemble vectors, both invariant in `burnin`. If the counts
//! diverge, a per-pass allocation has crept into the hot loop.
//!
//! Everything runs in a **single** `#[test]` function on one thread. The counter
//! is process-global, so a second test measuring concurrently (or the harness
//! printing another test's result mid-measurement) could corrupt an exact count;
//! one sequential test removes that race entirely. It runs at `jobs = 1`, so no
//! rayon pool is built and no worker thread allocates behind the measurement.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

use st3_core::{
    CollapseMethod, GibbsParams, SimConfig, Simulation, predict_sinks, simulate_two_source,
};

/// Count of `alloc` calls since process start.
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

/// A `System`-delegating allocator that counts allocations.
///
/// Only `alloc`/`dealloc` are overridden; `realloc` and `alloc_zeroed` inherit
/// the `GlobalAlloc` default implementations, which route through `alloc` — so
/// `Vec` growth (a realloc) is counted too, not silently missed.
struct Counting;

// SAFETY: every method forwards to the `System` allocator with the same layout,
// so the allocator contract is upheld; the only added behaviour is an atomic
// increment, which cannot affect memory safety.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Run `f` and return `(allocations_it_performed, its_result)`.
fn allocations_of<T>(f: impl FnOnce() -> T) -> (usize, T) {
    let before = ALLOCS.load(Ordering::Relaxed);
    let value = f();
    let after = ALLOCS.load(Ordering::Relaxed);
    (after - before, value)
}

/// Sampler params with a chosen `burnin` and contingency flag; everything else
/// (restarts, draws, delay, collapse) is fixed so only the named axis varies.
fn params(burnin: u32, contingency: bool) -> GibbsParams {
    GibbsParams {
        alpha1: 0.001,
        alpha2: 0.1,
        beta: 10.0,
        restarts: 4,
        draws_per_restart: 2,
        burnin,
        delay: 1,
        collapse: CollapseMethod::Sum,
        contingency,
    }
}

/// A small deterministic workload: 2 sources, 3 sinks, shallow depth — enough to
/// exercise the full restart/burnin loop while staying fast even at burnin=500.
fn dataset() -> Simulation {
    let cfg = SimConfig {
        n_taxa: 50,
        samples_per_source: 1,
        seqs_per_sample: 100,
        n_trials: 3,
        sink_depth: 100,
        concentration: 0.2,
    };
    simulate_two_source(&cfg, 7).expect("simulation is valid")
}

#[test]
fn per_pass_loop_allocates_nothing() {
    let sim = dataset();
    let table = sim.table();
    let ctx = sim.context();

    // Warm up once so any first-call lazy initialization lands outside the
    // measured windows and cannot skew the first count.
    let warm = predict_sinks(table, ctx, &params(5, false), 1, 1).expect("warmup");
    black_box(&warm);
    drop(warm);

    // (1) Burnin invariance: identical everything but burnin ⇒ identical allocs,
    // because extra burnin passes run the per-pass loop but record no draws and
    // (if the loop is alloc-free) allocate nothing.
    let short = params(5, false);
    let long = params(500, false);
    let (c_short, r_short) =
        allocations_of(|| predict_sinks(table, ctx, &short, 1, 1).expect("short burnin"));
    black_box(&r_short);
    let (c_long, r_long) =
        allocations_of(|| predict_sinks(table, ctx, &long, 1, 1).expect("long burnin"));
    black_box(&r_long);
    assert_eq!(
        c_short, c_long,
        "per-pass loop is not allocation-free: burnin=5 did {c_short} allocs, \
         burnin=500 did {c_long} (the extra 495 passes should allocate nothing)"
    );

    // (2) Skip-storage win (design §15.4): with the contingency tally off, the
    // sampler must allocate no more than with it on (it skips the per-sink
    // V×τ tally and its clones). A generous `<=` — the point is that off never
    // *exceeds* on.
    let off = params(20, false);
    let on = params(20, true);
    let (c_off, r_off) =
        allocations_of(|| predict_sinks(table, ctx, &off, 1, 1).expect("contingency off"));
    black_box(&r_off);
    let (c_on, r_on) =
        allocations_of(|| predict_sinks(table, ctx, &on, 1, 1).expect("contingency on"));
    black_box(&r_on);
    assert!(
        c_off <= c_on,
        "contingency off ({c_off} allocs) allocated more than on ({c_on}); \
         the skip-assignment-storage path should never allocate more"
    );
}
