// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Deterministic, reentrant fan-out over independent work items.
//!
//! The estimator's work is embarrassingly parallel: each sink (and each held-out
//! sample in leave-one-out) is a self-contained unit whose randomness derives
//! only from `(seed, item_index)` via [`crate::rng::rng_for_item`]. [`map_items`]
//! runs those units across a **per-call scoped** rayon thread pool sized by the
//! caller's `jobs`, rather than the rayon global pool — two concurrent callers
//! never fight over pool sizing, which keeps the library reentrant under a
//! threaded host.
//!
//! Two invariants make the output independent of `jobs` (and of scheduling):
//! results are reassembled in item order, so no cross-item floating-point
//! reduction is reordered; and when a unit fails, the **lowest-index** error is
//! returned, exactly as the serial loop would. A small-input guard runs serially
//! (no pool, no task-spawn overhead) when there is at most one item or `jobs`
//! is one.

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::error::{Error, Result};

/// Apply `f` to each index `0..n`, returning the results in index order.
///
/// `jobs` sizes a per-call scoped thread pool: `0` uses all logical cores, `1`
/// runs serially, and `n` uses exactly `n` worker threads. Regardless of `jobs`,
/// the returned vector is in ascending index order and, on failure, carries the
/// error of the **lowest** failing index — so the result is byte-identical across
/// thread counts (given an `f` whose per-item output depends only on the index).
///
/// # Errors
/// The lowest-index [`Err`] returned by `f`, or [`Error::ThreadPool`] if the
/// scoped pool cannot be built.
pub(crate) fn map_items<T, F>(jobs: usize, n: usize, f: F) -> Result<Vec<T>>
where
    F: Fn(usize) -> Result<T> + Sync + Send,
    T: Send,
{
    // Small-input / serial guard: no pool, no task-spawn overhead. `map` is
    // ordered, so `?` surfaces the lowest-index error.
    if jobs == 1 || n < 2 {
        return (0..n).map(&f).collect();
    }

    let pool = build_pool(jobs)?;
    // Rayon's indexed `collect` preserves order, so `results[i]` is `f(i)`. We
    // collect every result (not short-circuiting) and pick the lowest-index error
    // afterwards, so error selection does not depend on which thread finished.
    let results: Vec<Result<T>> = pool.install(|| (0..n).into_par_iter().map(&f).collect());
    results.into_iter().collect()
}

/// Build a scoped rayon thread pool with `jobs` workers (`0` = all cores).
fn build_pool(jobs: usize) -> Result<rayon::ThreadPool> {
    let mut builder = rayon::ThreadPoolBuilder::new();
    if jobs != 0 {
        builder = builder.num_threads(jobs);
    }
    builder.build().map_err(|e| Error::ThreadPool {
        reason: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // (a) order is preserved and results match across job counts.
    #[test]
    fn preserves_order_across_job_counts() {
        for jobs in [0usize, 1, 2, 8] {
            let out = map_items(jobs, 5, |i| Ok::<usize, Error>(i * 10)).unwrap();
            assert_eq!(out, vec![0, 10, 20, 30, 40], "jobs={jobs}");
        }
    }

    // (a) the lowest-index error is returned regardless of job count.
    #[test]
    fn returns_lowest_index_error() {
        for jobs in [1usize, 2, 8] {
            let err = map_items(jobs, 6, |i| {
                if i == 2 || i == 4 {
                    Err(Error::EmptySink { sample_index: i })
                } else {
                    Ok(i)
                }
            })
            .unwrap_err();
            assert_eq!(err, Error::EmptySink { sample_index: 2 }, "jobs={jobs}");
        }
    }

    // (b) small-input guard: n < 2 never spawns the pool. We can only observe it
    // indirectly, so assert the trivial cases still produce correct results.
    #[test]
    fn small_input_is_handled_serially() {
        assert_eq!(
            map_items(8, 0, Ok::<usize, Error>).unwrap(),
            Vec::<usize>::new()
        );
        assert_eq!(
            map_items(8, 1, |i| Ok::<usize, Error>(i + 1)).unwrap(),
            vec![1]
        );
    }

    // Every index is visited exactly once (no dropped or duplicated work).
    #[test]
    fn visits_every_index_once() {
        let counter = AtomicUsize::new(0);
        let out = map_items(4, 100, |i| {
            counter.fetch_add(1, Ordering::Relaxed);
            Ok::<usize, Error>(i)
        })
        .unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 100);
        assert_eq!(out, (0..100).collect::<Vec<_>>());
    }
}
