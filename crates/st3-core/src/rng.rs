// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Deterministic per-work-item PRNG seeding, shared across milestones.
//!
//! Every stochastic unit of work — a rarefaction column here, and (later) each
//! sink's Gibbs chains in the estimator — derives its own PRNG from the run
//! `seed` and the item's index. Because a stream depends only on
//! `(seed, item_index)` and never on iteration order, thread, or job count,
//! results are identical regardless of how the work is scheduled or
//! parallelized. This is the project's single reproducibility knob: one seed.

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// The concrete PRNG used for every stochastic work item.
///
/// Xoshiro256++ has tiny state, is fast, and seeds cleanly from a `u64`.
pub type ItemRng = Xoshiro256PlusPlus;

/// Odd 64-bit constant (fractional bits of the golden ratio) used to
/// decorrelate adjacent work-item indices before seeding.
const PHI64: u64 = 0x9E37_79B9_7F4A_7C15;

/// Derive an independent PRNG for work item `item_index` from the run `seed`.
///
/// The stream is a pure function of `(seed, item_index)`, so a given work item
/// produces the same randomness no matter when or on which thread it runs.
/// `seed_from_u64` applies a SplitMix64 avalanche internally, so even the
/// closely-spaced inputs `seed ^ (item_index * PHI64)` yield well-separated
/// generator states.
#[must_use]
pub fn rng_for_item(seed: u64, item_index: u64) -> ItemRng {
    Xoshiro256PlusPlus::seed_from_u64(seed ^ item_index.wrapping_mul(PHI64))
}

/// Stage tag for [`stage_seed`]: the per-sink subsampling of a rarefied run.
pub(crate) const STAGE_RAREFY_SINKS: u64 = 0x5349_4e4b_5241_5245;
/// Stage tag for [`stage_seed`]: the source subsampling of a rarefied run (the
/// collapsed environments in sink mode, the source samples in leave-one-out).
pub(crate) const STAGE_RAREFY_SOURCES: u64 = 0x5352_4352_4152_4546;

/// Derive the seed for one stochastic *stage* of a run from the run `seed`.
///
/// A rarefied run has up to three stochastic stages — sink subsampling, source
/// subsampling, and the sampler — and each seeds its own work items with
/// [`rng_for_item`]`(stage seed, index)`. Giving every stage its own seed keeps
/// their streams apart: column `s` of the sink subsampling never shares a stream
/// with sink item `s` of the sampler, and the sampler, which keeps the run seed
/// itself, draws the same randomness with or without rarefaction.
#[must_use]
pub(crate) fn stage_seed(seed: u64, stage: u64) -> u64 {
    rng_for_item(seed, stage).next_u64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn same_seed_and_index_give_identical_stream() {
        let mut a = rng_for_item(42, 7);
        let mut b = rng_for_item(42, 7);
        for _ in 0..8 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn distinct_indices_differ() {
        let mut a = rng_for_item(42, 0);
        let mut b = rng_for_item(42, 1);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn distinct_seeds_differ() {
        let mut a = rng_for_item(1, 0);
        let mut b = rng_for_item(2, 0);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    // Each stochastic stage of a rarefied run gets its own seed, so no two
    // stages (or a stage and the sampler) ever share a per-item stream.
    #[test]
    fn stage_seeds_are_distinct_from_the_run_seed_and_each_other() {
        let seed = 42;
        let sinks = stage_seed(seed, STAGE_RAREFY_SINKS);
        let sources = stage_seed(seed, STAGE_RAREFY_SOURCES);
        assert_ne!(sinks, seed);
        assert_ne!(sources, seed);
        assert_ne!(sinks, sources);
        // Deterministic: a pure function of (seed, stage).
        assert_eq!(sinks, stage_seed(seed, STAGE_RAREFY_SINKS));
        assert_ne!(sinks, stage_seed(seed + 1, STAGE_RAREFY_SINKS));
    }
}
