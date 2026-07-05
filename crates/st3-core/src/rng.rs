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

use rand::SeedableRng;
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
}
