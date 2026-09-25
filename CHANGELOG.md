# Changelog

All notable changes to SourceTracker3 are recorded here. Versions follow
[Semantic Versioning](https://semver.org/); the C ABI additionally carries its
own version (`st3_abi_version`, `ST3_CONFIG_V1`) that is bumped only on a
breaking ABI change.

## [Unreleased]

### Changed
- The generated C header is committed at `crates/st3-capi/include/st3.h`, so a
  consumer no longer has to locate cargo's `OUT_DIR`. `make header` refreshes
  it and the test gate fails if it drifts from the generated one. The header
  now states up front that the Arrow C Data Interface structs must be declared
  before including it (#1).
- Sink-mode rarefaction now follows SourceTracker2's order: the sources are
  collapsed first and each *collapsed environment* is then subsampled to the
  source depth, so every environment enters the sampler with exactly that many
  sequences. Previously each source *sample* was subsampled before collapse,
  which lowered the sampling variance of the pooled profile and, under mean
  collapse, left the environment below the depth. Leave-one-out was already in
  the reference order (per source sample, then per-fold collapse) and is
  unchanged; its sink depth is now ignored, as in the reference. New
  `predict_sinks_rarefied` and `predict_loo_rarefied` drivers in `st3-core`
  take a `RarefyConfig` and implement this order, `CollapsedSources::rarefy`
  subsamples a collapsed table, and `st3_run` lowers onto the new drivers. The
  sink-subsampling, source-subsampling, and sampler stages now draw from
  distinct seed-derived streams (#3).
- Rarefaction now fails fast on shallow input, as SourceTracker2 does: when a
  depth is set and a collapsed source environment (sink mode), a source sample
  (leave-one-out), or a sink cannot reach it, the run is refused before any
  sampling with `Error::ShallowSamples` in `st3-core` and the previously
  reserved `ST3_STATUS_ERR_SHALLOW_SAMPLE` across the C ABI; the last-error
  names the role, how many samples fall short, and the shallowest total.
  Previously such samples were passed through unchanged and the run reported
  success. The `rarefy` / `rarefy_per_sample` primitives keep their
  flag-and-pass-through behaviour for callers composing their own pipeline
  (#2).
- Empty sinks (sink mode) and empty held-out sources (leave-one-out) are now
  rejected before collapse, subsampling, model precompute, or the worker pool
  start, rather than inside the per-item work, where with `jobs > 1` other
  items could run their full Gibbs chains before the error surfaced. The
  reported error is unchanged: the lowest-index empty column (#5).

### Added
- A CI job that type-checks `st3-capi` for `wasm32-unknown-emscripten`, the
  target of duckdb-miint's DuckDB-Wasm build, so the crate stays buildable
  there (#6).

## [1.0.0] — 2026-07-06

First release. SourceTracker3 is a library-only, C/C++-facing reimplementation
of SourceTracker — Bayesian estimation, via a collapsed Gibbs sampler, of the
source environments that contributed to a microbial community ("sink"). It is
an independent, clean-room implementation of the algorithm described in Knights
et al., *Nature Methods* 2011, written in Rust with an Apache Arrow API boundary
and a `cbindgen`-generated C header. Licensed BSD 3-Clause.

The workspace is three crates with a one-way dependency direction
(`st3-capi → st3-arrow → st3-core`):

- **`st3-core`** — the pure algorithm (no FFI, Arrow, or I/O).
- **`st3-arrow`** — the Apache Arrow ⇄ core data boundary.
- **`st3-capi`** — the reentrant C ABI (`libst3.so` / `libst3.a` + `st3.h`).

### Features

**Estimation**
- Sink source-prediction (`predict_sinks`) and per-sample, source-scoped
  leave-one-out (`predict_loo`).
- Collapsed-Gibbs sampler behind a pluggable `SinkModel` estimator trait, so an
  alternate estimator can be added later without disturbing the output contract.
- Per-sink mixing means with **correct per-draw standard deviations** (the
  population standard deviation across retained draws), plus an opt-in per-sink
  source × taxon assignment tally emitted sparsely (COO).

**Preprocessing**
- Sparse compressed-sparse-column count table with a single, well-defined
  float→integer floor at COO ingest and eager, descriptive validation.
- Source/sink split and per-environment collapse, configurable `Mean` or `Sum`.
- Seeded rarefaction, with or without replacement, preserving the feature axis.

**Determinism & parallelism**
- Output is a function of the run seed alone: each work item (sink or held-out
  sample) is seeded from `(seed, index)`, never from thread or iteration order.
- Per-sink / per-fold parallelism over a per-call scoped `rayon` pool sized by
  `jobs` (`0` = all cores, `1` = serial, `n` = `n` threads); results are
  byte-identical across thread counts.

**Boundaries & interop**
- Arrow import from a COO `StructArray` plus a per-sample metadata table; dense
  `RecordBatch` exports of means and standard deviations; the contingency table
  streamed as one flat-COO batch per sink.
- A small, reentrant C ABI over the Apache Arrow C Data Interface: opaque
  handles, a versioned `St3Config`, a coarse `St3Status` return paired with a
  thread-local `st3_last_error` string, and an unwind guard on every entry so a
  panic can never cross the C boundary.

**Equivalence & correctness**
- Validated for statistical equivalence against committed reference numeric
  output (sum-collapse and mean-collapse) and against first-principles
  invariants, with a determinism guard asserting identical output across thread
  counts.
- An in-process equivalence harness generates two-source Dirichlet mixtures with
  known mixing weights and scores recovery (R² related to the Jensen–Shannon
  divergence between sources).

**Performance**
- Criterion benchmark suite, a machine-independent allocation guard in the test
  gate, and an opt-in wall-clock regression guard against a committed baseline.

**Documentation**
- `#![deny(missing_docs)]` on every crate, runnable doctests on the flagship
  API, a documented and compiled C example
  ([`crates/st3-capi/examples/usage.c`](crates/st3-capi/examples/usage.c)), a
  C-clean generated header, and a README "Building & using from C" guide with a
  `make header` convenience.

### Not included

- **No CLI** — SourceTracker3 is a library. QIIME2 integration is out of scope.
- α-tuning, SIMD multiversioning, and alternate estimators are deferred to a
  future release; the estimator seam and an evaluation module exist to support
  them.

[1.0.0]: https://github.com/the-miint/st3/releases/tag/v1.0.0
