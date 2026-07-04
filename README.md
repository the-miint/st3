# SourceTracker3 (`st3`)

A Rust-core, C/C++-facing library that reimplements SourceTracker: Bayesian
estimation, via a collapsed Gibbs sampler, of the source environments that
contributed to a microbial community ("sink").

- **Library only** — no CLI. Consumed from C/C++ via a `cbindgen`-generated header (`st3.h`).
- **Apache Arrow** at the API boundary; sparse **COO** input, dense + COO output.
- Rust workspace: `st3-core` (algorithm), `st3-arrow` (Arrow ⇄ core), `st3-capi` (C ABI).

## Goals

- **Verifiable statistical equivalence** to the R/Python SourceTracker lineage.
- A **clean, simple, well-documented** C API.
- **Performance** — benchmarked, with a regression guard.

## Status

Pre-implementation. Work proceeds milestone by milestone, each on its own
branch, test-driven (red/green/refactor), merged to `main` only when green.

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
