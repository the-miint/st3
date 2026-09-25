# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ABSOLUTE HARD REQUIREMENTS
- NEVER use `rm` without permission

## Project Overview

**SourceTracker3** (`st3`) is a library-only, C/C++-facing Rust reimplementation of
SourceTracker: Bayesian estimation, via a collapsed Gibbs sampler, of the source
environments that contributed to a microbial community ("sink"). It is a clean-room
implementation of the algorithm in Knights et al., *Nature Methods* 2011.

Three crates, one-way dependency direction (`st3-capi → st3-arrow → st3-core`):

- **`st3-core`** — the pure algorithm. No FFI, no Arrow, no I/O.
- **`st3-arrow`** — the Apache Arrow ⇄ core data boundary.
- **`st3-capi`** — the reentrant C ABI (`libst3.so` / `libst3.a` + generated `st3.h`).

No CLI. Consumers are C/C++ (via `st3.h`) or Rust (depending on `st3-core` directly).

## Priorities

1. Red/green/refactor Test Driven Development (TDD)
2. Verifiably correct code
3. Maintainable code, using Don't Repeat Yourself (DRY) and Keep It Simple Stupid (KISS)
4. Model quality — statistical fidelity and estimator improvements outrank raw speed
5. Performance

Priorities 4 and 5 are not in tension by default: the goal is both. When they do
conflict, a better model wins over a faster one.

## Rules

These rules apply to every task in this project unless explicitly overridden.

### Rule 1 — Think Before Coding
Bias: caution over speed on non-trivial work.
State assumptions explicitly. Ask rather than guess.
Push back when a simpler approach exists. Stop when confused.

### Rule 2 — Simplicity First
Minimum code that solves the problem. Nothing speculative.
No abstractions for single-use code.

### Rule 3 — Surgical Changes
Touch only what you must. Don't improve adjacent code.
Match existing style. Don't refactor what isn't broken.

### Rule 4 — Goal-Driven Execution
Define success criteria. Loop until verified.

### Rule 5 — Surface conflicts, don't average them
If two patterns contradict, pick one (more recent / more tested).
Explain why. Flag the other for cleanup.

### Rule 6 — Read before you write
Before adding code, read exports, immediate callers, shared utilities.
If unsure why existing code is structured a certain way, ask.

### Rule 7 — Tests verify intent, not just behavior
Tests must encode WHY behavior matters, not just WHAT it does.
A test that can't fail when business logic changes is wrong.

### Rule 8 — Checkpoint after every significant step
Summarize what was done, what's verified, what's left.
Don't continue from a state you can't describe back.

### Rule 9 — Match the codebase's conventions, even if you disagree
Conformance > taste inside the codebase.
If you think a convention is harmful, surface it. Don't fork silently.

### Rule 10 — Fail loud
"Completed" is wrong if anything was skipped silently.
Default to surfacing uncertainty, not hiding it.

## Build and test

`make test` is the green gate — the single command every milestone must leave
passing. It runs `cargo fmt --check`, `clippy -D warnings`, a workspace build,
the full test suite (unit, integration, doctests, and the compiled C example), a
docs build with warnings denied, and a benchmark compile check. The same gate
runs in CI (`.github/workflows/ci.yml`) on every push and PR to `main`.

```bash
make test         # the gate — must be green before any commit
make fmt          # auto-fix formatting
make header       # refresh the committed header crates/st3-capi/include/st3.h
make perf-guard   # opt-in wall-clock regression check (see below)
```

Enable the pre-commit hook once per checkout — git does not pick it up from a
fresh clone:

```bash
git config core.hooksPath .githooks
```

Two things about the gate that are easy to get wrong:

- **`cargo build` must precede `cargo test`.** The C tests (`c_example`,
  `capi_harness`) compile their `.c` against `target/<profile>/libst3.so`, and
  `cargo test` builds only the rlib its harness links — it never emits the
  cdylib. Running `cargo test` alone passes only where an earlier build left an
  `.so` behind, and fails on a clean checkout.
- **`make perf-guard` is deliberately not in `make test`.** It compares bench
  means against the committed `perf/baseline.json`, which is reference-machine
  relative and meaningless elsewhere. The machine-independent regression net is
  the allocation guard, which does run in the gate.

If a test produces an **incorrect expected value**: DO NOT change the expected
value without permission. The reference fixtures under `fixtures/` are oracles;
see their `PROVENANCE.md`.

## Rust edition and version

`edition = "2021"` and `rust-version = "1.85"` in `[workspace.package]`. Both are
deliberate; do not "modernize" them without reading this section.

**Edition 2021 is for consistency with `../duckdb-miint`**, where st3 is intended
to be consumed. All three Rust crates that co-build into that extension are
edition 2021 — `ext/rype` (which also declares `rust-version = "1.70"`),
`ext/sylph`, and `ext/miint-rust-glue` — and `ext/rype/rustfmt.toml` carries the
same `max_width`/`tab_spaces`/shorthand settings as ours. Keeping st3 on 2021
means one edition and one rustfmt style across everything linked into the same
extension. Note the cost: the 2021 style edition sorts `use` groups differently
from 2024 and indents some trailing comments worse.

**`rust-version = "1.85"` is a measured floor, not a guess.** Verified against
real toolchains: `cargo check --workspace --all-features` passes on 1.85 and
fails on 1.84. Edition 2021 by itself only needs 1.56, but the dependency graph
does not: `cbindgen` pulls in `toml_writer`, which is an edition-2024 crate, so
Cargo must understand edition 2024 to parse the graph at all. **Dropping to
edition 2021 therefore does not lower the toolchain floor** — if matching
rype's declared 1.70 ever matters, `cbindgen` is the blocker, not the edition.

The floor is two-tier, which is why CI has two jobs:

- **1.85** to *depend on* the crates — library targets only.
- **1.86** to run the benches and tests, because `criterion` 0.8 requires 1.86.

The CI MSRV job checks library targets deliberately without `--all-targets`;
`rust-version` is a promise to consumers, and dev-dependencies are outside it.
This is comfortable for duckdb-miint either way: its CI installs unpinned
current stable via rustup in the manylinux images, uses
`dtolnay/rust-toolchain@stable` on macOS/Windows, and the one pinned path in
`extension-ci-tools` pins 1.86.0.
