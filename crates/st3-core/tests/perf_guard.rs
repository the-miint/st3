// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! Opt-in, reference-machine-relative wall-clock regression guard.
//!
//! Design memory records **no fixed numeric SLA** for st3: performance is guarded
//! *relative to a committed baseline*, not against absolute wall-clock targets.
//! Criterion writes machine-specific nanosecond timings, so this guard is
//! **opt-in** and only meaningful on the same machine the committed
//! `perf/baseline.json` was captured on. It is therefore `#[ignore]`d and wired
//! into `make perf-guard` (which runs the benches first), never into `make test`.
//! The machine-independent regression net that *does* run in the default gate is
//! the allocation guard (`tests/alloc_guard.rs`).
//!
//! The check is pure file IO — it never shells out to cargo. It locates
//! `target/criterion` from the test binary's own path (the `capi_harness.rs`
//! idiom: the binary lives at `target/<profile>/deps/…`), reads each baselined
//! bench's `new/estimates.json`, and compares its `mean.point_estimate` (the same
//! field the baseline was distilled from) to the committed value. A bench slower
//! than [`THRESHOLD`]× its baseline fails the guard; the generous factor absorbs
//! run-to-run noise so only real regressions trip it. If the criterion output is
//! *entirely* absent (benches were never run) the guard skips gracefully; but if
//! the criterion dir exists yet a baselined bench has no output, that is id drift
//! (a rename/removal) or a partial run and the guard **fails** rather than
//! silently passing an unguarded bench. Adding or renaming a bench therefore
//! requires re-capturing `perf/baseline.json` from `target/criterion`.

use std::path::{Path, PathBuf};

/// A bench must exceed this multiple of its committed baseline to count as a
/// regression. Generous on purpose: criterion means wobble a few percent between
/// runs, and the baseline is single-machine, so the guard should catch
/// order-of-magnitude / structural slowdowns, not noise.
const THRESHOLD: f64 = 1.5;

/// `target/criterion`, resolved from this test binary's location.
///
/// The binary is at `target/<profile>/deps/<name>-<hash>`, so its
/// grandparent is `target/<profile>` and great-grandparent is `target`.
fn target_criterion_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe");
    let target = exe
        .parent() // deps/
        .and_then(Path::parent) // <profile>/
        .and_then(Path::parent) // target/
        .expect("test binary should live under target/<profile>/deps/");
    target.join("criterion")
}

/// The committed baseline at the repo root (`perf/baseline.json`), resolved from
/// this crate's manifest directory (`crates/st3-core`).
fn baseline_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../perf/baseline.json")
}

#[test]
#[ignore = "opt-in, reference-machine relative; run via `make perf-guard`"]
fn wall_clock_within_baseline() {
    // The baseline is committed, so a missing/invalid file is a hard error.
    let baseline_file = baseline_path();
    let baseline_text = std::fs::read_to_string(&baseline_file).unwrap_or_else(|e| {
        panic!(
            "cannot read committed baseline {}: {e}",
            baseline_file.display()
        )
    });
    let baseline: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&baseline_text).expect("perf/baseline.json is a JSON object");

    // Criterion output is optional: absent means the benches were never run.
    let criterion = target_criterion_dir();
    if !criterion.is_dir() {
        eprintln!(
            "no criterion output at {} — run `cargo bench --workspace` first (skipping guard)",
            criterion.display()
        );
        return;
    }

    let mut regressions: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for (id, base_value) in &baseline {
        let base_ns = base_value
            .as_f64()
            .unwrap_or_else(|| panic!("baseline[{id}] is not a number"));

        let estimates = criterion.join(id).join("new").join("estimates.json");
        let Ok(text) = std::fs::read_to_string(&estimates) else {
            missing.push(id.clone());
            continue;
        };
        let parsed: serde_json::Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("parsing {}: {e}", estimates.display()));
        let now_ns = parsed["mean"]["point_estimate"]
            .as_f64()
            .unwrap_or_else(|| panic!("{}: mean.point_estimate missing", estimates.display()));

        let ratio = now_ns / base_ns;
        let flag = if ratio > THRESHOLD {
            "  <== REGRESSION"
        } else {
            ""
        };
        eprintln!("  {id:<38} {now_ns:>15.0} ns  ({ratio:>4.2}x baseline){flag}");
        if ratio > THRESHOLD {
            regressions.push(format!(
                "{id}: {now_ns:.0} ns is {ratio:.2}x the {base_ns:.0} ns baseline (> {THRESHOLD}x)"
            ));
        }
    }

    // A baselined bench with no `estimates.json` while the criterion dir *exists*
    // is not the "benches never run" case (that skips whole, above) — it is
    // rename/removal drift, or a partial run. Fail loudly: a silent skip here
    // would let a renamed hot path regress unguarded while the guard still
    // reports green. Whenever a bench id is added or renamed, re-capture
    // `perf/baseline.json` (regenerate it from `target/criterion`).
    assert!(
        missing.is_empty(),
        "these baselined benches produced no estimates.json under {} — id drift or a partial \
         run. Run the full suite (`make perf-guard`) and re-capture perf/baseline.json if a \
         bench id changed:\n{}",
        criterion.display(),
        missing.join("\n")
    );
    assert!(
        regressions.is_empty(),
        "wall-clock regression vs perf/baseline.json (reference-machine relative):\n{}",
        regressions.join("\n")
    );
}
