// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! The build script generates `st3.h` via cbindgen. This asserts the header is
//! produced, declares the full v1 surface, matches the committed copy that C
//! consumers include, and compiles for a consumer of Arrow's own `abi.h`.
//! (Layout stability is guarded by the `offset_of!` tests in `config.rs`, not a
//! golden-file diff of the header.)

use std::path::{Path, PathBuf};
use std::process::Command;

/// Whether a working compiler is available.
fn have_compiler(compiler: &str) -> bool {
    Command::new(compiler)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Read the header cbindgen generated from the current sources. Its directory
/// (`ST3_HEADER_DIR`) is exported by build.rs via `cargo:rustc-env`.
fn read_generated_header() -> String {
    let path = Path::new(env!("ST3_HEADER_DIR")).join("st3.h");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading st3.h at {}: {e}", path.display()))
}

/// The committed copy of the header, the one C consumers build against.
fn committed_header_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("include")
        .join("st3.h")
}

#[test]
fn header_is_generated_and_declares_the_surface() {
    let content = read_generated_header();

    assert!(!content.trim().is_empty(), "st3.h is empty");

    // The nine C ABI entry points.
    for sym in [
        "st3_abi_version",
        "st3_table_from_arrow",
        "st3_table_free",
        "st3_run",
        "st3_result_free",
        "st3_result_means",
        "st3_result_stds",
        "st3_result_contingency_stream",
        "st3_last_error",
    ] {
        assert!(content.contains(sym), "st3.h is missing {sym}");
    }

    // The versioned config struct and the three enums (renamed constants).
    assert!(content.contains("St3Config"), "st3.h is missing St3Config");
    assert!(content.contains("#define ST3_CONFIG_V1"));
    assert!(content.contains("ST3_STATUS_OK"));
    assert!(content.contains("ST3_STATUS_ERR_INVALID_INPUT"));
    assert!(content.contains("ST3_COLLAPSE_MEAN"));
    assert!(content.contains("ST3_COLLAPSE_SUM"));
    assert!(content.contains("ST3_ESTIMATOR_KIND_GIBBS_COLLAPSED"));

    // The include guard.
    assert!(
        content.contains("ST3_H"),
        "st3.h is missing its include guard"
    );
}

/// The generated header must carry the item doc comments (cbindgen
/// `documentation = true`) and they must read as natural C — free of the
/// rustdoc-isms (intra-doc links, `crate::` paths, `§` section refs) that leak
/// from Rust doc comments. This locks in the C-clean documentation so a future
/// cbindgen config regression or a stray rustdoc link is caught.
#[test]
fn header_carries_c_clean_docs() {
    let content = read_generated_header();

    // Doc comments are present: distinctive phrases from a function, a struct,
    // and an enum doc. Their absence means `documentation = true` regressed.
    for phrase in [
        "Import a COO count table",        // st3_table_from_arrow
        "leave-one-out source prediction", // st3_run
        "per-draw standard deviations",    // st3_result_stds
        "Opaque handle",                   // St3Table / St3Result
        "Versioned run configuration",     // St3Config
        "The call succeeded",              // St3Status::Ok
    ] {
        assert!(
            content.contains(phrase),
            "st3.h is missing the doc phrase {phrase:?}; is cbindgen documentation still on?"
        );
    }

    // No rustdoc-isms survive into the C header.
    assert!(
        !content.contains("[`"),
        "st3.h contains a rustdoc intra-doc link (`[`...`]`)"
    );
    assert!(
        !content.contains("crate::"),
        "st3.h contains a `crate::` path from an intra-doc link"
    );
    assert!(
        !content.contains('§'),
        "st3.h contains a `§` corpus section reference"
    );
}

/// The committed copy at `include/st3.h` must be byte-identical to the header
/// cbindgen just generated from the current sources. C consumers build against
/// the committed copy (they cannot locate cargo's `OUT_DIR`), so a stale copy
/// would silently disagree with the `#[repr(C)]` types and the exported
/// functions. `make header` refreshes it.
#[test]
fn committed_header_is_current() {
    let generated = read_generated_header();
    let path = committed_header_path();
    let committed = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "reading the committed header at {}: {e}; run `make header` to create it",
            path.display()
        )
    });
    assert!(
        committed == generated,
        "the committed header {} differs from the one generated from the current \
         sources; run `make header` and commit the result",
        path.display()
    );
}

/// The header names each Arrow struct by tag and forward-declares the tag
/// before any prototype takes it (cbindgen's `[export.rename]` and
/// `after_includes`), which is what lets a C consumer of Arrow's `abi.h`
/// include it in any order. The compile test below proves the header works;
/// this check needs no compiler, so it still guards the header where that test
/// skips.
#[test]
fn header_forward_declares_the_arrow_struct_tags() {
    let content = read_generated_header();
    for name in ["ArrowArray", "ArrowSchema", "ArrowArrayStream"] {
        let decl = content
            .find(&format!("struct {name};"))
            .unwrap_or_else(|| panic!("st3.h does not forward-declare `struct {name}`"));
        let first_use = content
            .find(&format!("struct {name} *"))
            .unwrap_or_else(|| panic!("st3.h no longer takes a `struct {name} *`"));
        assert!(
            decl < first_use,
            "st3.h forward-declares `struct {name}` only after a prototype uses it"
        );
        let bare = format!("{name} *");
        for (at, _) in content.match_indices(&bare) {
            assert!(
                content[..at].ends_with("struct "),
                "st3.h uses the bare name `{bare}`, which a C consumer of abi.h cannot \
                 compile"
            );
        }
    }
}

/// Arrow's own `abi.h` and nanoarrow declare the C Data Interface structs by
/// tag only (`struct ArrowSchema { ... };`, no typedef), and a consumer may
/// include them before `st3.h`, after it, or not at all. `header_consumer.c`
/// passes consumer-side `struct Arrow...` pointers to every entry point that
/// takes one, and must compile warning-free in all three orders, as C and as
/// C++ (duckdb-miint includes `st3.h` from C++). A header that uses the bare
/// names (`ArrowSchema *`, which only C++ or a consumer's own typedefs supply),
/// or that leaves the structs undeclared, fails here. A missing compiler skips
/// its language.
#[test]
fn header_compiles_against_arrow_struct_tags_in_any_order() {
    let header_dir = env!("ST3_HEADER_DIR");
    let consumer = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("header_consumer.c");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());

    for (compiler, lang) in [
        (cc, ["-x", "c", "-std=c11"]),
        (cxx, ["-x", "c++", "-std=c++17"]),
    ] {
        if !have_compiler(&compiler) {
            eprintln!("no compiler ('{compiler}') found; skipping {lang:?}");
            continue;
        }
        for order in [
            "-DARROW_ABI_BEFORE",
            "-DARROW_ABI_AFTER",
            "-DARROW_ABI_NONE",
        ] {
            let out = Command::new(&compiler)
                .args(["-Wall", "-Wextra", "-Werror", "-fsyntax-only"])
                .arg(format!("-I{header_dir}"))
                .arg(order)
                .args(lang)
                .arg(&consumer)
                .output()
                .expect("failed to invoke the compiler");
            assert!(
                out.status.success(),
                "st3.h does not compile for an abi.h consumer ({compiler} {lang:?} {order}):\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
}
