// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! The build script generates `st3.h` via cbindgen. This asserts the header is
//! produced, declares the full v1 surface, and matches the committed copy that
//! C consumers include. (Layout stability is guarded by the `offset_of!` tests
//! in `config.rs`, not a golden-file diff of the header.)

use std::path::{Path, PathBuf};

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

/// The header refers to the Arrow C Data Interface structs by pointer without
/// declaring them (deliberately, so a consumer's own Arrow declarations are
/// used). The header must say so up front, so a consumer learns to declare them
/// before including it without reading the example.
#[test]
fn header_tells_consumers_to_declare_the_arrow_structs() {
    let content = read_generated_header();
    let note = content
        .find("Declare them before including this header")
        .expect("st3.h is missing the note that the Arrow structs must be declared first");
    let first_use = content
        .find("ArrowArray *")
        .expect("st3.h no longer refers to ArrowArray");
    assert!(
        note < first_use,
        "the Arrow-declaration note must precede the first use of the Arrow structs"
    );
}
