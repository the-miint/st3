// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! The build script generates `st3.h` via cbindgen. This asserts the header is
//! produced and declares the full v1 surface. (Layout stability is guarded by the
//! `offset_of!` tests in `config.rs`, not a golden-file diff of the header.)

use std::path::Path;

#[test]
fn header_is_generated_and_declares_the_surface() {
    // `ST3_HEADER_DIR` is exported by build.rs via `cargo:rustc-env`.
    let dir = env!("ST3_HEADER_DIR");
    let path = Path::new(dir).join("st3.h");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading st3.h at {}: {e}", path.display()));

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
