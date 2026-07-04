// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! SourceTracker3 Arrow boundary (COO <-> core conversion).

/// Identifies this crate and the core it is built against. Placeholder anchor
/// proving the Arrow crate links the core crate; replaced in the Arrow milestone.
pub fn backend_name() -> String {
    format!("st3-arrow over {}", st3_core::name())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrow_uses_core() {
        assert_eq!(backend_name(), "st3-arrow over st3-core");
    }
}
