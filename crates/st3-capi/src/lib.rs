// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, The SourceTracker3 Development Team

//! SourceTracker3 C ABI (cdylib + staticlib).

/// Returns the SourceTracker3 C ABI version. Placeholder anchor; the real ABI
/// (config structs, opaque handles, Arrow interop) lands in the C-ABI milestone.
#[unsafe(no_mangle)]
pub extern "C" fn st3_abi_version() -> u32 {
    0
}

#[cfg(test)]
mod tests {
    #[test]
    fn abi_version_is_zero() {
        assert_eq!(crate::st3_abi_version(), 0);
    }
}
