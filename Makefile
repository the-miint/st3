.PHONY: test fmt clippy doc bench-check bench perf-guard build header

# Full green gate — the single command every milestone must leave passing.
#
# `cargo build` runs before `cargo test` on purpose. The two C tests
# (`c_example`, `capi_harness`) compile their .c against
# target/<profile>/libst3.so, and `cargo test` builds only the rlib its harness
# links — it never emits the cdylib. Without an explicit build the C tests pass
# only where an earlier `cargo build` happened to leave an .so behind, so the
# gate was green on developer machines and red on a clean checkout.
test:
	cargo fmt --all --check
	cargo clippy --workspace --all-targets --all-features -- -D warnings
	cargo build --workspace --all-features
	cargo test --workspace --all-features
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
	cargo bench --workspace --no-run

fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features

bench-check:
	cargo bench --workspace --no-run

# Run the full criterion benchmark suite, writing target/criterion/**/estimates.json.
bench:
	cargo bench --workspace

# Opt-in wall-clock regression guard: run the benches, then compare their means
# against the committed perf/baseline.json and fail on a >1.5x slowdown. It is
# REFERENCE-MACHINE RELATIVE — only meaningful on the same machine the baseline
# was captured on — so it is deliberately NOT part of `make test`. The
# machine-independent regression net (the allocation guard) runs in `make test`.
perf-guard: bench
	cargo test -p st3-core --test perf_guard -- --ignored --nocapture

build:
	cargo build --workspace

# Emit the cbindgen-generated C header to a stable path for C consumers. The
# header is generated into the release build's OUT_DIR during compilation (it is
# not committed); this builds the C ABI crate, locates the freshest generated
# copy under target/release, and copies it to target/st3.h. Not part of
# `make test`.
header:
	cargo build --release -p st3-capi
	@hdr=$$(find target/release -path '*st3-capi*/out/st3.h' -printf '%T@ %p\n' \
		| sort -rn | head -1 | cut -d' ' -f2-); \
	if [ -z "$$hdr" ]; then \
		echo "st3.h not found under target/release; did the build run?" >&2; exit 1; \
	fi; \
	cp "$$hdr" target/st3.h; \
	echo "st3.h -> target/st3.h"
