.PHONY: test fmt clippy doc bench-check bench perf-guard build

# Full green gate — the single command every milestone must leave passing.
test:
	cargo fmt --all --check
	cargo clippy --workspace --all-targets --all-features -- -D warnings
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
