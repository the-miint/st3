.PHONY: test fmt clippy doc bench-check build

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

build:
	cargo build --workspace
