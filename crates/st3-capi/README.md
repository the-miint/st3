# `st3-capi` — SourceTracker3 C ABI

The C/C++ front door to SourceTracker3. Builds a shared and static library
(`libst3.so` / `libst3.a`) exposing a small, reentrant C ABI over the Apache
Arrow C Data Interface, plus a `cbindgen`-generated header `st3.h`.

The surface: opaque handles created and freed by the library, a versioned
`St3Config`, a coarse `St3Status` return paired with a thread-local
`st3_last_error` string, and an unwind guard on every entry so a panic can never
cross the C boundary.

## Build & use

```sh
cargo build --release          # -> target/release/libst3.{so,a}
make header                    # -> target/st3.h (generated, not committed)

cc -std=c11 -I target my_app.c -L target/release -lst3 -lm -o my_app
LD_LIBRARY_PATH=target/release ./my_app
```

See [`examples/usage.c`](examples/usage.c) for a complete, commented walkthrough:
build a dataset over the Arrow C Data Interface, run source attribution, read the
dense means and standard deviations, and drain the per-sink assignment stream.

Licensed BSD 3-Clause.
