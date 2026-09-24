# Contributing

Thanks for taking the time. [`AGENTS.md`](AGENTS.md) holds the working rules in more detail.

## Build

```sh
cargo build            # debug
cargo build --release  # target/release/kanon
just install           # cargo install --path . --locked, onto PATH
just                   # lists every recipe (the justfile mirrors CI)
```

Rust 2024 edition; the minimum supported version is the `rust-version` in `Cargo.toml`.

## Test

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

`cargo test` runs the unit tests, the integration tests, the golden corpus checks and the
report snapshots, all offline. `just update-golden` refreshes the pinned results after an
intended change; say why in the commit body.

## Pull requests

One change per pull request, with the commit body saying why. CI runs the lint and test gates
above on Linux and macOS, plus a build on the minimum supported Rust version.
