# Recipes for working on kanon. `just` lists them; `just check` is what CI runs offline.

set shell := ["bash", "-euo", "pipefail", "-c"]

# List the recipes.
default:
    @just --list --unsorted

# Debug build.
build:
    cargo build --all-targets

# Optimised build; the binary is target/release/kanon.
release:
    cargo build --release --locked

# Install the kanon binary from this checkout (cargo's bin directory, normally ~/.cargo/bin).
install:
    cargo install --path . --locked
    @echo "installed $(kanon --version) at $(command -v kanon)"

# Remove the binary `just install` put on PATH.
uninstall:
    cargo uninstall kanon

# Apply rustfmt (not just check it).
fmt:
    cargo fmt

# The lint gate exactly as CI runs it: rustfmt check, clippy with pedantic, rustdoc with warnings denied.
lint:
    cargo fmt --check
    cargo clippy --all-targets --all-features -- -D warnings
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

# Unit, integration, golden corpus and snapshot tests, plus the doctests; all offline.
test:
    cargo test --all-targets
    cargo test --doc

# Everything CI checks without the network: lint, then test.
check: lint test

# Refresh the pinned golden results and the report snapshots after an intended change (say why in the commit).
update-golden:
    UPDATE_GOLDEN=1 cargo test --test golden --test backend_tantivy_golden --test backend_dense_hybrid_golden
    UPDATE_SNAPSHOTS=1 cargo test

# RustSec advisories against Cargo.lock (needs `cargo install cargo-audit`).
audit:
    cargo audit
