# Documentation commands

# Build documentation for all crates
[group('doc')]
build:
    cargo doc --workspace --no-deps

# Build and open documentation
[group('doc')]
open:
    cargo doc --workspace --no-deps --open

# Build docs with private items
[group('doc')]
private:
    cargo doc --workspace --no-deps --document-private-items
