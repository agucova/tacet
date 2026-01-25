# Build commands

# Build entire workspace in release mode (default)
[group('build')]
release:
    cargo build --release --workspace

# Build in debug mode
[group('build')]
debug:
    cargo build --workspace

# Build with minimal features (no parallel, no PMU timers)
[group('build')]
minimal:
    cargo build --release --workspace --no-default-features

# Build only the main library
[group('build')]
lib:
    cargo build --release -p tacet --lib

# Build only the core crate
[group('build')]
core:
    cargo build --release -p tacet-core

# Build only the macros crate
[group('build')]
macros:
    cargo build --release -p tacet-macros
