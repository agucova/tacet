# Code quality commands

# Run all quality checks (fmt + lint)
[group('quality')]
check: fmt-check lint

# Run clippy lints on workspace
[group('quality')]
lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Run clippy with auto-fix
[group('quality')]
lint-fix:
    cargo clippy --workspace --all-targets --all-features --fix --allow-dirty

# Format all code
[group('quality')]
fmt:
    cargo fmt --all

# Check formatting
[group('quality')]
fmt-check:
    cargo fmt --all -- --check
