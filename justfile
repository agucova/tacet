# tacet workspace commands
# Run with: just <command> [subcommand]

# Crate directory
oracle_crate := "crates/tacet"

# Default output directories (at repo root)
data_dir := env("CALIBRATION_DATA_DIR", justfile_directory() + "/calibration_data")
plot_dir := env("CALIBRATION_PLOT_DIR", justfile_directory() + "/plots")
profile_dir := env("PROFILE_DIR", "/var/tmp/tacet-profile")

# Load subcommand modules
mod build
mod test
mod calibrate
mod bench
mod example
mod doc
mod quality
mod profile
mod bindings

# ============================================================================
# TOP-LEVEL COMMANDS
# ============================================================================

# Format all code in workspace
fmt:
    cargo fmt --all

# Check compilation without building
check:
    cargo check --workspace --all-targets

# Run all CI checks
ci: quality::check test::quick
    @echo ""
    @echo "CI checks passed!"

# Pre-commit check (fast)
pre-commit: quality::check check
    @echo "Pre-commit checks passed!"

# Pre-push check (thorough)
pre-push: ci test::crypto calibrate::quick
    @echo "Pre-push checks passed!"

# ============================================================================
# RELEASE
# ============================================================================

# Dry-run publish for all crates
publish-dry:
    cargo publish -p tacet-core --dry-run
    cargo publish -p tacet-macros --dry-run
    cargo publish -p tacet --dry-run

# Publish all crates (in dependency order)
publish:
    cargo publish -p tacet-core
    @echo "Waiting for crates.io to index tacet-core..."
    sleep 30
    cargo publish -p tacet-macros
    @echo "Waiting for crates.io to index tacet-macros..."
    sleep 30
    cargo publish -p tacet

# ============================================================================
# CLEANUP
# ============================================================================

# Remove calibration data and plots
clean-data:
    rm -rf {{data_dir}} {{plot_dir}}

# Cargo clean
clean-build:
    cargo clean

# Clean everything
clean: clean-data clean-build

# ============================================================================
# UTILITIES
# ============================================================================

# Show feature flags
features:
    @echo "Default features: parallel, kperf (macOS), perf (Linux), macros"
    @echo ""
    @echo "Build with specific features:"
    @echo "  cargo build --no-default-features                    # Minimal"
    @echo "  cargo build --no-default-features --features parallel # Parallel only"
    @echo "  cargo build --features macros                        # With proc macros"

# Show environment variables
env:
    @echo "Output verbosity (default is debug, set in .cargo/config.toml):"
    @echo "  just test quiet ...                   # Minimal output"
    @echo "  just test verbose ...                 # Preflight + diagnostics"
    @echo "  cargo test ... (default)              # Full debug output"
    @echo ""
    @echo "Calibration environment variables:"
    @echo "  CALIBRATION_TIER=iteration|quick|full|validation"
    @echo "  CALIBRATION_DATA_DIR=<path>           # Enable CSV export"
    @echo "  CALIBRATION_SEED=<u64>                # Fixed RNG seed"
    @echo "  CALIBRATION_DISABLED=1                # Skip calibration tests"
    @echo "  CALIBRATION_ENABLE_STRESS=1           # Enable stress tests"
    @echo ""
    @echo "Calibration tiers:"
    @echo "  iteration  - ~30 min, quick feedback during development"
    @echo "  quick      - ~1-2 hours, PR checks"
    @echo "  full       - ~2-3 hours, weekly validation"
    @echo "  validation - ~4+ hours, pre-release (includes ignored tests)"
    @echo ""
    @echo "Oracle environment variables:"
    @echo "  TO_SAMPLES=<n>                        # Override sample count"
    @echo "  TO_ALPHA=<f64>                        # Override alpha level"
    @echo "  TIMING_ORACLE_UNRELIABLE_POLICY=fail_closed  # Strict mode"

# Print version info
version:
    @cargo pkgid -p tacet | cut -d# -f2

# Show workspace members
workspace:
    @echo "Workspace members:"
    @echo "  tacet-core   - Core types and traits"
    @echo "  tacet-macros - Proc macros (timing_test!, timing_test_checked!)"
    @echo "  tacet        - Main library"

# Show all available recipes
help:
    @just --list --list-submodules
