# Calibration commands

set fallback

# Import variables from parent
data_dir := env("CALIBRATION_DATA_DIR", justfile_directory() + "/calibration_data")
plot_dir := env("CALIBRATION_PLOT_DIR", justfile_directory() + "/plots")
oracle_crate := "crates/timing-oracle"

# Run quick calibration tests (~1-2 min, no sudo)
[group('calibrate')]
quick:
    mkdir -p {{data_dir}}
    CALIBRATION_TIER=quick CALIBRATION_DATA_DIR={{data_dir}} \
        cargo test --release -p timing-oracle \
        --test calibration_power --test calibration_fpr --test calibration_coverage \
        -- --nocapture

# Run full calibration tests (~5-10 min, no sudo)
[group('calibrate')]
full:
    mkdir -p {{data_dir}}
    CALIBRATION_TIER=full CALIBRATION_DATA_DIR={{data_dir}} \
        cargo test --release -p timing-oracle \
        --test calibration_power --test calibration_fpr --test calibration_coverage \
        -- --nocapture

# Run full calibration tests with PMU timer (requires sudo)
[group('calibrate')]
full-pmu:
    mkdir -p {{data_dir}}
    sudo -E CALIBRATION_TIER=full CALIBRATION_DATA_DIR={{data_dir}} \
        cargo test --release -p timing-oracle \
        --test calibration_power --test calibration_fpr --test calibration_coverage \
        -- --nocapture

# Run validation curve tests (~20-30 min, requires sudo)
[group('calibrate')]
validation:
    mkdir -p {{data_dir}}
    sudo -E CALIBRATION_TIER=validation CALIBRATION_DATA_DIR={{data_dir}} \
        cargo test --release -p timing-oracle \
        --test calibration_power \
        --test calibration_fpr \
        --test calibration_coverage \
        -- --ignored --nocapture

# Run stress tests (requires CALIBRATION_ENABLE_STRESS=1)
[group('calibrate')]
stress:
    mkdir -p {{data_dir}}
    CALIBRATION_ENABLE_STRESS=1 CALIBRATION_DATA_DIR={{data_dir}} \
        cargo test --release -p timing-oracle --test calibration_stress -- --nocapture

# Generate plots from calibration data
[group('calibrate')]
plot:
    mkdir -p {{plot_dir}}
    uv run {{oracle_crate}}/scripts/plot_calibration.py {{data_dir}} --output {{plot_dir}}
    @echo "Plots saved to {{plot_dir}}"

# Open plots directory (macOS)
[macos]
[group('calibrate')]
plot-open: plot
    open {{plot_dir}}

# Open plots directory (Linux)
[linux]
[group('calibrate')]
plot-open: plot
    xdg-open {{plot_dir}}

# Full calibration + plots (no sudo, ~5-10 min)
[group('calibrate')]
run: full plot

# Full calibration with PMU + plots (requires sudo, ~5-10 min)
[group('calibrate')]
run-pmu: full-pmu plot

# Production run: validation tests + plots (requires sudo, ~30 min)
[group('calibrate')]
production: validation plot
    @echo ""
    @echo "Production calibration complete!"
    @echo "Data: {{data_dir}}"
    @echo "Plots: {{plot_dir}}"
