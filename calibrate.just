# Calibration commands

set fallback

# Import variables from parent
data_dir := env("CALIBRATION_DATA_DIR", justfile_directory() + "/calibration_data")
plot_dir := env("CALIBRATION_PLOT_DIR", justfile_directory() + "/plots")
oracle_crate := "crates/tacet"

# Run iteration calibration tests with PMU (~5-10 min, requires sudo) - for quick development feedback
[group('calibrate')]
iteration:
    mkdir -p {{data_dir}}
    sudo -E CALIBRATION_TIER=iteration CALIBRATION_DATA_DIR={{data_dir}} \
        cargo nextest run --profile calibration --release --no-capture

# Run quick calibration tests (~20 min, no sudo)
[group('calibrate')]
quick:
    mkdir -p {{data_dir}}
    CALIBRATION_TIER=quick CALIBRATION_DATA_DIR={{data_dir}} \
        cargo nextest run --profile calibration --release --no-capture

# Run full calibration tests (~2-3 hours, no sudo)
[group('calibrate')]
full:
    mkdir -p {{data_dir}}
    CALIBRATION_TIER=full CALIBRATION_DATA_DIR={{data_dir}} \
        cargo nextest run --profile calibration --release --no-capture

# Run full calibration tests with PMU timer (requires sudo)
[group('calibrate')]
full-pmu:
    mkdir -p {{data_dir}}
    sudo -E CALIBRATION_TIER=full CALIBRATION_DATA_DIR={{data_dir}} \
        cargo nextest run --profile calibration --release --no-capture

# Run validation tests (~4+ hours, includes #[ignore] tests, requires sudo)
[group('calibrate')]
validation:
    mkdir -p {{data_dir}}
    sudo -E CALIBRATION_TIER=validation CALIBRATION_DATA_DIR={{data_dir}} \
        cargo nextest run --profile calibration-full --run-ignored ignored-only --release --no-capture

# Run validation tests with PMU timer (requires sudo)
[group('calibrate')]
validation-pmu:
    mkdir -p {{data_dir}}
    sudo -E CALIBRATION_TIER=validation CALIBRATION_DATA_DIR={{data_dir}} \
        cargo nextest run --profile calibration-full --run-ignored ignored-only --release --no-capture

# Run ALL calibration tests - quick + validation (~4+ hours, requires sudo)
[group('calibrate')]
all:
    mkdir -p {{data_dir}}
    sudo -E CALIBRATION_DATA_DIR={{data_dir}} \
        cargo nextest run --profile calibration-full --run-ignored all --release --no-capture

# Run stress tests (requires CALIBRATION_ENABLE_STRESS=1)
[group('calibrate')]
stress:
    mkdir -p {{data_dir}}
    CALIBRATION_ENABLE_STRESS=1 CALIBRATION_DATA_DIR={{data_dir}} \
        cargo nextest run --profile calibration --release --no-capture -E 'test(/stress/)'

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

# Full calibration + plots (no sudo)
[group('calibrate')]
run: full plot

# Full calibration with PMU + plots (requires sudo)
[group('calibrate')]
run-pmu: full-pmu plot

# Production run: validation tests + plots (~4+ hours)
[group('calibrate')]
production: validation plot
    @echo ""
    @echo "Production calibration complete!"
    @echo "Data: {{data_dir}}"
    @echo "Plots: {{plot_dir}}"
