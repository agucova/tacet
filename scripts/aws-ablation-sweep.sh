#!/usr/bin/env bash
# Hyperparameter-sensitivity ablation sweep for the USENIX Sec'26 rebuttal.
#
# Runs tacet-only (9 configs × reduced-medium grid) to quantify sensitivity
# of FPR / detection / Inconclusive-rate to four hyperparameters:
#   π₀ (prior target exceedance), α / 1-β (decision thresholds),
#   kl_min (DataTooNoisy gate), ν_ℓ (Student-t likelihood df).
#
# Per-config grid:
#   6 effect sizes × 2 patterns (shift, tail) × 4 AR(1) noise levels
#   × 2 attacker models (AdjacentNetwork θ=100ns, SharedHardware θ≈0.4ns)
#   × 30 datasets × (null+IID sigma-sweep = +4× additional null rows for Heatmap 3)
#   ≈ 2,880 non-null + ~480 null trials per config.
#   `medium` preset drives the attacker-model and sigma-sweep dimensions; both
#   are preserved in the CSV and filtered at aggregation time so we report
#   per-attacker FPR/detection (see scripts/analyze_ablation.py).
#
# Aggregate: 9 configs ≈ 10–12 h wall-clock on 16-vCPU Graviton4 (plus compile).
# Each run is checkpointed via --resume, so interruption is safe.
#
# Usage:
#   ./scripts/aws-ablation-sweep.sh [OUTPUT_DIR]
#   OUTPUT_DIR defaults to ~/bench-results/ablation.
#
# Produces one subdirectory per config, each containing benchmark_results.csv.
# Post-process with scripts/analyze_ablation.py.

set -euo pipefail

OUTPUT_ROOT="${1:-$HOME/bench-results/ablation}"
BENCHMARK_BIN="${BENCHMARK_BIN:-}"

# Find/build the release binary so we do not re-compile for each config.
if [[ -z "$BENCHMARK_BIN" ]]; then
    if [[ -x "./target/release/benchmark" ]]; then
        BENCHMARK_BIN="$(pwd)/target/release/benchmark"
    else
        echo "[setup] Building release benchmark binary..."
        cargo build --release --bin benchmark
        BENCHMARK_BIN="$(pwd)/target/release/benchmark"
    fi
fi
echo "[setup] Using binary: $BENCHMARK_BIN"
echo "[setup] Output root:  $OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

# 16 configurations. Format: "LABEL PI0 ALPHA BETA KL_MIN NU_LIKELIHOOD NU_PRIOR"
# A value of "default" means: do NOT set the env var, use library default.
# The `*_extreme`, `*_cauchy`, and `combo_stress_*` rows are stress-tests near
# the boundaries of each knob's legitimate operating range; if they break,
# that's an honest disclosure of where the mechanism starts to degrade
# (stronger than "everything was fine").
CONFIGS=(
    "baseline            default  default  default  default  default  default"
    "pi0_low             0.50     default  default  default  default  default"
    "pi0_high            0.75     default  default  default  default  default"
    "pi0_extreme         0.85     default  default  default  default  default"
    "alpha_tight         default  0.01     0.99     default  default  default"
    "alpha_loose         default  0.10     0.90     default  default  default"
    "kl_loose            default  default  default  0.3      default  default"
    "kl_strict           default  default  default  1.5      default  default"
    "nu_low              default  default  default  default  4        default"
    "nu_high             default  default  default  default  16       default"
    "nu_ell_extreme      default  default  default  default  2.5      default"
    "nu_ell_cauchy       default  default  default  default  2.01     default"
    "nu_prior_low        default  default  default  default  default  2.5"
    "nu_prior_high       default  default  default  default  default  16"
    "combo_stress_heavy  0.85     0.01     0.99     1.5      2.5      2.5"
    "combo_stress_light  0.50     0.10     0.90     0.3      16       16"
)

# Grid shared across all configs; preset=medium iterates BOTH attacker models
# (AdjacentNetwork θ=100ns, SharedHardware θ≈0.4ns) and adds a sigma sweep at
# null+IID for Heatmap 3. Filtered at aggregation time (see analyze_ablation.py).
# 6 effects × 2 patterns × 4 noise × 2 attackers × 120 datasets ≈ 11,520 non-null
# + ~960 null trials per config. Datasets bumped 60 → 120 to halve Wilson CI
# widths (e.g., FPR upper bound 0.9% → 0.44% at n=880 null per (cfg, attacker)).
PATTERNS="shift,tail"
NOISE="iid,ar1-0.3,ar1-0.6,ar1-0.8"
EFFECTS="0,0.2,1.0,2.0,4.0,20.0"
DATASETS=120

run_config() {
    local label=$1 pi0=$2 alpha=$3 beta=$4 kl=$5 nu=$6 nu_prior=$7
    local outdir="$OUTPUT_ROOT/$label"
    mkdir -p "$outdir/logs"

    # Assemble env-var prefix (skip "default" values so the library uses defaults)
    local env_prefix=""
    [[ "$pi0"      != "default" ]] && env_prefix+="TACET_ABLATION_PI0=$pi0 "
    [[ "$alpha"    != "default" ]] && env_prefix+="TACET_ABLATION_ALPHA=$alpha "
    [[ "$beta"     != "default" ]] && env_prefix+="TACET_ABLATION_BETA=$beta "
    [[ "$kl"       != "default" ]] && env_prefix+="TACET_ABLATION_KL_MIN=$kl "
    [[ "$nu"       != "default" ]] && env_prefix+="TACET_ABLATION_NU_LIKELIHOOD=$nu "
    [[ "$nu_prior" != "default" ]] && env_prefix+="TACET_ABLATION_NU_PRIOR=$nu_prior "

    echo ""
    echo "=========================================="
    echo "=== Config: $label"
    echo "=== Env:    ${env_prefix:-<library defaults>}"
    echo "=== Output: $outdir"
    echo "=========================================="

    local start=$(date +%s)
    eval "$env_prefix $BENCHMARK_BIN \
        --preset medium \
        --tools tacet \
        --patterns $PATTERNS \
        --noise $NOISE \
        --effects $EFFECTS \
        --datasets $DATASETS \
        --output $outdir \
        --resume \
        -q" \
        2>&1 | tee "$outdir/logs/run.log"

    local end=$(date +%s)
    echo "[$label] completed in $((end - start))s"

    # Quick sanity: how many rows landed, what outcomes
    local rows=$(tail -n +2 "$outdir/benchmark_results.csv" 2>/dev/null | wc -l)
    local outcomes=$(tail -n +2 "$outdir/benchmark_results.csv" 2>/dev/null | awk -F, '{print $NF}' | sort | uniq -c | tr '\n' ' ')
    echo "[$label] rows=$rows outcomes: $outcomes"
}

for cfg in "${CONFIGS[@]}"; do
    # shellcheck disable=SC2086
    run_config $cfg
done

echo ""
echo "=== All ${#CONFIGS[@]} configs complete ==="
echo "Next: scripts/analyze_ablation.py $OUTPUT_ROOT"
