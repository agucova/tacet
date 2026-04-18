#!/usr/bin/env bash
# Thin wrapper to launch the crypto cross-tool benchmark on a RunPod / AWS box
# from inside devenv shell. Idempotent: resumes from the existing CSV.
#
# Usage (inside devenv shell):
#   bash scripts/run-crypto-cross-tool.sh [iterations] [output_dir]
#
# Environment overrides:
#   TOOLS         Comma-separated tool list (default: all).
#   TIER          Tier selector (default: all).
#   SAMPLES       samples_per_class (default: 10000).
#   R_WORKERS     SILENT/RTLF pool size (default: 24 for 32 vCPUs).
#   PY_WORKERS    tlsfuzzer pool size (default: 6).
#   SEED          Base RNG seed (default: 20260418).
#
# The binary is built in release mode if missing. Raw samples are persisted
# alongside the CSV for offline re-analysis (~200 KB per iteration).

set -euo pipefail

ITERATIONS="${1:-20}"
OUTPUT_DIR="${2:-$HOME/bench-results/crypto-cross-tool/x86_64}"
TOOLS="${TOOLS:-all}"
TIER="${TIER:-all}"
SAMPLES="${SAMPLES:-10000}"
R_WORKERS="${R_WORKERS:-24}"
PY_WORKERS="${PY_WORKERS:-6}"
SEED="${SEED:-20260418}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Crypto cross-tool benchmark"
echo "=========================================="
echo "Iterations:     $ITERATIONS"
echo "Tier:           $TIER"
echo "Tools:          $TOOLS"
echo "Samples/class:  $SAMPLES"
echo "R workers:      $R_WORKERS"
echo "Python workers: $PY_WORKERS"
echo "Seed:           $SEED"
echo "Output dir:     $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR/raw"

if [[ ! -x target/release/crypto_benchmark ]]; then
    echo "[build] cargo build --release -p tacet-bench --bin crypto_benchmark"
    cargo build --release -p tacet-bench --bin crypto_benchmark
fi

exec ./target/release/crypto_benchmark \
    --tier "$TIER" \
    --iterations "$ITERATIONS" \
    --tools "$TOOLS" \
    --samples-per-class "$SAMPLES" \
    --r-pool-workers "$R_WORKERS" \
    --python-pool-workers "$PY_WORKERS" \
    --seed "$SEED" \
    --output "$OUTPUT_DIR/results.csv" \
    --raw-samples-out "$OUTPUT_DIR/raw"
