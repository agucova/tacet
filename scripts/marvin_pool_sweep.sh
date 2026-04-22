#!/usr/bin/env bash
# MARVIN input-pool-size sweep (Reviewer B Q2: input-generation sensitivity).
#
# Fixed sample budget = §5.6's 62 000 samples/class. Sweeps sample-class
# pool size N ∈ {1, 10, 100, 1000}. 20 seeds per pool size. Emits one CSV
# row per (pool_size, seed) to $OUTPUT_DIR/results.csv. Resumable.
#
# Usage:
#   bash scripts/marvin_pool_sweep.sh [OUTPUT_DIR] [SEEDS] [PARALLEL]
# Defaults:
#   OUTPUT_DIR  = $HOME/marvin-pool-sweep
#   SEEDS       = 20
#   PARALLEL    = 4

set -euo pipefail

OUTPUT_DIR="${1:-$HOME/marvin-pool-sweep}"
SEEDS="${2:-20}"
PARALLEL="${3:-4}"
BASE_SEED="${BASE_SEED:-20260422}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${REPO_ROOT}/target/release/marvin_budget_sweep"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: binary not found: $BIN"
  echo "Build first: cd ${REPO_ROOT} && cargo build --release --bin marvin_budget_sweep -p tacet-bench"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_CSV="${OUTPUT_DIR}/results.csv"

# Fixed budget: §5.6's 62 000 samples/class.
BUDGET_LABEL="1.0x"
SAMPLES_PER_CLASS=62000

# Pool size axis.
POOL_SIZES=(1 10 100 1000)

# Capture instance metadata once per sweep.
CONDITIONS="${OUTPUT_DIR}/pool_conditions.md"
if [[ ! -f "$CONDITIONS" ]]; then
  {
    echo "# MARVIN pool-size sweep: runtime conditions"
    echo
    echo "Generated: $(date -Iseconds)"
    echo
    echo "## uname"
    echo '```'
    uname -a
    echo '```'
    echo
    echo "## lscpu"
    echo '```'
    lscpu 2>/dev/null || sysctl -a | grep -E "^(hw|machdep)\." | head
    echo '```'
    echo
    echo "## governor"
    echo '```'
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "(unavailable)"
    echo '```'
    echo
    echo "## turbo"
    echo '```'
    cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo "(unavailable)"
    echo '```'
    echo
    echo "## perf_event_paranoid"
    echo '```'
    cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "(unavailable)"
    echo '```'
    echo
    echo "## load (1/5/15 min)"
    echo '```'
    uptime
    echo '```'
    echo
    echo "## rustc"
    echo '```'
    rustc --version
    echo '```'
    echo
    echo "## pool sizes"
    for ps in "${POOL_SIZES[@]}"; do
      echo "- $ps"
    done
    echo
    echo "## config"
    echo "- budget: ${BUDGET_LABEL} (${SAMPLES_PER_CLASS} samples/class)"
    echo "- seeds: $SEEDS"
    echo "- parallelism: $PARALLEL"
    echo "- base_seed: $BASE_SEED"
  } > "$CONDITIONS"
  echo "[conditions] captured to $CONDITIONS"
fi

# Per-iteration seed derivation. Namespaced by pool_size so cross-pool seeds
# are independent (matches §8's per-budget independence convention).
derive_seed() {
  local pool_size="$1"
  local iter="$2"
  local h
  h=$(printf '%s|marvin-pool|%s|%s' "$BASE_SEED" "$pool_size" "$iter" | md5sum | awk '{print $1}')
  printf '%u' "0x${h:0:15}"
}

# Build the full work list as "pool_size|iter|seed" lines.
WORK_FILE="$(mktemp)"
trap 'rm -f "$WORK_FILE"' EXIT
for ps in "${POOL_SIZES[@]}"; do
  for ((iter=1; iter<=SEEDS; iter++)); do
    seed=$(derive_seed "$ps" "$iter")
    echo "${ps}|${iter}|${seed}"
  done
done > "$WORK_FILE"

total=$(wc -l < "$WORK_FILE")
echo "[sweep] total runs: $total (pool_sizes=${#POOL_SIZES[@]}, seeds=$SEEDS)"
echo "[sweep] parallelism: $PARALLEL"
echo "[sweep] output: $OUTPUT_CSV"

export BIN OUTPUT_CSV BUDGET_LABEL SAMPLES_PER_CLASS

# Detect available cores so taskset groups fit the box (§8 assumes 32 vCPU
# → 8-core groups). 16 vCPU → 4-core groups. We pin via physical count.
TOTAL_CPUS=$(nproc)
GROUP_SIZE=$(( TOTAL_CPUS / PARALLEL ))
if (( GROUP_SIZE < 1 )); then GROUP_SIZE=1; fi
export GROUP_SIZE

run_one() {
  local line="$1"
  local pool_size="${line%%|*}"
  local rest="${line#*|}"
  local iter="${rest%%|*}"
  local seed="${rest#*|}"

  local worker=$(( seed % 4 ))
  local lo=$(( worker * GROUP_SIZE ))
  local hi=$(( lo + GROUP_SIZE - 1 ))

  taskset -c "${lo}-${hi}" \
    "$BIN" \
      --samples-per-class "$SAMPLES_PER_CLASS" \
      --seed "$seed" \
      --budget-label "$BUDGET_LABEL" \
      --iteration "$iter" \
      --pool-size "$pool_size" \
      --marvin-mode "${MARVIN_MODE:-cache}" \
      --output "$OUTPUT_CSV" \
      --resume 2>&1 | sed -u "s/^/[w${worker} pool=${pool_size} iter${iter}] /"
}
export -f run_one

< "$WORK_FILE" xargs -n 1 -P "$PARALLEL" -I {} bash -c 'run_one "$@"' _ {}

echo "[sweep] complete. Rows in CSV:"
wc -l "$OUTPUT_CSV"
