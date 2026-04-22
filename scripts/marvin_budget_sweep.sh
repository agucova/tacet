#!/usr/bin/env bash
# MARVIN budget-scaling sweep.
#
# Drives `marvin_budget_sweep` across a grid of (budget × seed) runs on
# the current machine. Emits one CSV row per (budget, seed) to
# $OUTPUT_DIR/results.csv. Resumable — existing rows are skipped.
#
# Budgets (samples/class) are multipliers of the §5.6 baseline 62,000.
# Ladder: 0.5x, 1x, 1.5x, 2x, 2.5x, 3x, 5x  (7 points).
# Seeds per budget: 20 (default).
# Parallelism: PARALLEL seeds run concurrently via xargs -P.
#
# Usage:
#   bash scripts/marvin_budget_sweep.sh [OUTPUT_DIR] [SEEDS] [PARALLEL]
# Defaults:
#   OUTPUT_DIR  = $HOME/marvin-sweep
#   SEEDS       = 20
#   PARALLEL    = 4

set -euo pipefail

OUTPUT_DIR="${1:-$HOME/marvin-sweep}"
SEEDS="${2:-20}"
PARALLEL="${3:-4}"
BASE_SEED="${BASE_SEED:-20260422}"

# Resolve repo root: assumes this script lives in $REPO/scripts/.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${REPO_ROOT}/target/release/marvin_budget_sweep"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: binary not found: $BIN"
  echo "Build first: cd ${REPO_ROOT} && cargo build --release --bin marvin_budget_sweep -p tacet-bench"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_CSV="${OUTPUT_DIR}/results.csv"

# Budget ladder (samples/class). 62,000 is §5.6's baseline.
BUDGETS=(
  "0.5x:31000"
  "1.0x:62000"
  "1.5x:93000"
  "2.0x:124000"
  "2.5x:155000"
  "3.0x:186000"
  "5.0x:310000"
)

# Capture instance metadata once per sweep (header of conditions file).
CONDITIONS="${OUTPUT_DIR}/marvin_conditions.md"
if [[ ! -f "$CONDITIONS" ]]; then
  {
    echo "# MARVIN budget sweep: runtime conditions"
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
    echo "## ladder"
    for entry in "${BUDGETS[@]}"; do
      echo "- $entry"
    done
    echo
    echo "## config"
    echo "- seeds: $SEEDS"
    echo "- parallelism: $PARALLEL"
    echo "- base_seed: $BASE_SEED"
  } > "$CONDITIONS"
  echo "[conditions] captured to $CONDITIONS"
fi

# Stable per-iteration seed derivation (must match a hypothetical harness).
# python hash would be nondeterministic; use awk+md5sum for reproducibility.
derive_seed() {
  local budget_label="$1"
  local iter="$2"
  local h
  h=$(printf '%s|marvin|%s|%s' "$BASE_SEED" "$budget_label" "$iter" | md5sum | awk '{print $1}')
  # Take low 16 hex chars -> u64
  printf '%u' "0x${h:0:15}"
}

# Build the full work list as "budget_label:samples:iter:seed" lines.
WORK_FILE="$(mktemp)"
trap 'rm -f "$WORK_FILE"' EXIT
for entry in "${BUDGETS[@]}"; do
  label="${entry%%:*}"
  samples="${entry##*:}"
  for ((iter=1; iter<=SEEDS; iter++)); do
    seed=$(derive_seed "$label" "$iter")
    echo "${label}|${samples}|${iter}|${seed}"
  done
done > "$WORK_FILE"

total=$(wc -l < "$WORK_FILE")
echo "[sweep] total runs: $total (budgets=${#BUDGETS[@]}, seeds=$SEEDS)"
echo "[sweep] parallelism: $PARALLEL"
echo "[sweep] output: $OUTPUT_CSV"

# Each xargs worker runs one marvin_budget_sweep invocation.
# --resume makes each invocation idempotent — replays the whole list safely.
export BIN OUTPUT_CSV

# Use taskset to pin each worker to a dedicated core group so measurement
# runs don't contaminate each other. 32 vCPUs / 4 workers = 8 cores/worker.
# Workers 0..3 get cores 0-7, 8-15, 16-23, 24-31 respectively.
run_one() {
  local line="$1"
  local label="${line%%|*}"
  local rest="${line#*|}"
  local samples="${rest%%|*}"
  local rest2="${rest#*|}"
  local iter="${rest2%%|*}"
  local seed="${rest2#*|}"

  # Pick core group by worker index (XARGS_PARALLEL exports an index via $?)
  # simpler: round-robin by seed mod PARALLEL.
  local worker=$(( seed % 4 ))
  local lo=$(( worker * 8 ))
  local hi=$(( lo + 7 ))

  taskset -c "${lo}-${hi}" \
    "$BIN" \
      --samples-per-class "$samples" \
      --seed "$seed" \
      --budget-label "$label" \
      --iteration "$iter" \
      --marvin-mode "${MARVIN_MODE:-cache}" \
      --output "$OUTPUT_CSV" \
      --resume 2>&1 | sed -u "s/^/[w${worker} ${label} iter${iter}] /"
}
export -f run_one

< "$WORK_FILE" xargs -n 1 -P "$PARALLEL" -I {} bash -c 'run_one "$@"' _ {}

echo "[sweep] complete. Rows in CSV:"
wc -l "$OUTPUT_CSV"
