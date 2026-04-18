#!/usr/bin/env bash
# Measure True Positive Rate (TPR) across all known-leaky crypto tests.
#
# Usage: ./scripts/measure_tpr.sh [iterations] [output_file]
#
# Environment variables:
#   TPR_USE_SUDO=1   Run each test under `sudo -E` (required for kperf on
#                    macOS or perf_event on Linux to get sub-nanosecond
#                    timers). Off by default so local smoke runs do not
#                    trigger TouchID prompts. On AWS/Linux set to 1 for
#                    accurate timing.
#
# Discovers all tests in the `leaky` binary, runs each test N times to compute
# empirical TP rate. All included tests are on KNOWN-VULNERABLE code, so the
# CSV-outcome interpretation is INVERTED relative to measure_fpr.sh:
#
#   CSV outcome | Leaky-test interpretation
#   ------------+--------------------------
#   PASS        | true positive  (leak correctly detected)
#   FAIL        | false negative (missed detection)
#   INCONCLUSIVE| underpowered   (no verdict)
#   SKIP        | Unmeasurable / Research-mode / explicit skip
#
# The CSV schema matches measure_fpr.sh exactly so existing analysis tooling
# can consume both files. scripts/analyze_tpr.py applies the flipped semantics.
#
# Requires: cargo build --release -p tacet --test leaky
#
# Supports resume: re-running with the same $OUTPUT_FILE will skip
# (test_name, iteration) pairs already present in the file.

set -euo pipefail

ITERATIONS="${1:-10}"
OUTPUT_FILE="${2:-tpr_results_$(date +%Y%m%d_%H%M%S).csv}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USE_SUDO="${TPR_USE_SUDO:-0}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()         { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
log_success() { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[$(date +%H:%M:%S)]${NC} $*"; }
log_error()   { echo -e "${RED}[$(date +%H:%M:%S)]${NC} $*"; }

# Find the release leaky test binary dynamically.
# Filter out .d, .o, .rcgu.o companion files produced by cargo.
find_test_binary() {
    local bin
    bin=$(find "$REPO_ROOT/target/release/deps" \
        -name 'leaky-*' -type f \
        ! -name '*.d' ! -name '*.o' ! -name '*.rcgu.o' \
        2>/dev/null | head -1)
    if [[ -z "$bin" ]]; then
        log_error "No leaky test binary found in target/release/deps/"
        log_error "Build first: cargo build --release -p tacet --test leaky"
        exit 1
    fi
    echo "$bin"
}

# Discover all tests, filtering out sanity checks.
discover_tests() {
    local binary="$1"
    "$binary" --list 2>/dev/null \
        | grep ': test$' \
        | sed 's/: test$//' \
        | grep -v 'sanity_check'
}

# Extract test "family" (first mod segment) — used for per-family aggregates.
# e.g. "injected::injected_shift_100ns" → "injected"
extract_family() {
    local test_name="$1"
    echo "$test_name" | cut -d: -f1
}

# Family → leak-geometry label for human-readable reporting.
family_geometry() {
    case "$1" in
        injected)    echo "shift" ;;
        cache_miss)  echo "bimodal" ;;
        marvin)      echo "shift+persistence" ;;
        modpow)      echo "shift" ;;
        *)           echo "unknown" ;;
    esac
}

# Parse outcome from test output. Identical to measure_fpr.sh:parse_outcome
# so both scripts use the same recognizer.
parse_outcome() {
    local output="$1"
    local exit_code="$2"
    local outcome="UNKNOWN"
    local leak_prob="0.0"
    local samples="0"

    if echo "$output" | grep -q "Test passed:"; then
        outcome="PASS"
    elif echo "$output" | grep -q "test result:.*FAILED\|FAILED\|panicked"; then
        outcome="FAIL"
    elif echo "$output" | grep -q "Inconclusive:\|\[SKIPPED\].*inconclusive"; then
        outcome="INCONCLUSIVE"
    elif echo "$output" | grep -q "Skipping:\|Unmeasurable:\|\[SKIPPED\]"; then
        outcome="SKIP"
    elif [[ "$exit_code" -ne 0 ]]; then
        outcome="FAIL"
    fi

    leak_prob=$(echo "$output" | grep -o 'P(leak)=[0-9.]*%' | head -1 | sed 's/P(leak)=//;s/%//' || echo "0.0")
    if [[ -z "$leak_prob" ]]; then
        leak_prob="0.0"
    fi

    samples=$(echo "$output" | grep -o 'Samples: [0-9]*' | head -1 | sed 's/Samples: //' || echo "0")
    if [[ -z "$samples" ]]; then
        samples="0"
    fi

    echo "$outcome,$leak_prob,$samples"
}

# Initialize CSV (or resume if file exists with matching header).
init_csv() {
    local header="ecosystem,library,test_name,iteration,outcome,leak_probability,samples,elapsed_sec,timestamp"
    if [[ -f "$OUTPUT_FILE" ]]; then
        local existing_header
        existing_header=$(head -1 "$OUTPUT_FILE")
        if [[ "$existing_header" != "$header" ]]; then
            log_error "Existing file has incompatible header; refusing to overwrite."
            log_error "  Expected: $header"
            log_error "  Got:      $existing_header"
            exit 1
        fi
        local completed
        completed=$(tail -n +2 "$OUTPUT_FILE" | wc -l | tr -d ' ')
        log "Resuming from existing file: $OUTPUT_FILE ($completed rows already present)"
    else
        echo "$header" > "$OUTPUT_FILE"
        log "Initialized output file: $OUTPUT_FILE"
    fi
}

# True iff "$test_name,$iteration" is already in the CSV (robust to column
# count by anchoring on the two fields most likely to be unique together).
is_completed() {
    local test_name="$1"
    local iteration="$2"
    grep -q ",${test_name},${iteration}," "$OUTPUT_FILE" 2>/dev/null
}

# Generate summary report with Wilson CIs, using LEAKY-test interpretation:
#   PASS in CSV = true positive (detection).
#   FAIL in CSV = false negative (missed).
#   INCONCLUSIVE = underpowered.
generate_report() {
    log ""
    log "=========================================="
    log "True Positive Rate Report (Leaky-test interpretation)"
    log "=========================================="
    log "Iterations per test: $ITERATIONS"
    log "Output file: $OUTPUT_FILE"
    log ""

    local total_runs tp fn inc skip unk
    total_runs=$(tail -n +2 "$OUTPUT_FILE" | wc -l | tr -d ' ')
    tp=$(grep -c ",PASS," "$OUTPUT_FILE" || true); tp=${tp:-0}
    fn=$(grep -c ",FAIL," "$OUTPUT_FILE" || true); fn=${fn:-0}
    inc=$(grep -c ",INCONCLUSIVE," "$OUTPUT_FILE" || true); inc=${inc:-0}
    skip=$(grep -c ",SKIP," "$OUTPUT_FILE" || true); skip=${skip:-0}
    unk=$(grep -c ",UNKNOWN," "$OUTPUT_FILE" || true); unk=${unk:-0}

    log "Overall Results:"
    log "  Total runs:      $total_runs"
    log "  True positive:   $tp  ($(awk "BEGIN {if ($total_runs>0) printf \"%.1f\", 100*$tp/$total_runs; else print \"0.0\"}")%)"
    log "  False negative:  $fn  ($(awk "BEGIN {if ($total_runs>0) printf \"%.1f\", 100*$fn/$total_runs; else print \"0.0\"}")%)"
    log "  Inconclusive:    $inc ($(awk "BEGIN {if ($total_runs>0) printf \"%.1f\", 100*$inc/$total_runs; else print \"0.0\"}")%)"
    log "  Skipped:         $skip"
    log "  Unknown:         $unk"
    log ""

    # Wilson 95% CI on TP rate (TP / (TP+FN+INC), i.e., excluding SKIP/UNK).
    local definitive=$((tp + fn + inc))
    if [[ $definitive -gt 0 ]]; then
        python3 -c "
import math
n = $definitive
x = $tp
z = 1.96
p = x / n
denom = 1 + z**2 / n
center = (p + z**2 / (2*n)) / denom
margin = z * math.sqrt((p*(1-p)/n + z**2/(4*n**2))) / denom
lo = max(0, center - margin)
hi = min(1, center + margin)
print(f'True Positive Rate: {p:.4f} ({100*p:.2f}%) on {n} non-skip trials')
print(f'Wilson 95% CI: [{100*lo:.2f}%, {100*hi:.2f}%]')
" 2>/dev/null || log_warn "  (Python 3 not available for CI calculation)"
    fi
    log ""
}

# Main
main() {
    log "=========================================="
    log "Known-Leaky Crypto TP Rate Measurement"
    log "=========================================="
    log "Iterations: $ITERATIONS"
    log "Output: $OUTPUT_FILE"
    log ""

    local test_binary
    test_binary=$(find_test_binary)
    log "Test binary: $test_binary"

    local tests
    tests=$(discover_tests "$test_binary")
    local test_count
    test_count=$(echo "$tests" | wc -l | tr -d ' ')
    log "Discovered $test_count leaky tests"
    log ""

    # Optional sudo credential keeper. Enable via TPR_USE_SUDO=1 when you
    # want PMU-grade timers (kperf on macOS, perf_event on Linux).
    # Off by default to avoid TouchID prompts in casual local runs.
    if [[ "$USE_SUDO" == "1" ]]; then
        log "TPR_USE_SUDO=1: priming sudo credential cache..."
        sudo -v
        (while true; do sleep 50; sudo -n true 2>/dev/null || exit; done) &
        SUDO_KEEPER_PID=$!
        trap "kill $SUDO_KEEPER_PID 2>/dev/null" EXIT
    else
        log "Running without sudo (default timer). Set TPR_USE_SUDO=1 for PMU timers."
    fi

    init_csv

    local completed=0
    local skipped_resume=0
    local total=$((test_count * ITERATIONS))

    while IFS= read -r test_name; do
        local family ecosystem
        family=$(extract_family "$test_name")
        ecosystem="Rust"

        log "[$((completed / ITERATIONS + 1))/$test_count] $ecosystem / $family / $test_name"

        for ((i=1; i<=ITERATIONS; i++)); do
            completed=$((completed + 1))

            if is_completed "$test_name" "$i"; then
                skipped_resume=$((skipped_resume + 1))
                log "  Iteration $i/$ITERATIONS: already completed, skipping"
                continue
            fi

            log "  Iteration $i/$ITERATIONS ($completed/$total total)..."

            local start_time exit_code output
            start_time=$(date +%s)
            exit_code=0
            if [[ "$USE_SUDO" == "1" ]]; then
                output=$(sudo -E "$test_binary" "$test_name" --nocapture --test-threads=1 2>&1) || exit_code=$?
            else
                output=$("$test_binary" "$test_name" --nocapture --test-threads=1 2>&1) || exit_code=$?
            fi

            local elapsed timestamp parsed outcome leak_prob samples
            elapsed=$(($(date +%s) - start_time))
            timestamp=$(date -Iseconds)

            parsed=$(parse_outcome "$output" "$exit_code")
            outcome=$(echo "$parsed" | cut -d, -f1)
            leak_prob=$(echo "$parsed" | cut -d, -f2)
            samples=$(echo "$parsed" | cut -d, -f3)

            case "$outcome" in
                PASS)          log_success "    → TP (P=${leak_prob}%, ${elapsed}s)" ;;
                FAIL)          log_error   "    → FN (P=${leak_prob}%)" ;;
                INCONCLUSIVE)  log_warn    "    → INCONCLUSIVE (P=${leak_prob}%)" ;;
                SKIP)          log_warn    "    → SKIP" ;;
                *)             log_warn    "    → $outcome" ;;
            esac

            echo "$ecosystem,$family,$test_name,$i,$outcome,$leak_prob,$samples,$elapsed,$timestamp" >> "$OUTPUT_FILE"
        done
    done <<< "$tests"

    if [[ $skipped_resume -gt 0 ]]; then
        log "Resumed: $skipped_resume previously-completed iterations skipped"
    fi

    generate_report
}

main
