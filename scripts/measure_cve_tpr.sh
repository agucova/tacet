#!/usr/bin/env bash
# Measure CVE detection rate across previously-reported-vulnerable crypto
# implementations and ecosystems. tacet-only (no cross-tool fanout).
#
# Usage: ./scripts/measure_cve_tpr.sh [iterations] [output_file]
#
# Environment variables:
#   CVE_USE_SUDO=1   Run each invocation under `sudo -E` so PMU cycle counters
#                    (kperf on macOS, perf_event on Linux) are available.
#                    Off by default for local smoke tests.
#   CVE_SKIP_RUST    Non-empty: skip Rust CVE probes.
#   CVE_SKIP_GO      Non-empty: skip Go CVE probes.
#   CVE_SKIP_JS      Non-empty: skip JS CVE probes.
#
# CSV schema (one row per (cve_id, iteration)):
#   cve_id,ecosystem,library,test_name,iteration,outcome,leak_probability,
#   samples,elapsed_sec,timestamp
#
# Outcome interpretation (leaky tests, identical semantics to measure_tpr.sh):
#   PASS         = detection       (tacet returned Fail)   "Test passed:" token
#   FAIL         = missed detection (tacet returned Pass)  "FAILED:" token
#   INCONCLUSIVE = underpowered                            "Inconclusive:" token
#   SKIP         = unmeasurable / skipped                  "Skipping:" token
#
# Downstream analyze_cve_tpr.py applies the three-way verdict split
#   {Detect, Inconclusive, Miss}. See that script for Wilson CIs.
#
# Resume support: re-running with the same OUTPUT_FILE skips
# (cve_id, iteration) pairs already present.

set -euo pipefail

ITERATIONS="${1:-20}"
OUTPUT_FILE="${2:-cve_tpr_results_$(date +%Y%m%d_%H%M%S).csv}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USE_SUDO="${CVE_USE_SUDO:-0}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()         { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
log_success() { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[$(date +%H:%M:%S)]${NC} $*"; }
log_error()   { echo -e "${RED}[$(date +%H:%M:%S)]${NC} $*"; }

# -------- CSV init / resume --------

init_csv() {
    local header="cve_id,ecosystem,library,test_name,iteration,outcome,leak_probability,samples,elapsed_sec,timestamp"
    if [[ -f "$OUTPUT_FILE" ]]; then
        local existing
        existing=$(head -1 "$OUTPUT_FILE")
        if [[ "$existing" != "$header" ]]; then
            log_error "Incompatible CSV header in $OUTPUT_FILE"
            log_error "  Expected: $header"
            log_error "  Got:      $existing"
            exit 1
        fi
        local n
        n=$(tail -n +2 "$OUTPUT_FILE" | wc -l | tr -d ' ')
        log "Resuming: $n rows already in $OUTPUT_FILE"
    else
        echo "$header" > "$OUTPUT_FILE"
        log "Initialized $OUTPUT_FILE"
    fi
}

is_completed() {
    local cve_id="$1" iteration="$2"
    grep -q "^${cve_id},.*,${iteration}," "$OUTPUT_FILE" 2>/dev/null
}

record_row() {
    local cve_id="$1" ecosystem="$2" library="$3" test_name="$4" iteration="$5"
    local outcome="$6" leak_prob="$7" samples="$8" elapsed="$9" timestamp="${10}"
    echo "$cve_id,$ecosystem,$library,$test_name,$iteration,$outcome,$leak_prob,$samples,$elapsed,$timestamp" >> "$OUTPUT_FILE"
}

# -------- Outcome parsers --------

# Rust/Go/JS test harnesses emit output shaped differently. Each parser
# normalizes to the standard outcome tokens (PASS/FAIL/INCONCLUSIVE/SKIP).

# Rust test binaries driven via `cargo test` print "Test passed:" /
# "Inconclusive:" / "Skipping:" tokens when wrapped in the leaky/assert.rs
# recorder. But some CVE tests (like investigation/rsa_vulnerability.rs::
# exp_p1_padding_oracle_basic) use plain assertion patterns — we parse the
# tacet Outcome from its stdout format instead.
parse_rust_output() {
    local output="$1" exit_code="$2"
    local outcome="UNKNOWN" leak_prob="0.0" samples="0"

    # Priority: explicit tokens from leaky/assert.rs, then fall back to
    # raw Outcome keywords, then exit code.
    if echo "$output" | grep -q "Test passed: Leak detected"; then
        outcome="PASS"
    elif echo "$output" | grep -q "FAILED: No leak detected"; then
        outcome="FAIL"
    elif echo "$output" | grep -q "🚨 PADDING ORACLE DETECTED\|Fail\s*(P="; then
        outcome="PASS"
    elif echo "$output" | grep -q "Inconclusive:"; then
        outcome="INCONCLUSIVE"
    elif echo "$output" | grep -q "Skipping:\|Unmeasurable:"; then
        outcome="SKIP"
    elif echo "$output" | grep -q "No padding oracle.*detected\|Pass\s*(P="; then
        outcome="FAIL"
    elif [[ "$exit_code" -ne 0 ]]; then
        outcome="FAIL"
    fi

    leak_prob=$(echo "$output" | grep -oE 'P\(?leak\)?[ =][0-9.]+%?' | head -1 | sed -E 's/.*[ =]([0-9.]+)%?/\1/')
    [[ -z "$leak_prob" ]] && leak_prob="0.0"
    samples=$(echo "$output" | grep -oE 'Samples: [0-9]+' | head -1 | sed 's/Samples: //')
    [[ -z "$samples" ]] && samples="0"

    echo "$outcome,$leak_prob,$samples"
}

# Go test binaries emit:
#   Outcome: fail | pass | inconclusive | unmeasurable
#   P(leak): XX.XX%
# from the custom t.Logf in stdlib_crypto_test.go's leaky variants. The
# assertion (t.Errorf) only triggers on MISS, so exit_code is useful but
# the Outcome: line is more reliable.
parse_go_output() {
    local output="$1" exit_code="$2"
    local outcome="UNKNOWN" leak_prob="0.0" samples="0"

    local raw_outcome
    raw_outcome=$(echo "$output" | grep -oE 'Outcome: [a-zA-Z]+' | head -1 | sed 's/Outcome: //')
    case "$(echo "$raw_outcome" | tr '[:upper:]' '[:lower:]')" in
        fail)          outcome="PASS" ;;
        pass)          outcome="FAIL" ;;
        inconclusive)  outcome="INCONCLUSIVE" ;;
        unmeasurable)  outcome="SKIP" ;;
        *)
            if [[ "$exit_code" -ne 0 ]]; then
                outcome="FAIL"
            fi
            ;;
    esac

    leak_prob=$(echo "$output" | grep -oE 'P\(leak\): [0-9.]+%' | head -1 | sed 's/P(leak): //;s/%//')
    [[ -z "$leak_prob" ]] && leak_prob="0.0"
    samples=$(echo "$output" | grep -oE 'Samples[: ]+[0-9]+' | head -1 | sed -E 's/[^0-9]+([0-9]+)/\1/')
    [[ -z "$samples" ]] && samples="0"

    echo "$outcome,$leak_prob,$samples"
}

# JS (bun:test) output for node-forge test: emits
#   RSA PKCS#1 v1.5 decryption: fail | pass | inconclusive
#   Leak probability: XX.XX%
# We invert for leaky-test semantics.
parse_js_output() {
    local output="$1" exit_code="$2"
    local outcome="UNKNOWN" leak_prob="0.0" samples="0"

    local raw
    raw=$(echo "$output" | grep -oE 'decryption: [a-zA-Z]+' | head -1 | awk '{print tolower($NF)}')
    case "$raw" in
        fail)          outcome="PASS" ;;
        pass)          outcome="FAIL" ;;
        inconclusive)  outcome="INCONCLUSIVE" ;;
        *)
            if [[ "$exit_code" -ne 0 ]]; then
                outcome="FAIL"
            fi
            ;;
    esac

    leak_prob=$(echo "$output" | grep -oE 'Leak probability: [0-9.]+%' | head -1 | sed 's/Leak probability: //;s/%//')
    [[ -z "$leak_prob" ]] && leak_prob="0.0"
    samples=$(echo "$output" | grep -oE 'Samples used: [0-9]+' | head -1 | sed 's/Samples used: //')
    [[ -z "$samples" ]] && samples="0"

    echo "$outcome,$leak_prob,$samples"
}

# -------- Per-ecosystem test launchers --------

run_rust_cargo_test() {
    # Args: cve_id library test_path test_fn iteration
    local cve_id="$1" library="$2" test_path="$3" test_fn="$4" iter="$5"
    local t0 ts output exit_code=0

    t0=$(date +%s)
    if [[ "$USE_SUDO" == "1" ]]; then
        output=$(sudo -E cargo test --release -p tacet --test "$test_path" -- "$test_fn" \
            --ignored --nocapture --test-threads=1 2>&1) || exit_code=$?
    else
        output=$(cargo test --release -p tacet --test "$test_path" -- "$test_fn" \
            --ignored --nocapture --test-threads=1 2>&1) || exit_code=$?
    fi
    local elapsed=$(($(date +%s) - t0))
    ts=$(date -Iseconds)

    local parsed
    parsed=$(parse_rust_output "$output" "$exit_code")
    local outcome leak_prob samples
    outcome=$(echo "$parsed" | cut -d, -f1)
    leak_prob=$(echo "$parsed" | cut -d, -f2)
    samples=$(echo "$parsed" | cut -d, -f3)

    case "$outcome" in
        PASS)         log_success "    → Detect (P=${leak_prob}%, ${elapsed}s)" ;;
        FAIL)         log_error   "    → Miss (P=${leak_prob}%)" ;;
        INCONCLUSIVE) log_warn    "    → INCONCLUSIVE (P=${leak_prob}%)" ;;
        SKIP)         log_warn    "    → SKIP" ;;
        *)            log_warn    "    → $outcome" ;;
    esac

    record_row "$cve_id" "Rust" "$library" "$test_fn" "$iter" \
               "$outcome" "$leak_prob" "$samples" "$elapsed" "$ts"
}

run_go_test() {
    # Args: cve_id library test_fn iteration
    local cve_id="$1" library="$2" test_fn="$3" iter="$4"
    local t0 ts output exit_code=0

    if [[ ! -d "$REPO_ROOT/crates/tacet-go" ]]; then
        log_warn "  crates/tacet-go not present; skipping $cve_id"
        return
    fi

    t0=$(date +%s)
    pushd "$REPO_ROOT/crates/tacet-go" > /dev/null
    if [[ "$USE_SUDO" == "1" ]]; then
        output=$(sudo -E go test -v -run "^${test_fn}\$" -count=1 2>&1) || exit_code=$?
    else
        output=$(go test -v -run "^${test_fn}\$" -count=1 2>&1) || exit_code=$?
    fi
    popd > /dev/null
    local elapsed=$(($(date +%s) - t0))
    ts=$(date -Iseconds)

    local parsed
    parsed=$(parse_go_output "$output" "$exit_code")
    local outcome leak_prob samples
    outcome=$(echo "$parsed" | cut -d, -f1)
    leak_prob=$(echo "$parsed" | cut -d, -f2)
    samples=$(echo "$parsed" | cut -d, -f3)

    case "$outcome" in
        PASS)         log_success "    → Detect (P=${leak_prob}%, ${elapsed}s)" ;;
        FAIL)         log_error   "    → Miss (P=${leak_prob}%)" ;;
        INCONCLUSIVE) log_warn    "    → INCONCLUSIVE (P=${leak_prob}%)" ;;
        SKIP)         log_warn    "    → SKIP" ;;
        *)            log_warn    "    → $outcome" ;;
    esac

    record_row "$cve_id" "Go" "$library" "$test_fn" "$iter" \
               "$outcome" "$leak_prob" "$samples" "$elapsed" "$ts"
}

run_js_test() {
    # Args: cve_id library test_file test_name iteration
    local cve_id="$1" library="$2" test_file="$3" test_name="$4" iter="$5"
    local t0 ts output exit_code=0

    if [[ ! -d "$REPO_ROOT/crates/tacet-wasm" ]]; then
        log_warn "  crates/tacet-wasm not present; skipping $cve_id"
        return
    fi

    t0=$(date +%s)
    pushd "$REPO_ROOT/crates/tacet-wasm" > /dev/null
    output=$(bun test "tests/${test_file}" -t "$test_name" 2>&1) || exit_code=$?
    popd > /dev/null
    local elapsed=$(($(date +%s) - t0))
    ts=$(date -Iseconds)

    local parsed
    parsed=$(parse_js_output "$output" "$exit_code")
    local outcome leak_prob samples
    outcome=$(echo "$parsed" | cut -d, -f1)
    leak_prob=$(echo "$parsed" | cut -d, -f2)
    samples=$(echo "$parsed" | cut -d, -f3)

    case "$outcome" in
        PASS)         log_success "    → Detect (P=${leak_prob}%, ${elapsed}s)" ;;
        FAIL)         log_error   "    → Miss (P=${leak_prob}%)" ;;
        INCONCLUSIVE) log_warn    "    → INCONCLUSIVE (P=${leak_prob}%)" ;;
        SKIP)         log_warn    "    → SKIP" ;;
        *)            log_warn    "    → $outcome" ;;
    esac

    record_row "$cve_id" "JS" "$library" "$test_name" "$iter" \
               "$outcome" "$leak_prob" "$samples" "$elapsed" "$ts"
}

# -------- Entry point --------

run_iteration() {
    local iter="$1"

    # B1: MARVIN (CVE-2023-49092) via the investigation binary.
    if [[ -z "${CVE_SKIP_RUST:-}" ]]; then
        local cve_id="CVE-2023-49092"
        log "[$cve_id] iter $iter: Rust rsa 0.9.9 MARVIN"
        if is_completed "$cve_id" "$iter"; then
            log_warn "    already completed, skipping"
        else
            run_rust_cargo_test "$cve_id" "rsa-0.9.9" \
                "investigation" "exp_p1_padding_oracle_basic" "$iter"
        fi
    fi

    # B3: Go stdlib RSA PKCS1v15 known limitation.
    if [[ -z "${CVE_SKIP_GO:-}" ]]; then
        local cve_id="Go-RSA-PKCS1v15-KnownLimit"
        log "[$cve_id] iter $iter: Go stdlib crypto/rsa"
        if is_completed "$cve_id" "$iter"; then
            log_warn "    already completed, skipping"
        else
            run_go_test "$cve_id" "crypto/rsa" \
                "TestGoStdlibRSA_PKCS1v15_KnownLimitation_AssertLeak" "$iter"
        fi
    fi

    # B5: node-forge RSA PKCS#1 v1.5 decryption (MARVIN-class, CVE-2025-12816).
    if [[ -z "${CVE_SKIP_JS:-}" ]]; then
        local cve_id="CVE-2025-12816"
        log "[$cve_id] iter $iter: JS node-forge"
        if is_completed "$cve_id" "$iter"; then
            log_warn "    already completed, skipping"
        else
            run_js_test "$cve_id" "node-forge" "node-forge.test.ts" \
                "RSA PKCS#1 v1.5 decryption timing (MARVIN-class)" "$iter"
        fi
    fi
}

main() {
    log "=========================================="
    log "CVE True Positive Rate (tacet-only)"
    log "=========================================="
    log "Iterations: $ITERATIONS"
    log "Output: $OUTPUT_FILE"
    log "Repo root: $REPO_ROOT"
    log ""

    if [[ "$USE_SUDO" == "1" ]]; then
        log "CVE_USE_SUDO=1: priming sudo cache"
        sudo -v
        (while true; do sleep 50; sudo -n true 2>/dev/null || exit; done) &
        local keeper=$!
        trap "kill $keeper 2>/dev/null" EXIT
    fi

    init_csv

    for ((i=1; i<=ITERATIONS; i++)); do
        log ""
        log "================ iteration $i/$ITERATIONS ================"
        run_iteration "$i"
    done

    log ""
    log "Done. $(tail -n +2 "$OUTPUT_FILE" | wc -l | tr -d ' ') rows in $OUTPUT_FILE"
}

main
