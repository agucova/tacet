# Profiling commands

profile_dir := env("PROFILE_DIR", "/var/tmp/tacet-profile")

profile-dir:
    @echo "Profile output dir: {{profile_dir}}"

# =============================================================================
# PROFILING TARGETS
# =============================================================================

# Profile the test suite (release mode)
# macOS: uses xctrace -> inferno-collapse-xctrace
# Linux: uses perf -> inferno-collapse-perf
[group('profile')]
tests:
    set -euo pipefail
    mkdir -p {{profile_dir}}
    echo "Profiling tests; output dir: {{profile_dir}}"
    if [ "$(uname)" = "Darwin" ]; then \
        xctrace record --template 'Time Profiler' --time-limit 60s \
            --output "{{profile_dir}}/tests.trace" --target-stdout - --launch -- \
            cargo nextest run --release --workspace --no-capture; \
        xctrace export --input "{{profile_dir}}/tests.trace" \
            --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
            > "{{profile_dir}}/tests.xml"; \
        nix-shell -p inferno --run "inferno-collapse-xctrace {{profile_dir}}/tests.xml > {{profile_dir}}/tests.folded"; \
    else \
        sudo sysctl -w kernel.perf_event_max_sample_rate=1000; \
        sudo nix-shell -p linuxPackages.perf inferno --run "perf record -g --call-graph dwarf -F 999 -o {{profile_dir}}/perf-tests.data -- cargo nextest run --release --workspace --no-capture"; \
        sudo nix-shell -p linuxPackages.perf inferno --run "perf script -i {{profile_dir}}/perf-tests.data | inferno-collapse-perf" > "{{profile_dir}}/tests.folded"; \
    fi
    echo "Wrote {{profile_dir}}/tests.folded"
    wc -l "{{profile_dir}}/tests.folded"

# Profile the benchmark crate (quick preset, tacet only)
[group('profile')]
bench:
    set -euo pipefail
    mkdir -p {{profile_dir}}
    echo "Profiling benchmark; output dir: {{profile_dir}}"
    cargo build --release -p tacet-bench
    if [ "$(uname)" = "Darwin" ]; then \
        xctrace record --template 'Time Profiler' --time-limit 10s \
            --output "{{profile_dir}}/bench.trace" --target-stdout - --launch -- \
            ./target/release/benchmark --preset quick --tools tacet \
            --samples 1000 --datasets 3 --effects 0,0.1 --patterns shift --noise iid; \
        xctrace export --input "{{profile_dir}}/bench.trace" \
            --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
            > "{{profile_dir}}/bench.xml"; \
        nix-shell -p inferno --run "inferno-collapse-xctrace {{profile_dir}}/bench.xml > {{profile_dir}}/bench.folded"; \
    else \
        sudo sysctl -w kernel.perf_event_max_sample_rate=1000; \
        sudo nix-shell -p linuxPackages.perf inferno --run "perf record -g --call-graph dwarf -F 999 -o {{profile_dir}}/perf-bench.data -- ./target/release/benchmark --preset quick --tools tacet --samples 1000 --datasets 3 --effects 0,0.1 --patterns shift --noise iid"; \
        sudo nix-shell -p linuxPackages.perf inferno --run "perf script -i {{profile_dir}}/perf-bench.data | inferno-collapse-perf" > "{{profile_dir}}/bench.folded"; \
    fi
    echo "Wrote {{profile_dir}}/bench.folded"
    wc -l "{{profile_dir}}/bench.folded"

# Profile tests + bench
[group('profile')]
all: tests bench

# =============================================================================
# PROFILING UTILITIES
# =============================================================================

[group('profile')]
summary target="bench":
    #!/usr/bin/env python3
    from collections import Counter
    import os
    import sys

    profile_dir = "{{profile_dir}}"
    target = "{{target}}"
    path = os.path.join(profile_dir, target + ".folded")

    if not os.path.exists(path):
        print("Missing " + path + ". Run: just profile " + target)
        sys.exit(1)

    counter = Counter()
    total = 0

    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            stack, count = line.rsplit(" ", 1)
            count = int(count)
            total += count
            leaf = stack.split(";")[-1]
            counter[leaf] += count

    print("Total samples:", total)
    print("Top leaf functions:")
    for name, count in counter.most_common(20):
        pct = (count / total * 100.0) if total else 0.0
        print("%8d %5.1f%% %s" % (count, pct, name))

[group('profile')]
clean:
    rm -rf "{{profile_dir}}"/*.folded "{{profile_dir}}"/*.xml "{{profile_dir}}"/*.trace "{{profile_dir}}"/perf-*.data
