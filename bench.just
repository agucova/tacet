# Benchmark commands

# Run comparison benchmarks (vs DudeCT)
[group('bench')]
comparison:
    cargo bench -p timing-oracle --bench comparison

# Run oracle benchmarks
[group('bench')]
oracle:
    cargo bench -p timing-oracle --bench oracle

# Run all benchmarks
[group('bench')]
all:
    cargo bench --workspace
