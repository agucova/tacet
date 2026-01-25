# Benchmark commands

# Run comparison benchmarks (vs DudeCT)
[group('bench')]
comparison:
    cargo bench -p tacet --bench comparison

# Run oracle benchmarks
[group('bench')]
oracle:
    cargo bench -p tacet --bench oracle

# Run all benchmarks
[group('bench')]
all:
    cargo bench --workspace
