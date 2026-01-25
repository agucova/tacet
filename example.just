# Example commands

# Run simple example
[group('example')]
simple:
    cargo run --release -p tacet --example simple

# Run comparison example
[group('example')]
compare:
    cargo run --release -p tacet --example compare

# Run AES example
[group('example')]
aes:
    cargo run --release -p tacet --example aes

# Run XOR test example
[group('example')]
xor:
    cargo run --release -p tacet --example test_xor

# Run oracle delay test
[group('example')]
delay:
    cargo run --release -p tacet --example test_oracle_delay
