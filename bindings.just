# Binding generation and test commands
# All bindings are now based on cbindgen C bindings as the foundation
#
# Architecture:
#   Rust Core -> C (cbindgen) -> C++ (header-only wrapper)
#                            -> Go (CGo)
#
# NOTE: Run from within devenv shell for toolchains:
#   devenv shell
#   just bindings all

# Output directories
bindings_dir := justfile_directory() + "/bindings"
cpp_bindings_dir := bindings_dir + "/cpp"

# C bindings paths (cbindgen-based, in-crate)
c_crate_dir := justfile_directory() + "/crates/tacet-c"
c_include_dir := c_crate_dir + "/include"
c_examples_dir := c_crate_dir + "/examples"

# Go bindings path (CGo-based)
go_dir := justfile_directory() + "/crates/tacet-go"

# Library paths
c_lib := justfile_directory() + "/target/release/libtacet_c.a"
c_dylib := justfile_directory() + "/target/release/libtacet_c.dylib"
c_so := justfile_directory() + "/target/release/libtacet_c.so"

# ============================================================================
# MAIN COMMANDS
# ============================================================================

# Build all bindings and run tests
[group('bindings')]
all: c cpp-test go-test
    @echo ""
    @echo "All bindings built and tested successfully!"
    @echo "  C:   {{c_include_dir}}/tacet.h"
    @echo "  C++: {{cpp_bindings_dir}}/tacet.hpp"
    @echo "  Go:  {{go_dir}}"

# Build C library (foundation for all bindings)
[group('bindings')]
build: c

# ============================================================================
# C BINDINGS (cbindgen-based, foundation layer)
# ============================================================================

# Build C bindings (generates header via cbindgen)
[group('bindings')]
c:
    @echo "Building tacet-c library..."
    cargo build --release -p tacet-c
    @echo "C header generated at {{c_include_dir}}/tacet.h"
    @echo "Static library at {{c_lib}}"

# Build C example: simple (one-shot analysis)
[group('bindings')]
c-example-simple: c
    @echo "Building simple.c example..."
    cc -O3 -Wall -Wextra -std=c11 \
        -I{{c_crate_dir}} \
        -o {{c_examples_dir}}/simple \
        {{c_examples_dir}}/simple.c \
        -L{{justfile_directory()}}/target/release \
        -ltacet_c -lpthread -lm
    @echo "Built: {{c_examples_dir}}/simple"

# Build C example: adaptive (adaptive sampling loop)
[group('bindings')]
c-example-adaptive: c
    @echo "Building adaptive.c example..."
    cc -O3 -Wall -Wextra -std=c11 \
        -I{{c_crate_dir}} \
        -o {{c_examples_dir}}/adaptive \
        {{c_examples_dir}}/adaptive.c \
        -L{{justfile_directory()}}/target/release \
        -ltacet_c -lpthread -lm
    @echo "Built: {{c_examples_dir}}/adaptive"

# Build C example: simple_callback (callback-based API)
[group('bindings')]
c-example-simple-callback: c
    @echo "Building simple_callback.c example..."
    cc -O3 -Wall -Wextra -std=c11 \
        -I{{c_crate_dir}} \
        -o {{c_examples_dir}}/simple_callback \
        {{c_examples_dir}}/simple_callback.c \
        -L{{justfile_directory()}}/target/release \
        -ltacet_c -lpthread -lm
    @echo "Built: {{c_examples_dir}}/simple_callback"

# Build all C examples
[group('bindings')]
c-examples: c-example-simple c-example-adaptive c-example-simple-callback
    @echo "All C examples built"

# Run C simple example
[group('bindings')]
c-run-simple: c-example-simple
    @echo "Running simple example..."
    @cd {{c_examples_dir}} && DYLD_LIBRARY_PATH={{justfile_directory()}}/target/release LD_LIBRARY_PATH={{justfile_directory()}}/target/release ./simple

# Run C adaptive example
[group('bindings')]
c-run-adaptive: c-example-adaptive
    @echo "Running adaptive example..."
    @cd {{c_examples_dir}} && DYLD_LIBRARY_PATH={{justfile_directory()}}/target/release LD_LIBRARY_PATH={{justfile_directory()}}/target/release ./adaptive

# Run C simple_callback example
[group('bindings')]
c-run-simple-callback: c-example-simple-callback
    @echo "Running simple_callback example..."
    @cd {{c_examples_dir}} && DYLD_LIBRARY_PATH={{justfile_directory()}}/target/release LD_LIBRARY_PATH={{justfile_directory()}}/target/release ./simple_callback

# Build and run C tests (CMocka)
[group('bindings')]
c-tests: c
    @echo "Building C tests..."
    mkdir -p {{c_crate_dir}}/tests/build
    cd {{c_crate_dir}}/tests/build && cmake -DCMAKE_BUILD_TYPE=Release ..
    cd {{c_crate_dir}}/tests/build && make
    @echo "Running fast C tests..."
    cd {{c_crate_dir}}/tests/build && DYLD_LIBRARY_PATH={{justfile_directory()}}/target/release LD_LIBRARY_PATH={{justfile_directory()}}/target/release ./tacet_tests fast

# Run all C tests including slow integration tests
[group('bindings')]
c-tests-all: c
    @echo "Building C tests..."
    mkdir -p {{c_crate_dir}}/tests/build
    cd {{c_crate_dir}}/tests/build && cmake -DCMAKE_BUILD_TYPE=Release ..
    cd {{c_crate_dir}}/tests/build && make
    @echo "Running all C tests (including slow tests)..."
    cd {{c_crate_dir}}/tests/build && DYLD_LIBRARY_PATH={{justfile_directory()}}/target/release LD_LIBRARY_PATH={{justfile_directory()}}/target/release ./tacet_tests all

# Run C tests with AddressSanitizer
[group('bindings')]
c-tests-asan: c
    @echo "Building C tests with AddressSanitizer..."
    mkdir -p {{c_crate_dir}}/tests/build-asan
    cd {{c_crate_dir}}/tests/build-asan && cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS="-fsanitize=address -fno-omit-frame-pointer" ..
    cd {{c_crate_dir}}/tests/build-asan && make
    @echo "Running C tests with ASan..."
    cd {{c_crate_dir}}/tests/build-asan && DYLD_LIBRARY_PATH={{justfile_directory()}}/target/release LD_LIBRARY_PATH={{justfile_directory()}}/target/release ./tacet_tests fast

# ============================================================================
# C++ BINDINGS (header-only wrapper over C)
# ============================================================================

# Compile C++ wrapper (verification only)
[group('bindings')]
cpp-compile: c
    @echo "Compiling C++ wrapper..."
    clang++ -std=c++20 -stdlib=libc++ \
        -I{{c_include_dir}} \
        -I{{cpp_bindings_dir}} \
        -c {{cpp_bindings_dir}}/test_cpp_wrapper.cpp \
        -o /tmp/test_cpp_wrapper.o
    @echo "C++ wrapper compiles successfully"

# Build and test C++ bindings
[group('bindings')]
cpp-test: c
    @echo "Building C++ test..."
    clang++ -std=c++20 -stdlib=libc++ \
        -I{{c_include_dir}} \
        -I{{cpp_bindings_dir}} \
        {{cpp_bindings_dir}}/test_cpp_wrapper.cpp \
        -L{{justfile_directory()}}/target/release \
        -ltacet_c \
        -o /tmp/test_cpp_wrapper
    @echo "Running C++ tests..."
    DYLD_LIBRARY_PATH={{justfile_directory()}}/target/release LD_LIBRARY_PATH={{justfile_directory()}}/target/release /tmp/test_cpp_wrapper

# ============================================================================
# GO BINDINGS (CGo wrapper over C)
# ============================================================================

# Build Go bindings
[group('bindings')]
go-build: c
    @echo "Building Go bindings..."
    cd {{go_dir}} && \
        CGO_LDFLAGS="-L{{justfile_directory()}}/target/release" \
        CGO_CFLAGS="-I{{c_include_dir}}" \
        go build ./...
    @echo "Go bindings built successfully"

# Test Go bindings
[group('bindings')]
go-test: c
    @echo "Testing Go bindings..."
    cd {{go_dir}} && \
        DYLD_LIBRARY_PATH={{justfile_directory()}}/target/release \
        LD_LIBRARY_PATH={{justfile_directory()}}/target/release \
        CGO_LDFLAGS="-L{{justfile_directory()}}/target/release" \
        CGO_CFLAGS="-I{{c_include_dir}}" \
        go test -v -timeout 5m ./...

# Test Go bindings in short mode (skip slow tests)
[group('bindings')]
go-test-short: c
    @echo "Testing Go bindings (short mode)..."
    cd {{go_dir}} && \
        DYLD_LIBRARY_PATH={{justfile_directory()}}/target/release \
        LD_LIBRARY_PATH={{justfile_directory()}}/target/release \
        CGO_LDFLAGS="-L{{justfile_directory()}}/target/release" \
        CGO_CFLAGS="-I{{c_include_dir}}" \
        go test -v -short ./...

# ============================================================================
# NODE.JS/BUN TESTS (NAPI bindings - separate from cbindgen chain)
# ============================================================================

# Path to NAPI crate
napi_crate_dir := justfile_directory() + "/crates/tacet-napi"

# Run Node.js/Bun binding tests
[group('bindings')]
napi-test:
    @echo "Running Node.js binding tests..."
    cd {{napi_crate_dir}} && bun test

# Run Node.js integration tests only
[group('bindings')]
napi-test-integration:
    @echo "Running Node.js integration tests..."
    cd {{napi_crate_dir}} && bun test test/integration.test.ts

# ============================================================================
# UTILITIES
# ============================================================================

# Clean generated/built bindings
[group('bindings')]
clean:
    rm -f {{c_examples_dir}}/simple {{c_examples_dir}}/adaptive
    rm -rf {{c_crate_dir}}/tests/build {{c_crate_dir}}/tests/build-asan
    rm -f /tmp/test_cpp_wrapper /tmp/test_cpp_wrapper.o
    @echo "Bindings cleaned"

# Check if bindings are up to date
[group('bindings')]
check:
    #!/usr/bin/env bash
    set -euo pipefail

    # Portable function to get modification time
    get_mtime() {
        stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo "0"
    }

    # Check C library
    if [[ -f "{{c_dylib}}" ]]; then
        lib_file="{{c_dylib}}"
    elif [[ -f "{{c_so}}" ]]; then
        lib_file="{{c_so}}"
    else
        echo "WARNING: C library not found. Run 'just bindings c' first."
        exit 0
    fi

    lib_time=$(get_mtime "$lib_file")
    header_file="{{c_include_dir}}/tacet.h"

    if [[ -f "$header_file" ]]; then
        header_time=$(get_mtime "$header_file")
        echo "C bindings: built (library: $lib_file)"
    else
        echo "WARNING: C header not found. Run 'just bindings c' to generate."
    fi

    # Check C++ wrapper
    cpp_file="{{cpp_bindings_dir}}/tacet.hpp"
    if [[ -f "$cpp_file" ]]; then
        echo "C++ wrapper: present (header-only, no generation needed)"
    else
        echo "WARNING: C++ wrapper not found at $cpp_file"
    fi

    # Check Go bindings
    go_file="{{go_dir}}/tacet.go"
    if [[ -f "$go_file" ]]; then
        echo "Go bindings: present (CGo-based)"
    else
        echo "WARNING: Go bindings not found at {{go_dir}}"
    fi

# Show versions
[group('bindings')]
versions:
    @echo "Toolchain versions:"
    @echo -n "  clang++: " && clang++ --version 2>/dev/null | head -1 || echo "not installed"
    @echo -n "  go: " && go version 2>/dev/null || echo "not installed"
    @echo -n "  cargo: " && cargo --version 2>/dev/null || echo "not installed"
