# Makefile for timing-oracle C examples
#
# Usage:
#   make              Build all examples
#   make simple       Build single example
#   make clean        Remove built binaries
#   make WITH_OPENSSL=1  Build examples requiring OpenSSL

# Paths (relative to examples directory)
LIB_DIR ?= ../../../target/release
INCLUDE_DIR ?= ../include

# Compiler settings
CC ?= cc
CFLAGS = -Wall -Wextra -Wno-unused-parameter -O2 -std=c11 -I$(INCLUDE_DIR)
LDFLAGS = -L$(LIB_DIR)

# Platform detection
UNAME := $(shell uname)

# Library name
ifeq ($(UNAME), Darwin)
    # macOS: static library with framework dependencies
    LIBS = -ltiming_oracle_c -framework Security -framework CoreFoundation
    # Also need libiconv on macOS
    LIBS += -liconv
else ifeq ($(UNAME), Linux)
    # Linux: static library with system dependencies
    LIBS = -ltiming_oracle_c -lpthread -ldl -lm
else
    # Generic Unix
    LIBS = -ltiming_oracle_c -lpthread -lm
endif

# Core examples (no external dependencies)
EXAMPLES = simple compare attacker_models result_handling custom_config no_std_embedded

# Optional examples requiring OpenSSL
ifdef WITH_OPENSSL
    EXAMPLES += crypto_hmac
    CFLAGS += -DWITH_OPENSSL
    ifeq ($(UNAME), Darwin)
        # Homebrew OpenSSL location
        OPENSSL_PREFIX ?= $(shell brew --prefix openssl 2>/dev/null || echo /usr/local/opt/openssl)
        CFLAGS += -I$(OPENSSL_PREFIX)/include
        LDFLAGS += -L$(OPENSSL_PREFIX)/lib
    endif
    LIBS += -lssl -lcrypto
endif

.PHONY: all clean help check-lib

# Default target
all: check-lib $(EXAMPLES)

# Help target
help:
	@echo "timing-oracle C examples"
	@echo ""
	@echo "Usage:"
	@echo "  make                 Build all core examples"
	@echo "  make WITH_OPENSSL=1  Build all examples including crypto_hmac"
	@echo "  make <example>       Build a single example"
	@echo "  make clean           Remove built binaries"
	@echo ""
	@echo "Before building, ensure the library is compiled:"
	@echo "  cd ../../.. && cargo build --release -p timing-oracle-c"
	@echo ""
	@echo "Examples:"
	@echo "  simple          - Basic usage"
	@echo "  compare         - Leaky vs constant-time code"
	@echo "  attacker_models - Threat model selection"
	@echo "  result_handling - Full result interpretation"
	@echo "  custom_config   - Configuration options"
	@echo "  no_std_embedded - External time tracking"
	@echo "  crypto_hmac     - Real crypto (requires OpenSSL)"

# Check that the library exists
check-lib:
	@if [ ! -f "$(LIB_DIR)/libtiming_oracle_c.a" ]; then \
		echo "Error: Library not found at $(LIB_DIR)/libtiming_oracle_c.a"; \
		echo "Build it first with:"; \
		echo "  cd ../../.. && cargo build --release -p timing-oracle-c"; \
		exit 1; \
	fi

# Pattern rule for building examples
%: %.c common.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

# Clean up
clean:
	rm -f $(EXAMPLES) crypto_hmac
	rm -rf *.dSYM

# Individual targets (for explicit dependencies)
simple: simple.c common.h
compare: compare.c common.h
attacker_models: attacker_models.c common.h
result_handling: result_handling.c common.h
custom_config: custom_config.c common.h
no_std_embedded: no_std_embedded.c common.h
crypto_hmac: crypto_hmac.c common.h
