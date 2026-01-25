# CMakeLists.txt for timing-oracle C API tests
#
# Build instructions:
#   mkdir build && cd build
#   cmake ..
#   make
#   ./timing_oracle_tests
#
# Or with ctest:
#   ctest --output-on-failure
#
# Prerequisites:
#   - CMake 3.14+
#   - cmocka library (brew install cmocka / apt install libcmocka-dev)
#   - libtiming_oracle_c built (cargo build in parent crate)

cmake_minimum_required(VERSION 3.14)
project(timing_oracle_c_tests C)

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find cmocka library
# Try pkg-config first, fall back to find_library
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(CMOCKA QUIET cmocka)
endif()

if(NOT CMOCKA_FOUND)
    # Manual search for cmocka
    find_library(CMOCKA_LIBRARY
        NAMES cmocka
        HINTS
            ENV CMOCKA_ROOT
            /usr/local/lib
            /opt/homebrew/lib
    )
    find_path(CMOCKA_INCLUDE_DIR
        NAMES cmocka.h
        HINTS
            ENV CMOCKA_ROOT
            /usr/local/include
            /opt/homebrew/include
    )
    if(CMOCKA_LIBRARY AND CMOCKA_INCLUDE_DIR)
        set(CMOCKA_LIBRARIES ${CMOCKA_LIBRARY})
        set(CMOCKA_INCLUDE_DIRS ${CMOCKA_INCLUDE_DIR})
        set(CMOCKA_FOUND TRUE)
    endif()
endif()

if(NOT CMOCKA_FOUND)
    message(FATAL_ERROR
        "Could not find cmocka library. "
        "Install with: brew install cmocka (macOS) or apt install libcmocka-dev (Linux) "
        "or nix-shell -p cmocka"
    )
endif()

# Find the timing-oracle-c library
# Look in parent crate's target directory
set(TIMING_ORACLE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(TIMING_ORACLE_INCLUDE "${TIMING_ORACLE_ROOT}/include")

# Try to find the library in various locations
find_library(TIMING_ORACLE_LIB
    NAMES timing_oracle_c libtiming_oracle_c
    HINTS
        "${TIMING_ORACLE_ROOT}/../../target/release"
        "${TIMING_ORACLE_ROOT}/../../target/debug"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../target/release"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../target/debug"
    NO_DEFAULT_PATH
)

if(NOT TIMING_ORACLE_LIB)
    message(FATAL_ERROR
        "Could not find libtiming_oracle_c. "
        "Please build the Rust library first with: cargo build --release"
    )
endif()

message(STATUS "Found timing_oracle_c: ${TIMING_ORACLE_LIB}")
message(STATUS "Include directory: ${TIMING_ORACLE_INCLUDE}")
message(STATUS "CMocka include: ${CMOCKA_INCLUDE_DIRS}")
message(STATUS "CMocka libraries: ${CMOCKA_LIBRARIES}")

# Test sources
set(TEST_SOURCES
    test_main.c
    test_helpers.c
    test_config.c
    test_one_shot.c
    test_adaptive.c
    test_error_handling.c
    test_memory_safety.c
    test_known_leaky.c
    test_known_safe.c
)

# Create test executable
add_executable(timing_oracle_tests ${TEST_SOURCES})

# Include directories
target_include_directories(timing_oracle_tests PRIVATE
    ${TIMING_ORACLE_INCLUDE}
    ${CMOCKA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries
target_link_libraries(timing_oracle_tests PRIVATE
    ${TIMING_ORACLE_LIB}
    ${CMOCKA_LIBRARIES}
)

# Platform-specific settings
if(APPLE)
    # macOS: Link against system frameworks needed by Rust
    target_link_libraries(timing_oracle_tests PRIVATE
        "-framework Security"
        "-framework CoreFoundation"
    )
    # Set rpath for finding dylib
    set_target_properties(timing_oracle_tests PROPERTIES
        BUILD_RPATH "@loader_path;${CMAKE_CURRENT_SOURCE_DIR}/../../../target/release;${CMAKE_CURRENT_SOURCE_DIR}/../../../target/debug"
        INSTALL_RPATH "@loader_path"
    )
elseif(UNIX)
    # Linux: Link against pthread and dl
    target_link_libraries(timing_oracle_tests PRIVATE
        pthread
        dl
        m
    )
    # Set rpath for finding shared library
    set_target_properties(timing_oracle_tests PROPERTIES
        BUILD_RPATH "$ORIGIN;${CMAKE_CURRENT_SOURCE_DIR}/../../../target/release;${CMAKE_CURRENT_SOURCE_DIR}/../../../target/debug"
        INSTALL_RPATH "$ORIGIN"
    )
endif()

# Compiler warnings
target_compile_options(timing_oracle_tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
)

# Enable testing
enable_testing()

# Add test target for ctest
add_test(NAME timing_oracle_fast_tests
    COMMAND timing_oracle_tests fast
)

add_test(NAME timing_oracle_all_tests
    COMMAND timing_oracle_tests all
)

# Set test properties
set_tests_properties(timing_oracle_fast_tests PROPERTIES
    TIMEOUT 120
    LABELS "fast"
)

set_tests_properties(timing_oracle_all_tests PROPERTIES
    TIMEOUT 300
    LABELS "slow"
)

# Individual test group targets
add_test(NAME timing_oracle_config_tests
    COMMAND timing_oracle_tests config
)
add_test(NAME timing_oracle_one_shot_tests
    COMMAND timing_oracle_tests one_shot
)
add_test(NAME timing_oracle_adaptive_tests
    COMMAND timing_oracle_tests adaptive
)
add_test(NAME timing_oracle_error_tests
    COMMAND timing_oracle_tests error
)
add_test(NAME timing_oracle_memory_tests
    COMMAND timing_oracle_tests memory
)
add_test(NAME timing_oracle_known_leaky_tests
    COMMAND timing_oracle_tests known_leaky
)
add_test(NAME timing_oracle_known_safe_tests
    COMMAND timing_oracle_tests known_safe
)

# Set labels for individual tests
set_tests_properties(timing_oracle_config_tests PROPERTIES LABELS "fast")
set_tests_properties(timing_oracle_one_shot_tests PROPERTIES LABELS "fast")
set_tests_properties(timing_oracle_adaptive_tests PROPERTIES LABELS "fast")
set_tests_properties(timing_oracle_error_tests PROPERTIES LABELS "fast")
set_tests_properties(timing_oracle_memory_tests PROPERTIES LABELS "fast")
set_tests_properties(timing_oracle_known_leaky_tests PROPERTIES LABELS "slow" TIMEOUT 120)
set_tests_properties(timing_oracle_known_safe_tests PROPERTIES LABELS "slow" TIMEOUT 120)

# Custom target for fast tests
add_custom_target(test_fast
    COMMAND ${CMAKE_CTEST_COMMAND} -L fast --output-on-failure
    DEPENDS timing_oracle_tests
    COMMENT "Running fast tests..."
)

# Custom target for all tests
add_custom_target(test_all
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS timing_oracle_tests
    COMMENT "Running all tests..."
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== timing-oracle C tests configuration ===")
message(STATUS "Library: ${TIMING_ORACLE_LIB}")
message(STATUS "Include: ${TIMING_ORACLE_INCLUDE}")
message(STATUS "CMocka: ${CMOCKA_LIBRARIES}")
message(STATUS "")
message(STATUS "Build with: cmake --build .")
message(STATUS "Run fast tests: ./timing_oracle_tests fast")
message(STATUS "Run all tests: ./timing_oracle_tests all")
message(STATUS "Or use ctest: ctest -L fast --output-on-failure")
message(STATUS "")
