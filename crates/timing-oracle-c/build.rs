//! Build script for timing-oracle-c
//!
//! Compiles csrc/to_measure.c which provides the timing-critical measurement loop.

fn main() {
    // Compile the C measurement loop
    cc::Build::new()
        .file("csrc/to_measure.c")
        .warnings(true)
        .extra_warnings(true)
        .opt_level(3) // Maximum optimization for timing-critical code
        .flag_if_supported("-fno-omit-frame-pointer") // For profiling
        .compile("to_measure");

    // Re-run build if C source changes
    println!("cargo:rerun-if-changed=csrc/to_measure.c");
    println!("cargo:rerun-if-changed=csrc/to_measure.h");
    println!("cargo:rerun-if-changed=include/timing_oracle.h");
}
