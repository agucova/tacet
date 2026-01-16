// Test: Duplicate field produces error
use timing_oracle::timing_test;

fn main() {
    let _result = timing_test! {
        baseline: || 42u8,
        baseline: || 43u8,  // Duplicate!
        sample: || rand::random::<u8>(),
        measure: |input| {
            std::hint::black_box(input);
        },
    };
}
