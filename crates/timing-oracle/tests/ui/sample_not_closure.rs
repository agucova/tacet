// Test: `sample` field must be a closure, not a value
use timing_oracle::timing_test;

fn main() {
    let value = 42u8;
    let _result = timing_test! {
        baseline: || 42u8,
        sample: value,  // Should be || value
        measure: |input| {
            std::hint::black_box(input);
        },
    };
}
