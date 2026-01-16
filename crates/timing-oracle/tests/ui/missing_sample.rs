// Test: Missing required `sample` field produces helpful error
use timing_oracle::timing_test;

fn main() {
    let _result = timing_test! {
        baseline: || 42u8,
        measure: |input| {
            std::hint::black_box(input);
        },
    };
}
