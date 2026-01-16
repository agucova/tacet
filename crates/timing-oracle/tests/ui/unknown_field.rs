// Test: Unknown field produces helpful error
use timing_oracle::timing_test;

fn main() {
    let _result = timing_test! {
        baseline: || 42u8,
        sampl: || rand::random::<u8>(),  // Typo: should be `sample`
        measure: |input| {
            std::hint::black_box(input);
        },
    };
}
