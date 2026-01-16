// Test: `measure` field must be a closure
use timing_oracle::timing_test;

fn my_test(input: &u8) {
    std::hint::black_box(input);
}

fn main() {
    let _result = timing_test! {
        baseline: || 42u8,
        sample: || rand::random::<u8>(),
        measure: my_test,  // Should be |input| my_test(input)
    };
}
