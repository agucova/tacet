// Test: Missing required `measure` field produces helpful error
use timing_oracle::timing_test;

fn main() {
    let _result = timing_test! {
        baseline: || 42u8,
        sample: || rand::random::<u8>(),
    };
}
