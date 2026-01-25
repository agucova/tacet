// Test: Missing required `measure` field produces helpful error
use tacet::timing_test;

fn main() {
    let _result = timing_test! {
        baseline: || 42u8,
        sample: || rand::random::<u8>(),
    };
}
