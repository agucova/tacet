// Test: Missing required `baseline` field produces helpful error
use timing_oracle::timing_test;

fn main() {
    let _result = timing_test! {
        sample: || rand::random::<u8>(),
        measure: |input| {
            std::hint::black_box(input);
        },
    };
}
