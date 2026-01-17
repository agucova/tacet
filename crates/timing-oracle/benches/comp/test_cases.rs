//! Shared test cases for comparison benchmarks.
//!
//! Provides standardized test cases that can be used across different
//! timing analysis tools for fair comparison.
//!
//! Test cases are categorized as:
//! - **Clear leaky**: Obvious timing leaks (early exit, branches)
//! - **Clear safe**: Obviously constant-time operations
//! - **Subtle leaky**: Small timing differences that stress detection
//! - **Noisy safe**: High variance but no actual leak
//! - **Edge cases**: Platform-dependent or borderline cases

use rand::Rng;

/// A test case that can be run by different timing analysis tools.
pub trait TestCase: Send + Sync {
    /// Name of the test case
    fn name(&self) -> &str;

    /// Whether this test case is expected to show a timing leak
    fn expected_leaky(&self) -> bool;

    /// Category for grouping in reports
    fn category(&self) -> TestCategory {
        if self.expected_leaky() {
            TestCategory::ClearLeaky
        } else {
            TestCategory::ClearSafe
        }
    }

    /// Generate the fixed input operation
    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync>;

    /// Generate the random input operation
    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync>;

    /// Generate Rust code for the fixed operation (for dudect code generation)
    fn fixed_code(&self) -> String;

    /// Generate Rust code for the random operation (for dudect code generation)
    fn random_code(&self) -> String;

    /// Generate helper code needed by both operations (for dudect code generation)
    fn helper_code(&self) -> String;
}

/// Categories for test cases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestCategory {
    /// Obvious timing leaks
    ClearLeaky,
    /// Obviously constant-time
    ClearSafe,
    /// Subtle timing differences
    SubtleLeaky,
    /// High variance but safe
    NoisySafe,
    /// Platform-dependent behavior
    EdgeCase,
}

// =============================================================================
// CLEAR LEAKY TEST CASES
// =============================================================================

/// Early-exit comparison (KNOWN LEAKY)
pub struct EarlyExitCompare;

impl TestCase for EarlyExitCompare {
    fn name(&self) -> &str {
        "early_exit_comparison"
    }

    fn expected_leaky(&self) -> bool {
        true
    }

    fn category(&self) -> TestCategory {
        TestCategory::ClearLeaky
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        let secret = [0u8; 512];
        let input = [0u8; 512];
        Box::new(move || {
            std::hint::black_box(early_exit_compare(&secret, &input));
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        let secret = [0u8; 512];
        Box::new(move || {
            let input = rand_bytes_512();
            std::hint::black_box(early_exit_compare(&secret, &input));
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let secret = [0u8; 512];
            let input = [0u8; 512];
            std::hint::black_box(early_exit_compare(&secret, &input));
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let secret = [0u8; 512];
            let input = rand_bytes_512();
            std::hint::black_box(early_exit_compare(&secret, &input));
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        r#"
fn early_exit_compare(a: &[u8], b: &[u8]) -> bool {
    for i in 0..a.len().min(b.len()) {
        if a[i] != b[i] {
            return false;
        }
    }
    a.len() == b.len()
}

fn rand_bytes_512() -> [u8; 512] {
    use dudect_bencher::rand::thread_rng;
    let mut arr = [0u8; 512];
    thread_rng().fill(&mut arr[..]);
    arr
}
        "#
        .to_string()
    }
}

/// Branch-on-zero timing (KNOWN LEAKY)
pub struct BranchOnZero;

impl TestCase for BranchOnZero {
    fn name(&self) -> &str {
        "branch_on_zero"
    }

    fn expected_leaky(&self) -> bool {
        true
    }

    fn category(&self) -> TestCategory {
        TestCategory::ClearLeaky
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            let x = 0u8;
            std::hint::black_box(branch_on_zero(x));
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            let x = rand::rng().random::<u8>() | 1; // Never zero
            std::hint::black_box(branch_on_zero(x));
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let x = 0u8;
            std::hint::black_box(helper_branch_on_zero(x));
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            use dudect_bencher::rand::thread_rng;
            let x = thread_rng().gen::<u8>() | 1; // Never zero
            std::hint::black_box(helper_branch_on_zero(x));
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        r#"
fn helper_branch_on_zero(x: u8) -> u8 {
    if x == 0 {
        // Simulate expensive operation
        std::hint::black_box(0u8);
        for _ in 0..1000 {
            std::hint::black_box(0u8);
        }
        0
    } else {
        x
    }
}
        "#
        .to_string()
    }
}

// =============================================================================
// CLEAR SAFE TEST CASES
// =============================================================================

/// XOR-based constant-time comparison (KNOWN SAFE)
pub struct XorCompare;

impl TestCase for XorCompare {
    fn name(&self) -> &str {
        "xor_compare"
    }

    fn expected_leaky(&self) -> bool {
        false
    }

    fn category(&self) -> TestCategory {
        TestCategory::ClearSafe
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        let secret = [0xABu8; 32];
        let input = [0x00u8; 32];
        Box::new(move || {
            std::hint::black_box(constant_time_compare(&secret, &input));
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        let secret = [0xABu8; 32];
        Box::new(move || {
            let input = rand_bytes_32();
            std::hint::black_box(constant_time_compare(&secret, &input));
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let secret = [0xABu8; 32];
            let input = [0x00u8; 32];
            std::hint::black_box(constant_time_compare(&secret, &input));
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let secret = [0xABu8; 32];
            let input = rand_bytes_32();
            std::hint::black_box(constant_time_compare(&secret, &input));
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        r#"
fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    let mut acc = 0u8;
    for i in 0..a.len().min(b.len()) {
        acc |= a[i] ^ b[i];
    }
    acc == 0 && a.len() == b.len()
}

fn rand_bytes_32() -> [u8; 32] {
    use dudect_bencher::rand::thread_rng;
    let mut arr = [0u8; 32];
    thread_rng().fill(&mut arr[..]);
    arr
}
        "#
        .to_string()
    }
}

/// Simple XOR operation (KNOWN SAFE)
pub struct XorOperation;

impl TestCase for XorOperation {
    fn name(&self) -> &str {
        "xor_operation"
    }

    fn expected_leaky(&self) -> bool {
        false
    }

    fn category(&self) -> TestCategory {
        TestCategory::ClearSafe
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        let a = [0xABu8; 32];
        let b = [0x00u8; 32];
        Box::new(move || {
            std::hint::black_box(xor_bytes(&a, &b));
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        let a = [0xABu8; 32];
        Box::new(move || {
            let b = rand_bytes_32();
            std::hint::black_box(xor_bytes(&a, &b));
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let a = [0xABu8; 32];
            let b = [0x00u8; 32];
            std::hint::black_box(xor_bytes(&a, &b));
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let a = [0xABu8; 32];
            let b = rand_bytes_32();
            std::hint::black_box(xor_bytes(&a, &b));
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        r#"
fn xor_bytes(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    for i in 0..32 {
        result[i] = a[i] ^ b[i];
    }
    result
}

fn rand_bytes_32() -> [u8; 32] {
    use dudect_bencher::rand::thread_rng;
    let mut arr = [0u8; 32];
    thread_rng().fill(&mut arr[..]);
    arr
}
        "#
        .to_string()
    }
}

// =============================================================================
// SUBTLE LEAKY TEST CASES
// =============================================================================

/// Micro-leak: Only 1-5 extra operations (SUBTLE LEAKY)
/// Tests detection sensitivity for very small timing differences
pub struct MicroLeak;

impl TestCase for MicroLeak {
    fn name(&self) -> &str {
        "micro_leak"
    }

    fn expected_leaky(&self) -> bool {
        true
    }

    fn category(&self) -> TestCategory {
        TestCategory::SubtleLeaky
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Fixed class: do 5 extra operations
            let mut x = 0u64;
            for _ in 0..5 {
                x = std::hint::black_box(x.wrapping_add(1));
            }
            std::hint::black_box(x);
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Random class: no extra operations
            let x = 0u64;
            std::hint::black_box(x);
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let mut x = 0u64;
            for _ in 0..5 {
                x = std::hint::black_box(x.wrapping_add(1));
            }
            std::hint::black_box(x);
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let x = 0u64;
            std::hint::black_box(x);
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        String::new()
    }
}

/// Variable iteration count: n vs n+1 loop iterations (SUBTLE LEAKY)
pub struct VariableIteration;

impl TestCase for VariableIteration {
    fn name(&self) -> &str {
        "variable_iteration"
    }

    fn expected_leaky(&self) -> bool {
        true
    }

    fn category(&self) -> TestCategory {
        TestCategory::SubtleLeaky
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Fixed: 100 iterations
            let mut acc = 0u64;
            for i in 0..100 {
                acc = std::hint::black_box(acc.wrapping_add(i as u64));
            }
            std::hint::black_box(acc);
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Random: 101 iterations (just 1 more)
            let mut acc = 0u64;
            for i in 0..101 {
                acc = std::hint::black_box(acc.wrapping_add(i as u64));
            }
            std::hint::black_box(acc);
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let mut acc = 0u64;
            for i in 0..100 {
                acc = std::hint::black_box(acc.wrapping_add(i as u64));
            }
            std::hint::black_box(acc);
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let mut acc = 0u64;
            for i in 0..101 {
                acc = std::hint::black_box(acc.wrapping_add(i as u64));
            }
            std::hint::black_box(acc);
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        String::new()
    }
}

/// Table lookup with potential cache timing (SUBTLE LEAKY)
/// Simulates S-box lookups that might have cache-dependent timing
pub struct TableLookup;

impl TestCase for TableLookup {
    fn name(&self) -> &str {
        "table_lookup"
    }

    fn expected_leaky(&self) -> bool {
        true // Cache timing can leak
    }

    fn category(&self) -> TestCategory {
        TestCategory::SubtleLeaky
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Fixed: Always access same cache line (indices 0-15)
            let table = sbox_table();
            let mut acc = 0u8;
            for value in table.iter().take(16) {
                acc ^= std::hint::black_box(*value);
            }
            std::hint::black_box(acc);
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Random: Access scattered cache lines
            let table = sbox_table();
            let mut acc = 0u8;
            let indices = [
                0, 64, 128, 192, 16, 80, 144, 208, 32, 96, 160, 224, 48, 112, 176, 240,
            ];
            for &i in &indices {
                acc ^= std::hint::black_box(table[i]);
            }
            std::hint::black_box(acc);
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let table = sbox_table();
            let mut acc = 0u8;
            for i in 0..16 {
                acc ^= std::hint::black_box(table[i]);
            }
            std::hint::black_box(acc);
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let table = sbox_table();
            let mut acc = 0u8;
            let indices = [0, 64, 128, 192, 16, 80, 144, 208, 32, 96, 160, 224, 48, 112, 176, 240];
            for &i in &indices {
                acc ^= std::hint::black_box(table[i]);
            }
            std::hint::black_box(acc);
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        r#"
fn sbox_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    for i in 0..256 {
        // Simple S-box: multiplicative inverse approximation
        table[i] = ((i as u8).wrapping_mul(0x1B)) ^ ((i as u8).rotate_left(3));
    }
    table
}
        "#
        .to_string()
    }
}

/// Cache-line crossing: Access patterns that cross cache line boundaries (SUBTLE LEAKY)
pub struct CacheLineCrossing;

impl TestCase for CacheLineCrossing {
    fn name(&self) -> &str {
        "cache_line_crossing"
    }

    fn expected_leaky(&self) -> bool {
        true
    }

    fn category(&self) -> TestCategory {
        TestCategory::SubtleLeaky
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Aligned access within single cache line
            let data = [0u64; 128]; // 1KB, multiple cache lines
            let mut acc = 0u64;
            // Access 8 consecutive u64s (64 bytes = 1 cache line)
            for value in data.iter().take(8) {
                acc = acc.wrapping_add(std::hint::black_box(*value));
            }
            std::hint::black_box(acc);
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Strided access crossing cache lines
            let data = [0u64; 128];
            let mut acc = 0u64;
            // Access every 8th u64 (crosses cache lines)
            for value in data.iter().step_by(8).take(8) {
                acc = acc.wrapping_add(std::hint::black_box(*value));
            }
            std::hint::black_box(acc);
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let data = [0u64; 128];
            let mut acc = 0u64;
            for i in 0..8 {
                acc = acc.wrapping_add(std::hint::black_box(data[i]));
            }
            std::hint::black_box(acc);
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let data = [0u64; 128];
            let mut acc = 0u64;
            for i in 0..8 {
                acc = acc.wrapping_add(std::hint::black_box(data[i * 8]));
            }
            std::hint::black_box(acc);
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        String::new()
    }
}

/// Branch predictor effects: Predictable vs unpredictable branches (SUBTLE LEAKY)
pub struct BranchPredictor;

impl TestCase for BranchPredictor {
    fn name(&self) -> &str {
        "branch_predictor"
    }

    fn expected_leaky(&self) -> bool {
        true
    }

    fn category(&self) -> TestCategory {
        TestCategory::SubtleLeaky
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Predictable pattern: always true
            let mut acc = 0u64;
            for i in 0..100 {
                if i < 100 {
                    // Always taken
                    acc = acc.wrapping_add(1);
                }
            }
            std::hint::black_box(acc);
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Unpredictable pattern: based on bit pattern
            let mut acc = 0u64;
            let pattern = 0xAAAAAAAAAAAAAAAAu64; // Alternating bits
            for i in 0..64 {
                if (pattern >> i) & 1 == 1 {
                    acc = acc.wrapping_add(1);
                }
            }
            // Pad to similar iteration count
            for _ in 64..100 {
                std::hint::black_box(0u64);
            }
            std::hint::black_box(acc);
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let mut acc = 0u64;
            for i in 0..100 {
                if i < 100 {
                    acc = acc.wrapping_add(1);
                }
            }
            std::hint::black_box(acc);
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let mut acc = 0u64;
            let pattern = 0xAAAAAAAAAAAAAAAAu64;
            for i in 0..64 {
                if (pattern >> i) & 1 == 1 {
                    acc = acc.wrapping_add(1);
                }
            }
            for _ in 64..100 {
                std::hint::black_box(0u64);
            }
            std::hint::black_box(acc);
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        String::new()
    }
}

/// Modular exponentiation timing (SUBTLE LEAKY)
/// Square-and-multiply with different exponent Hamming weights
pub struct ModularExp;

impl TestCase for ModularExp {
    fn name(&self) -> &str {
        "modular_exp"
    }

    fn expected_leaky(&self) -> bool {
        true
    }

    fn category(&self) -> TestCategory {
        TestCategory::SubtleLeaky
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Exponent with low Hamming weight (few multiplications)
            let base = 3u64;
            let exp = 0b10000001u64; // HW = 2
            let modulus = 1000000007u64;
            std::hint::black_box(mod_exp(base, exp, modulus));
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Exponent with high Hamming weight (many multiplications)
            let base = 3u64;
            let exp = 0b11111111u64; // HW = 8
            let modulus = 1000000007u64;
            std::hint::black_box(mod_exp(base, exp, modulus));
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let base = 3u64;
            let exp = 0b10000001u64;
            let modulus = 1000000007u64;
            std::hint::black_box(mod_exp(base, exp, modulus));
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let base = 3u64;
            let exp = 0b11111111u64;
            let modulus = 1000000007u64;
            std::hint::black_box(mod_exp(base, exp, modulus));
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        r#"
fn mod_exp(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result.wrapping_mul(base) % modulus;
        }
        exp >>= 1;
        base = base.wrapping_mul(base) % modulus;
    }
    result
}
        "#
        .to_string()
    }
}

/// HMAC-style comparison: partial vs full compare (SUBTLE LEAKY)
pub struct HmacCompare;

impl TestCase for HmacCompare {
    fn name(&self) -> &str {
        "hmac_compare"
    }

    fn expected_leaky(&self) -> bool {
        true
    }

    fn category(&self) -> TestCategory {
        TestCategory::SubtleLeaky
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Vulnerable: early exit after first N bytes match
            let expected = [0xABu8; 32];
            let input = [0xABu8; 32]; // Matches fully
            std::hint::black_box(insecure_hmac_verify(&expected, &input));
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Input differs at byte 0, exits immediately
            let expected = [0xABu8; 32];
            let mut input = [0x00u8; 32];
            input[0] = 0xCD; // Differs at first byte
            std::hint::black_box(insecure_hmac_verify(&expected, &input));
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let expected = [0xABu8; 32];
            let input = [0xABu8; 32];
            std::hint::black_box(insecure_hmac_verify(&expected, &input));
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let expected = [0xABu8; 32];
            let mut input = [0x00u8; 32];
            input[0] = 0xCD;
            std::hint::black_box(insecure_hmac_verify(&expected, &input));
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        r#"
fn insecure_hmac_verify(expected: &[u8], input: &[u8]) -> bool {
    if expected.len() != input.len() {
        return false;
    }
    for i in 0..expected.len() {
        if expected[i] != input[i] {
            return false; // Early exit - timing leak!
        }
    }
    true
}
        "#
        .to_string()
    }
}

/// RSA padding oracle simulation (SUBTLE LEAKY)
pub struct RsaPaddingCheck;

impl TestCase for RsaPaddingCheck {
    fn name(&self) -> &str {
        "rsa_padding_check"
    }

    fn expected_leaky(&self) -> bool {
        true
    }

    fn category(&self) -> TestCategory {
        TestCategory::SubtleLeaky
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Valid PKCS#1 v1.5 padding: 0x00 0x02 [random] 0x00 [message]
            let mut padded = [0u8; 128];
            padded[0] = 0x00;
            padded[1] = 0x02;
            for value in padded.iter_mut().take(120).skip(2) {
                *value = 0xFF; // Non-zero padding
            }
            padded[120] = 0x00; // Separator
            std::hint::black_box(check_pkcs1_padding(&padded));
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Invalid padding: wrong first byte
            let mut padded = [0u8; 128];
            padded[0] = 0x01; // Wrong! Should be 0x00
            padded[1] = 0x02;
            for value in padded.iter_mut().take(120).skip(2) {
                *value = 0xFF;
            }
            padded[120] = 0x00;
            std::hint::black_box(check_pkcs1_padding(&padded));
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let mut padded = [0u8; 128];
            padded[0] = 0x00;
            padded[1] = 0x02;
            for i in 2..120 {
                padded[i] = 0xFF;
            }
            padded[120] = 0x00;
            std::hint::black_box(check_pkcs1_padding(&padded));
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let mut padded = [0u8; 128];
            padded[0] = 0x01;
            padded[1] = 0x02;
            for i in 2..120 {
                padded[i] = 0xFF;
            }
            padded[120] = 0x00;
            std::hint::black_box(check_pkcs1_padding(&padded));
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        r#"
fn check_pkcs1_padding(data: &[u8]) -> bool {
    if data.len() < 11 {
        return false;
    }
    // Vulnerable: early exit on each check
    if data[0] != 0x00 {
        return false;
    }
    if data[1] != 0x02 {
        return false;
    }
    // Find separator (first 0x00 after position 2)
    for i in 2..data.len() {
        if data[i] == 0x00 {
            return i >= 10; // At least 8 bytes of padding
        }
    }
    false
}
        "#
        .to_string()
    }
}

// =============================================================================
// NOISY BUT SAFE TEST CASES
// =============================================================================

/// High-variance constant-time operation (NOISY SAFE)
/// Both classes have same timing distribution but high variance
pub struct HighVarianceSafe;

impl TestCase for HighVarianceSafe {
    fn name(&self) -> &str {
        "high_variance_safe"
    }

    fn expected_leaky(&self) -> bool {
        false
    }

    fn category(&self) -> TestCategory {
        TestCategory::NoisySafe
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Add random jitter (same distribution as random class)
            let jitter = (rand::rng().random::<u8>() % 50) as usize;
            let mut acc = 0u64;
            for i in 0..(100 + jitter) {
                acc = acc.wrapping_add(std::hint::black_box(i as u64));
            }
            std::hint::black_box(acc);
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Same random jitter distribution
            let jitter = (rand::rng().random::<u8>() % 50) as usize;
            let mut acc = 0u64;
            for i in 0..(100 + jitter) {
                acc = acc.wrapping_add(std::hint::black_box(i as u64));
            }
            std::hint::black_box(acc);
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            use dudect_bencher::rand::thread_rng;
            let jitter = (thread_rng().gen::<u8>() % 50) as usize;
            let mut acc = 0u64;
            for i in 0..(100 + jitter) {
                acc = acc.wrapping_add(std::hint::black_box(i as u64));
            }
            std::hint::black_box(acc);
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            use dudect_bencher::rand::thread_rng;
            let jitter = (thread_rng().gen::<u8>() % 50) as usize;
            let mut acc = 0u64;
            for i in 0..(100 + jitter) {
                acc = acc.wrapping_add(std::hint::black_box(i as u64));
            }
            std::hint::black_box(acc);
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        String::new()
    }
}

/// Memory allocation noise (NOISY SAFE)
/// Both classes allocate/deallocate with equal overhead
/// Uses explicit memset to avoid calloc optimization differences
pub struct MemoryAllocationNoise;

impl TestCase for MemoryAllocationNoise {
    fn name(&self) -> &str {
        "memory_allocation_noise"
    }

    fn expected_leaky(&self) -> bool {
        false
    }

    fn category(&self) -> TestCategory {
        TestCategory::NoisySafe
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Allocate uninit, then fill with 0x42 (avoiding calloc optimization)
            let mut data: Vec<u8> = Vec::with_capacity(1024);
            data.resize(1024, 0x42);
            let sum: u64 = data.iter().map(|&x| x as u64).sum();
            std::hint::black_box(sum);
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Same allocation pattern, same fill value = identical work
            let mut data: Vec<u8> = Vec::with_capacity(1024);
            data.resize(1024, 0x42);
            let sum: u64 = data.iter().map(|&x| x as u64).sum();
            std::hint::black_box(sum);
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let mut data: Vec<u8> = Vec::with_capacity(1024);
            data.resize(1024, 0x42);
            let sum: u64 = data.iter().map(|&x| x as u64).sum();
            std::hint::black_box(sum);
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let mut data: Vec<u8> = Vec::with_capacity(1024);
            data.resize(1024, 0x42);
            let sum: u64 = data.iter().map(|&x| x as u64).sum();
            std::hint::black_box(sum);
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        String::new()
    }
}

/// Interleaved operations (NOISY SAFE)
/// Mix of fast and slow operations, equally distributed
pub struct InterleavedOps;

impl TestCase for InterleavedOps {
    fn name(&self) -> &str {
        "interleaved_ops"
    }

    fn expected_leaky(&self) -> bool {
        false
    }

    fn category(&self) -> TestCategory {
        TestCategory::NoisySafe
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Same computation as random - only final XOR differs (constant-time)
            let mut acc = 42u64;
            for i in 0..50 {
                // Fast op
                acc = acc.wrapping_add(std::hint::black_box(i));
                // Slow op
                for _ in 0..10 {
                    acc = std::hint::black_box(acc.wrapping_mul(3));
                }
            }
            std::hint::black_box(acc ^ 0xDEADBEEF);
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Same computation as fixed - only final XOR differs (constant-time)
            let mut acc = 42u64;
            for i in 0..50 {
                // Fast op
                acc = acc.wrapping_add(std::hint::black_box(i));
                // Slow op
                for _ in 0..10 {
                    acc = std::hint::black_box(acc.wrapping_mul(3));
                }
            }
            std::hint::black_box(acc ^ 0xCAFEBABE);
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let mut acc = 42u64;
            for i in 0..50 {
                acc = acc.wrapping_add(std::hint::black_box(i));
                for _ in 0..10 {
                    acc = std::hint::black_box(acc.wrapping_mul(3));
                }
            }
            std::hint::black_box(acc ^ 0xDEADBEEF);
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let mut acc = 42u64;
            for i in 0..50 {
                acc = acc.wrapping_add(std::hint::black_box(i));
                for _ in 0..10 {
                    acc = std::hint::black_box(acc.wrapping_mul(3));
                }
            }
            std::hint::black_box(acc ^ 0xCAFEBABE);
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        String::new()
    }
}

// =============================================================================
// EDGE CASE TEST CASES
// =============================================================================

/// Division timing (EDGE CASE)
/// Some CPUs have data-dependent division timing
pub struct DivisionTiming;

impl TestCase for DivisionTiming {
    fn name(&self) -> &str {
        "division_timing"
    }

    fn expected_leaky(&self) -> bool {
        true // Platform-dependent, but often leaky
    }

    fn category(&self) -> TestCategory {
        TestCategory::EdgeCase
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Small divisor (potentially faster on some CPUs)
            let dividend = 1000000u64;
            let divisor = 3u64;
            let mut acc = 0u64;
            for _ in 0..100 {
                acc = acc.wrapping_add(std::hint::black_box(dividend / divisor));
            }
            std::hint::black_box(acc);
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Large divisor (potentially slower)
            let dividend = 1000000u64;
            let divisor = 999983u64; // Large prime
            let mut acc = 0u64;
            for _ in 0..100 {
                acc = acc.wrapping_add(std::hint::black_box(dividend / divisor));
            }
            std::hint::black_box(acc);
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let dividend = 1000000u64;
            let divisor = 3u64;
            let mut acc = 0u64;
            for _ in 0..100 {
                acc = acc.wrapping_add(std::hint::black_box(dividend / divisor));
            }
            std::hint::black_box(acc);
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let dividend = 1000000u64;
            let divisor = 999983u64;
            let mut acc = 0u64;
            for _ in 0..100 {
                acc = acc.wrapping_add(std::hint::black_box(dividend / divisor));
            }
            std::hint::black_box(acc);
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        String::new()
    }
}

/// Population count timing (EDGE CASE)
/// Hardware popcnt is constant-time, but software might not be
pub struct PopulationCount;

impl TestCase for PopulationCount {
    fn name(&self) -> &str {
        "population_count"
    }

    fn expected_leaky(&self) -> bool {
        false // Hardware popcnt should be constant-time
    }

    fn category(&self) -> TestCategory {
        TestCategory::EdgeCase
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Low Hamming weight
            let value = 0x8000000000000001u64; // HW = 2
            let mut acc = 0u32;
            for _ in 0..100 {
                acc = acc.wrapping_add(std::hint::black_box(value.count_ones()));
            }
            std::hint::black_box(acc);
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // High Hamming weight
            let value = 0xFFFFFFFFFFFFFFFFu64; // HW = 64
            let mut acc = 0u32;
            for _ in 0..100 {
                acc = acc.wrapping_add(std::hint::black_box(value.count_ones()));
            }
            std::hint::black_box(acc);
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let value = 0x8000000000000001u64;
            let mut acc = 0u32;
            for _ in 0..100 {
                acc = acc.wrapping_add(std::hint::black_box(value.count_ones()));
            }
            std::hint::black_box(acc);
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let value = 0xFFFFFFFFFFFFFFFFu64;
            let mut acc = 0u32;
            for _ in 0..100 {
                acc = acc.wrapping_add(std::hint::black_box(value.count_ones()));
            }
            std::hint::black_box(acc);
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        String::new()
    }
}

/// Floating point denormals (EDGE CASE)
/// Denormal numbers can cause massive slowdowns on some hardware
pub struct FloatingPointDenormals;

impl TestCase for FloatingPointDenormals {
    fn name(&self) -> &str {
        "fp_denormals"
    }

    fn expected_leaky(&self) -> bool {
        true // Denormals are typically much slower
    }

    fn category(&self) -> TestCategory {
        TestCategory::EdgeCase
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Normal floats
            let mut x = 1.0f64;
            for _ in 0..100 {
                x = std::hint::black_box(x * 0.99);
            }
            std::hint::black_box(x);
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            // Denormal floats (very small numbers)
            let mut x = 1e-308f64; // Near denormal range
            for _ in 0..100 {
                x = std::hint::black_box(x * 0.5);
            }
            std::hint::black_box(x);
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let mut x = 1.0f64;
            for _ in 0..100 {
                x = std::hint::black_box(x * 0.99);
            }
            std::hint::black_box(x);
        "#
        .to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let mut x = 1e-308f64;
            for _ in 0..100 {
                x = std::hint::black_box(x * 0.5);
            }
            std::hint::black_box(x);
        "#
        .to_string()
    }

    fn helper_code(&self) -> String {
        String::new()
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn early_exit_compare(a: &[u8], b: &[u8]) -> bool {
    for i in 0..a.len().min(b.len()) {
        if a[i] != b[i] {
            return false;
        }
    }
    a.len() == b.len()
}

fn branch_on_zero(x: u8) -> u8 {
    if x == 0 {
        std::hint::black_box(0u8);
        for _ in 0..1000 {
            std::hint::black_box(0u8);
        }
        0
    } else {
        x
    }
}

fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    let mut acc = 0u8;
    for i in 0..a.len().min(b.len()) {
        acc |= a[i] ^ b[i];
    }
    acc == 0 && a.len() == b.len()
}

fn xor_bytes(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    for i in 0..32 {
        result[i] = a[i] ^ b[i];
    }
    result
}

fn sbox_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    for (i, value) in table.iter_mut().enumerate() {
        let byte = i as u8;
        *value = byte.wrapping_mul(0x1B) ^ byte.rotate_left(3);
    }
    table
}

fn mod_exp(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result.wrapping_mul(base) % modulus;
        }
        exp >>= 1;
        base = base.wrapping_mul(base) % modulus;
    }
    result
}

fn insecure_hmac_verify(expected: &[u8], input: &[u8]) -> bool {
    if expected.len() != input.len() {
        return false;
    }
    for i in 0..expected.len() {
        if expected[i] != input[i] {
            return false;
        }
    }
    true
}

fn check_pkcs1_padding(data: &[u8]) -> bool {
    if data.len() < 11 {
        return false;
    }
    if data[0] != 0x00 {
        return false;
    }
    if data[1] != 0x02 {
        return false;
    }
    for (i, &value) in data.iter().enumerate().skip(2) {
        if value == 0x00 {
            return i >= 10;
        }
    }
    false
}

fn rand_bytes_32() -> [u8; 32] {
    let mut arr = [0u8; 32];
    rand::rng().fill(&mut arr[..]);
    arr
}

fn rand_bytes_512() -> [u8; 512] {
    let mut arr = [0u8; 512];
    rand::rng().fill(&mut arr[..]);
    arr
}

// =============================================================================
// TEST CASE REGISTRY
// =============================================================================

/// Get all available test cases
pub fn all_test_cases() -> Vec<Box<dyn TestCase>> {
    vec![
        // Clear leaky
        Box::new(EarlyExitCompare),
        Box::new(BranchOnZero),
        // Clear safe
        Box::new(XorCompare),
        Box::new(XorOperation),
        // Subtle leaky
        Box::new(MicroLeak),
        Box::new(VariableIteration),
        Box::new(TableLookup),
        Box::new(CacheLineCrossing),
        Box::new(BranchPredictor),
        Box::new(ModularExp),
        Box::new(HmacCompare),
        Box::new(RsaPaddingCheck),
        // Noisy safe
        Box::new(HighVarianceSafe),
        Box::new(MemoryAllocationNoise),
        Box::new(InterleavedOps),
        // Edge cases
        Box::new(DivisionTiming),
        Box::new(PopulationCount),
        Box::new(FloatingPointDenormals),
    ]
}

/// Get only known-leaky test cases
pub fn leaky_test_cases() -> Vec<Box<dyn TestCase>> {
    all_test_cases()
        .into_iter()
        .filter(|tc| tc.expected_leaky())
        .collect()
}

/// Get only known-safe test cases
pub fn safe_test_cases() -> Vec<Box<dyn TestCase>> {
    all_test_cases()
        .into_iter()
        .filter(|tc| !tc.expected_leaky())
        .collect()
}

/// Get test cases by category
pub fn test_cases_by_category(category: TestCategory) -> Vec<Box<dyn TestCase>> {
    all_test_cases()
        .into_iter()
        .filter(|tc| tc.category() == category)
        .collect()
}

/// Get subtle test cases (harder to detect)
pub fn subtle_test_cases() -> Vec<Box<dyn TestCase>> {
    all_test_cases()
        .into_iter()
        .filter(|tc| {
            matches!(
                tc.category(),
                TestCategory::SubtleLeaky | TestCategory::NoisySafe | TestCategory::EdgeCase
            )
        })
        .collect()
}
