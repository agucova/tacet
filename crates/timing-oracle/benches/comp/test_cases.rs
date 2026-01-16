//! Shared test cases for comparison benchmarks.
//!
//! Provides standardized test cases that can be used across different
//! timing analysis tools for fair comparison.

use rand::Rng;

/// A test case that can be run by different timing analysis tools.
pub trait TestCase {
    /// Name of the test case
    fn name(&self) -> &str;

    /// Whether this test case is expected to show a timing leak
    fn expected_leaky(&self) -> bool;

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

/// Early-exit comparison (KNOWN LEAKY)
pub struct EarlyExitCompare;

impl TestCase for EarlyExitCompare {
    fn name(&self) -> &str {
        "early_exit_comparison"
    }

    fn expected_leaky(&self) -> bool {
        true
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
        "#.to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let secret = [0u8; 512];
            let input = rand_bytes_512();
            std::hint::black_box(early_exit_compare(&secret, &input));
        "#.to_string()
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
        "#.to_string()
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

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            let x = 0u8;
            std::hint::black_box(branch_on_zero(x));
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        Box::new(|| {
            let x = rand::rng().gen::<u8>() | 1; // Never zero
            std::hint::black_box(branch_on_zero(x));
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let x = 0u8;
            std::hint::black_box(helper_branch_on_zero(x));
        "#.to_string()
    }

    fn random_code(&self) -> String {
        r#"
            use dudect_bencher::rand::thread_rng;
            let x = thread_rng().gen::<u8>() | 1; // Never zero
            std::hint::black_box(helper_branch_on_zero(x));
        "#.to_string()
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
        "#.to_string()
    }
}

/// XOR-based constant-time comparison (KNOWN SAFE)
pub struct XorCompare;

impl TestCase for XorCompare {
    fn name(&self) -> &str {
        "xor_compare"
    }

    fn expected_leaky(&self) -> bool {
        false
    }

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        let secret = [0xABu8; 32];
        let input = [0x00u8; 32];  // Class 0: all zeros (both return false)
        Box::new(move || {
            std::hint::black_box(constant_time_compare(&secret, &input));
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        let secret = [0xABu8; 32];
        Box::new(move || {
            let input = rand_bytes_32();  // Class 1: random (both return false)
            std::hint::black_box(constant_time_compare(&secret, &input));
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let secret = [0xABu8; 32];
            let input = [0x00u8; 32];  // Class 0: all zeros (both return false)
            std::hint::black_box(constant_time_compare(&secret, &input));
        "#.to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let secret = [0xABu8; 32];
            let input = rand_bytes_32();  // Class 1: random (both return false)
            std::hint::black_box(constant_time_compare(&secret, &input));
        "#.to_string()
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
        "#.to_string()
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

    fn fixed_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        let a = [0xABu8; 32];
        let b = [0x00u8; 32];  // Class 0: XOR with all zeros
        Box::new(move || {
            std::hint::black_box(xor_bytes(&a, &b));
        })
    }

    fn random_operation(&self) -> Box<dyn Fn() + Send + Sync> {
        let a = [0xABu8; 32];
        Box::new(move || {
            let b = rand_bytes_32();  // Class 1: XOR with random
            std::hint::black_box(xor_bytes(&a, &b));
        })
    }

    fn fixed_code(&self) -> String {
        r#"
            let a = [0xABu8; 32];
            let b = [0x00u8; 32];  // Class 0: XOR with all zeros
            std::hint::black_box(xor_bytes(&a, &b));
        "#.to_string()
    }

    fn random_code(&self) -> String {
        r#"
            let a = [0xABu8; 32];
            let b = rand_bytes_32();  // Class 1: XOR with random
            std::hint::black_box(xor_bytes(&a, &b));
        "#.to_string()
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
        "#.to_string()
    }
}

// Helper functions (implementations from test files)

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

fn rand_bytes_32() -> [u8; 32] {
    let mut arr = [0u8; 32];
    let mut rng = rand::rng();
    rng.fill(&mut arr[..]);
    arr
}

fn rand_bytes_512() -> [u8; 512] {
    let mut arr = [0u8; 512];
    let mut rng = rand::rng();
    rng.fill(&mut arr[..]);
    arr
}

/// Get all available test cases
pub fn all_test_cases() -> Vec<Box<dyn TestCase>> {
    vec![
        Box::new(EarlyExitCompare),
        Box::new(BranchOnZero),
        Box::new(XorCompare),
        Box::new(XorOperation),
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
