//! Cross-validation of RTLF Native vs R implementation
use timing_oracle_bench::adapters::{RtlfAdapter, RtlfNativeAdapter, ToolAdapter};
use timing_oracle_bench::BlockedData;

fn main() {
    println!("=== RTLF Cross-Validation ===\n");
    
    let native = RtlfNativeAdapter::default();
    let r_adapter = RtlfAdapter::default();
    
    // Test 1: Null case (identical distributions) - should NOT detect
    println!("Test 1: Null case (same distribution, n=1000)");
    let mut rng_state = 42u64;
    let mut rand = || -> u64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) % 1000 + 500
    };
    
    let null_data = BlockedData {
        baseline: (0..1000).map(|_| rand()).collect(),
        test: (0..1000).map(|_| rand()).collect(),
    };
    
    let native_result = native.analyze_blocked(&null_data);
    println!("  Native:   detected={:<5} time={}ms  {}", 
        native_result.detected_leak, native_result.decision_time_ms, native_result.status);
    
    let r_result = r_adapter.analyze_blocked(&null_data);
    println!("  R:        detected={:<5} time={}ms  {}", 
        r_result.detected_leak, r_result.decision_time_ms, r_result.status);
    
    let agree = native_result.detected_leak == r_result.detected_leak;
    println!("  Agreement: {} | Expected: No detection", if agree { "YES" } else { "NO" });
    println!();
    
    // Test 2: Clear difference (shifted distribution) - should detect
    println!("Test 2: Alternative case (shifted by 200, n=1000)");
    let alt_data = BlockedData {
        baseline: (0..1000).map(|_| rand()).collect(),
        test: (0..1000).map(|_| rand() + 200).collect(),
    };
    
    let native_result = native.analyze_blocked(&alt_data);
    println!("  Native:   detected={:<5} time={}ms  {}", 
        native_result.detected_leak, native_result.decision_time_ms, native_result.status);
    
    let r_result = r_adapter.analyze_blocked(&alt_data);
    println!("  R:        detected={:<5} time={}ms  {}", 
        r_result.detected_leak, r_result.decision_time_ms, r_result.status);
    
    let agree = native_result.detected_leak == r_result.detected_leak;
    println!("  Agreement: {} | Expected: Detection", if agree { "YES" } else { "NO" });
    println!();
    
    // Test 3: Multiple null trials to check FPR
    println!("Test 3: FPR check (20 null trials, should be ~9% detection rate)");
    let mut native_detections = 0;
    let mut r_detections = 0;

    for trial in 0..20 {
        let mut trial_rng = 1000 + trial as u64;
        let mut trial_rand = || -> u64 {
            trial_rng = trial_rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (trial_rng >> 33) % 1000 + 500
        };
        let data = BlockedData {
            baseline: (0..500).map(|_| trial_rand()).collect(),
            test: (0..500).map(|_| trial_rand()).collect(),
        };
        
        if native.analyze_blocked(&data).detected_leak {
            native_detections += 1;
        }
        if r_adapter.analyze_blocked(&data).detected_leak {
            r_detections += 1;
        }
    }
    
    println!("  Native FPR: {}/20 = {:.0}%", native_detections, native_detections as f64 / 20.0 * 100.0);
    println!("  R FPR:      {}/20 = {:.0}%", r_detections, r_detections as f64 / 20.0 * 100.0);
    println!("  Expected:   ~9% (alpha=0.09)");
}
