//! Shared recorder for known-leaky crypto tests.
//!
//! Emits stdout tokens that `scripts/measure_fpr.sh`'s parser recognizes, so
//! the existing parser + CSV pipeline can be reused unchanged. The test
//! function itself never panics; `cargo test` always reports success. The
//! CSV captures the oracle's raw verdict, and `scripts/analyze_tpr.py` flips
//! the interpretation at analysis time.
//!
//! | `Outcome` variant    | Emitted token       | CSV record   | Interpretation |
//! |----------------------|---------------------|--------------|----------------|
//! | `Fail`               | `Test passed:`      | PASS         | true positive  |
//! | `Pass`               | `FAILED:`           | FAIL         | false negative |
//! | `Inconclusive`       | `Inconclusive:`     | INCONCLUSIVE | underpowered   |
//! | `Unmeasurable`       | `Skipping:`         | SKIP         | skipped        |
//! | `Research(_)`        | `Skipping:`         | SKIP         | skipped        |
//!
//! The token priority in the parser (`scripts/measure_fpr.sh` lines 86–98) is:
//! `Test passed:` > `FAILED|panicked` > `Inconclusive:` > `Skipping:` > exit code.
//! We never emit `panicked` so the exit code stays 0 and `cargo test` exits
//! cleanly — the failure signal is encoded in stdout tokens only.
//!
//! The helper always prints explicit `P(leak)=XX.X%` and `Samples: N` lines
//! so the parser's regexes capture them regardless of upstream formatting
//! changes.

#![allow(dead_code)] // each test uses a subset of these helpers

use tacet::Outcome;

/// Record a leaky-test outcome in parser-compatible form. Never panics.
///
/// This deliberately avoids `panic!` on `Outcome::Pass` so that sub-floor
/// delays (where the hardware genuinely cannot inject the configured effect)
/// produce a clean CSV row rather than a `cargo test` failure. The correct
/// downstream interpretation — "Pass on a leaky test == false negative" —
/// is applied by `scripts/analyze_tpr.py`.
pub fn record_detection_outcome(outcome: &Outcome, test_name: &str) {
    eprintln!("\n[{test_name}]");
    eprintln!("{}", tacet::output::format_outcome(outcome));

    let leak_prob = outcome.leak_probability().unwrap_or(0.0);
    let samples = outcome.samples_used().unwrap_or(0);

    match outcome {
        Outcome::Fail { .. } => {
            eprintln!(
                "Test passed: Leak detected as expected. P(leak)={:.1}%",
                leak_prob * 100.0
            );
            eprintln!("Samples: {samples}");
        }
        Outcome::Inconclusive { reason, .. } => {
            eprintln!(
                "Inconclusive: {reason:?}. P(leak)={:.1}%",
                leak_prob * 100.0
            );
            eprintln!("Samples: {samples}");
        }
        Outcome::Unmeasurable { recommendation, .. } => {
            eprintln!("Skipping: Unmeasurable. {recommendation}");
            eprintln!("Samples: {samples}");
        }
        Outcome::Research(_) => {
            eprintln!("Skipping: Research mode returned no definitive verdict.");
            eprintln!("Samples: {samples}");
        }
        Outcome::Pass { .. } => {
            // Emit FAILED token for the parser — recorded as FAIL in the CSV,
            // interpreted as a false-negative by analyze_tpr.py. Do NOT panic:
            // this is normal for sub-floor injections and we want clean runs.
            eprintln!(
                "FAILED: No leak detected (missed detection). P(leak)={:.1}%",
                leak_prob * 100.0
            );
            eprintln!("Samples: {samples}");
        }
    }
}
