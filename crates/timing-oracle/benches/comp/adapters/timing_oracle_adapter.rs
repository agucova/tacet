//! Adapter for timing-oracle library.

#![allow(dead_code)]

use super::super::test_cases::TestCase;
use super::{DetectionResult, Detector, RawData};
use std::any::Any;
use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;
use timing_oracle::{helpers::InputPair, AttackerModel, Outcome, TimingOracle};

/// Adapter for timing-oracle
pub struct TimingOracleDetector {
    /// Attacker model to use
    attacker_model: AttackerModel,
    /// Current test case operations (set via prepare_test_case)
    current_ops: RefCell<Option<PreparedOps>>,
}

/// Prepared operations from a test case
struct PreparedOps {
    fixed: Arc<dyn Fn() + Send + Sync>,
    random: Arc<dyn Fn() + Send + Sync>,
}

impl TimingOracleDetector {
    pub fn new() -> Self {
        Self {
            // Use SharedHardware for cycle-level sensitivity comparable to dudect
            // This sets threshold to ~0.6ns (~2 cycles at 3GHz)
            attacker_model: AttackerModel::SharedHardware,
            current_ops: RefCell::new(None),
        }
    }

    pub fn with_attacker_model(mut self, model: AttackerModel) -> Self {
        self.attacker_model = model;
        self
    }

    /// Set the current test case before calling detect()
    ///
    /// Must be called before detect().
    pub fn prepare_test_case(&self, test_case: &dyn TestCase) {
        let fixed = test_case.fixed_operation();
        let random = test_case.random_operation();
        *self.current_ops.borrow_mut() = Some(PreparedOps {
            fixed: Arc::from(fixed),
            random: Arc::from(random),
        });
    }
}

impl Default for TimingOracleDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Detector for TimingOracleDetector {
    fn name(&self) -> &str {
        "timing-oracle"
    }

    fn detect(&self, _fixed: &dyn Fn(), _random: &dyn Fn(), samples: usize) -> DetectionResult {
        let start = Instant::now();

        // Get the prepared operations
        let ops = self
            .current_ops
            .borrow()
            .as_ref()
            .map(|o| (Arc::clone(&o.fixed), Arc::clone(&o.random)));

        let (fixed_op, random_op) = ops.expect("Must call prepare_test_case() before detect()");

        let oracle = TimingOracle::for_attacker(self.attacker_model);

        // Use a boolean to indicate which class: false = baseline (fixed), true = sample (random)
        let inputs = InputPair::new(|| false, || true);

        let outcome = oracle.max_samples(samples).test(inputs, move |is_sample| {
            if *is_sample {
                random_op();
            } else {
                fixed_op();
            }
        });

        let duration = start.elapsed();

        match outcome {
            Outcome::Pass {
                leak_probability,
                samples_used,
                ..
            } => DetectionResult {
                detected_leak: false,
                confidence_metric: leak_probability,
                samples_used,
                duration,
                raw_data: Some(RawData::TimingOracle {
                    leak_probability,
                    ci_gate_passed: true,
                }),
            },
            Outcome::Fail {
                leak_probability,
                samples_used,
                ..
            } => DetectionResult {
                detected_leak: true,
                confidence_metric: leak_probability,
                samples_used,
                duration,
                raw_data: Some(RawData::TimingOracle {
                    leak_probability,
                    ci_gate_passed: false,
                }),
            },
            Outcome::Inconclusive {
                leak_probability,
                samples_used,
                ..
            } => DetectionResult {
                detected_leak: leak_probability > 0.5,
                confidence_metric: leak_probability,
                samples_used,
                duration,
                raw_data: Some(RawData::TimingOracle {
                    leak_probability,
                    ci_gate_passed: leak_probability < 0.5,
                }),
            },
            Outcome::Unmeasurable { .. } => DetectionResult {
                detected_leak: false,
                confidence_metric: 0.0,
                samples_used: 0,
                duration,
                raw_data: None,
            },
        }
    }

    fn default_threshold(&self) -> f64 {
        0.5
    }

    fn exceeds_threshold(&self, confidence_metric: f64, threshold: f64) -> bool {
        confidence_metric > threshold
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
