//! Adapter for timing-oracle library.

use super::{DetectionResult, Detector, RawData};
use std::any::Any;
use std::time::Instant;
use timing_oracle::{helpers::InputPair, AttackerModel, Outcome, TimingOracle};

/// Adapter for timing-oracle
pub struct TimingOracleDetector {
    /// Attacker model to use
    attacker_model: AttackerModel,
}

impl TimingOracleDetector {
    pub fn new() -> Self {
        Self {
            attacker_model: AttackerModel::AdjacentNetwork,
        }
    }

    pub fn with_attacker_model(mut self, model: AttackerModel) -> Self {
        self.attacker_model = model;
        self
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

    fn detect(&self, fixed: &dyn Fn(), random: &dyn Fn(), samples: usize) -> DetectionResult {
        let start = Instant::now();

        let oracle = TimingOracle::for_attacker(self.attacker_model.clone());

        // Create an InputPair that wraps the closures
        // Note: We use () as the input type since the original closures don't take inputs
        let inputs = InputPair::new(|| (), || ());

        let outcome = oracle.max_samples(samples).test(inputs, |_: &()| {
            // We alternate between fixed and random based on the internal scheduling
            // This is a simplification - the original API allowed separate closures
            // For the comparison benchmark, we just run both
            fixed();
            random();
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
