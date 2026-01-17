//! Generator cost measurement utility.
//!
//! This module provides the platform-specific `measure_generator_cost` function
//! that requires `std::time::Instant`.
//!
//! The generator cost check logic itself is in `timing-oracle-core::preflight::generator`.

/// Measure generator cost by running the generator multiple times.
///
/// # Arguments
///
/// * `generator` - Closure that generates an input
/// * `iterations` - Number of iterations to average over
///
/// # Returns
///
/// Average time per generation in nanoseconds.
pub fn measure_generator_cost<F, T>(mut generator: F, iterations: usize) -> f64
where
    F: FnMut() -> T,
{
    if iterations == 0 {
        return 0.0;
    }

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(generator());
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / iterations as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measure_generator_cost() {
        // Measure a simple generator
        let cost = measure_generator_cost(|| 42u32, 1000);
        // Should be very small (< 1us)
        assert!(cost < 1000.0, "Simple generator should be fast: {}ns", cost);
    }

    #[test]
    fn test_zero_iterations() {
        let cost = measure_generator_cost(|| 42u32, 0);
        assert!(cost == 0.0, "Zero iterations should return 0");
    }
}
