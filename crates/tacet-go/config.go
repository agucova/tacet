package tacet

import (
	"time"
)

// Config holds the configuration for timing analysis.
type Config struct {
	attackerModel        AttackerModel
	customThresholdNs    float64
	maxSamples           int
	timeBudget           time.Duration
	passThreshold        float64
	failThreshold        float64
	seed                 uint64
	calibrationSamples   int
	batchSize            int
	disableAdaptiveBatch bool
}

// Option is a functional option for configuring timing tests.
type Option func(*Config)

// defaultConfig returns the default configuration.
func defaultConfig() *Config {
	return &Config{
		attackerModel:      AdjacentNetwork,
		maxSamples:         100_000,
		timeBudget:         30 * time.Second,
		passThreshold:      0.05,
		failThreshold:      0.95,
		calibrationSamples: 5000,
		batchSize:          1000,
	}
}

// WithAttacker sets the attacker model (threat model).
func WithAttacker(model AttackerModel) Option {
	return func(c *Config) {
		c.attackerModel = model
	}
}

// WithCustomThreshold sets a custom timing threshold in nanoseconds.
// This overrides the attacker model's default threshold.
func WithCustomThreshold(thresholdNs float64) Option {
	return func(c *Config) {
		c.customThresholdNs = thresholdNs
	}
}

// WithMaxSamples sets the maximum number of samples per class.
func WithMaxSamples(n int) Option {
	return func(c *Config) {
		c.maxSamples = n
	}
}

// WithTimeBudget sets the maximum time to spend on the test.
func WithTimeBudget(d time.Duration) Option {
	return func(c *Config) {
		c.timeBudget = d
	}
}

// WithPassThreshold sets the posterior probability threshold for passing.
// Default is 0.05 (pass if P(leak) < 5%).
func WithPassThreshold(p float64) Option {
	return func(c *Config) {
		c.passThreshold = p
	}
}

// WithFailThreshold sets the posterior probability threshold for failing.
// Default is 0.95 (fail if P(leak) > 95%).
func WithFailThreshold(p float64) Option {
	return func(c *Config) {
		c.failThreshold = p
	}
}

// WithSeed sets the random seed for reproducibility.
// Default (0) uses system entropy.
func WithSeed(seed uint64) Option {
	return func(c *Config) {
		c.seed = seed
	}
}

// WithCalibrationSamples sets the number of samples for calibration phase.
// Default is 5000.
func WithCalibrationSamples(n int) Option {
	return func(c *Config) {
		c.calibrationSamples = n
	}
}

// WithBatchSize sets the number of samples per adaptive batch.
// Default is 1000.
func WithBatchSize(n int) Option {
	return func(c *Config) {
		c.batchSize = n
	}
}

// WithoutAdaptiveBatching disables adaptive batching.
// This is useful for debugging or when the operation is known to be slow.
func WithoutAdaptiveBatching() Option {
	return func(c *Config) {
		c.disableAdaptiveBatch = true
	}
}
