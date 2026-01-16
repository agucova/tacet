//! Parser for dudect-bencher stdout output.
//!
//! Extracts statistics from dudect's terminal output format.

use regex::Regex;

/// Parsed output from a dudect-bencher run
#[derive(Debug, Clone)]
pub struct DudectOutput {
    /// Number of samples processed (per class)
    pub n_samples: usize,
    /// Maximum t-statistic observed
    pub max_t: f64,
    /// Maximum tau (normalized effect size)
    pub max_tau: f64,
}

/// Errors that can occur during parsing
#[derive(Debug)]
pub enum ParseError {
    NoMatchFound,
    InvalidFormat(String),
    ParseFloat(std::num::ParseFloatError),
    ParseInt(std::num::ParseIntError),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::NoMatchFound => write!(f, "No dudect output pattern found in stdout"),
            ParseError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            ParseError::ParseFloat(e) => write!(f, "Failed to parse float: {}", e),
            ParseError::ParseInt(e) => write!(f, "Failed to parse int: {}", e),
        }
    }
}

impl std::error::Error for ParseError {}

impl From<std::num::ParseFloatError> for ParseError {
    fn from(e: std::num::ParseFloatError) -> Self {
        ParseError::ParseFloat(e)
    }
}

impl From<std::num::ParseIntError> for ParseError {
    fn from(e: std::num::ParseIntError) -> Self {
        ParseError::ParseInt(e)
    }
}

/// Parse dudect-bencher stdout to extract statistics
///
/// Expected format (from dudect-bencher):
/// ```text
/// n == +0.046M, max t = +61.61472, max tau = +0.28863, (5/tau)^2 = 300
/// ```
///
/// Or with different number formatting:
/// ```text
/// n == +46000, max t = -2.35, max tau = +0.01, (5/tau)^2 = 50000
/// ```
pub fn parse_dudect_output(stdout: &str) -> Result<DudectOutput, ParseError> {
    // Pattern to match dudect output line
    // Supports formats like:
    // - n == +0.046M (millions)
    // - n == +46000 (raw number)
    // - max t = +61.61472 or max t = -2.35
    let re = Regex::new(
        r"n\s*==\s*([+-]?[\d.]+)([KM]?),\s*max t\s*=\s*([+-]?[\d.]+),\s*max tau\s*=\s*([+-]?[\d.]+)"
    ).unwrap();

    // Find the last matching line (dudect outputs multiple progress lines)
    let mut last_match = None;

    for line in stdout.lines() {
        if let Some(captures) = re.captures(line) {
            last_match = Some(captures);
        }
    }

    let captures = last_match.ok_or(ParseError::NoMatchFound)?;

    // Parse n (number of samples)
    let n_str = captures.get(1).unwrap().as_str();
    let n_multiplier = captures.get(2).unwrap().as_str();

    let n_base: f64 = n_str.parse()?;
    let n_samples = match n_multiplier {
        "K" => (n_base * 1_000.0) as usize,
        "M" => (n_base * 1_000_000.0) as usize,
        "" => n_base as usize,
        _ => {
            return Err(ParseError::InvalidFormat(format!(
                "Unknown multiplier: {}",
                n_multiplier
            )))
        }
    };

    // Parse max t
    let max_t: f64 = captures.get(3).unwrap().as_str().parse()?;

    // Parse max tau
    let max_tau: f64 = captures.get(4).unwrap().as_str().parse()?;

    Ok(DudectOutput {
        n_samples,
        max_t,
        max_tau,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_millions_format() {
        let stdout = r#"
meas: 26.68 M, var: 14.85732685, max t: +0.47924, max tau: +0.00029, (5/tau)^2: 29890844
meas: 26.77 M, var: 14.85688837, max t: +0.67175, max tau: +0.00041, (5/tau)^2: 14873906
n == +0.046M, max t = +61.61472, max tau = +0.28863, (5/tau)^2 = 300
"#;

        let result = parse_dudect_output(stdout).unwrap();
        assert_eq!(result.n_samples, 46_000);
        assert!((result.max_t - 61.61472).abs() < 0.001);
        assert!((result.max_tau - 0.28863).abs() < 0.00001);
    }

    #[test]
    fn test_parse_thousands_format() {
        let stdout = "n == +25K, max t = -2.35, max tau = +0.01, (5/tau)^2 = 50000";

        let result = parse_dudect_output(stdout).unwrap();
        assert_eq!(result.n_samples, 25_000);
        assert!((result.max_t + 2.35).abs() < 0.001);
        assert!((result.max_tau - 0.01).abs() < 0.0001);
    }

    #[test]
    fn test_parse_raw_number() {
        let stdout = "n == +1500, max t = +3.14, max tau = +0.05, (5/tau)^2 = 10000";

        let result = parse_dudect_output(stdout).unwrap();
        assert_eq!(result.n_samples, 1_500);
        assert!((result.max_t - 3.14).abs() < 0.001);
        assert!((result.max_tau - 0.05).abs() < 0.0001);
    }

    #[test]
    fn test_parse_negative_t() {
        let stdout = "n == +1.5M, max t = -15.23, max tau = +0.12, (5/tau)^2 = 1735";

        let result = parse_dudect_output(stdout).unwrap();
        assert_eq!(result.n_samples, 1_500_000);
        assert!((result.max_t + 15.23).abs() < 0.001);
    }

    #[test]
    fn test_multiple_lines() {
        let stdout = r#"
Starting benchmark...
n == +100, max t = +1.0, max tau = +0.001, (5/tau)^2 = 25000000
n == +200, max t = +2.0, max tau = +0.002, (5/tau)^2 = 6250000
n == +1000, max t = +5.5, max tau = +0.01, (5/tau)^2 = 250000
"#;

        let result = parse_dudect_output(stdout).unwrap();
        // Should take the last matching line
        assert_eq!(result.n_samples, 1_000);
        assert!((result.max_t - 5.5).abs() < 0.001);
    }

    #[test]
    fn test_no_match() {
        let stdout = "Some random output with no dudect stats";

        let result = parse_dudect_output(stdout);
        assert!(matches!(result, Err(ParseError::NoMatchFound)));
    }
}
