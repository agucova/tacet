//! System-level preflight checks.
//!
//! Platform-specific checks to ensure the system is configured
//! optimally for timing measurements.
//!
//! **Severity**: Informational
//!
//! System configuration issues add variance to measurements but don't
//! invalidate the statistical analysis. The Bayesian model's assumptions
//! are still valid; you just need more samples to reach the same confidence.

use serde::{Deserialize, Serialize};

use timing_oracle_core::result::{PreflightCategory, PreflightSeverity, PreflightWarningInfo};

/// Warning from system checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemWarning {
    /// CPU frequency scaling is not set to performance mode.
    ///
    /// **Severity**: Informational
    CpuGovernorNotPerformance {
        /// Current governor setting.
        current: String,
        /// Recommended governor.
        recommended: String,
    },

    /// Could not read CPU governor (permission or path issue).
    ///
    /// **Severity**: Informational
    CpuGovernorUnreadable {
        /// Error message.
        reason: String,
    },

    /// Turbo boost is enabled (can cause timing variability).
    ///
    /// **Severity**: Informational
    TurboBoostEnabled,

    /// Hyperthreading detected (can affect timing measurements).
    ///
    /// **Severity**: Informational
    HyperthreadingEnabled,

    /// Running in a virtual machine.
    ///
    /// **Severity**: Informational
    VirtualMachineDetected {
        /// Type of VM if known.
        vm_type: Option<String>,
    },

    /// High system load detected.
    ///
    /// **Severity**: Informational
    HighSystemLoad {
        /// Current load average.
        load_average: f64,
        /// Threshold exceeded.
        threshold: f64,
    },
}

impl SystemWarning {
    /// Check if this warning undermines result confidence.
    ///
    /// System warnings are always informational - they add variance but
    /// don't invalidate results.
    pub fn is_result_undermining(&self) -> bool {
        false
    }

    /// Check if this warning indicates a critical issue.
    ///
    /// Deprecated: Use `is_result_undermining()` instead.
    #[deprecated(note = "Use is_result_undermining() instead")]
    pub fn is_critical(&self) -> bool {
        false
    }

    /// Get the severity of this warning.
    pub fn severity(&self) -> PreflightSeverity {
        // All system warnings are informational
        PreflightSeverity::Informational
    }

    /// Get a human-readable description of the warning.
    pub fn description(&self) -> String {
        match self {
            SystemWarning::CpuGovernorNotPerformance {
                current,
                recommended,
            } => {
                format!(
                    "CPU governor is '{}', recommend '{}'.",
                    current, recommended
                )
            }
            SystemWarning::CpuGovernorUnreadable { reason } => {
                format!("Could not check CPU governor: {}.", reason)
            }
            SystemWarning::TurboBoostEnabled => {
                "Turbo boost enabled - can cause timing variability.".to_string()
            }
            SystemWarning::HyperthreadingEnabled => {
                "Hyperthreading detected - consider pinning to physical cores.".to_string()
            }
            SystemWarning::VirtualMachineDetected { vm_type } => {
                let vm_info = vm_type
                    .as_ref()
                    .map(|t| format!(" ({})", t))
                    .unwrap_or_default();
                format!(
                    "Running in a virtual machine{}. Timing measurements may be less reliable.",
                    vm_info
                )
            }
            SystemWarning::HighSystemLoad {
                load_average,
                threshold,
            } => {
                format!(
                    "High system load: {:.1} (threshold: {:.1}).",
                    load_average, threshold
                )
            }
        }
    }

    /// Get guidance for addressing this warning.
    pub fn guidance(&self) -> Option<String> {
        match self {
            SystemWarning::CpuGovernorNotPerformance { .. } => {
                Some("Set with: sudo cpufreq-set -g performance".to_string())
            }
            SystemWarning::CpuGovernorUnreadable { .. } => None,
            SystemWarning::TurboBoostEnabled => {
                Some("Consider disabling for more stable measurements.".to_string())
            }
            SystemWarning::HyperthreadingEnabled => {
                Some("Pin to physical cores for more stable timing.".to_string())
            }
            SystemWarning::VirtualMachineDetected { .. } => {
                Some("Consider running on bare metal for more reliable measurements.".to_string())
            }
            SystemWarning::HighSystemLoad { .. } => {
                Some("Reduce background processes for more stable measurements.".to_string())
            }
        }
    }

    /// Convert to a PreflightWarningInfo.
    pub fn to_warning_info(&self) -> PreflightWarningInfo {
        match self.guidance() {
            Some(guidance) => PreflightWarningInfo::with_guidance(
                PreflightCategory::System,
                self.severity(),
                self.description(),
                guidance,
            ),
            None => PreflightWarningInfo::new(
                PreflightCategory::System,
                self.severity(),
                self.description(),
            ),
        }
    }
}

/// Perform all system checks.
///
/// Returns a vector of warnings for any issues detected.
/// On unsupported platforms, returns an empty vector.
pub fn system_check() -> Vec<SystemWarning> {
    #[allow(unused_mut)]
    let mut warnings = Vec::new();

    // Run platform-specific checks
    #[cfg(target_os = "linux")]
    {
        if let Some(warning) = check_cpu_governor_linux() {
            warnings.push(warning);
        }
        if let Some(warning) = check_turbo_boost_linux() {
            warnings.push(warning);
        }
        if let Some(warning) = check_hyperthreading_linux() {
            warnings.push(warning);
        }
        if let Some(warning) = check_vm_detection_linux() {
            warnings.push(warning);
        }
        if let Some(warning) = check_load_linux() {
            warnings.push(warning);
        }
    }

    #[cfg(target_os = "macos")]
    {
        // macOS-specific checks could go here
        let _ = check_macos_power_settings();
    }

    #[cfg(target_os = "windows")]
    {
        // Windows-specific checks could go here
        let _ = check_windows_power_settings();
    }

    warnings
}

/// Check CPU frequency governor on Linux.
///
/// Tries multiple sysfs paths to handle different kernel configurations:
/// - Standard per-CPU path: `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`
/// - Policy-based path: `/sys/devices/system/cpu/cpufreq/policy0/scaling_governor`
///
/// On systems without cpufreq (common on ARM64 cloud instances where firmware
/// handles frequency scaling), silently returns None.
#[cfg(target_os = "linux")]
fn check_cpu_governor_linux() -> Option<SystemWarning> {
    // Try multiple possible paths for the governor file
    // Different kernel versions and configurations use different paths
    let governor_paths = [
        // Standard per-CPU path (most common)
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
        // Policy-based path (some newer kernels, especially with intel_pstate)
        "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor",
    ];

    // Try each path in order
    for path in &governor_paths {
        match std::fs::read_to_string(path) {
            Ok(governor) => {
                let governor = governor.trim().to_lowercase();
                if governor != "performance" {
                    return Some(SystemWarning::CpuGovernorNotPerformance {
                        current: governor,
                        recommended: "performance".to_string(),
                    });
                } else {
                    return None; // Governor is "performance", all good
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Path doesn't exist, try next one
                continue;
            }
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                // Permission denied - worth warning about
                return Some(SystemWarning::CpuGovernorUnreadable {
                    reason: format!("Permission denied reading {}", path),
                });
            }
            Err(_) => {
                // Other error, try next path
                continue;
            }
        }
    }

    // No paths worked - cpufreq likely not available on this system
    // This is common on:
    // - ARM64 cloud instances (AWS Graviton, Ampere, etc.) where firmware handles scaling
    // - Virtual machines without cpufreq passthrough
    // - Embedded systems without cpufreq support
    // Silently skip - nothing actionable for the user
    None
}

/// Placeholder for macOS power settings check.
#[cfg(target_os = "macos")]
fn check_macos_power_settings() -> Option<SystemWarning> {
    // TODO: Check macOS power settings using pmset or similar
    None
}

/// Placeholder for Windows power settings check.
#[cfg(target_os = "windows")]
fn check_windows_power_settings() -> Option<SystemWarning> {
    // TODO: Check Windows power plan using powercfg or WMI
    None
}

#[cfg(target_os = "linux")]
fn check_turbo_boost_linux() -> Option<SystemWarning> {
    let intel_path = "/sys/devices/system/cpu/intel_pstate/no_turbo";
    if let Ok(value) = std::fs::read_to_string(intel_path) {
        if value.trim() == "0" {
            return Some(SystemWarning::TurboBoostEnabled);
        }
        return None;
    }

    let generic_path = "/sys/devices/system/cpu/cpufreq/boost";
    if let Ok(value) = std::fs::read_to_string(generic_path) {
        if value.trim() == "1" {
            return Some(SystemWarning::TurboBoostEnabled);
        }
    }

    None
}

#[cfg(target_os = "linux")]
fn check_hyperthreading_linux() -> Option<SystemWarning> {
    let path = "/sys/devices/system/cpu/smt/active";
    if let Ok(value) = std::fs::read_to_string(path) {
        if value.trim() == "1" {
            return Some(SystemWarning::HyperthreadingEnabled);
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn check_vm_detection_linux() -> Option<SystemWarning> {
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    if cpuinfo.to_lowercase().contains("hypervisor") {
        return Some(SystemWarning::VirtualMachineDetected { vm_type: None });
    }
    None
}

#[cfg(target_os = "linux")]
fn check_load_linux() -> Option<SystemWarning> {
    let loadavg = std::fs::read_to_string("/proc/loadavg").ok()?;
    let load = loadavg
        .split_whitespace()
        .next()
        .and_then(|val| val.parse::<f64>().ok())?;

    let threshold = 1.0;
    if load > threshold {
        Some(SystemWarning::HighSystemLoad {
            load_average: load,
            threshold,
        })
    } else {
        None
    }
}

#[allow(dead_code)]
fn check_system_load() -> Option<SystemWarning> {
    const LOAD_THRESHOLD: f64 = 1.0;

    // TODO: Read load average
    // Linux: /proc/loadavg
    // macOS: getloadavg() or sysctl
    // Windows: Performance counters

    let _ = LOAD_THRESHOLD;
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_check_runs() {
        // Just verify it doesn't panic
        let _warnings = system_check();
    }

    #[test]
    fn test_warning_descriptions() {
        let warning = SystemWarning::CpuGovernorNotPerformance {
            current: "powersave".to_string(),
            recommended: "performance".to_string(),
        };
        let desc = warning.description();
        assert!(desc.contains("powersave"));
        assert!(desc.contains("performance"));

        let warning = SystemWarning::TurboBoostEnabled;
        let desc = warning.description();
        assert!(desc.contains("Turbo boost"));

        let warning = SystemWarning::VirtualMachineDetected {
            vm_type: Some("QEMU".to_string()),
        };
        let desc = warning.description();
        assert!(desc.contains("virtual machine"));
        assert!(desc.contains("QEMU"));

        let warning = SystemWarning::HighSystemLoad {
            load_average: 2.5,
            threshold: 1.0,
        };
        let desc = warning.description();
        assert!(desc.contains("2.5"));
    }

    #[test]
    fn test_warning_is_not_critical() {
        let warnings = vec![
            SystemWarning::CpuGovernorNotPerformance {
                current: "powersave".to_string(),
                recommended: "performance".to_string(),
            },
            SystemWarning::TurboBoostEnabled,
            SystemWarning::HyperthreadingEnabled,
            SystemWarning::VirtualMachineDetected { vm_type: None },
            SystemWarning::HighSystemLoad {
                load_average: 2.0,
                threshold: 1.0,
            },
        ];

        for warning in warnings {
            assert!(
                !warning.is_result_undermining(),
                "System warnings should not undermine results"
            );
            assert_eq!(warning.severity(), PreflightSeverity::Informational);
        }
    }

    #[test]
    fn test_severity() {
        let governor = SystemWarning::CpuGovernorNotPerformance {
            current: "powersave".to_string(),
            recommended: "performance".to_string(),
        };
        assert_eq!(governor.severity(), PreflightSeverity::Informational);
        assert!(!governor.is_result_undermining());
    }
}
