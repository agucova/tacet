//! System-level preflight checks.
//!
//! Platform-specific checks to ensure the system is configured
//! optimally for timing measurements.

use serde::{Deserialize, Serialize};

/// Warning from system checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemWarning {
    /// CPU frequency scaling is not set to performance mode.
    CpuGovernorNotPerformance {
        /// Current governor setting.
        current: String,
        /// Recommended governor.
        recommended: String,
    },

    /// Could not read CPU governor (permission or path issue).
    CpuGovernorUnreadable {
        /// Error message.
        reason: String,
    },

    /// Turbo boost is enabled (can cause timing variability).
    TurboBoostEnabled,

    /// Hyperthreading detected (can affect timing measurements).
    HyperthreadingEnabled,

    /// Running in a virtual machine.
    VirtualMachineDetected {
        /// Type of VM if known.
        vm_type: Option<String>,
    },

    /// High system load detected.
    HighSystemLoad {
        /// Current load average.
        load_average: f64,
        /// Threshold exceeded.
        threshold: f64,
    },
}

impl SystemWarning {
    /// Check if this warning indicates a critical issue.
    pub fn is_critical(&self) -> bool {
        // System warnings are generally not critical, just informational
        false
    }

    /// Get a human-readable description of the warning.
    pub fn description(&self) -> String {
        match self {
            SystemWarning::CpuGovernorNotPerformance {
                current,
                recommended,
            } => {
                format!(
                    "CPU frequency governor is '{}', recommend '{}' for stable timing. \
                     Set with: sudo cpufreq-set -g performance",
                    current, recommended
                )
            }
            SystemWarning::CpuGovernorUnreadable { reason } => {
                format!(
                    "Could not check CPU governor: {}. \
                     This may indicate limited permissions or unsupported platform.",
                    reason
                )
            }
            SystemWarning::TurboBoostEnabled => {
                "Turbo boost is enabled. This can cause timing variability. \
                 Consider disabling for more stable measurements."
                    .to_string()
            }
            SystemWarning::HyperthreadingEnabled => {
                "Hyperthreading detected. Consider pinning to physical cores \
                 for more stable timing measurements."
                    .to_string()
            }
            SystemWarning::VirtualMachineDetected { vm_type } => {
                let vm_info = vm_type
                    .as_ref()
                    .map(|t| format!(" ({})", t))
                    .unwrap_or_default();
                format!(
                    "Running in a virtual machine{}. Timing measurements may be \
                     less reliable due to VM overhead and scheduling.",
                    vm_info
                )
            }
            SystemWarning::HighSystemLoad {
                load_average,
                threshold,
            } => {
                format!(
                    "High system load detected: {:.2} (threshold: {:.2}). \
                     This may affect timing measurement stability.",
                    load_average, threshold
                )
            }
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
        // TODO: Check for macOS-specific power settings
        let _ = check_macos_power_settings();
    }

    #[cfg(target_os = "windows")]
    {
        // Windows-specific checks could go here
        // TODO: Check power plan settings
        let _ = check_windows_power_settings();
    }

    warnings
}

/// Check CPU frequency governor on Linux.
#[cfg(target_os = "linux")]
fn check_cpu_governor_linux() -> Option<SystemWarning> {
    let governor_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor";

    match std::fs::read_to_string(governor_path) {
        Ok(governor) => {
            let governor = governor.trim().to_lowercase();
            if governor != "performance" {
                Some(SystemWarning::CpuGovernorNotPerformance {
                    current: governor,
                    recommended: "performance".to_string(),
                })
            } else {
                None
            }
        }
        Err(e) => Some(SystemWarning::CpuGovernorUnreadable {
            reason: e.to_string(),
        }),
    }
}

/// Placeholder for macOS power settings check.
#[cfg(target_os = "macos")]
fn check_macos_power_settings() -> Option<SystemWarning> {
    // TODO: Check macOS power settings using pmset or similar
    // For now, this is a no-op
    None
}

/// Placeholder for Windows power settings check.
#[cfg(target_os = "windows")]
fn check_windows_power_settings() -> Option<SystemWarning> {
    // TODO: Check Windows power plan using powercfg or WMI
    // For now, this is a no-op
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
        assert!(desc.contains("2.50"));
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
                !warning.is_critical(),
                "System warnings should not be critical"
            );
        }
    }
}
