use tacet::measurement::TimerSpec;

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
use tacet::measurement::kperf::PmuTimer;

#[test]
fn test_pmu_initialization() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "kperf"))]
    {
        match PmuTimer::new() {
            Ok(_) => eprintln!("✓ PMU timer initialized successfully"),
            Err(e) => eprintln!("✗ Failed to initialize PMU timer: {:?}", e),
        }
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "kperf")))]
    {
        eprintln!("PMU timer not available on this platform");
    }

    eprintln!(
        "\nTimerSpec::cycle_accurate_available() = {}",
        TimerSpec::cycle_accurate_available()
    );
}
