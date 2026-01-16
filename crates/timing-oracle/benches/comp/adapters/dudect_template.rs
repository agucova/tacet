//! Binary template generation and compilation for dudect-bencher.
//!
//! Since dudect-bencher is designed for macro-based standalone binaries,
//! we generate source code, compile it, and run it as a subprocess.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;
use tempfile::TempDir;

/// Error types for compilation and execution
#[derive(Debug)]
pub enum CompileError {
    IoError(std::io::Error),
    CargoFailed(String),
    BinaryNotFound,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::IoError(e) => write!(f, "IO error: {}", e),
            CompileError::CargoFailed(msg) => write!(f, "Cargo build failed: {}", msg),
            CompileError::BinaryNotFound => write!(f, "Compiled binary not found"),
        }
    }
}

impl std::error::Error for CompileError {}

impl From<std::io::Error> for CompileError {
    fn from(e: std::io::Error) -> Self {
        CompileError::IoError(e)
    }
}

#[derive(Debug)]
pub enum RunError {
    IoError(std::io::Error),
    Timeout,
    NonZeroExit(i32),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::IoError(e) => write!(f, "IO error: {}", e),
            RunError::Timeout => write!(f, "Execution timed out"),
            RunError::NonZeroExit(code) => write!(f, "Process exited with code {}", code),
        }
    }
}

impl std::error::Error for RunError {}

impl From<std::io::Error> for RunError {
    fn from(e: std::io::Error) -> Self {
        RunError::IoError(e)
    }
}

/// Generates dudect-bencher source code for a test case
pub fn generate_binary_source(
    test_case_name: &str,
    fixed_code: &str,
    random_code: &str,
    helper_code: &str,
) -> String {
    format!(
        r#"use dudect_bencher::{{ctbench_main, BenchRng, Class, CtRunner}};
use dudect_bencher::rand::Rng;

{}

fn {}(runner: &mut CtRunner, rng: &mut BenchRng) {{
    const BATCH_SIZE: usize = 1000;

    for _ in 0..BATCH_SIZE {{
        let class = if rng.gen::<bool>() {{ Class::Left }} else {{ Class::Right }};

        runner.run_one(class, || {{
            match class {{
                Class::Left => {{ {} }},
                Class::Right => {{ {} }},
            }}
        }});
    }}
}}

ctbench_main!({});
"#,
        helper_code, test_case_name, fixed_code, random_code, test_case_name
    )
}

/// Generate Cargo.toml for a dudect benchmark binary
fn generate_cargo_toml(binary_name: &str) -> String {
    format!(
        r#"[package]
name = "dudect-{}"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "{}"
path = "src/main.rs"

[dependencies]
dudect-bencher = "0.6"
rand = "0.9"
"#,
        binary_name, binary_name
    )
}

/// Represents a compiled dudect binary
pub struct CompiledBinary {
    pub path: PathBuf,
    pub test_case: String,
    _temp_dir: Option<TempDir>, // Keep temp dir alive
}

impl CompiledBinary {
    /// Compile a dudect binary from generated source code
    pub fn compile(
        test_case_name: &str,
        source: &str,
        persist: bool,
    ) -> Result<Self, CompileError> {
        let target_dir = if persist {
            // Use persistent directory in target/
            let dir = PathBuf::from("target/dudect-benchmarks");
            std::fs::create_dir_all(&dir)?;
            dir
        } else {
            // Use temporary directory
            let temp = TempDir::new()?;
            temp.path().to_path_buf()
        };

        let project_dir = target_dir.join(test_case_name);
        let src_dir = project_dir.join("src");

        // Create project structure
        std::fs::create_dir_all(&src_dir)?;

        // Write Cargo.toml
        let cargo_toml = generate_cargo_toml(test_case_name);
        std::fs::write(project_dir.join("Cargo.toml"), cargo_toml)?;

        // Write source code
        std::fs::write(src_dir.join("main.rs"), source)?;

        // Compile with cargo
        let output = Command::new("cargo")
            .arg("build")
            .arg("--release")
            .arg("--manifest-path")
            .arg(project_dir.join("Cargo.toml"))
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CompileError::CargoFailed(stderr.to_string()));
        }

        // Find the compiled binary
        let binary_path = project_dir
            .join("target")
            .join("release")
            .join(test_case_name);

        if !binary_path.exists() {
            return Err(CompileError::BinaryNotFound);
        }

        Ok(CompiledBinary {
            path: binary_path,
            test_case: test_case_name.to_string(),
            _temp_dir: if persist {
                None
            } else {
                Some(TempDir::new()?)
            },
        })
    }

    /// Run the compiled binary and capture output
    pub fn run(&self, timeout_secs: u64) -> Result<String, RunError> {
        let mut child = Command::new(&self.path).stdout(std::process::Stdio::piped()).spawn()?;

        // Wait with timeout
        let timeout = Duration::from_secs(timeout_secs);
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > timeout {
                child.kill()?;
                return Err(RunError::Timeout);
            }

            match child.try_wait()? {
                Some(status) => {
                    if !status.success() {
                        return Err(RunError::NonZeroExit(status.code().unwrap_or(-1)));
                    }

                    // Read stdout
                    let output = child.wait_with_output()?;
                    return Ok(String::from_utf8_lossy(&output.stdout).to_string());
                }
                None => {
                    // Still running, sleep briefly
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }
    }
}

/// Cache for compiled binaries
pub struct BinaryCache {
    cache: HashMap<String, CompiledBinary>,
}

impl BinaryCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Get or compile a binary for a test case
    pub fn get_or_compile(
        &mut self,
        test_case_name: &str,
        source: &str,
    ) -> Result<&CompiledBinary, CompileError> {
        if !self.cache.contains_key(test_case_name) {
            let binary = CompiledBinary::compile(test_case_name, source, true)?;
            self.cache.insert(test_case_name.to_string(), binary);
        }

        Ok(self.cache.get(test_case_name).unwrap())
    }

    /// Compile all binaries from a list of (name, source) pairs
    pub fn compile_all(&mut self, sources: Vec<(String, String)>) -> Vec<CompileError> {
        let mut errors = Vec::new();

        for (name, source) in sources {
            eprintln!("  Compiling dudect binary: {}", name);
            match CompiledBinary::compile(&name, &source, true) {
                Ok(binary) => {
                    self.cache.insert(name.clone(), binary);
                    eprintln!("    ✓ {}", name);
                }
                Err(e) => {
                    eprintln!("    ✗ {}: {}", name, e);
                    errors.push(e);
                }
            }
        }

        errors
    }

    /// Get a compiled binary by name
    pub fn get(&self, test_case_name: &str) -> Option<&CompiledBinary> {
        self.cache.get(test_case_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_source() {
        let source = generate_binary_source(
            "test_bench",
            "let x = 0u8;",
            "let x = 1u8;",
            "// helper code",
        );

        assert!(source.contains("fn test_bench"));
        assert!(source.contains("ctbench_main!"));
        assert!(source.contains("let x = 0u8;"));
        assert!(source.contains("let x = 1u8;"));
    }

    #[test]
    fn test_generate_cargo_toml() {
        let toml = generate_cargo_toml("my_bench");
        assert!(toml.contains("name = \"dudect-my_bench\""));
        assert!(toml.contains("dudect-bencher = \"0.6\""));
    }
}
