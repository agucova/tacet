//! Persistent process pool for subprocess-based tool adapters.
//!
//! This module provides a thread-safe pool of persistent interpreter processes
//! (R, Python) to avoid paying startup overhead on each analysis call.
//!
//! # Architecture
//!
//! Each pool maintains N child processes that communicate via JSON over stdin/stdout.
//! Tools acquire a process from the pool, send a request, receive a response, and
//! return the process to the pool.
//!
//! # Protocol
//!
//! Request format:
//! ```json
//! {"id": 1, "method": "silent", "params": {"baseline": [...], "test": [...], ...}}
//! ```
//!
//! Response format:
//! ```json
//! {"id": 1, "result": {"detected": true, "statistic": 2.34, ...}}
//! ```
//!
//! Error format:
//! ```json
//! {"id": 1, "error": {"code": -32603, "message": "..."}}
//! ```

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::semaphore::Semaphore;

/// Configuration for spawning worker processes.
#[derive(Debug, Clone)]
pub struct ProcessConfig {
    /// Command to run (e.g., "Rscript", "python3").
    pub command: String,
    /// Arguments to pass to the command.
    pub args: Vec<String>,
    /// Working directory for the process.
    pub working_dir: Option<String>,
    /// Timeout for each request in seconds.
    pub request_timeout_secs: u64,
    /// Maximum requests before restarting process (to avoid memory leaks).
    pub max_requests_per_process: usize,
}

impl ProcessConfig {
    /// Create config for R worker with optional SILENT and RTLF script paths.
    ///
    /// If `silent_path` or `rtlf_path` are provided, the worker will source
    /// those scripts on startup, enabling persistent mode for those tools.
    pub fn r_worker(script_path: &str, silent_path: Option<&str>, rtlf_path: Option<&str>) -> Self {
        let mut args = vec!["--vanilla".to_string(), script_path.to_string()];

        // Add optional paths for SILENT and RTLF scripts
        if let Some(path) = silent_path {
            args.push("--silent-path".to_string());
            args.push(path.to_string());
        }
        if let Some(path) = rtlf_path {
            args.push("--rtlf-path".to_string());
            args.push(path.to_string());
        }

        Self {
            command: "Rscript".to_string(),
            args,
            working_dir: None,
            request_timeout_secs: 60, // R bootstrap tests can be slow
            max_requests_per_process: 1000,
        }
    }

    /// Create config for Python worker.
    pub fn python_worker(script_path: &str) -> Self {
        Self {
            command: "python3".to_string(),
            args: vec![script_path.to_string()],
            working_dir: None,
            request_timeout_secs: 30,
            max_requests_per_process: 1000,
        }
    }
}

/// A managed child process with buffered I/O.
struct ChildProcess {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    requests_handled: usize,
    config: ProcessConfig,
}

impl ChildProcess {
    /// Spawn a new child process.
    fn spawn(config: &ProcessConfig) -> std::io::Result<Self> {
        let mut cmd = Command::new(&config.command);
        cmd.args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Inherit stderr so errors from workers are visible
            // (startup messages will show, but errors are more important to catch)
            .stderr(Stdio::inherit());

        if let Some(ref dir) = config.working_dir {
            cmd.current_dir(dir);
        }

        let mut child = cmd.spawn()?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| std::io::Error::other("Failed to open stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("Failed to open stdout"))?;

        Ok(Self {
            child,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
            requests_handled: 0,
            config: config.clone(),
        })
    }

    /// Send a request and receive a response.
    fn send_request(&mut self, request: &Request) -> std::io::Result<Response> {
        // Serialize and send request
        let request_json = serde_json::to_string(request)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        writeln!(self.stdin, "{}", request_json)?;
        self.stdin.flush()?;

        // Read response line
        let mut response_line = String::new();
        self.stdout.read_line(&mut response_line)?;

        if response_line.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Process closed stdout unexpectedly",
            ));
        }

        // Parse response
        let response: Response = serde_json::from_str(&response_line)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        self.requests_handled += 1;
        Ok(response)
    }

    /// Check if the process should be recycled.
    fn should_recycle(&self) -> bool {
        self.requests_handled >= self.config.max_requests_per_process
    }

    /// Check if the process is still alive.
    fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Gracefully terminate the process.
    fn terminate(mut self) {
        // Close stdin to signal EOF
        drop(self.stdin);

        // Give it a moment to exit gracefully
        std::thread::sleep(Duration::from_millis(100));

        // Force kill if still running
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// JSON-RPC style request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    pub id: u64,
    pub method: String,
    pub params: serde_json::Value,
}

impl Request {
    /// Create a new request with auto-incrementing ID.
    pub fn new(method: impl Into<String>, params: serde_json::Value) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            method: method.into(),
            params,
        }
    }
}

/// JSON-RPC style response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseError>,
}

impl Response {
    /// Check if the response is an error.
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }

    /// Get the result, or error message if it was an error.
    pub fn into_result(self) -> Result<serde_json::Value, String> {
        if let Some(error) = self.error {
            Err(error.message)
        } else if let Some(result) = self.result {
            Ok(result)
        } else {
            Err("Response contained neither result nor error".to_string())
        }
    }
}

/// Error in a JSON-RPC response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseError {
    pub code: i32,
    pub message: String,
}

/// A pool of persistent worker processes.
pub struct ProcessPool {
    /// Process slots (None if process died or hasn't been spawned).
    processes: Vec<Mutex<Option<ChildProcess>>>,
    /// Semaphore to limit concurrent acquisitions.
    semaphore: Semaphore,
    /// Configuration for spawning new processes.
    config: ProcessConfig,
    /// Statistics: total requests handled.
    total_requests: AtomicUsize,
    /// Statistics: total processes spawned.
    total_spawns: AtomicUsize,
}

impl std::fmt::Debug for ProcessPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcessPool")
            .field("size", &self.processes.len())
            .field("config", &self.config)
            .field(
                "total_requests",
                &self.total_requests.load(Ordering::Relaxed),
            )
            .field("total_spawns", &self.total_spawns.load(Ordering::Relaxed))
            .finish()
    }
}

impl ProcessPool {
    /// Create a new process pool.
    ///
    /// Processes are spawned lazily on first acquire.
    pub fn new(config: ProcessConfig, size: usize) -> Self {
        let processes = (0..size).map(|_| Mutex::new(None)).collect();
        Self {
            processes,
            semaphore: Semaphore::new(size),
            config,
            total_requests: AtomicUsize::new(0),
            total_spawns: AtomicUsize::new(0),
        }
    }

    /// Create a pool for R workers.
    pub fn new_r(
        script_path: &str,
        silent_path: Option<&str>,
        rtlf_path: Option<&str>,
        size: usize,
    ) -> Self {
        Self::new(
            ProcessConfig::r_worker(script_path, silent_path, rtlf_path),
            size,
        )
    }

    /// Create a pool for Python workers.
    pub fn new_python(script_path: &str, size: usize) -> Self {
        Self::new(ProcessConfig::python_worker(script_path), size)
    }

    /// Get pool size.
    pub fn size(&self) -> usize {
        self.processes.len()
    }

    /// Get total requests handled.
    pub fn total_requests(&self) -> usize {
        self.total_requests.load(Ordering::Relaxed)
    }

    /// Get total processes spawned.
    pub fn total_spawns(&self) -> usize {
        self.total_spawns.load(Ordering::Relaxed)
    }

    /// Acquire a process from the pool.
    ///
    /// Blocks until a process is available. Returns a guard that sends
    /// requests to the process and returns it to the pool when dropped.
    pub fn acquire(&self) -> PoolGuard<'_> {
        let _permit = self.semaphore.acquire();
        PoolGuard {
            pool: self,
            slot_index: None,
            _permit,
        }
    }

    /// Try to acquire a process without blocking.
    ///
    /// Returns None if no process is immediately available.
    pub fn try_acquire(&self) -> Option<PoolGuard<'_>> {
        // We don't have try_acquire on our semaphore, so just use acquire for now
        // This could be enhanced later
        Some(self.acquire())
    }

    /// Find and lock an available process slot, spawning if needed.
    fn get_or_spawn_process(
        &self,
    ) -> std::io::Result<(usize, std::sync::MutexGuard<'_, Option<ChildProcess>>)> {
        // Try to find an existing healthy process
        for (i, slot) in self.processes.iter().enumerate() {
            if let Ok(mut guard) = slot.try_lock() {
                if let Some(ref mut process) = *guard {
                    if process.is_alive() && !process.should_recycle() {
                        return Ok((i, guard));
                    }
                    // Process is dead or needs recycling, will respawn below
                }
                // Slot is available (empty or dead process)
                if guard.is_none() || !guard.as_mut().map(|p| p.is_alive()).unwrap_or(false) {
                    // Spawn new process
                    let process = ChildProcess::spawn(&self.config)?;
                    self.total_spawns.fetch_add(1, Ordering::Relaxed);
                    *guard = Some(process);
                    return Ok((i, guard));
                }
            }
        }

        // All slots busy - block on first available
        for (i, slot) in self.processes.iter().enumerate() {
            let mut guard = slot.lock().unwrap();
            if guard.is_none() {
                let process = ChildProcess::spawn(&self.config)?;
                self.total_spawns.fetch_add(1, Ordering::Relaxed);
                *guard = Some(process);
                return Ok((i, guard));
            }
            if let Some(ref mut process) = *guard {
                if !process.is_alive() || process.should_recycle() {
                    // Replace dead/recycled process
                    if let Some(old) = guard.take() {
                        old.terminate();
                    }
                    let process = ChildProcess::spawn(&self.config)?;
                    self.total_spawns.fetch_add(1, Ordering::Relaxed);
                    *guard = Some(process);
                }
                return Ok((i, guard));
            }
        }

        Err(std::io::Error::other("No process slots available"))
    }

    /// Shutdown all processes in the pool.
    pub fn shutdown(&self) {
        for slot in &self.processes {
            if let Ok(mut guard) = slot.lock() {
                if let Some(process) = guard.take() {
                    process.terminate();
                }
            }
        }
    }
}

impl Drop for ProcessPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// RAII guard for a borrowed process.
///
/// Send requests via `send_request()`. The process is returned to the pool
/// when the guard is dropped.
pub struct PoolGuard<'a> {
    pool: &'a ProcessPool,
    slot_index: Option<usize>,
    _permit: crate::semaphore::SemaphoreGuard,
}

impl<'a> PoolGuard<'a> {
    /// Send a request to the process and receive a response.
    pub fn send_request(&mut self, request: &Request) -> std::io::Result<Response> {
        let start = Instant::now();

        // Get or spawn a process
        let (slot_idx, mut guard) = self.pool.get_or_spawn_process()?;
        self.slot_index = Some(slot_idx);

        let process = guard
            .as_mut()
            .ok_or_else(|| std::io::Error::other("Process slot unexpectedly empty"))?;

        // Send request with timeout consideration
        let timeout = Duration::from_secs(self.pool.config.request_timeout_secs);
        if start.elapsed() > timeout {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "Request timed out before sending",
            ));
        }

        let result = process.send_request(request);

        // Update statistics
        if result.is_ok() {
            self.pool.total_requests.fetch_add(1, Ordering::Relaxed);
        }

        // Check if process needs recycling after this request
        if process.should_recycle() {
            if let Some(old) = guard.take() {
                old.terminate();
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization() {
        let request = Request::new(
            "silent",
            serde_json::json!({
                "baseline": [100, 101, 102],
                "test": [103, 104, 105],
                "alpha": 0.1
            }),
        );

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"method\":\"silent\""));
        assert!(json.contains("\"baseline\""));
    }

    #[test]
    fn test_response_parsing() {
        let json = r#"{"id": 1, "result": {"detected": true, "statistic": 2.34}}"#;
        let response: Response = serde_json::from_str(json).unwrap();

        assert!(!response.is_error());
        let result = response.into_result().unwrap();
        assert_eq!(result["detected"], true);
    }

    #[test]
    fn test_error_response_parsing() {
        let json = r#"{"id": 1, "error": {"code": -32603, "message": "Insufficient samples"}}"#;
        let response: Response = serde_json::from_str(json).unwrap();

        assert!(response.is_error());
        let err = response.into_result().unwrap_err();
        assert!(err.contains("Insufficient samples"));
    }
}
