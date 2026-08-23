fn main() {
    // Configure library paths for FFI tests.
    // Each library is behind a feature gate so builds don't fail when the
    // system library isn't installed.

    // === mbedTLS Configuration (feature: test-mbedtls) ===
    if std::env::var("CARGO_FEATURE_TEST_MBEDTLS").is_ok() {
        configure_mbedtls();
    }

    // === wolfSSL Configuration (feature: test-wolfssl) ===
    if std::env::var("CARGO_FEATURE_TEST_WOLFSSL").is_ok() {
        configure_wolfssl();
    }

    // === Botan Configuration (feature: test-botan) ===
    if std::env::var("CARGO_FEATURE_TEST_BOTAN").is_ok() {
        configure_botan();
    }
}

fn configure_mbedtls() {
    if let Ok(mbedtls_dir) = std::env::var("MBEDTLS_DIR") {
        let lib_path = format!("{}/lib", mbedtls_dir);
        println!("cargo:rustc-link-search=native={}", lib_path);
        println!("cargo:warning=Using MBEDTLS_DIR: {}", mbedtls_dir);
        return;
    }

    // Try pkg-config first
    if try_pkg_config("mbedcrypto", "mbedTLS") {
        return;
    }

    // Scan Nix store for mbedTLS
    if let Ok(entries) = std::fs::read_dir("/nix/store") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.contains("mbedtls-") && !name_str.contains(".drv") {
                let lib_dir = entry.path().join("lib");
                if lib_dir.join("libmbedcrypto.a").exists()
                    || lib_dir.join("libmbedcrypto.so").exists()
                    || lib_dir.join("libmbedcrypto.dylib").exists()
                {
                    println!("cargo:rustc-link-search=native={}", lib_dir.display());
                    println!("cargo:warning=Found mbedTLS at: {}", entry.path().display());
                    return;
                }
            }
        }
    }

    println!(
        "cargo:warning=mbedTLS not found. Set MBEDTLS_DIR or install via your package manager."
    );
}

fn configure_wolfssl() {
    if let Ok(wolfssl_dir) = std::env::var("WOLFSSL_DIR") {
        let lib_path = format!("{}/lib", wolfssl_dir);
        println!("cargo:rustc-link-search=native={}", lib_path);
        println!("cargo:rustc-link-lib=wolfssl");
        println!("cargo:warning=Using WOLFSSL_DIR: {}", wolfssl_dir);
        return;
    }

    // Try pkg-config first
    if try_pkg_config("wolfssl", "wolfSSL") {
        return;
    }

    // Try common library paths
    let search_paths: &[&str] = if cfg!(target_os = "macos") {
        &["/opt/homebrew/lib", "/usr/local/lib"]
    } else {
        &["/usr/lib", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu"]
    };

    let lib_ext = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };

    for path in search_paths {
        let lib_file = format!("{}/libwolfssl.{}", path, lib_ext);
        if std::path::Path::new(&lib_file).exists() {
            println!("cargo:rustc-link-search=native={}", path);
            println!("cargo:rustc-link-lib=wolfssl");
            println!("cargo:warning=Found wolfSSL at: {}", path);
            return;
        }
    }

    println!(
        "cargo:warning=wolfSSL not found. Set WOLFSSL_DIR or install via your package manager."
    );
}

fn configure_botan() {
    if let Ok(botan_dir) = std::env::var("BOTAN_DIR") {
        let lib_path = format!("{}/lib", botan_dir);
        println!("cargo:rustc-link-search=native={}", lib_path);
        println!("cargo:rustc-link-lib=botan-3");
        println!("cargo:warning=Using BOTAN_DIR: {}", botan_dir);
        return;
    }

    // Try pkg-config first
    if try_pkg_config("botan-3", "Botan") {
        return;
    }

    // Try common library paths
    let search_paths: &[&str] = if cfg!(target_os = "macos") {
        &["/opt/homebrew/lib", "/usr/local/lib"]
    } else {
        &["/usr/lib", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu"]
    };

    let lib_ext = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };

    for path in search_paths {
        let lib_file = format!("{}/libbotan-3.{}", path, lib_ext);
        if std::path::Path::new(&lib_file).exists() {
            println!("cargo:rustc-link-search=native={}", path);
            println!("cargo:rustc-link-lib=botan-3");
            println!("cargo:warning=Found Botan at: {}", path);
            return;
        }
    }

    println!("cargo:warning=Botan not found. Set BOTAN_DIR or install via your package manager.");
}

/// Try to use pkg-config to find a library. Returns true if successful.
fn try_pkg_config(lib_name: &str, display_name: &str) -> bool {
    let output = std::process::Command::new("pkg-config")
        .args(["--libs-only-L", lib_name])
        .output();

    if let Ok(output) = output {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for flag in stdout.split_whitespace() {
                if let Some(path) = flag.strip_prefix("-L") {
                    println!("cargo:rustc-link-search=native={}", path);
                }
            }
            println!("cargo:rustc-link-lib={}", lib_name);
            println!("cargo:warning=Found {} via pkg-config", display_name);
            return true;
        }
    }
    false
}
