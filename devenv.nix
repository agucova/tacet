{ pkgs, lib, ... }:

let
  # R environment with RTLF and SILENT dependencies
  rWithPackages = pkgs.rWrapper.override {
    packages = with pkgs.rPackages; [
      # RTLF deps
      tidyverse
      optparse
      jsonlite
      # SILENT deps
      cli
      glue
      np
      robcp
      Qtools
    ];
  };

  # RTLF timing leakage detection tool (USENIX Security 2024)
  rtlf = pkgs.stdenv.mkDerivation {
    pname = "rtlf";
    version = "unstable-2024-01-19";

    src = pkgs.fetchFromGitHub {
      owner = "tls-attacker";
      repo = "RTLF";
      rev = "2497d660e4b23fc9acd21b077a35e82178fcd728";
      hash = "sha256-mbQaXGHhsSDWqkfRgiHIIZdKBLiaeRnkc0wLr0KEro8=";
    };

    nativeBuildInputs = [ pkgs.makeWrapper ];

    installPhase = ''
      mkdir -p $out/bin $out/share/rtlf
      cp -r . $out/share/rtlf/

      # Create wrapper script that uses our R environment
      makeWrapper ${rWithPackages}/bin/Rscript $out/bin/rtlf \
        --add-flags "$out/share/rtlf/rtlf.R"
    '';

    meta = {
      description = "RTLF - Statistical timing leakage detection (USENIX Security 2024)";
      homepage = "https://github.com/tls-attacker/RTLF";
      license = lib.licenses.asl20;
    };
  };

  # tlslite-ng - TLS library required by tlsfuzzer
  tlslite-ng = pkgs.python3Packages.buildPythonPackage {
    pname = "tlslite-ng";
    version = "unstable-2026-08-03";
    format = "setuptools";

    src = pkgs.fetchFromGitHub {
      owner = "tlsfuzzer";
      repo = "tlslite-ng";
      rev = "7e95ea1a86c0d42d8d931ff14134d31f01ce1d76";
      hash = "sha256-l0M/pR0RUF2e03dpLh/qL8no22/owGn0dgGTq/yV/Y4=";
    };

    propagatedBuildInputs = with pkgs.python3Packages; [
      ecdsa
    ];

    doCheck = false; # Tests require network

    meta = {
      description = "TLS implementation in Python";
      homepage = "https://github.com/tlsfuzzer/tlslite-ng";
      license = lib.licenses.lgpl21;
    };
  };

  # tlsfuzzer - TLS timing analysis tool
  tlsfuzzer = pkgs.python3Packages.buildPythonApplication {
    pname = "tlsfuzzer";
    version = "unstable-2026-07-19";
    format = "setuptools";

    src = pkgs.fetchFromGitHub {
      owner = "tlsfuzzer";
      repo = "tlsfuzzer";
      rev = "5eebc4464e5197a7f7392fb9acda99cfc32441f7";
      hash = "sha256-+8Lf1sbuu8D7fiWt6iyNgY5RZv3bKXbwuEZARj2NXPQ=";
    };

    propagatedBuildInputs = [
      tlslite-ng
      pkgs.python3Packages.ecdsa
      # Required for timing analysis (analysis.py)
      pkgs.python3Packages.numpy
      pkgs.python3Packages.scipy
      pkgs.python3Packages.pandas
      pkgs.python3Packages.matplotlib
    ];

    doCheck = false; # Tests require TLS server

    meta = {
      description = "TLS test suite and fuzzer";
      homepage = "https://github.com/tlsfuzzer/tlsfuzzer";
      license = lib.licenses.gpl2;
    };
  };

  # SILENT timing leakage detection tool (arXiv 2024)
  silent = pkgs.stdenv.mkDerivation {
    pname = "silent";
    version = "unstable-2024-01-19";

    src = pkgs.fetchFromGitHub {
      owner = "tls-attacker";
      repo = "SILENT";
      rev = "129f13d61330ac598674ee197e5ff06dfc5351b1";
      hash = "sha256-NNGUvNIpG5TQZZHyHAAumDofwnmlDGL3DaHTZOVP2gg=";
    };

    nativeBuildInputs = [ pkgs.makeWrapper ];

    installPhase = ''
      mkdir -p $out/bin $out/share/silent
      cp -r . $out/share/silent/

      # Create wrapper script that uses our R environment
      makeWrapper ${rWithPackages}/bin/Rscript $out/bin/silent \
        --add-flags "$out/share/silent/scripts/SILENT.R"
    '';

    meta = {
      description = "SILENT - Bootstrap-based timing leakage detection (arXiv 2024)";
      homepage = "https://github.com/tls-attacker/SILENT";
      license = lib.licenses.asl20;
    };
  };

  # Custom TexLive with packages needed for the paper
  texlive-paper = pkgs.texlive.combine {
    inherit (pkgs.texlive) scheme-full;
  };

  # wolfSSL for the test-wolfssl feature. Nixpkgs dropped the package in
  # April 2026, so it is built here. The configure flags match the upstream
  # Homebrew build: crates/tacet/tests/crypto/c_libraries/wolfssl.rs hardcodes
  # the sizes of opaque wolfSSL structs (RsaKey = 8448, ecc_key = 4320 bytes),
  # and builds using SP math (--enable-sp-math-all, as nixpkgs did) make those
  # structs larger, which would overflow the Rust-side buffers.
  wolfssl = pkgs.stdenv.mkDerivation (finalAttrs: {
    pname = "wolfssl";
    version = "5.8.4";

    src = pkgs.fetchFromGitHub {
      owner = "wolfSSL";
      repo = "wolfssl";
      tag = "v${finalAttrs.version}-stable";
      hash = "sha256-vfJKmDdM0r591t5GnuSS7NyiUYXCQOTKbWLVydB3N9s=";
    };

    postPatch = ''
      patchShebangs ./scripts
    '';

    configureFlags = [
      "--enable-all"
      "--enable-reproducible-build"
    ];

    nativeBuildInputs = [ pkgs.autoreconfHook ];

    # Upstream test suite spawns TLS servers and takes minutes
    doCheck = false;

    meta = {
      description = "wolfSSL - embedded TLS/cryptography library";
      homepage = "https://www.wolfssl.com/";
      license = lib.licenses.gpl2Plus;
    };
  });
in
{
  languages.rust = {
    enable = true;
    channel = "stable";
    components = [ "rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" "rust-src" ];
    # thumbv7em-none-eabihf is a bare-metal target: it guards tacet-core's
    # no_std build, which a host-target check can pass even when std leaks in
    targets = [ "wasm32-unknown-unknown" "wasm32-wasip1" "thumbv7em-none-eabihf" ];
  };

  languages.go.enable = true;

  # C++ toolchain for building/testing C++ bindings
  languages.cplusplus.enable = true;

  # Python for analysis notebooks and tlsfuzzer
  languages.python = {
    enable = true;
    package = pkgs.python3;
    venv.enable = true;
    venv.requirements = ''
      pandas
      numpy
      matplotlib
      seaborn
      scipy
      jupyter
      nbconvert
    '';
  };

  packages = with pkgs; [
    cargo-nextest
    cargo-edit
    rWithPackages
    rtlf
    silent
    tlsfuzzer
    # Note: Mona (Crosby box test) is implemented in pure Rust - no external tool needed

    # C binding tests
    cmocka
    cmake
    pkg-config

    # C/C++ cryptographic libraries for cross-language testing
    libressl.dev  # Include dev outputs for headers
    libsodium.dev  # Include dev outputs for headers
    mbedtls  # Headers and libraries share a single output
    botan3.out  # Library output (botan3 defaults to its `bin` output)
    botan3.dev  # Include dev outputs for headers
    wolfssl  # Built above (no longer in nixpkgs)

    # Documentation website (Starlight + CF Workers)
    bun
    wrangler

    # WASM development
    wasmtime
    wasm-pack
    wasm-bindgen-cli

    # LaTeX for paper compilation (USENIX Security 2026)
    texlive-paper
    aspell
    aspellDicts.en
    poppler-utils  # for pdfinfo
  ];

  env = {
    RUST_BACKTRACE = "1";

    # C crypto library locations for crates/tacet/build.rs, which links
    # <DIR>/lib for the test-mbedtls, test-wolfssl and test-botan features
    MBEDTLS_DIR = "${pkgs.mbedtls}";
    WOLFSSL_DIR = "${wolfssl}";
    BOTAN_DIR = "${pkgs.botan3.out}";
  };

  enterShell = ''
    command -v cargo-llvm-cov &> /dev/null || cargo install cargo-llvm-cov --quiet
    command -v cargo-nextest &> /dev/null || cargo install cargo-nextest --quiet
  '';
}
