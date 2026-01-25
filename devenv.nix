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
    version = "unstable-2024-01-19";
    format = "setuptools";

    src = pkgs.fetchFromGitHub {
      owner = "tlsfuzzer";
      repo = "tlslite-ng";
      rev = "master";
      hash = "sha256-Z0nVC6XcnA3I4H8s0KkKd5gkJ1kfWd4d/2/2eJyo7lQ=";
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
    version = "unstable-2024-01-19";
    format = "setuptools";

    src = pkgs.fetchFromGitHub {
      owner = "tlsfuzzer";
      repo = "tlsfuzzer";
      rev = "master";
      hash = "sha256-MQ7iFsCsTyIa4T4LcfRHpDD24UxkcpAAryyBsPRKctE=";
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
in
{
  languages.rust = {
    enable = true;
    channel = "stable";
    components = [ "rustc" "cargo" "clippy" "rustfmt" "rust-analyzer" "rust-src" ];
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

    # Documentation website (Starlight + CF Workers)
    bun
    nodePackages.wrangler
  ];

  env = {
    RUST_BACKTRACE = "1";
  };

  enterShell = ''
    command -v cargo-llvm-cov &> /dev/null || cargo install cargo-llvm-cov --quiet
    command -v cargo-nextest &> /dev/null || cargo install cargo-nextest --quiet
  '';
}
