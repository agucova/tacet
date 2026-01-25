#!/bin/bash
# Downloads the pre-built timing-oracle-c static library for the current platform.
# Called by go:generate or manually before building.

set -euo pipefail

# Configuration
REPO="agucova/timing-oracle"
VERSION="${TIMING_ORACLE_VERSION:-latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="${SCRIPT_DIR}/../internal/ffi/lib"

# Detect platform
OS="$(go env GOOS)"
ARCH="$(go env GOARCH)"
PLATFORM="${OS}_${ARCH}"

# Map to release asset names
case "${PLATFORM}" in
    darwin_arm64)
        ASSET="libtiming_oracle_c-darwin-arm64.a"
        ;;
    darwin_amd64)
        ASSET="libtiming_oracle_c-darwin-amd64.a"
        ;;
    linux_arm64)
        ASSET="libtiming_oracle_c-linux-arm64.a"
        ;;
    linux_amd64)
        ASSET="libtiming_oracle_c-linux-amd64.a"
        ;;
    *)
        echo "Error: Unsupported platform ${PLATFORM}" >&2
        echo "Supported platforms: darwin_arm64, darwin_amd64, linux_arm64, linux_amd64" >&2
        echo "" >&2
        echo "You can build from source instead:" >&2
        echo "  cargo build -p timing-oracle-c --release" >&2
        echo "  mkdir -p ${LIB_DIR}/${PLATFORM}" >&2
        echo "  cp target/release/libtiming_oracle_c.a ${LIB_DIR}/${PLATFORM}/" >&2
        exit 1
        ;;
esac

# Create lib directory
mkdir -p "${LIB_DIR}/${PLATFORM}"
OUTPUT="${LIB_DIR}/${PLATFORM}/libtiming_oracle_c.a"

# Check if already downloaded
if [[ -f "${OUTPUT}" ]]; then
    echo "Library already exists at ${OUTPUT}"
    echo "To re-download, remove it first: rm ${OUTPUT}"
    exit 0
fi

# Determine download URL
if [[ "${VERSION}" == "latest" ]]; then
    RELEASE_URL="https://api.github.com/repos/${REPO}/releases/latest"
    echo "Fetching latest release info..."
    DOWNLOAD_URL=$(curl -fsSL "${RELEASE_URL}" | grep -o "https://[^\"]*${ASSET}" | head -1)
else
    DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${VERSION}/${ASSET}"
fi

if [[ -z "${DOWNLOAD_URL}" ]]; then
    echo "Error: Could not find release asset ${ASSET}" >&2
    echo "" >&2
    echo "The pre-built library may not be available yet." >&2
    echo "You can build from source instead:" >&2
    echo "  cargo build -p timing-oracle-c --release" >&2
    echo "  strip -S target/release/libtiming_oracle_c.a" >&2
    echo "  mkdir -p ${LIB_DIR}/${PLATFORM}" >&2
    echo "  cp target/release/libtiming_oracle_c.a ${LIB_DIR}/${PLATFORM}/" >&2
    exit 1
fi

echo "Downloading ${ASSET} for ${PLATFORM}..."
echo "URL: ${DOWNLOAD_URL}"

# Download with progress
if command -v wget &> /dev/null; then
    wget -q --show-progress -O "${OUTPUT}" "${DOWNLOAD_URL}"
elif command -v curl &> /dev/null; then
    curl -fSL --progress-bar -o "${OUTPUT}" "${DOWNLOAD_URL}"
else
    echo "Error: Neither curl nor wget found" >&2
    exit 1
fi

echo "Downloaded to ${OUTPUT}"
echo "Library size: $(du -h "${OUTPUT}" | cut -f1)"
