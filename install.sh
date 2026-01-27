#!/bin/bash
set -e

# Tacet C/C++ Library Installer
# Downloads and installs pre-built tacet libraries from GitHub releases

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}➜${NC} $*"
}

warn() {
    echo -e "${YELLOW}⚠${NC} $*"
}

error() {
    echo -e "${RED}✗${NC} $*" >&2
}

# Detect platform
detect_platform() {
    local os arch target

    os=$(uname -s | tr '[:upper:]' '[:lower:]')
    arch=$(uname -m)

    # Normalize architecture names
    case "$arch" in
        arm64|aarch64)
            arch="arm64"
            ;;
        x86_64|amd64)
            arch="amd64"
            ;;
        *)
            error "Unsupported architecture: $arch"
            error "Supported: arm64, amd64"
            exit 1
            ;;
    esac

    # Normalize OS names
    case "$os" in
        darwin)
            target="darwin-${arch}"
            ;;
        linux)
            target="linux-${arch}"
            ;;
        *)
            error "Unsupported operating system: $os"
            error "Supported: macOS (darwin), Linux"
            exit 1
            ;;
    esac

    echo "$target"
}

# Get latest release version from GitHub API
get_latest_version() {
    local api_url="https://api.github.com/repos/agucova/tacet/releases/latest"

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$api_url" | grep '"tag_name"' | sed -E 's/.*"tag_name": "([^"]+)".*/\1/'
    else
        error "curl is required but not installed"
        exit 1
    fi
}

# Download file from URL
download_file() {
    local url=$1
    local output=$2

    info "Downloading $(basename "$output")..."

    if ! curl -fSL --progress-bar -o "$output" "$url"; then
        error "Failed to download: $url"
        return 1
    fi

    return 0
}

# Check if we have write permissions to a directory
check_write_permission() {
    local dir=$1

    if [ -w "$dir" ] || touch "${dir}/.write_test" 2>/dev/null; then
        rm -f "${dir}/.write_test" 2>/dev/null || true
        return 0
    else
        return 1
    fi
}

# Check sudo requirement and exit with instructions if needed
check_sudo_requirement() {
    local prefix=$1

    if ! check_write_permission "$prefix"; then
        error "Installation to $prefix requires elevated privileges"
        echo ""
        echo "Please re-run the installation with sudo:"
        echo ""
        if [ -f "$0" ]; then
            echo "  sudo -E bash \"$0\""
        else
            echo "  curl -fsSL https://raw.githubusercontent.com/agucova/tacet/main/install.sh | sudo -E bash"
        fi
        echo ""
        echo "Or choose a different PREFIX where you have write access:"
        echo ""
        echo "  PREFIX=\$HOME/.local bash <(curl -fsSL https://raw.githubusercontent.com/agucova/tacet/main/install.sh)"
        echo ""
        exit 1
    fi
}

# Main installation function
main() {
    info "Tacet C/C++ Library Installer"
    echo

    # Configuration
    VERSION="${VERSION:-latest}"
    PREFIX="${PREFIX:-/usr/local}"
    TMPDIR=$(mktemp -d)

    # Ensure cleanup on exit
    trap 'rm -rf "$TMPDIR"' EXIT

    # Detect platform
    TARGET=$(detect_platform)
    info "Detected platform: $TARGET"

    # Resolve version
    if [ "$VERSION" = "latest" ]; then
        VERSION=$(get_latest_version)
        if [ -z "$VERSION" ]; then
            error "Failed to determine latest version"
            exit 1
        fi
        info "Latest version: $VERSION"
    else
        info "Installing version: $VERSION"
    fi

    # Check write permissions (exit if sudo required)
    check_sudo_requirement "$PREFIX"

    # Download artifacts
    BASE_URL="https://github.com/agucova/tacet/releases/download/${VERSION}"

    cd "$TMPDIR"

    download_file "${BASE_URL}/libtacet_c-${TARGET}.a" "libtacet_c.a" || exit 1
    download_file "${BASE_URL}/tacet.h" "tacet.h" || exit 1
    download_file "${BASE_URL}/tacet.hpp" "tacet.hpp" || exit 1
    download_file "${BASE_URL}/tacet-${TARGET}.pc" "tacet.pc.template" || exit 1

    # Create installation directories
    info "Creating directories..."
    mkdir -p "${PREFIX}/lib"
    mkdir -p "${PREFIX}/include/tacet"
    mkdir -p "${PREFIX}/lib/pkgconfig"

    # Install files
    info "Installing files to $PREFIX..."
    cp libtacet_c.a "${PREFIX}/lib/"
    cp tacet.h "${PREFIX}/include/tacet/"
    cp tacet.hpp "${PREFIX}/include/tacet/"

    # Generate final pkg-config file with actual paths
    sed -e "s|@PREFIX@|${PREFIX}|g" \
        -e "s|@LIBDIR@|${PREFIX}/lib|g" \
        -e "s|@INCLUDEDIR@|${PREFIX}/include/tacet|g" \
        tacet.pc.template > "${PREFIX}/lib/pkgconfig/tacet.pc"

    # Get file sizes for display
    LIB_SIZE=$(du -h "${PREFIX}/lib/libtacet_c.a" | cut -f1)

    echo
    info "Installation complete!"
    echo
    echo "Files installed:"
    echo "  ${PREFIX}/lib/libtacet_c.a ($LIB_SIZE)"
    echo "  ${PREFIX}/include/tacet/tacet.h"
    echo "  ${PREFIX}/include/tacet/tacet.hpp"
    echo "  ${PREFIX}/lib/pkgconfig/tacet.pc"
    echo
    echo "Verify installation:"
    echo "  pkg-config --modversion tacet"
    echo
    echo "Usage in your project:"
    echo "  cc myfile.c \$(pkg-config --cflags --libs tacet) -o myapp"
}

# Run main function
main "$@"
