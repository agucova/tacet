package main

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	repo           = "agucova/tacet"
	modulePath     = "github.com/agucova/tacet/crates/tacet-go"
	defaultVersion = "latest"
)

func main() {
	version := flag.String("version", defaultVersion, "Version to install (e.g., v0.2.3 or 'latest')")
	flag.Parse()

	if err := install(*version); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func install(version string) error {
	platform := fmt.Sprintf("%s_%s", runtime.GOOS, runtime.GOARCH)

	// Map platform to asset name
	var assetName string
	switch platform {
	case "darwin_arm64":
		assetName = "libtacet_c-darwin-arm64.a"
	case "darwin_amd64":
		assetName = "libtacet_c-darwin-amd64.a"
	case "linux_arm64":
		assetName = "libtacet_c-linux-arm64.a"
	case "linux_amd64":
		assetName = "libtacet_c-linux-amd64.a"
	default:
		return fmt.Errorf("unsupported platform: %s\n\nSupported platforms:\n  - darwin_arm64 (macOS Apple Silicon)\n  - darwin_amd64 (macOS Intel)\n  - linux_arm64 (Linux ARM64)\n  - linux_amd64 (Linux x86_64)", platform)
	}

	// Download the library
	fmt.Printf("Downloading %s for %s...\n", assetName, platform)
	libData, err := downloadLibrary(version, assetName)
	if err != nil {
		return err
	}

	// Find Go module cache directory
	modCacheDir, err := getModuleCacheDir()
	if err != nil {
		fmt.Printf("Warning: could not find module cache: %v\n", err)
		fmt.Println("The library will be installed to ~/.tacet/lib/")
		fmt.Println("You'll need to set CGO_LDFLAGS when building.")
	} else {
		// Install to module cache
		targetDir := filepath.Join(modCacheDir, "internal", "ffi", "lib", platform)
		if err := installToDir(targetDir, libData, "module cache"); err != nil {
			return err
		}
	}

	// Also install to standard location as backup
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get home directory: %w", err)
	}
	homeLibDir := filepath.Join(homeDir, ".tacet", "lib", platform)
	if err := installToDir(homeLibDir, libData, "home directory"); err != nil {
		return err
	}

	fmt.Printf("\n✓ Successfully installed tacet native library (%.1f MB)\n", float64(len(libData))/(1024*1024))
	fmt.Println("\nYou can now use tacet in your Go projects:")
	fmt.Println("  import tacet \"github.com/agucova/tacet/crates/tacet-go\"")

	return nil
}

func downloadLibrary(version, assetName string) ([]byte, error) {
	var downloadURL string
	if version == "latest" {
		downloadURL = fmt.Sprintf("https://github.com/%s/releases/latest/download/%s", repo, assetName)
	} else {
		downloadURL = fmt.Sprintf("https://github.com/%s/releases/download/%s/%s", repo, version, assetName)
	}

	fmt.Printf("URL: %s\n", downloadURL)

	resp, err := http.Get(downloadURL)
	if err != nil {
		return nil, fmt.Errorf("failed to download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("download failed: HTTP %d\n\nPossible issues:\n  - Release %s doesn't exist\n  - Asset %s not found in release\n  - Network connectivity issues", resp.StatusCode, version, assetName)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	return data, nil
}

func installToDir(dir string, data []byte, location string) error {
	outputPath := filepath.Join(dir, "libtacet_c.a")

	// Check if already exists
	if _, err := os.Stat(outputPath); err == nil {
		fmt.Printf("Skipping %s (already exists): %s\n", location, outputPath)
		return nil
	}

	// Create directory
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create %s directory: %w", location, err)
	}

	// Write file
	if err := os.WriteFile(outputPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write to %s: %w", location, err)
	}

	fmt.Printf("Installed to %s: %s\n", location, outputPath)
	return nil
}

func getModuleCacheDir() (string, error) {
	// Run: go list -m -f '{{.Dir}}' github.com/agucova/tacet/crates/tacet-go
	cmd := exec.Command("go", "list", "-m", "-f", "{{.Dir}}", modulePath)
	output, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("go list failed: %w", err)
	}

	dir := strings.TrimSpace(string(output))
	if dir == "" {
		return "", fmt.Errorf("module not found in cache")
	}

	return dir, nil
}
