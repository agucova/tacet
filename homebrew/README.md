# Homebrew Formula for Tacet

This directory contains the Homebrew formula for installing tacet-c on macOS.

## For Users

### Installation

Install directly from URL (no tap required):

```bash
brew install https://raw.githubusercontent.com/agucova/tacet/main/homebrew/Formula/tacet-c.rb
```

Alternatively, if you prefer to create a local tap:

```bash
brew tap agucova/tacet https://github.com/agucova/tacet
brew install tacet-c
```

### Verify Installation

```bash
pkg-config --modversion tacet
pkg-config --cflags --libs tacet
```

### Usage in Your Project

#### With pkg-config

```bash
cc myfile.c $(pkg-config --cflags --libs tacet) -o myapp
```

#### With CMake

```cmake
cmake_minimum_required(VERSION 3.14)
project(my_project)

find_package(PkgConfig REQUIRED)
pkg_check_modules(TACET REQUIRED tacet)

add_executable(my_app main.c)
target_include_directories(my_app PRIVATE ${TACET_INCLUDE_DIRS})
target_link_libraries(my_app PRIVATE ${TACET_LIBRARIES})
```

## For Maintainers

The formula lives in the main tacet repository at `homebrew/Formula/tacet-c.rb`. Users install directly from the URL.

### Updating the Formula for a New Release

When releasing a new version:

1. Download the release artifacts:
   ```bash
   cd /tmp
   VERSION=v0.3.0  # Update this
   curl -LO "https://github.com/agucova/tacet/releases/download/${VERSION}/libtacet_c-darwin-arm64.a"
   curl -LO "https://github.com/agucova/tacet/releases/download/${VERSION}/libtacet_c-darwin-amd64.a"
   curl -LO "https://github.com/agucova/tacet/releases/download/${VERSION}/tacet.h"
   curl -LO "https://github.com/agucova/tacet/releases/download/${VERSION}/tacet.hpp"
   ```

2. Calculate SHA256 checksums:
   ```bash
   shasum -a 256 libtacet_c-darwin-arm64.a
   shasum -a 256 libtacet_c-darwin-amd64.a
   shasum -a 256 tacet.h
   shasum -a 256 tacet.hpp
   ```

3. Update `homebrew/Formula/tacet-c.rb`:
   - Change `version "0.3.0"` to the new version
   - Replace SHA256 placeholders with actual checksums
   - Test locally: `brew install --build-from-source ./homebrew/Formula/tacet-c.rb`
   - Run audit: `brew audit --strict tacet-c`

4. Commit and push to main tacet repository

### Testing the Formula

```bash
# Install from local formula
brew install --build-from-source ./homebrew/Formula/tacet-c.rb

# Run built-in tests
brew test tacet-c

# Audit formula
brew audit --strict tacet-c

# Uninstall
brew uninstall tacet-c
```

## Links

- Main repository: https://github.com/agucova/tacet
- Documentation: https://tacet.dev
- Homebrew documentation: https://docs.brew.sh
