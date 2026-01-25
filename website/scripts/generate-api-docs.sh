#!/bin/bash
# Generate API documentation from rustdoc JSON output
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "Generating rustdoc JSON..."

# Generate rustdoc JSON (requires nightly)
RUSTC_BOOTSTRAP=1 RUSTDOCFLAGS="-Z unstable-options --output-format json" \
  cargo +nightly doc --no-deps -p tacet -p tacet-core 2>/dev/null || {
    echo "Note: rustdoc JSON generation requires nightly Rust"
    echo "Install with: rustup toolchain install nightly"
    echo "Skipping rustdoc-md generation..."
    exit 0
}

# Check if rustdoc-md is installed
if ! command -v rustdoc-md &> /dev/null; then
    echo "rustdoc-md not found. Install with: cargo install rustdoc-md"
    echo "Skipping API doc generation..."
    exit 0
fi

echo "Converting to markdown..."

OUTPUT_FILE="website/src/content/docs/api/generated.mdx"

# Generate markdown from JSON
rustdoc-md --path target/doc/tacet.json \
  --output /tmp/tacet_api.md

# Add frontmatter and convert to mdx
cat > "$OUTPUT_FILE" << 'EOF'
---
title: Generated API (rustdoc)
description: Auto-generated API documentation from Rust source
---

import { Aside } from '@astrojs/starlight/components';

<Aside type="note">
This page is auto-generated from rustdoc. For curated API documentation, see the [Rust API Reference](/api/rust).
</Aside>

EOF

cat /tmp/tacet_api.md >> "$OUTPUT_FILE"

echo "Generated: $OUTPUT_FILE"
