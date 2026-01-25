# Documentation commands

# Build documentation for all crates
[group('doc')]
build:
    cargo doc --workspace --no-deps

# Build and open documentation
[group('doc')]
open:
    cargo doc --workspace --no-deps --open

# Build docs with private items
[group('doc')]
private:
    cargo doc --workspace --no-deps --document-private-items

# Website dev server
[group('doc')]
website-dev:
    cd website && bun run dev

# Build website
[group('doc')]
website-build:
    cd website && bun run build

# Generate API docs from rustdoc
[group('doc')]
api-docs:
    ./website/scripts/generate-api-docs.sh

# Deploy to Cloudflare Workers
[group('doc')]
website-deploy:
    cd website && bun run deploy

# Install website dependencies
[group('doc')]
website-install:
    cd website && bun install
