//go:generate go tool cgo -godefs types_cgo.go > types_generated_raw.go
//go:generate go run ../../scripts/postprocess_types.go types_generated_raw.go types_generated.go

package ffi

// This file triggers code generation for FFI types.
// Run: go generate ./...
//
// The generation process:
// 1. cgo -godefs extracts Go struct definitions from C types
// 2. postprocess_types.go converts field names to Go style (snake_case -> CamelCase)
