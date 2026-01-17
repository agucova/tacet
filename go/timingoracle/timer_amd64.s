//go:build amd64

#include "textflag.h"

// func rdtsc() uint64
// Reads the Time Stamp Counter with serialization.
// Uses LFENCE before RDTSC for proper serialization on AMD and Intel.
TEXT ·rdtsc(SB), NOSPLIT, $0-8
	// LFENCE serializes - ensures all previous instructions complete
	// before reading TSC. This is recommended over CPUID for lower overhead.
	LFENCE
	RDTSC
	// RDTSC returns low 32 bits in AX, high 32 bits in DX
	SHLQ	$32, DX
	ORQ	DX, AX
	MOVQ	AX, ret+0(FP)
	RET
