//go:build arm64

#include "textflag.h"

// func cntvct() uint64
// Reads the virtual counter (CNTVCT_EL0) with serialization.
TEXT ·cntvct(SB), NOSPLIT, $0-8
	// ISB serializes the instruction stream, ensuring all previous
	// instructions complete before reading the counter.
	ISB	$15
	// Read CNTVCT_EL0 (virtual counter)
	// System register encoding: op0=3, op1=3, CRn=14, CRm=0, op2=2
	MRS	CNTVCT_EL0, R0
	MOVD	R0, ret+0(FP)
	RET

// func cntfrq() uint64
// Reads the counter frequency (CNTFRQ_EL0).
// Returns the frequency in Hz (e.g., 24000000 for Apple Silicon).
TEXT ·cntfrq(SB), NOSPLIT, $0-8
	// Read CNTFRQ_EL0 (counter frequency)
	// System register encoding: op0=3, op1=3, CRn=14, CRm=0, op2=0
	MRS	CNTFRQ_EL0, R0
	MOVD	R0, ret+0(FP)
	RET
