module github.com/agucova/timing-oracle/go/timingoracle

go 1.21

// No external dependencies for the core library.
// The measurement loop and timers are implemented in pure Go + assembly.
// Statistical analysis is provided via CGo linking to timing-oracle-go.
