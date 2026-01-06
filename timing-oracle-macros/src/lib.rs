//! Proc macros for timing-oracle.
//!
//! This crate provides the `timing_test!` and `timing_test_checked!` macros for
//! writing timing side-channel tests with compile-time validation.
//!
//! See the `timing-oracle` crate documentation for usage examples.

use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

mod parse;

use parse::TimingTestInput;

/// Create a timing test that panics on unmeasurable operations.
///
/// This macro provides a declarative syntax for timing tests that prevents
/// common mistakes through compile-time checks. Returns `TestResult` directly,
/// panicking if the operation is too fast to measure reliably.
///
/// For explicit handling of unmeasurable operations, use `timing_test_checked!` instead.
///
/// # Returns
///
/// Returns `TestResult` directly. Panics if the operation is unmeasurable.
///
/// # Syntax
///
/// ```ignore
/// timing_test! {
///     // Optional: custom oracle configuration (defaults to balanced())
///     oracle: TimingOracle::quick(),
///
///     // Required: baseline input generator (closure returning the fixed/baseline value)
///     baseline: || [0u8; 32],
///
///     // Required: sample input generator (closure returning random values)
///     sample: || rand::random::<[u8; 32]>(),
///
///     // Required: measurement body (closure that receives input and performs the operation)
///     measure: |input| {
///         encrypt(&input);
///     },
/// }
/// ```
///
/// # Example
///
/// ```ignore
/// use timing_oracle::timing_test;
///
/// fn main() {
///     let result = timing_test! {
///         baseline: || [0u8; 32],
///         sample: || rand::random::<[u8; 32]>(),
///         measure: |input| {
///             let _ = std::hint::black_box(&input);
///         },
///     };
///
///     println!("Leak probability: {:.1}%", result.leak_probability * 100.0);
/// }
/// ```
#[proc_macro]
pub fn timing_test(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as TimingTestInput);
    expand_timing_test(input, false).into()
}

/// Create a timing test that returns `Outcome` for explicit handling.
///
/// This macro is identical to `timing_test!` but returns `Outcome` instead of
/// `TestResult`, allowing you to explicitly handle unmeasurable operations.
///
/// # Returns
///
/// Returns `Outcome` which is either `Outcome::Completed(TestResult)` or
/// `Outcome::Unmeasurable { ... }`.
///
/// # Example
///
/// ```ignore
/// use timing_oracle::{timing_test_checked, Outcome};
///
/// fn main() {
///     let outcome = timing_test_checked! {
///         baseline: || [0u8; 32],
///         sample: || rand::random::<[u8; 32]>(),
///         measure: |input| {
///             let _ = std::hint::black_box(&input);
///         },
///     };
///
///     match outcome {
///         Outcome::Completed(result) => {
///             println!("Leak probability: {:.1}%", result.leak_probability * 100.0);
///         }
///         Outcome::Unmeasurable { recommendation, .. } => {
///             println!("Operation too fast: {}", recommendation);
///         }
///     }
/// }
/// ```
#[proc_macro]
pub fn timing_test_checked(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as TimingTestInput);
    expand_timing_test(input, true).into()
}

fn expand_timing_test(input: TimingTestInput, checked: bool) -> proc_macro2::TokenStream {
    let TimingTestInput {
        input_type: _,
        oracle,
        baseline,
        sample,
        measure,
    } = input;

    // Default oracle if not specified - use balanced() for reasonable speed
    let oracle_expr = oracle.unwrap_or_else(|| {
        syn::parse_quote!(::timing_oracle::TimingOracle::balanced())
    });

    // Generate the timing test code
    let test_call = quote! {
        {
            // Create InputPair from baseline and sample closures
            let __inputs = ::timing_oracle::helpers::InputPair::new(
                #baseline,
                #sample,
            );

            // Run the test with the new API
            #oracle_expr.test(__inputs, #measure)
        }
    };

    if checked {
        // For timing_test_checked!, return Outcome directly
        test_call
    } else {
        // For timing_test!, unwrap to TestResult and panic on unmeasurable
        quote! {
            {
                let __outcome = #test_call;
                match __outcome {
                    ::timing_oracle::Outcome::Completed(result) => result,
                    ::timing_oracle::Outcome::Unmeasurable { operation_ns, threshold_ns, platform, recommendation } => {
                        panic!(
                            "Operation too fast to measure reliably:\n  \
                             Operation: {:.2}ns\n  \
                             Threshold: {:.2}ns\n  \
                             Platform: {}\n  \
                             Recommendation: {}",
                            operation_ns, threshold_ns, platform, recommendation
                        );
                    }
                }
            }
        }
    }
}
