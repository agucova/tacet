//! Proc macros for tacet.
//!
//! This crate provides the `timing_test!` and `timing_test_checked!` macros for
//! writing timing side-channel tests with compile-time validation.
//!
//! See the `tacet` crate documentation for usage examples.

use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

mod parse;

use parse::TimingTestInput;

/// Create a timing test that returns `Outcome` for pattern matching.
///
/// This macro provides a declarative syntax for timing tests that prevents
/// common mistakes through compile-time checks. Returns `Outcome` which can be
/// `Pass`, `Fail`, `Inconclusive`, or `Unmeasurable`.
///
/// # Returns
///
/// Returns `Outcome` which is one of:
/// - `Outcome::Pass { leak_probability, effect, samples_used, quality, diagnostics }`
/// - `Outcome::Fail { leak_probability, effect, exploitability, samples_used, quality, diagnostics }`
/// - `Outcome::Inconclusive { reason, leak_probability, effect, samples_used, quality, diagnostics }`
/// - `Outcome::Unmeasurable { operation_ns, timer_resolution_ns, platform, recommendation }`
///
/// # Syntax
///
/// ```ignore
/// timing_test! {
///     // Optional: custom oracle configuration (defaults to AdjacentNetwork attacker model)
///     oracle: TimingOracle::for_attacker(AttackerModel::AdjacentNetwork),
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
/// use tacet::{timing_test, Outcome};
///
/// fn main() {
///     let outcome = timing_test! {
///         baseline: || [0u8; 32],
///         sample: || rand::random::<[u8; 32]>(),
///         measure: |input| {
///             let _ = std::hint::black_box(&input);
///         },
///     };
///
///     match outcome {
///         Outcome::Pass { leak_probability, .. } => {
///             println!("No leak detected (P={:.1}%)", leak_probability * 100.0);
///         }
///         Outcome::Fail { leak_probability, exploitability, .. } => {
///             println!("Leak detected! P={:.1}%, {:?}", leak_probability * 100.0, exploitability);
///         }
///         Outcome::Inconclusive { reason, .. } => {
///             println!("Inconclusive: {:?}", reason);
///         }
///         Outcome::Unmeasurable { recommendation, .. } => {
///             println!("Operation too fast: {}", recommendation);
///         }
///     }
/// }
/// ```
#[proc_macro]
pub fn timing_test(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as TimingTestInput);
    expand_timing_test(input, false).into()
}

/// Create a timing test that returns `Outcome` for explicit handling.
///
/// This macro is identical to `timing_test!` - both return `Outcome`.
/// It is kept for backwards compatibility.
///
/// # Returns
///
/// Returns `Outcome` which is one of `Pass`, `Fail`, `Inconclusive`, or `Unmeasurable`.
///
/// # Example
///
/// ```ignore
/// use tacet::{timing_test_checked, Outcome};
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
///         Outcome::Pass { leak_probability, .. } |
///         Outcome::Fail { leak_probability, .. } |
///         Outcome::Inconclusive { leak_probability, .. } => {
///             println!("Leak probability: {:.1}%", leak_probability * 100.0);
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

fn expand_timing_test(input: TimingTestInput, _checked: bool) -> proc_macro2::TokenStream {
    let TimingTestInput {
        oracle,
        baseline,
        sample,
        measure,
    } = input;

    // Default oracle if not specified - use AdjacentNetwork attacker model with 30s time budget
    let oracle_expr = oracle.unwrap_or_else(|| {
        syn::parse_quote!(::tacet::TimingOracle::for_attacker(
            ::tacet::AttackerModel::AdjacentNetwork
        )
        .time_budget(::std::time::Duration::from_secs(30)))
    });

    // Generate the timing test code - both macros now return Outcome directly
    quote! {
        {
            // Create InputPair from baseline and sample closures
            let __inputs = ::tacet::helpers::InputPair::new(
                #baseline,
                #sample,
            );

            // Run the test with the new API
            #oracle_expr.test(__inputs, #measure)
        }
    }
}
