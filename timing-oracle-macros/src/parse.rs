//! Parsing logic for timing_test! macro input.

use syn::parse::{Parse, ParseStream};
use syn::{Expr, ExprClosure, Ident, Result, Token};

/// Parsed input to the timing_test! macro.
pub struct TimingTestInput {
    /// Optional oracle configuration
    pub oracle: Option<Expr>,
    /// Required: baseline input generator (closure)
    pub baseline: ExprClosure,
    /// Required: sample input generator (closure)
    pub sample: ExprClosure,
    /// Required: measure body (closure)
    pub measure: ExprClosure,
}

impl Parse for TimingTestInput {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut oracle = None;
        let mut baseline: Option<ExprClosure> = None;
        let mut sample: Option<ExprClosure> = None;
        let mut measure = None;

        // Parse field: value pairs
        while !input.is_empty() {
            let field_name: Ident = input.parse()?;
            input.parse::<Token![:]>()?;

            match field_name.to_string().as_str() {
                "oracle" => {
                    if oracle.is_some() {
                        return Err(syn::Error::new(
                            field_name.span(),
                            "duplicate field `oracle`",
                        ));
                    }
                    oracle = Some(input.parse()?);
                }
                "baseline" => {
                    if baseline.is_some() {
                        return Err(syn::Error::new(
                            field_name.span(),
                            "duplicate field `baseline`",
                        ));
                    }
                    // Parse the expression first
                    let expr: Expr = input.parse()?;

                    // Verify it's a closure
                    match expr {
                        Expr::Closure(closure) => {
                            baseline = Some(closure);
                        }
                        _ => {
                            return Err(syn::Error::new_spanned(
                                &expr,
                                "`baseline` must be a closure.\n\n\
                                 Hint: Use `baseline: || expr` instead of `baseline: expr`.\n\n\
                                 This makes both `baseline` and `sample` symmetric - both are\n\
                                 closures that generate inputs.\n\n\
                                 Example:\n\
                                 timing_test! {\n\
                                     baseline: || [0u8; 32],\n\
                                     sample: || rand::random(),\n\
                                     measure: |input| { ... },\n\
                                 }",
                            ));
                        }
                    }
                }
                "sample" => {
                    if sample.is_some() {
                        return Err(syn::Error::new(
                            field_name.span(),
                            "duplicate field `sample`",
                        ));
                    }
                    // Parse the expression first
                    let expr: Expr = input.parse()?;

                    // Verify it's a closure
                    match expr {
                        Expr::Closure(closure) => {
                            sample = Some(closure);
                        }
                        _ => {
                            return Err(syn::Error::new_spanned(
                                &expr,
                                "`sample` must be a closure.\n\n\
                                 Hint: Use `sample: || expr` instead of `sample: expr`.\n\n\
                                 This is intentional - a closure makes it clear that the expression\n\
                                 is evaluated freshly for each sample, not captured once.\n\n\
                                 Example:\n\
                                 \n\
                                 // WRONG: Pre-evaluated value\n\
                                 let value = rand::random();\n\
                                 timing_test! { sample: value, ... }\n\
                                 \n\
                                 // CORRECT: Fresh value per sample\n\
                                 timing_test! { sample: || rand::random(), ... }",
                            ));
                        }
                    }
                }
                "measure" => {
                    if measure.is_some() {
                        return Err(syn::Error::new(
                            field_name.span(),
                            "duplicate field `measure`",
                        ));
                    }
                    // Parse the expression first
                    let expr: Expr = input.parse()?;

                    // Verify it's a closure
                    match expr {
                        Expr::Closure(closure) => {
                            measure = Some(closure);
                        }
                        _ => {
                            return Err(syn::Error::new_spanned(
                                &expr,
                                "`measure` must be a closure that takes the input.\n\n\
                                 Example:\n\
                                 measure: |input| { my_function(&input); }",
                            ));
                        }
                    }
                }
                unknown => {
                    return Err(syn::Error::new(
                        field_name.span(),
                        format!(
                            "unknown field `{}`\n\n\
                             Expected one of: `oracle`, `baseline`, `sample`, `measure`",
                            unknown
                        ),
                    ));
                }
            }

            // Parse trailing comma (optional for last field)
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        // Validate required fields
        let baseline = baseline.ok_or_else(|| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                "missing required field `baseline`\n\n\
                 The `baseline` field is a closure that generates the constant input\n\
                 used for the baseline class.\n\n\
                 Example:\n\
                 timing_test! {\n\
                     baseline: || [0u8; 32],\n\
                     sample: || rand::random::<[u8; 32]>(),\n\
                     measure: |input| { ... },\n\
                 }",
            )
        })?;

        let sample = sample.ok_or_else(|| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                "missing required field `sample`\n\n\
                 The `sample` field must be a closure that generates sample inputs.\n\n\
                 Example:\n\
                 timing_test! {\n\
                     baseline: || [0u8; 32],\n\
                     sample: || rand::random::<[u8; 32]>(),  // <-- Add this\n\
                     measure: |input| { ... },\n\
                 }",
            )
        })?;

        let measure = measure.ok_or_else(|| {
            syn::Error::new(
                proc_macro2::Span::call_site(),
                "missing required field `measure`\n\n\
                 The `measure` field is a closure that takes the input and performs\n\
                 the operation to be timed.\n\n\
                 Example:\n\
                 timing_test! {\n\
                     baseline: || [0u8; 32],\n\
                     sample: || rand::random::<[u8; 32]>(),\n\
                     measure: |input| { encrypt(&input); },  // <-- Add this\n\
                 }",
            )
        })?;

        Ok(TimingTestInput {
            oracle,
            baseline,
            sample,
            measure,
        })
    }
}
