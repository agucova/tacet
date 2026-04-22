# Synthetic vs. real-AWS timing characterisation

Raw streams: synthetic AR(1) × LogNormal from `crates/tacet-bench/src/synthetic.rs` and real crypto timings collected on `c8a.4xlarge` (16-vCPU AMD EPYC 9R45, rdtsc resolution 0.385 ns). Row counts = baseline + test × primitives × iterations × seeds. Reported statistic is the median across rows in each group. Tail reported in *robust* σ units (MAD × 1.4826) to prevent outlier inflation. Streams with <100 unique timing values are flagged `[quant]` and excluded from the aggregate (these are sub-100-cycle ops at the rdtsc resolution floor — the paper's `Unmeasurable` category).


### Synthetic (nominal AR(1) coefficient φ)

| Group | n | ρ₁ | ρ₅ | ρ₁₀ | P(z>3σ)_robust | p99.9 (σ_MAD) | IACT τ̂ | PW block (SB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synth null φ=0.0 | 10 | 0.002 | -0.003 | 0.003 | 0.0047 | 3.68 | 1.02 | 0.9 |
| synth null φ=0.3 | 10 | 0.147 | -0.002 | 0.004 | 0.0063 | 3.89 | 1.42 | 11.0 |
| synth null φ=0.6 | 10 | 0.292 | 0.033 | 0.003 | 0.0063 | 4.01 | 2.49 | 24.7 |
| synth null φ=0.8 | 10 | 0.392 | 0.156 | 0.051 | 0.0063 | 3.92 | 4.75 | 48.0 |
| synth shift φ=1sigma | 40 | 0.222 | 0.021 | 0.005 | 0.0059 | 3.87 | 1.87 | 16.9 |
| synth tail φ=1sigma | 40 | 0.159 | 0.016 | 0.001 | 0.0192 | 5.17 | 1.59 | 14.2 |

### AWS c8a by primitive

| Group | n | ρ₁ | ρ₅ | ρ₁₀ | P(z>3σ)_robust | p99.9 (σ_MAD) | IACT τ̂ | PW block (SB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aws-c8a idle c_libraries-libsodium-ed25519_sign | 6 | 0.010 | 0.007 | 0.009 | 0.1018 | 156.09 | 1.34 | 1.6 |
| aws-c8a idle dalek-x25519-scalar_mult | 6 | 0.031 | 0.016 | 0.019 | 0.1555 | 111.13 | 1.80 | 3.3 |
| aws-c8a idle pqcrypto-kyber-kyber768_decapsulate | 6 | 0.002 | -0.002 | 0.001 | 0.0214 | 59.53 | 1.06 | 1.5 |
| aws-c8a idle ring-aes_gcm-aes256gcm_seal [quant] | 6 | -0.005 | -0.001 | -0.001 | 0.0000 | 0.00 | 1.00 | 7.6 |
| aws-c8a idle rustcrypto-aes-aes128_encrypt [quant] | 6 | -0.007 | -0.001 | 0.000 | 0.0000 | 0.00 | 1.00 | 2.3 |
| aws-c8a idle rustcrypto-chacha20poly1305-encrypt [quant] | 6 | -0.000 | -0.000 | -0.000 | 0.0000 | 0.00 | 1.00 | 0.2 |
| aws-c8a idle rustcrypto-sha3-sha3_256 [quant] | 6 | -0.000 | -0.000 | -0.000 | 0.0000 | 0.00 | 1.00 | 0.2 |
| aws-c8a loaded c_libraries-libsodium-ed25519_sign | 6 | -0.003 | -0.003 | -0.003 | 0.2011 | 91169.59 | 1.00 | 0.7 |
| aws-c8a loaded dalek-x25519-scalar_mult | 6 | -0.002 | -0.002 | -0.002 | 0.2649 | 16464.09 | 1.00 | 0.6 |
| aws-c8a loaded pqcrypto-kyber-kyber768_decapsulate | 6 | -0.003 | -0.003 | -0.003 | 0.0555 | 16903.27 | 1.00 | 0.7 |
| aws-c8a loaded ring-aes_gcm-aes256gcm_seal [quant] | 6 | 0.017 | 0.002 | 0.003 | 0.0000 | 0.00 | 1.04 | 5.6 |
| aws-c8a loaded rustcrypto-aes-aes128_encrypt [quant] | 6 | 0.001 | -0.001 | 0.002 | 0.0000 | 0.00 | 1.03 | 2.1 |
| aws-c8a loaded rustcrypto-chacha20poly1305-encrypt [quant] | 6 | -0.000 | -0.000 | -0.000 | 0.1035 | 322.41 | 1.00 | 0.2 |
| aws-c8a loaded rustcrypto-sha3-sha3_256 [quant] | 6 | 0.000 | 0.001 | -0.000 | 0.0000 | 0.00 | 1.00 | 0.2 |

### AWS c8a aggregate (continuous streams only)

| Group | n | ρ₁ | ρ₅ | ρ₁₀ | P(z>3σ)_robust | p99.9 (σ_MAD) | IACT τ̂ | PW block (SB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aws-c8a idle (continuous streams) | 18 | 0.018 | 0.007 | 0.012 | 0.1018 | 111.13 | 1.39 | 2.4 |
| aws-c8a loaded (continuous streams) | 18 | -0.003 | -0.003 | -0.003 | 0.2011 | 22534.81 | 1.00 | 0.7 |

## Sanity checks

- AR(1) ρ=0.6 theoretical on the underlying noise: ρ₁=0.6, ρ₅≈0.078, IACT≈(1+ρ)/(1−ρ)=4.0.
- AR(1) ρ=0.8 theoretical on the underlying noise: ρ₁=0.8, ρ₅≈0.328, IACT=9.0.
- The synthetic generator adds AR(1) noise *multiplicatively in log-space* with a scale factor of 0.10, so measured ρ₁ on the final stream is diluted relative to nominal φ. This is expected — what matters is whether measured AWS ρ₁ lands inside the *measured* synthetic range, not the nominal one.
- IID theoretical: ρ₁≈0, IACT≈1, Gaussian p99.9≈3.09σ.
