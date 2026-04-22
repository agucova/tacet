# Tacet cross-tool runtime benchmark

_Inputs: /Users/agucova/repos/tacet/paper/author-response/crypto-cross-tool-runtime/runtime-tier1-N10000/results.csv, /Users/agucova/repos/tacet/paper/author-response/crypto-cross-tool-runtime/runtime-tier1-N30000/results.csv, /Users/agucova/repos/tacet/paper/author-response/crypto-cross-tool-runtime/runtime-tier1-N50000/results.csv, /Users/agucova/repos/tacet/paper/author-response/crypto-cross-tool-runtime/runtime-tier2/results.csv_
_Total rows: 1380_

## Table A — Per-primitive analysis latency at N=50,000

Median `decision_time_ms` per (tool, primitive) on identical blocked timing data. IQR in per-cell text; `n` = iterations.

| Tool      | AES-128    | AES-256-GCM | ChaCha20-Poly1305 | Ed25519    | ML-KEM-768 | SHA3-256   | X25519     |
|-----------|------------|-------------|-------------------|------------|------------|------------|------------|
| dudect    | 25 ms (n=10) | 37 ms (n=10) | 38 ms (n=10)      | 46 ms (n=10) | 46 ms (n=10) | 41 ms (n=10) | 46 ms (n=10) |
| rtlf      | 76.22 s (n=10) | 79.42 s (n=10) | 78.69 s (n=10)    | 97.83 s (n=10) | 101.64 s (n=10) | 79.54 s (n=10) | 109.22 s (n=10) |
| silent    | 1.76 s (n=10) | 2.06 s (n=10) | 1.58 s (n=10)     | 1.88 s (n=10) | 1.78 s (n=10) | 1.82 s (n=10) | 2.22 s (n=10) |
| tacet     | 1.41 s (n=10) | 1.43 s (n=10) | 1.48 s (n=10)     | 1.78 s (n=10) | 1.88 s (n=10) | 1.47 s (n=10) | 2.06 s (n=10) |
| tlsfuzzer | 329 ms (n=10) | 334 ms (n=10) | 339 ms (n=10)     | 350 ms (n=10) | 342 ms (n=10) | 336 ms (n=10) | 351 ms (n=10) |
| tvla      | 0.0 ms (n=10) | 0.0 ms (n=10) | 0.0 ms (n=10)     | 0.0 ms (n=10) | 0.0 ms (n=10) | 0.0 ms (n=10) | 0.0 ms (n=10) |

## Table B — Scaling: Tier-1 aggregate decision latency vs. N

Median `decision_time_ms` per tool across all 7 Tier-1 primitives, with percentile bootstrap 95 % CI.

| Tool      | N=10,000               | N=30,000               | N=50,000               |
|-----------|------------------------|------------------------|------------------------|
| dudect    | 8.0 ms [8.0 ms-9.0 ms] | 24 ms [23 ms-27 ms]    | 41 ms [38 ms-46 ms]    |
| rtlf      | 19.69 s [18.56 s-21.71 s] | 38.10 s [37.34 s-48.99 s] | 81.20 s [79.38 s-96.06 s] |
| silent    | 907 ms [866 ms-939 ms] | 1.45 s [1.37 s-1.47 s] | 1.81 s [1.74 s-1.91 s] |
| tacet     | 214 ms [198 ms-261 ms] | 736 ms [717 ms-908 ms] | 1.49 s [1.47 s-1.75 s] |
| tlsfuzzer | 282 ms [280 ms-283 ms] | 312 ms [310 ms-314 ms] | 340 ms [336 ms-344 ms] |
| tvla      | 0.0 ms [0.0 ms-0.0 ms] | 0.0 ms [0.0 ms-0.0 ms] | 0.0 ms [0.0 ms-0.0 ms] |

## Table C — MARVIN end-to-end (Tier 2)

Median end-to-end wall-clock on RustCrypto RSA-1024 PKCS#1v1.5 decrypt (CVE-2023-49092). `End-to-end = collection + decision`. Collection is shared across tools per iteration. Detection rate shows whether the tool actually caught the leak — a fast verdict that misses the CVE would be called out here.

| Tool       | N      | Collection (s) | Decision (ms) | End-to-end (s) | Detection rate (detect / n) |
|------------|-------:|---------------:|--------------:|---------------:|:----------------------------|
| dudect     | 50,000 |          18.56 |         47 ms |          18.61 | 10/20 |
| rtlf       | 50,000 |          18.56 |      124.19 s |         143.04 | 12/20 |
| silent     | 50,000 |          18.56 |        2.22 s |          20.75 | 12/20 |
| tacet      | 50,000 |          18.56 |        2.53 s |          21.08 | 0/20 (**all MISS**) |
| tlsfuzzer  | 50,000 |          18.56 |        352 ms |          18.93 | 13/20 |
| tvla       | 50,000 |          18.56 |        0.0 ms |          18.56 | 1/20 |

## Sanity

**Per-tool outcome distribution across all rows:**

| Tool | pass | fail | inconclusive | other |
|------|-----:|-----:|-------------:|------:|
| dudect | 59 | 171 | 0 | 0 |
| rtlf | 134 | 96 | 0 | 0 |
| silent | 99 | 94 | 0 | 37 |
| tacet | 195 | 0 | 35 | 0 |
| tlsfuzzer | 19 | 211 | 0 | 0 |
| tvla | 196 | 34 | 0 | 0 |

**Zero-decision-time rows (excluding tacet — flag if competitor):** tvla=230

## Drop-in paragraph for USENIX response

> On seven Tier-1 constant-time cryptographic primitives (AMD EPYC 32 vCPU, N = 50,000 samples/class, 10 iterations on identical raw timing data), median per-tool analysis latency was: dudect 41 ms, rtlf 81.20 s, silent 1.81 s, tacet 1.49 s, tlsfuzzer 340 ms, tvla 0.0 ms. Sample collection (shared across tools) took 0.21 s (median) per primitive. On the MARVIN RSA-1024 leaky test (CVE-2023-49092) at N = 50 000, end-to-end time-to-verdict was: dudect 18.61 s (10/20 detected); rtlf 143.04 s (12/20 detected); silent 20.75 s (12/20 detected); tacet 21.08 s (0/20 detected); tlsfuzzer 18.93 s (13/20 detected); tvla 18.56 s (1/20 detected).
