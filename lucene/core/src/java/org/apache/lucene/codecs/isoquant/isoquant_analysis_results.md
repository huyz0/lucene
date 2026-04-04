# IsoQuant vs Baseline: Benchmark Analysis Report

This document investigates the results of the SIFTSmall vector benchmark (10,000 vectors, 128 dimensions), analyzing the tradeoffs between indexing overhead, search speed, and storage footprint across different quantization algorithms.

## Benchmark Results Recap
```text
┌──────────────────────────────────────────────────────────────────────────────────────┐
│              Recall, Speed & Storage Benchmark (d=128, N=10000)                      │
├──────────────────────────┬────────────┬──────────────┬─────────────────┬─────────────┤
│ Configuration            │   Recall   │ Index Time   │ Search Time     │ Storage Size│
├──────────────────────────┼────────────┼──────────────┼─────────────────┼─────────────┤
│ Baseline (Raw FP32)      │   0.939    │   1185 ms    │      41 ms      │    5351 KB  │
│ Baseline (INT4 SQ)       │   0.830    │   1163 ms    │     104 ms      │    6134 KB  │
│ Baseline (INT8 SQ)       │   0.931    │   1028 ms    │      36 ms      │    6759 KB  │
│ IsoQuant (4-bit LM)      │   0.852    │   1033 ms    │      19 ms      │    6054 KB  │
│ IsoQuant (8-bit LM)      │   0.933    │   1240 ms    │      12 ms      │    6679 KB  │
└──────────────────────────┴────────────┴──────────────┴─────────────────┴─────────────┘
```

---

## 1. Search Time: Why is IsoQuant so Fast?

The most striking result is **Search Time**, where IsoQuant dramatically outpaces the baseline formats (12ms for 8-bit vs 36ms for INT8 SQ).

*   **Memory Alignment & AVX-512**: IsoQuant 8-bit is the fastest overall. Unlike bit-packing or generic scalar quantizers, an 8-bit IsoQuant vector corresponds perfectly to native 1-byte memory alignment (a `byte[]`). This allows it to funnel directly into Java's Panama SIMD `FloatVector.SPECIES_512` without any bitmasking overhead.
*   **The Cost of Bit-Unpacking**: Notice that `Baseline (INT4 SQ)` takes a staggering 104ms. Slicing upper and lower nibbles (4-bits) efficiently in Java adds cycle overhead. IsoQuant 4-bit mitigates this massively using SIMD `VectorShuffle` and avoiding intermediate scalar decoding, completing the same search in 19ms.
*   **No Pre-Query Bucket Math**: Baseline SQ formats are dynamic per-vector min/max buckets, meaning the search loop has to apply scaling algebra per coordinate mathematically. IsoQuant applies one rotation at the start, and then simple Lloyd-Max lookups during the scan. 

## 2. Storage Size: Why do Quantized Formats Take *More* Space?

At first glance, it seems counterintuitive that compressed formats (6.0MB – 6.7MB) consume more disk space than the uncompressed Raw FP32 baseline (5.3MB).

*   **Dual Storage Architecture**: Lucene's modern HNSW indexers (both `Lucene104` and `IsoQuant`) store the **quantized representations alongside the original FP32 vectors**. The `TestIsoQuantRecall` builds the graph using the compressed formats for navigating down the layers but retains the original flat FP32 vectors to calculate exact proximity rescoring on the final leaf candidates.
*   **The Math Checks Out**: Raw FP32 storage is ~5000KB. 
    *   An 8-bit vector drops the precision by 4x, equating to ~1250KB for 10,000 vectors. Thus, `5351 KB (FP32) + 1250 KB (8-bit payload) + minor metadata ≈ 6679 KB`.
    *   A 4-bit vector corresponds to ~625KB payload. `5351 KB (FP32) + 625 KB (4-bit payload) + minor metadata ≈ 6054 KB`.
*   **IsoQuant Metadata Density**: Notice `IsoQuant (8-bit)` uses slightly less space than `Baseline (INT8 SQ)` (6679 KB vs 6759 KB). ISO codes the global centroids uniformly once mathematically. Standard Lucene SQ stores unique `(min, scale)` float tuples for *every single vector*, eating a fractionally higher chunk of disk space.

## 3. Index Time: Graph Construction Economics

Index Time encompasses quantizing the vectors, persisting them, and building the HNSW navigational mesh (which requires computing millions of distance scores).

*   **Encoding Overhead**: IsoQuant 8-bit indexing (1240 ms) is mildly slower than Baseline INT8 SQ (1028 ms). This is because IsoQuant vectors must be mathematically rotated via the orthogonal matrix (`RotorQuant`) and computationally searched to find their closest Lloyd-Max centroids globally during insertion. Standard SQ just computes an array max/min element scale factor scalar.
*   **Fast Distance Math Saves Time**: Notice that `IsoQuant (4-bit)` builds relatively quickly (1033ms). While quantizing takes a hit during indexing, computing HNSW edge lengths relies on the `VectorScorer`. Because IsoQuant 4-bit is extremely fast during its scoring callbacks compared to standard 4-bit, the graph construction itself completes faster, somewhat negating its own initial mathematical encoding delay. 

### Conclusion

The data confirms the architectural goal: **IsoQuant aggressively shifts overhead out of the query execution path**. While indexing incurs a minor statistical mapping cost, the queries themselves execute up to 3x faster than traditional Lucene scalar quantization by bypassing per-vector floating-point arithmetic and deeply leveraging aligned memory and SIMD pipelines.
