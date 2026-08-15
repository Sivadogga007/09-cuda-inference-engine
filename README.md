# High-Performance C++/CUDA LLM Inference Engine

[![CUDA Inference Engine CI](https://github.com/Sivadogga007/cuda-inference-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Sivadogga007/cuda-inference-engine/actions/workflows/ci.yml)
[![CUDA](https://img.shields.io/badge/CUDA-12.4%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance C++/CUDA LLM inference engine featuring custom FlashAttention-2 kernels, Paged KV-Cache with Radix-tree prefix caching, INT4 AWQ group-wise dequantize-GEMM, and 2-GPU NCCL tensor parallelism optimized for NVIDIA RTX A5000 (sm_86) GPUs.

---

## Key Features

1. **Custom CUDA Kernels (`src/cuda_kernels.cu`)**:
   - Warp-shuffle accelerated `RMSNorm`.
   - Rotary Position Embeddings (`RoPE`).
   - Fused `SwiGLU` activation (`SiLU(gate) * up`).
   - Weight-Only INT4 AWQ Dequantize-GEMM fused with FP16 accumulators.

2. **Paged KV-Cache & Prefix Caching (`include/paged_kv_cache.hpp`)**:
   - 16-token physical block allocations eliminating external memory fragmentation.
   - Reference-counted prefix caching enabling instant prompt prefix reuse across concurrent requests.

3. **Multi-GPU Tensor Parallelism (`include/tensor_parallel.hpp`)**:
   - 2-GPU NCCL All-Reduce for distributed inference across dual NVIDIA RTX A5000 GPUs.
