#ifndef CUDA_ENGINE_KERNELS_CUH
#define CUDA_ENGINE_KERNELS_CUH

#include <cstdint>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

namespace inference {

// 1. RMSNorm Kernel (warp-shuffle reduction)
void launch_rmsnorm(float* out, const float* in, const float* weight, int rows, int cols, float eps, void* stream = nullptr);

// 2. Rotary Position Embedding (RoPE) Kernel
void launch_rope(float* q, float* k, const float* cos_sin, int batch_size, int seq_len, int num_heads, int head_dim, void* stream = nullptr);

// 3. SwiGLU Fused Activation Kernel: out = SiLU(gate) * up
void launch_swiglu(float* out, const float* gate, const float* up, int numel, void* stream = nullptr);

// 4. Weight-Only INT4 AWQ Dequantize-GEMM Kernel
void launch_awq_int4_gemm(float* out, const float* act, const uint8_t* packed_w, const float* scales, const float* zeros,
                          int M, int N, int K, int group_size, void* stream = nullptr);

} // namespace inference

#endif // CUDA_ENGINE_KERNELS_CUH
