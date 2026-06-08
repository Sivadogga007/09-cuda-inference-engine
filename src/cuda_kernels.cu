#include "cuda_kernels.cuh"
#include <cmath>
#include <algorithm>

#ifdef __CUDACC__
#include <cuda_runtime.h>

namespace inference {

// 1. RMSNorm CUDA Kernel
__global__ void rmsnorm_kernel(float* out, const float* in, const float* weight, int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row = in + row * cols;
    float* out_row = out + row * cols;

    float sum_sq = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        float val = in_row[col];
        sum_sq += val * val;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float s_variance;
    if (threadIdx.x == 0) {
        s_variance = rsqrtf((sum_sq / cols) + eps);
    }
    __syncthreads();

    float rsqrt_var = s_variance;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        out_row[col] = in_row[col] * rsqrt_var * weight[col];
    }
}

void launch_rmsnorm(float* out, const float* in, const float* weight, int rows, int cols, float eps, void* stream) {
    cudaStream_t s = stream ? reinterpret_cast<cudaStream_t>(stream) : 0;
    int threads = std::min(256, ((cols + 31) / 32) * 32);
    rmsnorm_kernel<<<rows, threads, 0, s>>>(out, in, weight, rows, cols, eps);
}

// 2. SwiGLU Fused Activation Kernel
__global__ void swiglu_kernel(float* out, const float* gate, const float* up, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float g = gate[idx];
        float silu_g = g / (1.0f + expf(-g));
        out[idx] = silu_g * up[idx];
    }
}

void launch_swiglu(float* out, const float* gate, const float* up, int numel, void* stream) {
    cudaStream_t s = stream ? reinterpret_cast<cudaStream_t>(stream) : 0;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads, 0, s>>>(out, gate, up, numel);
}

// 3. INT4 AWQ GEMM Kernel
__global__ void awq_gemm_kernel(float* out, const float* act, const uint8_t* packed_w,
                                const float* scales, const float* zeros,
                                int M, int N, int K, int group_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N

    if (row < M && col < N) {
        float acc = 0.0f;
        int num_groups = (K + group_size - 1) / group_size;

        for (int k = 0; k < K; ++k) {
            int g = k / group_size;
            float scale = scales[col * num_groups + g];
            float zero = zeros[col * num_groups + g];

            // Unpack 4-bit weight
            uint8_t byte_val = packed_w[col * (K / 2) + (k / 2)];
            int w_int = (k % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
            float dequant_w = static_cast<float>(w_int) * scale + zero;

            acc += act[row * K + k] * dequant_w;
        }
        out[row * N + col] = acc;
    }
}

void launch_awq_int4_gemm(float* out, const float* act, const uint8_t* packed_w, const float* scales, const float* zeros,
                          int M, int N, int K, int group_size, void* stream) {
    cudaStream_t s = stream ? reinterpret_cast<cudaStream_t>(stream) : 0;
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    awq_gemm_kernel<<<blocks, threads, 0, s>>>(out, act, packed_w, scales, zeros, M, N, K, group_size);
}

void launch_rope(float* q, float* k, const float* cos_sin, int batch_size, int seq_len, int num_heads, int head_dim, void* stream) {
    // RoPE kernel launcher
}

} // namespace inference

#else // Host / CPU Fallback Implementation

namespace inference {

void launch_rmsnorm(float* out, const float* in, const float* weight, int rows, int cols, float eps, void*) {
    for (int r = 0; r < rows; ++r) {
        const float* in_row = in + r * cols;
        float* out_row = out + r * cols;
        float sum_sq = 0.0f;
        for (int c = 0; c < cols; ++c) sum_sq += in_row[c] * in_row[c];
        float rsqrt_var = 1.0f / std::sqrt((sum_sq / cols) + eps);
        for (int c = 0; c < cols; ++c) out_row[c] = in_row[c] * rsqrt_var * weight[c];
    }
}

void launch_swiglu(float* out, const float* gate, const float* up, int numel, void*) {
    for (int i = 0; i < numel; ++i) {
        float g = gate[i];
        float silu_g = g / (1.0f + std::exp(-g));
        out[i] = silu_g * up[i];
    }
}

void launch_awq_int4_gemm(float* out, const float* act, const uint8_t* packed_w, const float* scales, const float* zeros,
                          int M, int N, int K, int group_size, void*) {
    int num_groups = (K + group_size - 1) / group_size;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                int g = k / group_size;
                float scale = scales[n * num_groups + g];
                float zero = zeros[n * num_groups + g];
                uint8_t byte_val = packed_w[n * (K / 2) + (k / 2)];
                int w_int = (k % 2 == 0) ? (byte_val & 0x0F) : ((byte_val >> 4) & 0x0F);
                float dequant_w = static_cast<float>(w_int) * scale + zero;
                acc += act[m * K + k] * dequant_w;
            }
            out[m * N + n] = acc;
        }
    }
}

void launch_rope(float*, float*, const float*, int, int, int, int, void*) {}

} // namespace inference

#endif
