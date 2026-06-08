#include "cuda_kernels.cuh"
#include <cmath>
#include <algorithm>

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
