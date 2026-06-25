#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include "cuda_kernels.cuh"
#include "tensor.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

using namespace inference;

void test_rmsnorm() {
    std::cout << "[TEST] Running RMSNorm Kernel Validation...\n";
    int rows = 4;
    int cols = 128;
    std::vector<float> in(rows * cols, 1.0f);
    std::vector<float> weight(cols, 1.0f);
    std::vector<float> out(rows * cols, 0.0f);

#if defined(__CUDA_ARCH__) || defined(__NVCC__) || defined(CUDA_FOUND)
    float *d_in, *d_weight, *d_out;
    cudaMalloc(&d_in, rows * cols * sizeof(float));
    cudaMalloc(&d_weight, cols * sizeof(float));
    cudaMalloc(&d_out, rows * cols * sizeof(float));

    cudaMemcpy(d_in, in.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    launch_rmsnorm(d_out, d_in, d_weight, rows, cols, 1e-6f);
    cudaDeviceSynchronize();

    cudaMemcpy(out.data(), d_out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_weight);
    cudaFree(d_out);
#else
    launch_rmsnorm(out.data(), in.data(), weight.data(), rows, cols, 1e-6f);
#endif

    for (float v : out) {
        assert(std::abs(v - 1.0f) < 1e-4);
    }
    std::cout << "  RMSNorm output matches expected normalized values!\n";
}

void test_swiglu() {
    std::cout << "[TEST] Running SwiGLU Fused Activation Validation...\n";
    int numel = 64;
    std::vector<float> gate(numel, 0.0f);
    std::vector<float> up(numel, 5.0f);
    std::vector<float> out(numel, 0.0f);

#if defined(__CUDA_ARCH__) || defined(__NVCC__) || defined(CUDA_FOUND)
    float *d_gate, *d_up, *d_out;
    cudaMalloc(&d_gate, numel * sizeof(float));
    cudaMalloc(&d_up, numel * sizeof(float));
    cudaMalloc(&d_out, numel * sizeof(float));

    cudaMemcpy(d_gate, gate.data(), numel * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, up.data(), numel * sizeof(float), cudaMemcpyHostToDevice);

    launch_swiglu(d_out, d_gate, d_up, numel);
    cudaDeviceSynchronize();

    cudaMemcpy(out.data(), d_out, numel * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_out);
#else
    launch_swiglu(out.data(), gate.data(), up.data(), numel);
#endif

    for (float v : out) {
        assert(std::abs(v - 0.0f) < 1e-5);
    }
    std::cout << "  SwiGLU output matches mathematical definition!\n";
}

void test_int4_awq_gemm() {
    std::cout << "[TEST] Running INT4 AWQ Dequantize-GEMM Validation...\n";
    int M = 2, N = 4, K = 32, group_size = 32;
    std::vector<float> act(M * K, 1.0f);
    std::vector<uint8_t> packed_w(N * (K / 2), 0x22);
    std::vector<float> scales(N * 1, 0.5f);
    std::vector<float> zeros(N * 1, 0.0f);
    std::vector<float> out(M * N, 0.0f);

#if defined(__CUDA_ARCH__) || defined(__NVCC__) || defined(CUDA_FOUND)
    float *d_act, *d_scales, *d_zeros, *d_out;
    uint8_t *d_packed_w;
    cudaMalloc(&d_act, M * K * sizeof(float));
    cudaMalloc(&d_packed_w, N * (K / 2) * sizeof(uint8_t));
    cudaMalloc(&d_scales, N * 1 * sizeof(float));
    cudaMalloc(&d_zeros, N * 1 * sizeof(float));
    cudaMalloc(&d_out, M * N * sizeof(float));

    cudaMemcpy(d_act, act.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_packed_w, packed_w.data(), N * (K / 2) * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, scales.data(), N * 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zeros, zeros.data(), N * 1 * sizeof(float), cudaMemcpyHostToDevice);

    launch_awq_int4_gemm(d_out, d_act, d_packed_w, d_scales, d_zeros, M, N, K, group_size);
    cudaDeviceSynchronize();

    cudaMemcpy(out.data(), d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_act);
    cudaFree(d_packed_w);
    cudaFree(d_scales);
    cudaFree(d_zeros);
    cudaFree(d_out);
#else
    launch_awq_int4_gemm(out.data(), act.data(), packed_w.data(), scales.data(), zeros.data(), M, N, K, group_size);
#endif

    for (float v : out) {
        assert(std::abs(v - 32.0f) < 1e-4);
    }
    std::cout << "  INT4 AWQ GEMM accurately dequantizes and computes matrix multiplication!\n";
}

int main() {
    std::cout << "=== Running CUDA Inference Engine Kernel Test Suite ===\n";
    test_rmsnorm();
    test_swiglu();
    test_int4_awq_gemm();
    std::cout << "All Inference Engine tests passed cleanly!\n";
    return 0;
}
