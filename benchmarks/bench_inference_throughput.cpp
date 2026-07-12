#include <iostream>
#include <vector>
#include <chrono>
#include "paged_kv_cache.hpp"
#include "cuda_kernels.cuh"

#if defined(__CUDACC__) || defined(CUDA_FOUND)
#include <cuda_runtime.h>
#endif

using namespace inference;

int main() {
    std::cout << "=======================================================================\n";
    std::cout << " PROJECT 9: C++/CUDA LLM INFERENCE ENGINE BENCHMARK SUITE\n";
    std::cout << "=======================================================================\n";

    int batch_size = 16;
    int seq_len = 512;
    int hidden_dim = 4096;

    std::cout << "  Configuration: Batch Size=" << batch_size << ", SeqLen=" << seq_len << ", Hidden=" << hidden_dim << "\n";

    // 1. RMSNorm Throughput
    std::vector<float> in(batch_size * hidden_dim, 1.0f);
    std::vector<float> weight(hidden_dim, 1.0f);
    std::vector<float> out(batch_size * hidden_dim, 0.0f);

#if defined(__CUDA_ARCH__) || defined(__NVCC__) || defined(CUDA_FOUND)
    float *d_in, *d_weight, *d_out;
    cudaMalloc(&d_in, batch_size * hidden_dim * sizeof(float));
    cudaMalloc(&d_weight, hidden_dim * sizeof(float));
    cudaMalloc(&d_out, batch_size * hidden_dim * sizeof(float));

    cudaMemcpy(d_in, in.data(), batch_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight.data(), hidden_dim * sizeof(float), cudaMemcpyHostToDevice);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        launch_rmsnorm(d_out, d_in, d_weight, batch_size, hidden_dim, 1e-6f);
    }
    cudaDeviceSynchronize();
    auto el_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - t0).count();

    cudaFree(d_in);
    cudaFree(d_weight);
    cudaFree(d_out);
#else
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        launch_rmsnorm(out.data(), in.data(), weight.data(), batch_size, hidden_dim, 1e-6f);
    }
    auto el_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - t0).count();
#endif

    double rmsnorm_lat_us = static_cast<double>(el_us) / 1000.0;
    std::cout << "  RMSNorm Kernel Latency:         " << rmsnorm_lat_us << " us / step\n";

    // 2. INT4 AWQ GEMM Throughput
    int M = batch_size, N = hidden_dim, K = hidden_dim, group_size = 128;
    std::vector<float> act(M * K, 1.0f);
    std::vector<uint8_t> packed_w(N * (K / 2), 0x33);
    std::vector<float> scales(N * (K / group_size), 0.25f);
    std::vector<float> zeros(N * (K / group_size), 0.0f);
    std::vector<float> gemm_out(M * N, 0.0f);

#if defined(__CUDA_ARCH__) || defined(__NVCC__) || defined(CUDA_FOUND)
    float *d_act, *d_scales, *d_zeros, *d_gemm_out;
    uint8_t *d_packed_w;
    cudaMalloc(&d_act, M * K * sizeof(float));
    cudaMalloc(&d_packed_w, N * (K / 2) * sizeof(uint8_t));
    cudaMalloc(&d_scales, N * (K / group_size) * sizeof(float));
    cudaMalloc(&d_zeros, N * (K / group_size) * sizeof(float));
    cudaMalloc(&d_gemm_out, M * N * sizeof(float));

    cudaMemcpy(d_act, act.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_packed_w, packed_w.data(), N * (K / 2) * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, scales.data(), N * (K / group_size) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zeros, zeros.data(), N * (K / group_size) * sizeof(float), cudaMemcpyHostToDevice);

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        launch_awq_int4_gemm(d_gemm_out, d_act, d_packed_w, d_scales, d_zeros, M, N, K, group_size);
    }
    cudaDeviceSynchronize();
    el_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - t0).count();

    cudaFree(d_act);
    cudaFree(d_packed_w);
    cudaFree(d_scales);
    cudaFree(d_zeros);
    cudaFree(d_gemm_out);
#else
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        launch_awq_int4_gemm(gemm_out.data(), act.data(), packed_w.data(), scales.data(), zeros.data(), M, N, K, group_size);
    }
    el_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - t0).count();
#endif

    double gemm_lat_ms = (static_cast<double>(el_us) / 100.0) / 1000.0;
    double tflops = (2.0 * M * N * K / 1e12) / (gemm_lat_ms / 1000.0);
    std::cout << "  INT4 AWQ GEMM Latency:          " << gemm_lat_ms << " ms (" << tflops << " TFLOPS effective)\n";

    std::cout << "=======================================================================\n";
    return 0;
}
