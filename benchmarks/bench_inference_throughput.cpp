#include <iostream>
#include <vector>
#include <chrono>
#include "paged_kv_cache.hpp"
#include "cuda_kernels.cuh"

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

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        launch_rmsnorm(out.data(), in.data(), weight.data(), batch_size, hidden_dim, 1e-6f);
    }
    auto el_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - t0).count();

    double rmsnorm_lat_us = static_cast<double>(el_us) / 1000.0;
    std::cout << "  RMSNorm Kernel Latency:         " << rmsnorm_lat_us << " us / step\n";

    // 2. INT4 AWQ GEMM Throughput
    int M = batch_size, N = hidden_dim, K = hidden_dim, group_size = 128;
    std::vector<float> act(M * K, 1.0f);
    std::vector<uint8_t> packed_w(N * (K / 2), 0x33);
    std::vector<float> scales(N * (K / group_size), 0.25f);
    std::vector<float> zeros(N * (K / group_size), 0.0f);
    std::vector<float> gemm_out(M * N, 0.0f);

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        launch_awq_int4_gemm(gemm_out.data(), act.data(), packed_w.data(), scales.data(), zeros.data(), M, N, K, group_size);
    }
    el_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - t0).count();

    double gemm_lat_ms = (static_cast<double>(el_us) / 100.0) / 1000.0;
    double tflops = (2.0 * M * N * K / 1e12) / (gemm_lat_ms / 1000.0);
    std::cout << "  INT4 AWQ GEMM Latency:          " << gemm_lat_ms << " ms (" << tflops << " TFLOPS effective)\n";

    std::cout << "=======================================================================\n";
    return 0;
}
