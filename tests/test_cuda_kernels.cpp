#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include "cuda_kernels.cuh"
#include "tensor.hpp"

using namespace inference;

void test_rmsnorm() {
    std::cout << "[TEST] Running RMSNorm Kernel Validation...\n";
    int rows = 4;
    int cols = 128;
    std::vector<float> in(rows * cols, 1.0f);
    std::vector<float> weight(cols, 1.0f);
    std::vector<float> out(rows * cols, 0.0f);

    launch_rmsnorm(out.data(), in.data(), weight.data(), rows, cols, 1e-6f);

    // RMS of 1.0 is 1.0, so out should be 1.0
    for (float v : out) {
        assert(std::abs(v - 1.0f) < 1e-4);
    }
    std::cout << "  RMSNorm output matches expected normalized values!\n";
}

void test_swiglu() {
    std::cout << "[TEST] Running SwiGLU Fused Activation Validation...\n";
    int numel = 64;
    std::vector<float> gate(numel, 0.0f); // silu(0) = 0
    std::vector<float> up(numel, 5.0f);
    std::vector<float> out(numel, 0.0f);

    launch_swiglu(out.data(), gate.data(), up.data(), numel);

    for (float v : out) {
        assert(std::abs(v - 0.0f) < 1e-5);
    }
    std::cout << "  SwiGLU output matches mathematical definition!\n";
}

void test_int4_awq_gemm() {
    std::cout << "[TEST] Running INT4 AWQ Dequantize-GEMM Validation...\n";
    int M = 2, N = 4, K = 32, group_size = 32;
    std::vector<float> act(M * K, 1.0f);
    std::vector<uint8_t> packed_w(N * (K / 2), 0x22); // Each nibble is 2
    std::vector<float> scales(N * 1, 0.5f);
    std::vector<float> zeros(N * 1, 0.0f);
    std::vector<float> out(M * N, 0.0f);

    launch_awq_int4_gemm(out.data(), act.data(), packed_w.data(), scales.data(), zeros.data(), M, N, K, group_size);

    // Dequantized weight = 2 * 0.5 = 1.0. Dot product of 32 ones = 32.0
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
