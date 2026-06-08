#ifndef CUDA_ENGINE_TENSOR_PARALLEL_HPP
#define CUDA_ENGINE_TENSOR_PARALLEL_HPP

#include <iostream>
#include <vector>

#ifdef USE_NCCL
#include <nccl.h>
#endif

namespace inference {

class TensorParallelCommunicator {
public:
    TensorParallelCommunicator(int rank = 0, int world_size = 1)
        : rank_(rank), world_size_(world_size) {}

    void all_reduce_sum(float* buffer, size_t count, void* stream = nullptr) {
#ifdef USE_NCCL
        if (world_size_ > 1 && comm_) {
            cudaStream_t s = stream ? reinterpret_cast<cudaStream_t>(stream) : 0;
            ncclAllReduce(buffer, buffer, count, ncclFloat, ncclSum, comm_, s);
        }
#endif
    }

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }

private:
    int rank_{0};
    int world_size_{1};
#ifdef USE_NCCL
    ncclComm_t comm_{nullptr};
#endif
};

} // namespace inference

#endif // CUDA_ENGINE_TENSOR_PARALLEL_HPP
