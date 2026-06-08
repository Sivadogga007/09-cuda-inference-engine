#ifndef CUDA_ENGINE_TENSOR_HPP
#define CUDA_ENGINE_TENSOR_HPP

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <cstdint>
#include <cassert>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

namespace inference {

enum class DataType {
    FP32,
    FP16,
    INT8,
    INT4
};

inline size_t element_size(DataType type) {
    switch (type) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::INT8: return 1;
        case DataType::INT4: return 1; // Packed 2 per byte
    }
    return 4;
}

class Tensor {
public:
    Tensor() : data_(nullptr), is_cuda_(false), size_bytes_(0) {}

    Tensor(std::vector<int64_t> shape, DataType dtype, bool is_cuda = false)
        : shape_(std::move(shape)), dtype_(dtype), is_cuda_(is_cuda) {
        size_t total_elements = 1;
        for (int64_t s : shape_) total_elements *= s;
        size_bytes_ = (dtype == DataType::INT4) ? (total_elements + 1) / 2 : total_elements * element_size(dtype);

        if (is_cuda_) {
#ifdef __CUDACC__
            cudaMalloc(&data_, size_bytes_);
#else
            data_ = std::malloc(size_bytes_);
#endif
        } else {
            data_ = std::malloc(size_bytes_);
        }
    }

    ~Tensor() {
        if (data_) {
            if (is_cuda_) {
#ifdef __CUDACC__
                cudaFree(data_);
#else
                std::free(data_);
#endif
            } else {
                std::free(data_);
            }
            data_ = nullptr;
        }
    }

    // Disable copy, enable move
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_)), dtype_(other.dtype_),
          is_cuda_(other.is_cuda_), data_(other.data_), size_bytes_(other.size_bytes_) {
        other.data_ = nullptr;
        other.size_bytes_ = 0;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (data_) {
                if (is_cuda_) {
#ifdef __CUDACC__
                    cudaFree(data_);
#else
                    std::free(data_);
#endif
                } else {
                    std::free(data_);
                }
            }
            shape_ = std::move(other.shape_);
            dtype_ = other.dtype_;
            is_cuda_ = other.is_cuda_;
            data_ = other.data_;
            size_bytes_ = other.size_bytes_;
            other.data_ = nullptr;
            other.size_bytes_ = 0;
        }
        return *this;
    }

    void* data() { return data_; }
    const void* data() const { return data_; }
    const std::vector<int64_t>& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    bool is_cuda() const { return is_cuda_; }
    size_t size_bytes() const { return size_bytes_; }

    int64_t numel() const {
        int64_t count = 1;
        for (int64_t s : shape_) count *= s;
        return count;
    }

private:
    std::vector<int64_t> shape_;
    DataType dtype_{DataType::FP32};
    bool is_cuda_{false};
    void* data_{nullptr};
    size_t size_bytes_{0};
};

} // namespace inference

#endif // CUDA_ENGINE_TENSOR_HPP
