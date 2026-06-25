cmake_minimum_required(VERSION 3.18)
project(CUDALlmInferenceEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

include_directories(include)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_compile_definitions(CUDA_FOUND)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -arch=sm_86")
    set(ENGINE_SOURCES src/cuda_kernels.cu)
else()
    message(STATUS "CUDA not found - compiling with Host C++ CPU Fallback")
    set(ENGINE_SOURCES src/cuda_kernels.cpp)
endif()

# Test Executables
add_executable(test_cuda_kernels tests/test_cuda_kernels.cpp ${ENGINE_SOURCES})
add_executable(test_paged_cache tests/test_paged_cache.cpp ${ENGINE_SOURCES})

# Benchmark Executables
add_executable(bench_inference benchmarks/bench_inference_throughput.cpp ${ENGINE_SOURCES})

enable_testing()
add_test(NAME CudaKernelsTest COMMAND test_cuda_kernels)
add_test(NAME PagedCacheTest COMMAND test_paged_cache)
