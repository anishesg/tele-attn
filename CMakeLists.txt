cmake_minimum_required(VERSION 3.22)
project(tele_attn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

find_package(CUDAToolkit REQUIRED)

# Compile flags
set(CUDA_COMMON_FLAGS
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -use_fast_math
    -Xcompiler=-Wall
)

# Static library of all kernels
add_library(tele_attn_lib STATIC
    src/representative.cu
    src/promotion.cu
    src/reference.cu
    src/fused_tele_attn.cu
)
target_include_directories(tele_attn_lib PUBLIC src)
target_compile_options(tele_attn_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMMON_FLAGS}>)
target_link_libraries(tele_attn_lib PUBLIC CUDA::cudart CUDA::cuda_driver)

# Correctness tests
add_executable(test_correctness tests/test_correctness.cu)
target_include_directories(test_correctness PRIVATE src)
target_compile_options(test_correctness PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMMON_FLAGS}>)
target_link_libraries(test_correctness PRIVATE tele_attn_lib CUDA::cudart)

# Memory-quality Pareto benchmark
add_executable(bench_tradeoff benchmarks/bench_tradeoff.cu)
target_include_directories(bench_tradeoff PRIVATE src)
target_compile_options(bench_tradeoff PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMMON_FLAGS}>)
target_link_libraries(bench_tradeoff PRIVATE tele_attn_lib CUDA::cudart)

# Latency benchmark
add_executable(bench_latency benchmarks/bench_latency.cu)
target_include_directories(bench_latency PRIVATE src)
target_compile_options(bench_latency PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMMON_FLAGS}>)
target_link_libraries(bench_latency PRIVATE tele_attn_lib CUDA::cudart)

# Build type defaults
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
