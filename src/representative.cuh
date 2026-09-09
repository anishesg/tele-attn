#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include "cache.cuh"

// Select the KV entry with the highest importance score from a window of W consecutive
// entries, writing the result to an output array. Each thread block handles one head
// of one window. The output multiplicity equals the sum of all input multiplicities.
//
// grid:  (num_windows, num_heads)
// block: (32)  -- one warp
__global__ void representative_selection_kernel(
    const __half* __restrict__ in_k,         // [count, num_heads, head_dim] fp16
    const __half* __restrict__ in_v,         // [count, num_heads, head_dim] fp16
    const float*  __restrict__ in_importance,// [count] float32
    const uint16_t* __restrict__ in_mult,    // [count] uint16
    __half* __restrict__       out_k,        // [num_windows, num_heads, head_dim] fp16
    __half* __restrict__       out_v,        // [num_windows, num_heads, head_dim] fp16
    float*  __restrict__       out_importance,// [num_windows] float32
    uint16_t* __restrict__     out_mult,     // [num_windows] uint16
    int window_size,
    int num_heads,
    int head_dim
);

// Host wrapper: selects representatives for all windows in a contiguous input array.
// Outputs num_windows = count / window_size representatives.
void launch_representative_selection(
    const __half*   in_k,
    const __half*   in_v,
    const float*    in_importance,
    const uint16_t* in_mult,
    __half*         out_k,
    __half*         out_v,
    float*          out_importance,
    uint16_t*       out_mult,
    int count,          // must be divisible by window_size
    int window_size,
    int num_heads,
    int head_dim,
    cudaStream_t stream = 0
);
