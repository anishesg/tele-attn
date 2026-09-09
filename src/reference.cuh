#pragma once
#include <cuda_fp16.h>
#include <cstdint>

// Dense FP16 reference attention operating on a flat KV cache with no telescoping.
// Computes: O[h] = softmax(Q[h] . K^T / sqrt(d)) . V  for each head h.
//
// grid:  (num_heads)
// block: (128)  -- 4 warps for memory-level parallelism on the head_dim copy
//
// Dynamic shared memory: (2 * head_dim) * sizeof(float) bytes per block
//   [0..head_dim)       : K tile (float)
//   [head_dim..2*head_dim): online softmax accumulator scratch (not actually stored,
//                            but reserved to keep the head_dim uniform)
__global__ void reference_attention_kernel(
    const __half* __restrict__ Q,         // [num_heads, head_dim]
    const __half* __restrict__ K,         // [seq_len, num_heads, head_dim]
    const __half* __restrict__ V,         // [seq_len, num_heads, head_dim]
    __half* __restrict__       out,       // [num_heads, head_dim]
    int seq_len,
    int num_heads,
    int head_dim,
    float inv_sqrt_d
);

// Host wrapper: allocates and launches reference attention.
// out must be pre-allocated as [num_heads * head_dim] fp16.
void launch_reference_attention(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half*       out,
    int seq_len,
    int num_heads,
    int head_dim,
    cudaStream_t stream = 0
);
