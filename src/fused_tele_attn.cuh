#pragma once
#include <cuda_fp16.h>
#include "cache.cuh"

// Fused multi-resolution attention: attends over all cache levels with
// multiplicity-weighted online softmax, scores the new entry, appends it
// to level 0, and triggers promotion if level 0 is full.
//
// Q:   [num_heads, head_dim] fp16  (single new query token)
// new_k, new_v: [num_heads, head_dim] fp16  (new KV to append after attention)
// out: [num_heads, head_dim] fp16  (attention output, allocated by caller)
void launch_fused_tele_attn(
    const __half*      Q,
    const __half*      new_k,
    const __half*      new_v,
    __half*            out,
    MultiResolutionCache* cache,
    cudaStream_t       stream = 0
);
