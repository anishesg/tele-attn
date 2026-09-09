#include "representative.cuh"
#include <cuda_fp16.h>
#include <cstdio>

// One warp processes one (window, head) pair.
// Thread 0 copies the selected K/V vectors to output.
__global__ void representative_selection_kernel(
    const __half*   __restrict__ in_k,
    const __half*   __restrict__ in_v,
    const float*    __restrict__ in_importance,
    const uint16_t* __restrict__ in_mult,
    __half*   __restrict__ out_k,
    __half*   __restrict__ out_v,
    float*    __restrict__ out_importance,
    uint16_t* __restrict__ out_mult,
    int window_size,
    int num_heads,
    int head_dim)
{
    const int window_idx = blockIdx.x;
    const int head_idx   = blockIdx.y;
    const int lane       = threadIdx.x;  // 0..31

    const int window_start = window_idx * window_size;

    // --- Phase 1: find argmax importance and sum multiplicities via warp reduction ---
    float  best_val  = -1e38f;
    int    best_pos  = window_start;   // index into the flat input arrays
    uint32_t total_mult = 0;

    for (int i = lane; i < window_size; i += 32) {
        int pos = window_start + i;
        float imp = in_importance[pos];
        total_mult += static_cast<uint32_t>(in_mult[pos]);
        if (imp > best_val) {
            best_val = imp;
            best_pos = pos;
        }
    }

    // Warp reduction: find global argmax importance.
    for (int offset = 16; offset >= 1; offset >>= 1) {
        float  other_val = __shfl_xor_sync(0xffffffff, best_val, offset);
        int    other_pos = __shfl_xor_sync(0xffffffff, best_pos, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_pos = other_pos;
        }
    }

    // Warp reduction: sum multiplicities.
    for (int offset = 16; offset >= 1; offset >>= 1) {
        total_mult += __shfl_xor_sync(0xffffffff, total_mult, offset);
    }

    // After reduction, all lanes agree on best_pos and total_mult.

    // --- Phase 2: copy K and V of the winner to output ---
    // Source: in_k[best_pos, head_idx, :]
    // Dest:   out_k[window_idx, head_idx, :]
    const __half* src_k = in_k + (static_cast<ptrdiff_t>(best_pos) * num_heads + head_idx) * head_dim;
    const __half* src_v = in_v + (static_cast<ptrdiff_t>(best_pos) * num_heads + head_idx) * head_dim;
    __half*       dst_k = out_k + (static_cast<ptrdiff_t>(window_idx) * num_heads + head_idx) * head_dim;
    __half*       dst_v = out_v + (static_cast<ptrdiff_t>(window_idx) * num_heads + head_idx) * head_dim;

    for (int d = lane; d < head_dim; d += 32) {
        dst_k[d] = src_k[d];
        dst_v[d] = src_v[d];
    }

    // Thread 0 writes scalar outputs.
    if (lane == 0) {
        // Only write once per window: use head 0 as the writer for scalars.
        // But importance and multiplicity are per-window (not per-head), so only
        // one block per window should write. We use head_idx == 0 as the guard.
        if (head_idx == 0) {
            out_importance[window_idx] = best_val;
            out_mult[window_idx]       = static_cast<uint16_t>(min(total_mult, static_cast<uint32_t>(65535)));
        }
    }
}

void launch_representative_selection(
    const __half*   in_k,
    const __half*   in_v,
    const float*    in_importance,
    const uint16_t* in_mult,
    __half*         out_k,
    __half*         out_v,
    float*          out_importance,
    uint16_t*       out_mult,
    int count,
    int window_size,
    int num_heads,
    int head_dim,
    cudaStream_t stream)
{
    if (count == 0 || count % window_size != 0) {
        fprintf(stderr, "launch_representative_selection: count=%d not divisible by window_size=%d\n",
                count, window_size);
        return;
    }
    int num_windows = count / window_size;
    dim3 grid(num_windows, num_heads);
    dim3 block(32);

    representative_selection_kernel<<<grid, block, 0, stream>>>(
        in_k, in_v, in_importance, in_mult,
        out_k, out_v, out_importance, out_mult,
        window_size, num_heads, head_dim);
}
