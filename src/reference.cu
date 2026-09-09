#include "reference.cuh"
#include <cuda_fp16.h>
#include <cmath>

static constexpr int REF_TILE = 64;

// Each block handles one head. Processes K/V in tiles of REF_TILE.
// Online softmax with float32 accumulation.
__global__ void reference_attention_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__       out,
    int seq_len,
    int num_heads,
    int head_dim,
    float inv_sqrt_d)
{
    extern __shared__ float smem[];
    float* tile_k = smem;                  // [REF_TILE * head_dim] -- reused for V

    const int head = blockIdx.x;
    const int lane = threadIdx.x;         // 0..blockDim.x-1
    const int bsz  = blockDim.x;

    // Load query for this head into registers.
    // Each thread holds head_dim / bsz elements (assumes head_dim divisible by bsz).
    // For generality, threads may overlap.
    const __half* q_ptr = Q + head * head_dim;

    // Accumulate output in float.
    float acc[128] = {};  // max head_dim = 128; zero-initialized
    float running_max = -1e38f;
    float running_denom = 0.f;

    for (int tile_start = 0; tile_start < seq_len; tile_start += REF_TILE) {
        int tile_len = min(REF_TILE, seq_len - tile_start);

        // Load K tile into shared memory: [tile_len, head_dim]
        for (int i = lane; i < tile_len * head_dim; i += bsz) {
            int entry = i / head_dim;
            int d     = i % head_dim;
            int pos   = tile_start + entry;
            tile_k[i] = __half2float(K[(pos * num_heads + head) * head_dim + d]);
        }
        __syncthreads();

        // Each thread computes QK^T for a subset of tile entries.
        // Store scores in registers (tile_len <= 64, fits).
        float scores[REF_TILE];
        for (int e = lane; e < tile_len; e += bsz) {
            float dot = 0.f;
            const float* k_e = tile_k + e * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                dot += __half2float(q_ptr[d]) * k_e[d];
            }
            scores[e] = dot * inv_sqrt_d;
        }
        __syncthreads();

        // Reduce max score across all threads for this tile (warp-level then block).
        // For simplicity, thread 0 does sequential reduction after all threads write
        // their scores to shared memory.

        // Use tile_k (now free after K is no longer needed for this tile) as temp.
        float* score_buf = smem;  // reuse; we're done with K tile
        if (lane < tile_len) {
            // Only the owning thread wrote scores[lane] in the loop above.
            // Recompute for the sequential path.
        }
        // Simpler: let each thread write its scores sequentially.
        for (int e = lane; e < tile_len; e += bsz) {
            score_buf[e] = scores[e];
        }
        __syncthreads();

        // Thread 0 performs online softmax update and V accumulation.
        if (lane == 0) {
            // Find tile max.
            float tile_max = -1e38f;
            for (int e = 0; e < tile_len; ++e) tile_max = fmaxf(tile_max, score_buf[e]);

            // Rescale old accumulator.
            float exp_shift = expf(running_max - fmaxf(running_max, tile_max));
            float new_max   = fmaxf(running_max, tile_max);

            float tile_denom = 0.f;
            float tile_weights[REF_TILE];
            for (int e = 0; e < tile_len; ++e) {
                tile_weights[e] = expf(score_buf[e] - new_max);
                tile_denom += tile_weights[e];
            }

            // Rescale accumulated output by exp(old_max - new_max).
            for (int d = 0; d < head_dim; ++d) {
                acc[d] *= exp_shift;
            }
            running_denom  = running_denom * exp_shift + tile_denom;
            running_max    = new_max;

            // Accumulate V.
            for (int e = 0; e < tile_len; ++e) {
                int pos = tile_start + e;
                float w = tile_weights[e];
                for (int d = 0; d < head_dim; ++d) {
                    acc[d] += w * __half2float(V[(pos * num_heads + head) * head_dim + d]);
                }
            }
        }
        __syncthreads();
    }

    // Thread 0 writes normalized output.
    if (lane == 0) {
        __half* out_ptr = out + head * head_dim;
        float inv_denom = (running_denom > 0.f) ? 1.f / running_denom : 0.f;
        for (int d = 0; d < head_dim; ++d) {
            out_ptr[d] = __float2half(acc[d] * inv_denom);
        }
    }
}

void launch_reference_attention(
    const __half* Q,
    const __half* K,
    const __half* V,
    __half*       out,
    int seq_len,
    int num_heads,
    int head_dim,
    cudaStream_t stream)
{
    dim3 grid(num_heads);
    dim3 block(128);
    // Shared memory: tile_k = max(REF_TILE * head_dim, REF_TILE) floats
    size_t smem = static_cast<size_t>(REF_TILE) * head_dim * sizeof(float);

    reference_attention_kernel<<<grid, block, smem, stream>>>(
        Q, K, V, out, seq_len, num_heads, head_dim,
        1.f / sqrtf(static_cast<float>(head_dim)));
}
