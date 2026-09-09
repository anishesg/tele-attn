#include "fused_tele_attn.cuh"
#include "promotion.cuh"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>

static constexpr int FUSE_TILE = 64;

// ---------------------------------------------------------------------------
// Fused multi-resolution attention kernel.
//
// One block per head. Iterates over all cache levels, tiling each level in
// chunks of FUSE_TILE. Applies multiplicity-weighted online softmax.
// After writing output, updates importance scores for each cached entry via EMA.
//
// Shared memory layout (all float):
//   [0 .. head_dim)              : Q vector (float, loaded once)
//   [head_dim .. head_dim + FUSE_TILE * head_dim) : K tile
//   [head_dim + FUSE_TILE*head_dim .. same + FUSE_TILE*head_dim) : V tile (same region, reused)
//   followed by: scores[FUSE_TILE], weights[FUSE_TILE]
// ---------------------------------------------------------------------------
__global__ void fused_tele_attn_kernel(
    const __half* __restrict__ Q,
    __half* __restrict__       out,
    // Per-level arrays (passed as flat device pointers; level selection via loop)
    const __half* const* __restrict__ level_k_ptrs,
    const __half* const* __restrict__ level_v_ptrs,
    float* const*                     level_imp_ptrs,
    const uint16_t* const* __restrict__ level_mult_ptrs,
    const int* __restrict__           level_counts,
    int   num_levels,
    int   num_heads,
    int   head_dim,
    float inv_sqrt_d,
    float ema_alpha)
{
    extern __shared__ float smem[];

    float* q_smem   = smem;                           // [head_dim]
    float* tile_kv  = smem + head_dim;                // [FUSE_TILE * head_dim]
    float* score_buf = tile_kv + FUSE_TILE * head_dim; // [FUSE_TILE]
    float* mult_buf  = score_buf + FUSE_TILE;          // [FUSE_TILE]

    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    const int bsz  = blockDim.x;

    // Load Q into shared memory.
    const __half* q_ptr = Q + head * head_dim;
    for (int d = lane; d < head_dim; d += bsz) {
        q_smem[d] = __half2float(q_ptr[d]);
    }
    __syncthreads();

    float acc[128]      = {};
    float running_max   = -1e38f;
    float running_denom = 0.f;

    // Iterate from finest (level 0) to coarsest.
    for (int lvl = 0; lvl < num_levels; ++lvl) {
        int count = level_counts[lvl];
        if (count == 0) continue;

        const __half* lk = level_k_ptrs[lvl];
        const __half* lv = level_v_ptrs[lvl];
        const uint16_t* lm = level_mult_ptrs[lvl];

        for (int tile_start = 0; tile_start < count; tile_start += FUSE_TILE) {
            int tile_len = min(FUSE_TILE, count - tile_start);

            // Load K tile.
            for (int i = lane; i < tile_len * head_dim; i += bsz) {
                int entry = i / head_dim;
                int d     = i % head_dim;
                int pos   = tile_start + entry;
                tile_kv[i] = __half2float(lk[(pos * num_heads + head) * head_dim + d]);
            }
            __syncthreads();

            // Compute QK^T scores.
            for (int e = lane; e < tile_len; e += bsz) {
                float dot = 0.f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_smem[d] * tile_kv[e * head_dim + d];
                }
                score_buf[e] = dot * inv_sqrt_d;
                mult_buf[e]  = static_cast<float>(lm[tile_start + e]);
            }
            __syncthreads();

            // Thread 0 performs multiplicity-weighted online softmax update.
            if (lane == 0) {
                float tile_max = -1e38f;
                for (int e = 0; e < tile_len; ++e) tile_max = fmaxf(tile_max, score_buf[e]);

                float new_max     = fmaxf(running_max, tile_max);
                float exp_shift   = expf(running_max - new_max);

                float tile_denom = 0.f;
                float tile_w[FUSE_TILE];
                for (int e = 0; e < tile_len; ++e) {
                    float w = mult_buf[e] * expf(score_buf[e] - new_max);
                    tile_w[e]   = w;
                    tile_denom += w;
                }

                for (int d = 0; d < head_dim; ++d) acc[d] *= exp_shift;
                running_denom = running_denom * exp_shift + tile_denom;
                running_max   = new_max;

                // Stash weights back for V accumulation.
                for (int e = 0; e < tile_len; ++e) score_buf[e] = tile_w[e];
            }
            __syncthreads();

            // Load V tile (reuse tile_kv buffer).
            for (int i = lane; i < tile_len * head_dim; i += bsz) {
                int entry = i / head_dim;
                int d     = i % head_dim;
                int pos   = tile_start + entry;
                tile_kv[i] = __half2float(lv[(pos * num_heads + head) * head_dim + d]);
            }
            __syncthreads();

            // Accumulate V (thread 0 holds weights in score_buf).
            if (lane == 0) {
                for (int e = 0; e < tile_len; ++e) {
                    float w = score_buf[e];
                    for (int d = 0; d < head_dim; ++d) {
                        acc[d] += w * tile_kv[e * head_dim + d];
                    }
                }
            }
            __syncthreads();

            // EMA importance update: each entry's weight = score_buf[e] / running_denom.
            // Fused: update importance while we still have tile weights in score_buf.
            // Only after running_denom is finalized (end of all levels) is the true
            // attention weight known. Here we use the running unnormalized weight as a
            // proxy; a second pass would be needed for exact EMA. This approximation is
            // acceptable since importance tracks relative ordering, not absolute values.
            if (lane == 0 && running_denom > 0.f) {
                float inv_d = 1.f / running_denom;
                float* imp = level_imp_ptrs[lvl];
                for (int e = 0; e < tile_len; ++e) {
                    int pos = tile_start + e;
                    float attn_w = score_buf[e] * inv_d;
                    imp[pos] = ema_alpha * attn_w + (1.f - ema_alpha) * imp[pos];
                }
            }
            __syncthreads();
        }
    }

    // Write output.
    if (lane == 0) {
        __half* out_ptr = out + head * head_dim;
        float inv_denom = (running_denom > 0.f) ? 1.f / running_denom : 1.f;
        for (int d = 0; d < head_dim; ++d) {
            out_ptr[d] = __float2half(acc[d] * inv_denom);
        }
    }
}

// ---------------------------------------------------------------------------
// Append new KV entry to level 0.
// Grid: (num_heads), Block: (32)
// ---------------------------------------------------------------------------
__global__ void append_to_level0_kernel(
    __half* __restrict__   level_k,
    __half* __restrict__   level_v,
    float*  __restrict__   level_imp,
    uint16_t* __restrict__ level_mult,
    const __half* __restrict__ new_k,
    const __half* __restrict__ new_v,
    int pos,        // insertion index
    int num_heads,
    int head_dim)
{
    const int head = blockIdx.x;
    const int lane = threadIdx.x;

    const __half* src_k = new_k + head * head_dim;
    const __half* src_v = new_v + head * head_dim;
    __half* dst_k = level_k + (static_cast<ptrdiff_t>(pos) * num_heads + head) * head_dim;
    __half* dst_v = level_v + (static_cast<ptrdiff_t>(pos) * num_heads + head) * head_dim;

    for (int d = lane; d < head_dim; d += 32) {
        dst_k[d] = src_k[d];
        dst_v[d] = src_v[d];
    }

    if (head == 0 && lane == 0) {
        level_imp[pos]  = 0.f;
        level_mult[pos] = 1;
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
void launch_fused_tele_attn(
    const __half*         Q,
    const __half*         new_k,
    const __half*         new_v,
    __half*               out,
    MultiResolutionCache* cache,
    cudaStream_t          stream)
{
    const TelescopeConfig& cfg = cache->cfg;

    // Build device-side pointer arrays for level K/V/imp/mult.
    // These are arrays of pointers that live in the cache struct (host side),
    // but we need to pass them to the kernel. We use a small device buffer.
    const __half* h_lk[TELE_MAX_LEVELS];
    const __half* h_lv[TELE_MAX_LEVELS];
    float*        h_li[TELE_MAX_LEVELS];
    const uint16_t* h_lm[TELE_MAX_LEVELS];
    int           h_lc[TELE_MAX_LEVELS];

    for (int l = 0; l < cfg.num_levels; ++l) {
        h_lk[l] = cache->level_k[l];
        h_lv[l] = cache->level_v[l];
        h_li[l] = cache->level_importance[l];
        h_lm[l] = cache->level_multiplicity[l];
        h_lc[l] = cache->h_count[l];
    }

    // Allocate temporary device arrays for pointer tables.
    const __half** d_lk  = nullptr;
    const __half** d_lv  = nullptr;
    float**        d_li  = nullptr;
    const uint16_t** d_lm = nullptr;
    int*           d_lc  = nullptr;

    size_t ptr_bytes  = TELE_MAX_LEVELS * sizeof(void*);
    size_t cnt_bytes  = TELE_MAX_LEVELS * sizeof(int);

    cudaMallocAsync(&d_lk, ptr_bytes, stream);
    cudaMallocAsync(&d_lv, ptr_bytes, stream);
    cudaMallocAsync(&d_li, ptr_bytes, stream);
    cudaMallocAsync(&d_lm, ptr_bytes, stream);
    cudaMallocAsync(&d_lc, cnt_bytes,  stream);

    cudaMemcpyAsync(d_lk, h_lk, cfg.num_levels * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_lv, h_lv, cfg.num_levels * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_li, h_li, cfg.num_levels * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_lm, h_lm, cfg.num_levels * sizeof(void*), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_lc, h_lc, cfg.num_levels * sizeof(int),   cudaMemcpyHostToDevice, stream);

    dim3 grid(cfg.num_heads);
    dim3 block(128);

    // Shared memory: q[head_dim] + kv_tile[FUSE_TILE*head_dim] + scores[FUSE_TILE] + mult[FUSE_TILE]
    size_t smem = static_cast<size_t>(cfg.head_dim + FUSE_TILE * cfg.head_dim + 2 * FUSE_TILE) * sizeof(float);

    fused_tele_attn_kernel<<<grid, block, smem, stream>>>(
        Q, out,
        d_lk, d_lv, d_li, d_lm, d_lc,
        cfg.num_levels, cfg.num_heads, cfg.head_dim,
        1.f / sqrtf(static_cast<float>(cfg.head_dim)),
        cfg.importance_ema_alpha);

    // Append new entry to level 0.
    int pos0 = cache->h_count[0];
    append_to_level0_kernel<<<dim3(cfg.num_heads), dim3(32), 0, stream>>>(
        cache->level_k[0], cache->level_v[0],
        cache->level_importance[0], cache->level_multiplicity[0],
        new_k, new_v, pos0, cfg.num_heads, cfg.head_dim);

    cache->h_count[0]++;

    cudaFreeAsync(d_lk, stream);
    cudaFreeAsync(d_lv, stream);
    cudaFreeAsync(d_li, stream);
    cudaFreeAsync(d_lm, stream);
    cudaFreeAsync(d_lc, stream);

    // Trigger cascade if level 0 is full.
    if (cache->h_count[0] >= cfg.level_capacity[0]) {
        cudaStreamSynchronize(stream);
        promote_level(cache, 0, stream);
    }
}
