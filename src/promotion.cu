#include "promotion.cuh"
#include "representative.cuh"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Compact kernel: shift entries [n_remove, count) to [0, count-n_remove).
// Grid: (num_heads), Block: (32)
// Each block handles one head's K and V arrays.
// ---------------------------------------------------------------------------
__global__ void compact_level_kernel(
    __half*   level_k,
    __half*   level_v,
    float*    level_importance,
    uint16_t* level_mult,
    int       n_remove,
    int       count,
    int       num_heads,
    int       head_dim)
{
    const int head  = blockIdx.x;
    const int lane  = threadIdx.x;
    const int keep  = count - n_remove;

    // Shift K/V for this head.
    for (int entry = 0; entry < keep; ++entry) {
        int src = entry + n_remove;
        const __half* src_k = level_k + (static_cast<ptrdiff_t>(src)  * num_heads + head) * head_dim;
              __half* dst_k = level_k + (static_cast<ptrdiff_t>(entry) * num_heads + head) * head_dim;
        const __half* src_v = level_v + (static_cast<ptrdiff_t>(src)  * num_heads + head) * head_dim;
              __half* dst_v = level_v + (static_cast<ptrdiff_t>(entry) * num_heads + head) * head_dim;

        for (int d = lane; d < head_dim; d += 32) {
            dst_k[d] = src_k[d];
            dst_v[d] = src_v[d];
        }
    }

    // Thread 0 of head 0 shifts scalars (importance, multiplicity).
    if (head == 0 && lane == 0) {
        for (int entry = 0; entry < keep; ++entry) {
            level_importance[entry]   = level_importance[entry + n_remove];
            level_mult[entry]         = level_mult[entry + n_remove];
        }
    }
}

void compact_level(MultiResolutionCache* cache, int level, int n_remove, cudaStream_t stream) {
    int count = cache->h_count[level];
    if (n_remove <= 0 || n_remove > count) return;

    const TelescopeConfig& cfg = cache->cfg;
    dim3 grid(cfg.num_heads);
    dim3 block(32);

    compact_level_kernel<<<grid, block, 0, stream>>>(
        cache->level_k[level],
        cache->level_v[level],
        cache->level_importance[level],
        cache->level_multiplicity[level],
        n_remove, count, cfg.num_heads, cfg.head_dim);

    cache->h_count[level] -= n_remove;
}

// ---------------------------------------------------------------------------
// promote_level: promote oldest entries from level `from_level` to
// `from_level+1`, then cascade if needed.
// ---------------------------------------------------------------------------
int promote_level(MultiResolutionCache* cache, int from_level, cudaStream_t stream) {
    const TelescopeConfig& cfg = cache->cfg;

    if (from_level >= cfg.num_levels - 1) return 0;  // no higher level

    int count = cache->h_count[from_level];
    int ratio = cfg.downsample_ratio[from_level];

    // Number of complete windows we can promote.
    int num_windows = count / ratio;
    if (num_windows == 0) return 0;

    int n_promote = num_windows * ratio;

    // Target level.
    int to_level    = from_level + 1;
    int to_count    = cache->h_count[to_level];
    int to_capacity = cfg.level_capacity[to_level];

    // Temporary device buffer for representative output.
    size_t tmp_kv_bytes  = static_cast<size_t>(num_windows) * cfg.num_heads * cfg.head_dim * sizeof(__half);
    size_t tmp_imp_bytes = static_cast<size_t>(num_windows) * sizeof(float);
    size_t tmp_mul_bytes = static_cast<size_t>(num_windows) * sizeof(uint16_t);

    __half*   tmp_k   = nullptr;
    __half*   tmp_v   = nullptr;
    float*    tmp_imp = nullptr;
    uint16_t* tmp_mul = nullptr;

    cudaMalloc(&tmp_k,   tmp_kv_bytes);
    cudaMalloc(&tmp_v,   tmp_kv_bytes);
    cudaMalloc(&tmp_imp, tmp_imp_bytes);
    cudaMalloc(&tmp_mul, tmp_mul_bytes);

    // Select representatives from the oldest n_promote entries of from_level.
    launch_representative_selection(
        cache->level_k[from_level],
        cache->level_v[from_level],
        cache->level_importance[from_level],
        cache->level_multiplicity[from_level],
        tmp_k, tmp_v, tmp_imp, tmp_mul,
        n_promote, ratio, cfg.num_heads, cfg.head_dim, stream);

    // Append representatives to to_level.
    // If not enough space, only promote what fits.
    int slots_available = to_capacity - to_count;
    int actually_append = min(num_windows, slots_available);

    if (actually_append > 0) {
        size_t kv_elem_size = static_cast<size_t>(actually_append) * cfg.num_heads * cfg.head_dim;

        cudaMemcpyAsync(
            cache->level_k[to_level] + static_cast<ptrdiff_t>(to_count) * cfg.num_heads * cfg.head_dim,
            tmp_k, kv_elem_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(
            cache->level_v[to_level] + static_cast<ptrdiff_t>(to_count) * cfg.num_heads * cfg.head_dim,
            tmp_v, kv_elem_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(
            cache->level_importance[to_level] + to_count,
            tmp_imp, actually_append * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(
            cache->level_multiplicity[to_level] + to_count,
            tmp_mul, actually_append * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream);

        cache->h_count[to_level] += actually_append;
    }

    cudaFree(tmp_k);
    cudaFree(tmp_v);
    cudaFree(tmp_imp);
    cudaFree(tmp_mul);

    // Compact from_level by removing the promoted entries.
    // We always remove n_promote entries regardless of how many fit in to_level;
    // entries that didn't fit are dropped (oldest are least important).
    cudaStreamSynchronize(stream);
    compact_level(cache, from_level, n_promote, stream);

    int total_promotions = 1;

    // Cascade: if to_level now exceeds capacity, promote from it.
    if (cache->h_count[to_level] > to_capacity && to_level < cfg.num_levels - 1) {
        total_promotions += promote_level(cache, to_level, stream);
    }

    return total_promotions;
}
