#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "config.cuh"
#include "cache.cuh"
#include "reference.cuh"
#include "fused_tele_attn.cuh"

static constexpr int WARMUP_ITERS = 10;
static constexpr int BENCH_ITERS  = 100;

static void fill_rand(__half* d, int n) {
    std::vector<__half> h(n);
    for (int i = 0; i < n; ++i) h[i] = __float2half((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.f);
    cudaMemcpy(d, h.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
}

struct BenchResult {
    float lat_us;   // average per-step latency in microseconds
    float mem_mb;   // peak KV cache memory in MB
    float bw_gbps;  // effective memory bandwidth in GB/s
};

static BenchResult bench_dense(int seq_len, int num_heads, int head_dim) {
    // Pre-fill KV cache of size seq_len.
    size_t kv_sz = static_cast<size_t>(seq_len) * num_heads * head_dim;
    __half* d_K = nullptr; cudaMalloc(&d_K, kv_sz * sizeof(__half));
    __half* d_V = nullptr; cudaMalloc(&d_V, kv_sz * sizeof(__half));
    __half* d_Q = nullptr; cudaMalloc(&d_Q, num_heads * head_dim * sizeof(__half));
    __half* d_O = nullptr; cudaMalloc(&d_O, num_heads * head_dim * sizeof(__half));

    fill_rand(d_K, kv_sz); fill_rand(d_V, kv_sz); fill_rand(d_Q, num_heads * head_dim);

    // Warmup.
    for (int i = 0; i < WARMUP_ITERS; ++i)
        launch_reference_attention(d_Q, d_K, d_V, d_O, seq_len, num_heads, head_dim);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; ++i)
        launch_reference_attention(d_Q, d_K, d_V, d_O, seq_len, num_heads, head_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    float lat_us = ms * 1000.f / BENCH_ITERS;
    float mem_mb = 2.f * kv_sz * sizeof(__half) / (1024.f * 1024.f);  // K + V
    // Bandwidth: read K + V once per attention decode step.
    float bw_gbps = (2.f * kv_sz * sizeof(__half)) / (lat_us * 1e-6f) / 1e9f;

    cudaFree(d_K); cudaFree(d_V); cudaFree(d_Q); cudaFree(d_O);

    return {lat_us, mem_mb, bw_gbps};
}

static BenchResult bench_telescoped(int seq_len, int num_heads, int head_dim,
                                     float budget, int num_levels, int ratio) {
    TelescopeConfig cfg;
    cfg.num_levels = num_levels;
    cfg.num_heads  = num_heads;
    cfg.head_dim   = head_dim;
    cfg.importance_ema_alpha = 0.1f;

    cfg.level_capacity[0] = max(8, static_cast<int>(seq_len * budget));
    for (int l = 1; l < num_levels; ++l) {
        cfg.level_capacity[l]   = max(4, cfg.level_capacity[l-1] / ratio);
        cfg.downsample_ratio[l-1] = ratio;
    }
    cfg.downsample_ratio[num_levels - 1] = ratio;

    MultiResolutionCache* cache = MultiResolutionCache::allocate(cfg);

    __half* d_Q     = nullptr; cudaMalloc(&d_Q,     num_heads * head_dim * sizeof(__half));
    __half* d_new_k = nullptr; cudaMalloc(&d_new_k, num_heads * head_dim * sizeof(__half));
    __half* d_new_v = nullptr; cudaMalloc(&d_new_v, num_heads * head_dim * sizeof(__half));
    __half* d_O     = nullptr; cudaMalloc(&d_O,     num_heads * head_dim * sizeof(__half));

    // Pre-fill cache to steady-state by running seq_len/2 steps.
    srand(42);
    for (int i = 0; i < seq_len / 2; ++i) {
        fill_rand(d_Q, num_heads * head_dim);
        fill_rand(d_new_k, num_heads * head_dim);
        fill_rand(d_new_v, num_heads * head_dim);
        launch_fused_tele_attn(d_Q, d_new_k, d_new_v, d_O, cache);
        cudaDeviceSynchronize();
    }

    fill_rand(d_Q, num_heads * head_dim);
    fill_rand(d_new_k, num_heads * head_dim);
    fill_rand(d_new_v, num_heads * head_dim);

    // Warmup.
    for (int i = 0; i < WARMUP_ITERS; ++i)
        launch_fused_tele_attn(d_Q, d_new_k, d_new_v, d_O, cache);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITERS; ++i)
        launch_fused_tele_attn(d_Q, d_new_k, d_new_v, d_O, cache);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    float lat_us = ms * 1000.f / BENCH_ITERS;
    float mem_mb = static_cast<float>(cfg.peak_device_bytes()) / (1024.f * 1024.f);

    // Total cache entries visible during attention.
    int total_entries = 0;
    for (int l = 0; l < cfg.num_levels; ++l) total_entries += cache->h_count[l];
    float bw_gbps = (2.f * total_entries * num_heads * head_dim * sizeof(__half))
                    / (lat_us * 1e-6f) / 1e9f;

    MultiResolutionCache::free(cache);
    cudaFree(d_Q); cudaFree(d_new_k); cudaFree(d_new_v); cudaFree(d_O);

    return {lat_us, mem_mb, bw_gbps};
}

int main() {
    const int NUM_HEADS  = 32;
    const int HEAD_DIM   = 128;
    const int NUM_LEVELS = 3;
    const int RATIO      = 4;

    int seq_lengths[] = {1024, 4096, 8192, 16384, 32768, 65536};
    int n_seqs = sizeof(seq_lengths) / sizeof(seq_lengths[0]);

    float budgets[] = {1.0f, 0.80f, 0.50f, 0.30f};
    const char* budget_labels[] = {"Dense(100%)", "Tele-80%", "Tele-50%", "Tele-30%"};
    int n_budgets = 4;

    srand(0);

    printf("=== Latency Benchmark: Dense vs Telescoped ===\n");
    printf("num_heads=%d, head_dim=%d, num_levels=%d, ratio=%d\n\n", NUM_HEADS, HEAD_DIM, NUM_LEVELS, RATIO);
    printf("%-14s %-14s %-14s %-14s %-12s %-12s\n",
           "SeqLen", "Config", "Latency(us)", "MemCache(MB)", "Speedup", "BW(GB/s)");
    printf("%-14s %-14s %-14s %-14s %-12s %-12s\n",
           "------", "----------", "-----------", "----------", "-------", "--------");

    for (int si = 0; si < n_seqs; ++si) {
        int seq = seq_lengths[si];

        // Dense baseline.
        BenchResult ref = bench_dense(seq, NUM_HEADS, HEAD_DIM);

        for (int bi = 0; bi < n_budgets; ++bi) {
            BenchResult r;
            if (bi == 0) {
                r = ref;
            } else {
                r = bench_telescoped(seq, NUM_HEADS, HEAD_DIM, budgets[bi], NUM_LEVELS, RATIO);
            }
            float speedup = (bi == 0) ? 1.f : ref.lat_us / r.lat_us;
            printf("%-14d %-14s %-14.2f %-14.2f %-12.2f %-12.2f\n",
                   seq, budget_labels[bi], r.lat_us, r.mem_mb, speedup, r.bw_gbps);
        }
        printf("\n");
    }

    return 0;
}
