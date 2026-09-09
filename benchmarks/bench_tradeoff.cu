#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#include "config.cuh"
#include "cache.cuh"
#include "reference.cuh"
#include "fused_tele_attn.cuh"

static void fill_randn_device(__half* d, int n, unsigned int& seed) {
    std::vector<__half> h(n);
    for (int i = 0; i < n; ++i) {
        seed = seed * 1664525u + 1013904223u;
        float u1 = (static_cast<float>(seed >> 16) + 1.f) / 65537.f;
        seed = seed * 1664525u + 1013904223u;
        float u2 = (static_cast<float>(seed >> 16) + 1.f) / 65537.f;
        float z  = sqrtf(-2.f * logf(u1)) * cosf(2.f * 3.14159265f * u2);
        h[i] = __float2half(z);
    }
    cudaMemcpy(d, h.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
}

static float cosine_similarity_device(const __half* d_a, const __half* d_b, int n) {
    std::vector<__half> ha(n), hb(n);
    cudaMemcpy(ha.data(), d_a, n * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), d_b, n * sizeof(__half), cudaMemcpyDeviceToHost);
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        float a = __half2float(ha[i]), b = __half2float(hb[i]);
        dot += a * b; na += a * a; nb += b * b;
    }
    return (na == 0 || nb == 0) ? 0.f : static_cast<float>(dot / sqrt(na * nb));
}

struct TradeoffResult {
    float budget_pct;
    float avg_cosine;
    float worst_cosine;
    float peak_mb;
    float avg_promos_per_step;
};

static TradeoffResult run_budget_sweep(float budget_frac, int num_heads, int head_dim,
                                        int num_levels, int ratio, int seq_len) {
    TelescopeConfig cfg;
    cfg.num_levels = num_levels;
    cfg.num_heads  = num_heads;
    cfg.head_dim   = head_dim;
    cfg.importance_ema_alpha = 0.1f;

    int base_cap = static_cast<int>(seq_len * budget_frac);
    cfg.level_capacity[0] = max(8, base_cap);
    for (int l = 1; l < num_levels; ++l) {
        cfg.level_capacity[l]   = max(4, cfg.level_capacity[l-1] / ratio);
        cfg.downsample_ratio[l-1] = ratio;
    }
    cfg.downsample_ratio[num_levels - 1] = ratio;

    MultiResolutionCache* cache = MultiResolutionCache::allocate(cfg);

    // Allocate dense reference KV store.
    size_t kv_flat = static_cast<size_t>(seq_len) * num_heads * head_dim;
    __half* d_K_flat = nullptr; cudaMalloc(&d_K_flat, kv_flat * sizeof(__half));
    __half* d_V_flat = nullptr; cudaMalloc(&d_V_flat, kv_flat * sizeof(__half));

    __half* d_Q        = nullptr; cudaMalloc(&d_Q,        num_heads * head_dim * sizeof(__half));
    __half* d_new_k    = nullptr; cudaMalloc(&d_new_k,    num_heads * head_dim * sizeof(__half));
    __half* d_new_v    = nullptr; cudaMalloc(&d_new_v,    num_heads * head_dim * sizeof(__half));
    __half* d_out_fuse = nullptr; cudaMalloc(&d_out_fuse, num_heads * head_dim * sizeof(__half));
    __half* d_out_ref  = nullptr; cudaMalloc(&d_out_ref,  num_heads * head_dim * sizeof(__half));

    unsigned int seed = 0xdeadbeef ^ static_cast<unsigned int>(budget_frac * 1000);

    // We only need to accumulate K/V flat buffers in host.
    std::vector<float> h_K_flat_f, h_V_flat_f;
    h_K_flat_f.reserve(kv_flat);
    h_V_flat_f.reserve(kv_flat);
    std::vector<__half> h_K_flat_h, h_V_flat_h;

    float total_cosine = 0.f;
    float worst_cosine = 1.f;

    for (int step = 0; step < seq_len; ++step) {
        fill_randn_device(d_Q,     num_heads * head_dim, seed);
        fill_randn_device(d_new_k, num_heads * head_dim, seed);
        fill_randn_device(d_new_v, num_heads * head_dim, seed);

        // Append to flat host buffers.
        std::vector<__half> hk(num_heads * head_dim), hv(num_heads * head_dim);
        cudaMemcpy(hk.data(), d_new_k, hk.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        cudaMemcpy(hv.data(), d_new_v, hv.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        h_K_flat_h.insert(h_K_flat_h.end(), hk.begin(), hk.end());
        h_V_flat_h.insert(h_V_flat_h.end(), hv.begin(), hv.end());

        int cur_len = step + 1;
        cudaMemcpy(d_K_flat, h_K_flat_h.data(), cur_len * num_heads * head_dim * sizeof(__half),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_V_flat, h_V_flat_h.data(), cur_len * num_heads * head_dim * sizeof(__half),
                   cudaMemcpyHostToDevice);

        launch_reference_attention(d_Q, d_K_flat, d_V_flat, d_out_ref, cur_len, num_heads, head_dim);
        launch_fused_tele_attn(d_Q, d_new_k, d_new_v, d_out_fuse, cache);
        cudaDeviceSynchronize();

        float cs = cosine_similarity_device(d_out_fuse, d_out_ref, num_heads * head_dim);
        total_cosine += cs;
        worst_cosine  = fminf(worst_cosine, cs);
    }

    TradeoffResult res;
    res.budget_pct      = budget_frac * 100.f;
    res.avg_cosine      = total_cosine / seq_len;
    res.worst_cosine    = worst_cosine;
    res.peak_mb         = static_cast<float>(cfg.peak_device_bytes()) / (1024.f * 1024.f);
    res.avg_promos_per_step = 0.f;  // tracking skipped for brevity

    MultiResolutionCache::free(cache);
    cudaFree(d_K_flat); cudaFree(d_V_flat);
    cudaFree(d_Q); cudaFree(d_new_k); cudaFree(d_new_v);
    cudaFree(d_out_fuse); cudaFree(d_out_ref);

    return res;
}

int main() {
    const int SEQ_LEN   = 512;   // reduced for tractability; scale to 8192 on server
    const int NUM_HEADS = 32;
    const int HEAD_DIM  = 128;
    const int NUM_LEVELS = 3;
    const int RATIO      = 4;

    printf("=== Memory-Quality Pareto Sweep ===\n");
    printf("seq_len=%d, num_heads=%d, head_dim=%d, num_levels=%d, ratio=%d\n\n",
           SEQ_LEN, NUM_HEADS, HEAD_DIM, NUM_LEVELS, RATIO);

    printf("%-12s %-14s %-14s %-14s\n",
           "Budget(%)", "Avg Cosine", "Worst Cosine", "Cache MB");
    printf("%-12s %-14s %-14s %-14s\n",
           "-----------", "-------------", "-------------", "---------");

    for (int b = 100; b >= 20; b -= 5) {
        float budget = b / 100.f;
        TradeoffResult r = run_budget_sweep(budget, NUM_HEADS, HEAD_DIM, NUM_LEVELS, RATIO, SEQ_LEN);
        printf("%-12.1f %-14.6f %-14.6f %-14.3f\n",
               r.budget_pct, r.avg_cosine, r.worst_cosine, r.peak_mb);
    }

    printf("\n=== Level Count Sweep (50%% budget) ===\n");
    printf("%-12s %-14s %-14s %-14s\n",
           "Num Levels", "Avg Cosine", "Worst Cosine", "Cache MB");
    printf("%-12s %-14s %-14s %-14s\n",
           "-----------", "-------------", "-------------", "---------");

    for (int nl = 2; nl <= 5; ++nl) {
        TradeoffResult r = run_budget_sweep(0.50f, NUM_HEADS, HEAD_DIM, nl, RATIO, SEQ_LEN);
        printf("%-12d %-14.6f %-14.6f %-14.3f\n",
               nl, r.avg_cosine, r.worst_cosine, r.peak_mb);
    }

    return 0;
}
