#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <vector>

#include "config.cuh"
#include "cache.cuh"
#include "reference.cuh"
#include "fused_tele_attn.cuh"

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

static void fill_randn(__half* d_arr, int n, float mean = 0.f, float std = 1.f) {
    std::vector<__half> h(n);
    for (int i = 0; i < n; ++i) {
        // Box-Muller for a simple normal distribution.
        float u1 = (static_cast<float>(rand()) + 1.f) / (static_cast<float>(RAND_MAX) + 2.f);
        float u2 = (static_cast<float>(rand()) + 1.f) / (static_cast<float>(RAND_MAX) + 2.f);
        float z  = sqrtf(-2.f * logf(u1)) * cosf(2.f * 3.14159265358979f * u2);
        h[i] = __float2half(z * std + mean);
    }
    cudaMemcpy(d_arr, h.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
}

static float cosine_similarity(const __half* d_a, const __half* d_b, int n) {
    std::vector<__half> ha(n), hb(n);
    cudaMemcpy(ha.data(), d_a, n * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), d_b, n * sizeof(__half), cudaMemcpyDeviceToHost);

    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        float a = __half2float(ha[i]);
        float b = __half2float(hb[i]);
        dot += static_cast<double>(a) * b;
        na  += static_cast<double>(a) * a;
        nb  += static_cast<double>(b) * b;
    }
    if (na == 0 || nb == 0) return 0.f;
    return static_cast<float>(dot / (sqrt(na) * sqrt(nb)));
}

static __half* alloc_half(int n) {
    __half* p;
    cudaMalloc(&p, n * sizeof(__half));
    cudaMemset(p, 0, n * sizeof(__half));
    return p;
}

// ---------------------------------------------------------------------------
// Test 1: budget=100% (no downsampling needed, all level-0, num_steps <= capacity)
// Fused output vs dense reference must have cosine > 0.9999.
// ---------------------------------------------------------------------------
static void test1_full_budget(int num_heads, int head_dim) {
    printf("[Test1] full budget, num_heads=%d head_dim=%d ... ", num_heads, head_dim);

    const int STEPS = 64;

    TelescopeConfig cfg;
    cfg.num_levels = 1;
    cfg.num_heads  = num_heads;
    cfg.head_dim   = head_dim;
    cfg.level_capacity[0]   = STEPS + 8;  // never fill
    cfg.downsample_ratio[0] = 4;
    cfg.importance_ema_alpha = 0.1f;

    MultiResolutionCache* cache = MultiResolutionCache::allocate(cfg);
    assert(cache);

    // Dense KV accumulation on host side.
    std::vector<__half> h_K_full, h_V_full;
    h_K_full.reserve(STEPS * num_heads * head_dim);
    h_V_full.reserve(STEPS * num_heads * head_dim);

    __half* d_Q      = alloc_half(num_heads * head_dim);
    __half* d_new_k  = alloc_half(num_heads * head_dim);
    __half* d_new_v  = alloc_half(num_heads * head_dim);
    __half* d_out_fuse = alloc_half(num_heads * head_dim);
    __half* d_out_ref  = alloc_half(num_heads * head_dim);
    __half* d_K_flat   = nullptr;
    __half* d_V_flat   = nullptr;

    srand(42);

    float total_cosine = 0.f;
    for (int step = 0; step < STEPS; ++step) {
        fill_randn(d_Q,     num_heads * head_dim);
        fill_randn(d_new_k, num_heads * head_dim);
        fill_randn(d_new_v, num_heads * head_dim);

        // Append K/V to dense reference.
        std::vector<__half> hk(num_heads * head_dim), hv(num_heads * head_dim);
        cudaMemcpy(hk.data(), d_new_k, hk.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        cudaMemcpy(hv.data(), d_new_v, hv.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        h_K_full.insert(h_K_full.end(), hk.begin(), hk.end());
        h_V_full.insert(h_V_full.end(), hv.begin(), hv.end());

        int cur_len = step + 1;

        // Dense reference.
        cudaMalloc(&d_K_flat, h_K_full.size() * sizeof(__half));
        cudaMalloc(&d_V_flat, h_V_full.size() * sizeof(__half));
        cudaMemcpy(d_K_flat, h_K_full.data(), h_K_full.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V_flat, h_V_full.data(), h_V_full.size() * sizeof(__half), cudaMemcpyHostToDevice);

        launch_reference_attention(d_Q, d_K_flat, d_V_flat, d_out_ref, cur_len, num_heads, head_dim);

        // Fused (appends new_k/new_v inside).
        launch_fused_tele_attn(d_Q, d_new_k, d_new_v, d_out_fuse, cache);
        cudaDeviceSynchronize();

        float cs = cosine_similarity(d_out_fuse, d_out_ref, num_heads * head_dim);
        total_cosine += cs;

        cudaFree(d_K_flat);
        cudaFree(d_V_flat);
    }

    float avg_cs = total_cosine / STEPS;
    bool pass = avg_cs > 0.9999f;
    printf("%s (avg_cosine=%.6f)\n", pass ? "PASS" : "FAIL", avg_cs);
    if (!pass) exit(1);

    MultiResolutionCache::free(cache);
    cudaFree(d_Q); cudaFree(d_new_k); cudaFree(d_new_v);
    cudaFree(d_out_fuse); cudaFree(d_out_ref);
}

// ---------------------------------------------------------------------------
// Test 2 & 3: autoregressive steps with promotions; compare cosine similarity.
// ---------------------------------------------------------------------------
static void test_with_budget(int num_heads, int head_dim, int num_levels,
                              int total_cap, float budget,
                              float req_avg, float req_worst,
                              const char* label)
{
    printf("[%s] num_heads=%d head_dim=%d num_levels=%d budget=%.0f%% ... ",
           label, num_heads, head_dim, num_levels, budget * 100.f);

    const int STEPS = 512;
    const int ratio = 4;

    // Compute per-level capacities with budget.
    TelescopeConfig cfg;
    cfg.num_levels = num_levels;
    cfg.num_heads  = num_heads;
    cfg.head_dim   = head_dim;
    cfg.importance_ema_alpha = 0.1f;

    // Level 0 capacity = total_cap * budget, each subsequent level = prev / ratio.
    cfg.level_capacity[0] = max(8, static_cast<int>(total_cap * budget));
    for (int l = 1; l < num_levels; ++l) {
        cfg.level_capacity[l] = max(4, cfg.level_capacity[l-1] / ratio);
        cfg.downsample_ratio[l-1] = ratio;
    }
    cfg.downsample_ratio[num_levels - 1] = ratio;

    MultiResolutionCache* cache = MultiResolutionCache::allocate(cfg);
    assert(cache);

    std::vector<__half> h_K_full, h_V_full;
    h_K_full.reserve(STEPS * num_heads * head_dim);
    h_V_full.reserve(STEPS * num_heads * head_dim);

    __half* d_Q        = alloc_half(num_heads * head_dim);
    __half* d_new_k    = alloc_half(num_heads * head_dim);
    __half* d_new_v    = alloc_half(num_heads * head_dim);
    __half* d_out_fuse = alloc_half(num_heads * head_dim);
    __half* d_out_ref  = alloc_half(num_heads * head_dim);
    __half* d_K_flat   = nullptr;
    __half* d_V_flat   = nullptr;

    srand(123);

    float total_cosine = 0.f;
    float worst_cosine = 1.f;

    for (int step = 0; step < STEPS; ++step) {
        fill_randn(d_Q,     num_heads * head_dim);
        fill_randn(d_new_k, num_heads * head_dim);
        fill_randn(d_new_v, num_heads * head_dim);

        std::vector<__half> hk(num_heads * head_dim), hv(num_heads * head_dim);
        cudaMemcpy(hk.data(), d_new_k, hk.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        cudaMemcpy(hv.data(), d_new_v, hv.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        h_K_full.insert(h_K_full.end(), hk.begin(), hk.end());
        h_V_full.insert(h_V_full.end(), hv.begin(), hv.end());

        int cur_len = step + 1;
        cudaMalloc(&d_K_flat, h_K_full.size() * sizeof(__half));
        cudaMalloc(&d_V_flat, h_V_full.size() * sizeof(__half));
        cudaMemcpy(d_K_flat, h_K_full.data(), h_K_full.size() * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V_flat, h_V_full.data(), h_V_full.size() * sizeof(__half), cudaMemcpyHostToDevice);

        launch_reference_attention(d_Q, d_K_flat, d_V_flat, d_out_ref, cur_len, num_heads, head_dim);
        launch_fused_tele_attn(d_Q, d_new_k, d_new_v, d_out_fuse, cache);
        cudaDeviceSynchronize();

        float cs = cosine_similarity(d_out_fuse, d_out_ref, num_heads * head_dim);
        total_cosine += cs;
        worst_cosine = fminf(worst_cosine, cs);

        cudaFree(d_K_flat);
        cudaFree(d_V_flat);
    }

    float avg_cs = total_cosine / STEPS;
    bool pass = (avg_cs >= req_avg) && (worst_cosine >= req_worst);
    printf("%s (avg=%.4f worst=%.4f)\n", pass ? "PASS" : "FAIL", avg_cs, worst_cosine);
    if (!pass) {
        printf("  FAIL: required avg >= %.4f, worst >= %.4f\n", req_avg, req_worst);
        exit(1);
    }

    MultiResolutionCache::free(cache);
    cudaFree(d_Q); cudaFree(d_new_k); cudaFree(d_new_v);
    cudaFree(d_out_fuse); cudaFree(d_out_ref);
}

// ---------------------------------------------------------------------------
// Test 4: multiplicity invariant -- sum of all multiplicities == total appended.
// ---------------------------------------------------------------------------
static void test4_multiplicity_invariant(int num_heads, int head_dim, int num_levels) {
    printf("[Test4] multiplicity invariant, num_heads=%d head_dim=%d num_levels=%d ... ",
           num_heads, head_dim, num_levels);

    const int STEPS = 200;
    const int ratio = 4;

    TelescopeConfig cfg;
    cfg.num_levels = num_levels;
    cfg.num_heads  = num_heads;
    cfg.head_dim   = head_dim;
    cfg.importance_ema_alpha = 0.1f;
    cfg.level_capacity[0] = 32;
    for (int l = 1; l < num_levels; ++l) {
        cfg.level_capacity[l]   = max(4, cfg.level_capacity[l-1] / ratio);
        cfg.downsample_ratio[l-1] = ratio;
    }
    cfg.downsample_ratio[num_levels - 1] = ratio;

    MultiResolutionCache* cache = MultiResolutionCache::allocate(cfg);
    assert(cache);

    __half* d_Q     = alloc_half(num_heads * head_dim);
    __half* d_new_k = alloc_half(num_heads * head_dim);
    __half* d_new_v = alloc_half(num_heads * head_dim);
    __half* d_out   = alloc_half(num_heads * head_dim);

    srand(77);

    for (int step = 0; step < STEPS; ++step) {
        fill_randn(d_Q,     num_heads * head_dim);
        fill_randn(d_new_k, num_heads * head_dim);
        fill_randn(d_new_v, num_heads * head_dim);
        launch_fused_tele_attn(d_Q, d_new_k, d_new_v, d_out, cache);
        cudaDeviceSynchronize();
    }

    // Sum multiplicities across all levels.
    uint64_t total_mult = 0;
    for (int l = 0; l < cfg.num_levels; ++l) {
        int cnt = cache->h_count[l];
        std::vector<uint16_t> h_mult(cnt);
        cudaMemcpy(h_mult.data(), cache->level_multiplicity[l],
                   cnt * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        for (int i = 0; i < cnt; ++i) total_mult += h_mult[i];
    }

    bool pass = (total_mult == static_cast<uint64_t>(STEPS));
    printf("%s (sum_mult=%llu expected=%d)\n", pass ? "PASS" : "FAIL",
           static_cast<unsigned long long>(total_mult), STEPS);
    if (!pass) exit(1);

    MultiResolutionCache::free(cache);
    cudaFree(d_Q); cudaFree(d_new_k); cudaFree(d_new_v); cudaFree(d_out);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    printf("=== tele-attn correctness tests ===\n\n");

    // Test 1: full budget, multiple (num_heads, head_dim) configs.
    for (int nh : {8, 32}) {
        for (int hd : {64, 128}) {
            test1_full_budget(nh, hd);
        }
    }

    printf("\n");

    // Test 2: 80% budget, avg > 0.995, worst > 0.98
    for (int nh : {8, 32}) {
        for (int hd : {64, 128}) {
            for (int nl : {2, 3, 4}) {
                test_with_budget(nh, hd, nl, 128, 0.80f, 0.995f, 0.98f, "Test2-80%");
            }
        }
    }

    printf("\n");

    // Test 3: 50% budget, avg > 0.98, worst > 0.95
    for (int nh : {8, 32}) {
        for (int hd : {64, 128}) {
            for (int nl : {2, 3, 4}) {
                test_with_budget(nh, hd, nl, 128, 0.50f, 0.98f, 0.95f, "Test3-50%");
            }
        }
    }

    printf("\n");

    // Test 4: multiplicity invariant.
    for (int nh : {8, 32}) {
        for (int hd : {64, 128}) {
            for (int nl : {2, 3, 4}) {
                test4_multiplicity_invariant(nh, hd, nl);
            }
        }
    }

    printf("\n=== all tests passed ===\n");
    return 0;
}
