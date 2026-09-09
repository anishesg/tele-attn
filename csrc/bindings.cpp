#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "config.cuh"
#include "cache.cuh"
#include "reference.cuh"
#include "fused_tele_attn.cuh"
#include "promotion.cuh"

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

static void check_fp16_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.dtype() == torch::kFloat16,
                name, " must be float16, got ", t.dtype());
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.is_cuda(), name, " must be on CUDA");
}

static void check_shape(const torch::Tensor& t, std::initializer_list<int64_t> shape,
                         const char* name) {
    TORCH_CHECK(static_cast<int>(t.dim()) == static_cast<int>(shape.size()),
                name, " expected ", shape.size(), "D tensor, got ", t.dim(), "D");
    int i = 0;
    for (int64_t s : shape) {
        if (s >= 0) {
            TORCH_CHECK(t.size(i) == s,
                        name, " dim ", i, " expected ", s, " got ", t.size(i));
        }
        ++i;
    }
}

// ---------------------------------------------------------------------------
// Opaque cache handle: wrap MultiResolutionCache* in a Python capsule.
// ---------------------------------------------------------------------------

static void cache_destructor(void* ptr) {
    MultiResolutionCache::free(reinterpret_cast<MultiResolutionCache*>(ptr));
}

torch::Tensor create_cache(
    int num_levels,
    std::vector<int> level_capacities,
    std::vector<int> downsample_ratios,
    int num_heads,
    int head_dim,
    float importance_ema_alpha)
{
    TORCH_CHECK(num_levels >= 1 && num_levels <= TELE_MAX_LEVELS,
                "num_levels must be in [1, ", TELE_MAX_LEVELS, "]");
    TORCH_CHECK(static_cast<int>(level_capacities.size()) == num_levels,
                "level_capacities length must equal num_levels");
    TORCH_CHECK(static_cast<int>(downsample_ratios.size()) == num_levels - 1 ||
                static_cast<int>(downsample_ratios.size()) == num_levels,
                "downsample_ratios length must be num_levels-1 or num_levels");

    TelescopeConfig cfg;
    cfg.num_levels           = num_levels;
    cfg.num_heads            = num_heads;
    cfg.head_dim             = head_dim;
    cfg.importance_ema_alpha = importance_ema_alpha;
    for (int l = 0; l < num_levels; ++l) cfg.level_capacity[l] = level_capacities[l];
    for (int l = 0; l < num_levels - 1; ++l) cfg.downsample_ratio[l] = downsample_ratios[l];
    cfg.downsample_ratio[num_levels - 1] = (num_levels > 1) ? downsample_ratios[num_levels - 2] : 4;

    TORCH_CHECK(cfg.is_valid(), "Invalid TelescopeConfig");

    MultiResolutionCache* cache = MultiResolutionCache::allocate(cfg);
    TORCH_CHECK(cache != nullptr, "Failed to allocate MultiResolutionCache");

    // Wrap as a 1-element int64 tensor storing the pointer.
    // This lets Python hold it as a regular tensor.
    auto handle = torch::tensor(
        {reinterpret_cast<int64_t>(cache)},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    return handle;
}

void destroy_cache(torch::Tensor handle) {
    TORCH_CHECK(handle.dtype() == torch::kInt64 && handle.numel() == 1,
                "handle must be a 1-element int64 tensor");
    auto* cache = reinterpret_cast<MultiResolutionCache*>(handle.item<int64_t>());
    MultiResolutionCache::free(cache);
    // Zero out pointer to prevent double-free from Python.
    handle.fill_(0);
}

static MultiResolutionCache* get_cache(const torch::Tensor& handle) {
    TORCH_CHECK(handle.dtype() == torch::kInt64 && handle.numel() == 1,
                "handle must be a 1-element int64 tensor");
    auto* cache = reinterpret_cast<MultiResolutionCache*>(handle.item<int64_t>());
    TORCH_CHECK(cache != nullptr, "Cache handle is null (already destroyed?)");
    return cache;
}

// ---------------------------------------------------------------------------
// reference_attention(Q, K, V) -> output
//   Q: [num_heads, head_dim] fp16
//   K: [seq_len, num_heads, head_dim] fp16
//   V: [seq_len, num_heads, head_dim] fp16
// ---------------------------------------------------------------------------
torch::Tensor reference_attention(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V)
{
    check_fp16_contiguous(Q, "Q");
    check_fp16_contiguous(K, "K");
    check_fp16_contiguous(V, "V");

    int num_heads = Q.size(0);
    int head_dim  = Q.size(1);
    TORCH_CHECK(Q.dim() == 2, "Q must be [num_heads, head_dim]");
    TORCH_CHECK(K.dim() == 3, "K must be [seq_len, num_heads, head_dim]");
    TORCH_CHECK(K.size(1) == num_heads && K.size(2) == head_dim, "K/Q shape mismatch");
    TORCH_CHECK(V.sizes() == K.sizes(), "V must match K shape");

    int seq_len = static_cast<int>(K.size(0));
    auto out = torch::empty({num_heads, head_dim},
                             torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_reference_attention(
        reinterpret_cast<const __half*>(Q.data_ptr()),
        reinterpret_cast<const __half*>(K.data_ptr()),
        reinterpret_cast<const __half*>(V.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        seq_len, num_heads, head_dim, stream);

    return out;
}

// ---------------------------------------------------------------------------
// fused_telescoping_attention(Q, new_k, new_v, cache_handle) -> output
//   Q, new_k, new_v: [num_heads, head_dim] fp16
// ---------------------------------------------------------------------------
torch::Tensor fused_telescoping_attention(
    torch::Tensor Q,
    torch::Tensor new_k,
    torch::Tensor new_v,
    torch::Tensor cache_handle)
{
    check_fp16_contiguous(Q,     "Q");
    check_fp16_contiguous(new_k, "new_k");
    check_fp16_contiguous(new_v, "new_v");

    int num_heads = Q.size(0);
    int head_dim  = Q.size(1);
    TORCH_CHECK(new_k.sizes() == Q.sizes(), "new_k must match Q shape");
    TORCH_CHECK(new_v.sizes() == Q.sizes(), "new_v must match Q shape");

    MultiResolutionCache* cache = get_cache(cache_handle);
    TORCH_CHECK(cache->cfg.num_heads == num_heads,
                "Q num_heads mismatch with cache");
    TORCH_CHECK(cache->cfg.head_dim == head_dim,
                "Q head_dim mismatch with cache");

    auto out = torch::empty({num_heads, head_dim},
                             torch::TensorOptions().dtype(torch::kFloat16).device(Q.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fused_tele_attn(
        reinterpret_cast<const __half*>(Q.data_ptr()),
        reinterpret_cast<const __half*>(new_k.data_ptr()),
        reinterpret_cast<const __half*>(new_v.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        cache, stream);

    return out;
}

// ---------------------------------------------------------------------------
// promote_cache(cache_handle, from_level) -> num_promotions
// ---------------------------------------------------------------------------
int promote_cache(torch::Tensor cache_handle, int from_level) {
    MultiResolutionCache* cache = get_cache(cache_handle);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    return promote_level(cache, from_level, stream);
}

// ---------------------------------------------------------------------------
// cache_stats(cache_handle) -> dict with per-level occupancy and memory info.
// ---------------------------------------------------------------------------
std::vector<std::vector<int>> cache_stats(torch::Tensor cache_handle) {
    MultiResolutionCache* cache = get_cache(cache_handle);
    cache->sync_counts_to_host();

    std::vector<std::vector<int>> result;
    for (int l = 0; l < cache->cfg.num_levels; ++l) {
        int cnt = cache->h_count[l];
        int cap = cache->cfg.level_capacity[l];

        // Sum multiplicities for this level.
        std::vector<uint16_t> h_mult(cnt);
        if (cnt > 0) {
            cudaMemcpy(h_mult.data(), cache->level_multiplicity[l],
                       cnt * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        }
        int total_mult = 0;
        for (int i = 0; i < cnt; ++i) total_mult += h_mult[i];

        result.push_back({l, cnt, cap, total_mult});
    }
    return result;
}

// ---------------------------------------------------------------------------
// Pybind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create_cache",  &create_cache,
          "Allocate a MultiResolutionCache; returns int64 handle tensor",
          py::arg("num_levels"), py::arg("level_capacities"), py::arg("downsample_ratios"),
          py::arg("num_heads"), py::arg("head_dim"), py::arg("importance_ema_alpha") = 0.1f);

    m.def("destroy_cache", &destroy_cache,
          "Free a MultiResolutionCache by handle");

    m.def("reference_attention", &reference_attention,
          "Dense FP16 attention (Q:[H,D], K:[S,H,D], V:[S,H,D]) -> output:[H,D]");

    m.def("fused_telescoping_attention", &fused_telescoping_attention,
          "Fused multi-resolution attention with inline append and promotion",
          py::arg("Q"), py::arg("new_k"), py::arg("new_v"), py::arg("cache_handle"));

    m.def("promote_cache", &promote_cache,
          "Manually trigger promotion from from_level",
          py::arg("cache_handle"), py::arg("from_level") = 0);

    m.def("cache_stats", &cache_stats,
          "Returns list of [level, count, capacity, total_multiplicities] per level");
}
