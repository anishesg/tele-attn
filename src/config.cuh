#pragma once
#include <cstdint>
#include <cstddef>
#include <cassert>

// Maximum number of cache levels supported at compile time.
static constexpr int TELE_MAX_LEVELS = 8;

struct TelescopeConfig {
    int num_levels;
    int num_heads;
    int head_dim;

    // Maximum number of KV entries stored at each level.
    int level_capacity[TELE_MAX_LEVELS];

    // Number of consecutive entries at level L that are merged to produce one
    // representative entry at level L+1. Must be >= 2 for all levels except the last.
    int downsample_ratio[TELE_MAX_LEVELS];

    // EMA coefficient for importance score updates.
    float importance_ema_alpha;

    // Returns total number of float16 elements across all K (or V) arrays.
    __host__ __device__ size_t total_kv_elements() const {
        size_t total = 0;
        for (int l = 0; l < num_levels; ++l) {
            total += static_cast<size_t>(level_capacity[l]) * num_heads * head_dim;
        }
        return total;
    }

    // Returns peak device memory in bytes for one full cache (K + V + importance + multiplicity).
    __host__ size_t peak_device_bytes() const {
        size_t kv_bytes = 2 * total_kv_elements() * sizeof(__half);  // K and V

        size_t aux_entries = 0;
        for (int l = 0; l < num_levels; ++l) {
            aux_entries += static_cast<size_t>(level_capacity[l]);
        }
        size_t importance_bytes   = aux_entries * sizeof(float);
        size_t multiplicity_bytes = aux_entries * sizeof(uint16_t);
        size_t count_bytes        = num_levels  * sizeof(int32_t);

        // 128-byte alignment padding (conservative upper bound: 128 bytes per array)
        size_t padding = 128 * (2 * num_levels + 3);  // K/V per level + importance/multiplicity/count

        return kv_bytes + importance_bytes + multiplicity_bytes + count_bytes + padding;
    }

    // Validates that:
    //   - 1 <= num_levels <= TELE_MAX_LEVELS
    //   - level_capacity forms a non-increasing sequence (coarser levels are smaller)
    //   - downsample_ratio >= 2 for levels 0..num_levels-2
    __host__ bool is_valid() const {
        if (num_levels < 1 || num_levels > TELE_MAX_LEVELS) return false;
        if (num_heads < 1 || head_dim < 1)                  return false;
        if (importance_ema_alpha <= 0.f || importance_ema_alpha > 1.f) return false;

        for (int l = 0; l < num_levels; ++l) {
            if (level_capacity[l] < 1) return false;
            if (l > 0 && level_capacity[l] > level_capacity[l - 1]) return false;
        }
        for (int l = 0; l < num_levels - 1; ++l) {
            if (downsample_ratio[l] < 2) return false;
        }
        return true;
    }
};
