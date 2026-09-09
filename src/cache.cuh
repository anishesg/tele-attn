#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include "config.cuh"

// Align a byte offset to a 128-byte boundary.
static inline size_t align128(size_t offset) {
    return (offset + 127) & ~static_cast<size_t>(127);
}

// Flat device buffer holding all KV, importance, multiplicity, and count arrays
// for one multi-resolution cache instance.
struct MultiResolutionCache {
    // Backing allocation (single cudaMalloc'd block).
    void* base_ptr;
    size_t total_bytes;

    // Per-level K and V arrays: each [level_capacity[l] * num_heads * head_dim] float16.
    __half* level_k[TELE_MAX_LEVELS];
    __half* level_v[TELE_MAX_LEVELS];

    // Per-level importance scores: [level_capacity[l]] float32.
    float* level_importance[TELE_MAX_LEVELS];

    // Per-level multiplicity: [level_capacity[l]] uint16.
    // Counts how many original tokens each stored entry represents.
    uint16_t* level_multiplicity[TELE_MAX_LEVELS];

    // Per-level occupancy count (host-side shadow; device copy lives in d_counts).
    int32_t  h_count[TELE_MAX_LEVELS];
    int32_t* d_counts;  // device array of length num_levels

    TelescopeConfig cfg;

    // -----------------------------------------------------------------------
    // Device accessor helpers (callable from device code with __device__)
    // -----------------------------------------------------------------------

    __device__ __half* get_k(int level, int pos) const {
        return level_k[level] + static_cast<ptrdiff_t>(pos) * cfg.num_heads * cfg.head_dim;
    }

    __device__ __half* get_v(int level, int pos) const {
        return level_v[level] + static_cast<ptrdiff_t>(pos) * cfg.num_heads * cfg.head_dim;
    }

    __device__ float get_importance(int level, int pos) const {
        return level_importance[level][pos];
    }

    __device__ void set_importance(int level, int pos, float val) {
        level_importance[level][pos] = val;
    }

    __device__ uint16_t get_multiplicity(int level, int pos) const {
        return level_multiplicity[level][pos];
    }

    __device__ void set_multiplicity(int level, int pos, uint16_t val) {
        level_multiplicity[level][pos] = val;
    }

    // -----------------------------------------------------------------------
    // Host lifecycle
    // -----------------------------------------------------------------------

    static MultiResolutionCache* allocate(const TelescopeConfig& cfg) {
        // Compute layout for a single flat cudaMalloc.
        size_t offset = 0;

        // Reserve space for the struct itself at offset 0 (device-side copy).
        // We will cudaMemcpy the host struct into this location.
        offset = align128(sizeof(MultiResolutionCache));
        size_t struct_end = offset;

        // K arrays
        size_t k_offsets[TELE_MAX_LEVELS] = {};
        for (int l = 0; l < cfg.num_levels; ++l) {
            k_offsets[l] = offset;
            offset = align128(offset +
                static_cast<size_t>(cfg.level_capacity[l]) * cfg.num_heads * cfg.head_dim * sizeof(__half));
        }

        // V arrays
        size_t v_offsets[TELE_MAX_LEVELS] = {};
        for (int l = 0; l < cfg.num_levels; ++l) {
            v_offsets[l] = offset;
            offset = align128(offset +
                static_cast<size_t>(cfg.level_capacity[l]) * cfg.num_heads * cfg.head_dim * sizeof(__half));
        }

        // Importance arrays
        size_t imp_offsets[TELE_MAX_LEVELS] = {};
        for (int l = 0; l < cfg.num_levels; ++l) {
            imp_offsets[l] = offset;
            offset = align128(offset + static_cast<size_t>(cfg.level_capacity[l]) * sizeof(float));
        }

        // Multiplicity arrays
        size_t mul_offsets[TELE_MAX_LEVELS] = {};
        for (int l = 0; l < cfg.num_levels; ++l) {
            mul_offsets[l] = offset;
            offset = align128(offset + static_cast<size_t>(cfg.level_capacity[l]) * sizeof(uint16_t));
        }

        // d_counts array
        size_t counts_offset = offset;
        offset = align128(offset + cfg.num_levels * sizeof(int32_t));

        size_t total_bytes = offset;

        char* d_mem = nullptr;
        cudaError_t err = cudaMalloc(&d_mem, total_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "MultiResolutionCache::allocate cudaMalloc failed: %s\n",
                    cudaGetErrorString(err));
            return nullptr;
        }
        cudaMemset(d_mem, 0, total_bytes);

        // Build host struct to mirror device layout.
        MultiResolutionCache* cache = new MultiResolutionCache;
        cache->base_ptr   = d_mem;
        cache->total_bytes = total_bytes;
        cache->cfg        = cfg;

        for (int l = 0; l < cfg.num_levels; ++l) {
            cache->level_k[l]            = reinterpret_cast<__half*>   (d_mem + k_offsets[l]);
            cache->level_v[l]            = reinterpret_cast<__half*>   (d_mem + v_offsets[l]);
            cache->level_importance[l]   = reinterpret_cast<float*>    (d_mem + imp_offsets[l]);
            cache->level_multiplicity[l] = reinterpret_cast<uint16_t*> (d_mem + mul_offsets[l]);
            cache->h_count[l]            = 0;
        }
        cache->d_counts = reinterpret_cast<int32_t*>(d_mem + counts_offset);
        (void)struct_end;

        return cache;
    }

    static void free(MultiResolutionCache* cache) {
        if (!cache) return;
        cudaFree(cache->base_ptr);
        delete cache;
    }

    // Synchronize h_count from device d_counts.
    void sync_counts_to_host() {
        cudaMemcpy(h_count, d_counts, cfg.num_levels * sizeof(int32_t), cudaMemcpyDeviceToHost);
    }

    // Write h_count to device d_counts.
    void sync_counts_to_device() {
        cudaMemcpy(d_counts, h_count, cfg.num_levels * sizeof(int32_t), cudaMemcpyHostToDevice);
    }
};
