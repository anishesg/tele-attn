# tele-attn

Fused telescoping attention: progressive multi-resolution KV-cache with importance-weighted representative selection and multiplicity-correct single-pass decode.

## Problem: The Long-Context KV-Cache Memory Wall

Autoregressive decoding requires storing K and V tensors for every prior token. At 32 heads, head_dim=128, float16, a 100K-token context costs:

```
2 (K+V) * 32 * 128 * 2 bytes * 100000 = 1.6 GB per layer
```

For a 32-layer model this is 51 GB, exceeding the memory of most GPUs. Linear cache growth prevents serving long contexts at reasonable batch sizes.

## Why Naive Averaging Fails

The natural solution, temporally averaging K vectors in a window to produce a single representative, is mathematically incorrect. For a query q and window K_1, ..., K_W:

```
softmax(q . K_avg) != mean_i( softmax(q . K_i) )
```

This fails because softmax contains exp(), a strictly convex function. By Jensen's inequality:

```
exp(q . mean(K_i)) <= mean(exp(q . K_i))
```

The averaged K vector underestimates attention weight for tokens near the query and overestimates it for distant ones, producing systematically biased outputs that degrade with context length.

## Importance-Weighted Representative Selection

Instead of averaging, we retain the highest-importance KV pair as the window's representative, assigning it a multiplicity weight equal to the window size W. Importance is measured by cumulative attention weight received across all prior queries, tracked via exponential moving average:

```
importance[i] = alpha * attn_weight_received[i] + (1 - alpha) * importance[i]
```

This strategy has a concrete approximation bound: the selected representative achieves at least 1/W of the total attention weight of the window, versus O(1/W^2) for random selection. For peaked attention distributions (typical in practice), the best entry captures the majority of window weight.

## Multiplicity-Correct Online Softmax

Standard online softmax for a flat cache:

```
m_new = max(m_old, score_j)
d_new = d_old * exp(m_old - m_new) + exp(score_j - m_new)
o += exp(score_j - m_new) / d_new * V_j
```

With mixed-resolution entries each representing W_j original tokens, the denominator must account for multiplicity. Each entry's contribution to the partition function is scaled by its multiplicity:

```
m_new = max(m_old, score_j)
d_new = d_old * exp(m_old - m_new) + W_j * exp(score_j - m_new)
o += W_j * exp(score_j - m_new) / d_new * V_j
```

This correctly normalizes attention across all levels in a single pass without requiring separate passes per level.

## Multi-Level Cache Layout

```
Level 0: finest resolution, capacity C_0, downsample_ratio R_0
Level 1: coarser,           capacity C_1, downsample_ratio R_1
...
Level N-1: coarsest,        capacity C_{N-1}
```

When level L fills, the oldest R_L entries are grouped into one window, their representative is selected, and the representative is appended to level L+1 with multiplicity W = product of downsample ratios up to L. Promotion cascades if level L+1 also overflows.

## Memory Analysis

For a 3-level cache with downsample_ratio=4 and 50% budget at seq_len=100K:

```
Level 0: 12500 entries  (most recent, full resolution)
Level 1: 3125 entries   (each represents 4 original tokens)
Level 2: 781 entries    (each represents 16 original tokens)
Total:   16406 entries vs 100000 dense = 6.1x reduction
```

Expected quality (from sweep benchmarks): average cosine similarity > 0.98, worst-case > 0.95 for typical attention patterns.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./test_correctness
./bench_tradeoff
./bench_latency
```

Requires CUDA 11.8+ and sm_80+ (Ampere or newer).

## Python Extension

```bash
pip install -e .
```

```python
from tele_attn import TelescopingAttention
attn = TelescopingAttention(
    num_heads=32, head_dim=128,
    level_capacities=[4096, 1024, 256],
    downsample_ratios=[4, 4],
    importance_ema_alpha=0.1
)
output = attn.attend(q)  # q: [num_heads, head_dim] float16
stats = attn.cache_stats()
```
