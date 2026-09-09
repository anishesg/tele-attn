"""
pytest suite for the Python TelescopingAttention class.

Requires:
  - CUDA-capable GPU
  - tele_attn C extension built (pip install -e .)
"""

import math
import pytest
import torch

from tele_attn import TelescopingAttention


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two float16 tensors, cast to float32."""
    af = a.float().flatten()
    bf = b.float().flatten()
    return float((af @ bf) / (af.norm() * bf.norm() + 1e-12))


def make_attn(
    num_heads: int = 8,
    head_dim: int = 64,
    level_capacities: list = None,
    downsample_ratios: list = None,
) -> TelescopingAttention:
    if level_capacities is None:
        level_capacities = [128, 32, 8]
    if downsample_ratios is None:
        downsample_ratios = [4, 4]
    return TelescopingAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        level_capacities=level_capacities,
        downsample_ratios=downsample_ratios,
        importance_ema_alpha=0.1,
    )


def rand_token(num_heads: int, head_dim: int, device: str = "cuda") -> torch.Tensor:
    return torch.randn(num_heads, head_dim, dtype=torch.float16, device=device)


# ---------------------------------------------------------------------------
# Test 1: 512 random steps at 60% budget; average cosine > 0.99 vs reference.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cosine_quality_60pct_budget() -> None:
    num_heads, head_dim = 8, 64
    steps = 512

    # 60% budget: 307 entries at level 0 out of 512 total.
    attn = TelescopingAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        level_capacities=[307, 76, 19],
        downsample_ratios=[4, 4],
        importance_ema_alpha=0.1,
    )

    # Accumulate reference KV.
    ref_k_list: list[torch.Tensor] = []
    ref_v_list: list[torch.Tensor] = []

    total_cosine = 0.0
    torch.manual_seed(42)

    for _ in range(steps):
        q = rand_token(num_heads, head_dim)
        k = rand_token(num_heads, head_dim)
        v = rand_token(num_heads, head_dim)

        ref_k_list.append(k.unsqueeze(0))  # [1, H, D]
        ref_v_list.append(v.unsqueeze(0))

        full_k = torch.cat(ref_k_list, dim=0)  # [step+1, H, D]
        full_v = torch.cat(ref_v_list, dim=0)

        out_fuse = attn.attend(q, k, v)
        out_ref  = attn.reference_attend(q, full_k, full_v)

        total_cosine += cosine_similarity(out_fuse, out_ref)

    avg_cosine = total_cosine / steps
    assert avg_cosine > 0.99, f"average cosine {avg_cosine:.4f} < 0.99"


# ---------------------------------------------------------------------------
# Test 2: cache_stats multiplicities sum equals step count.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multiplicity_invariant() -> None:
    attn = make_attn()
    steps = 150
    torch.manual_seed(7)

    for _ in range(steps):
        q = rand_token(attn.num_heads, attn.head_dim)
        k = rand_token(attn.num_heads, attn.head_dim)
        v = rand_token(attn.num_heads, attn.head_dim)
        attn.attend(q, k, v)

    stats = attn.cache_stats()
    total_mult = stats["total_multiplicities"]
    assert total_mult == steps, (
        f"total multiplicities {total_mult} != step count {steps}"
    )


# ---------------------------------------------------------------------------
# Test 3: reset clears all levels.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_reset_clears_cache() -> None:
    attn = make_attn()
    torch.manual_seed(99)

    for _ in range(50):
        q = rand_token(attn.num_heads, attn.head_dim)
        k = rand_token(attn.num_heads, attn.head_dim)
        v = rand_token(attn.num_heads, attn.head_dim)
        attn.attend(q, k, v)

    attn.reset()
    stats = attn.cache_stats()

    assert stats["total_entries"] == 0, (
        f"after reset, total_entries={stats['total_entries']} != 0"
    )
    assert stats["total_multiplicities"] == 0, (
        f"after reset, total_multiplicities={stats['total_multiplicities']} != 0"
    )
    for lvl in stats["levels"]:
        assert lvl["count"] == 0, f"level {lvl['level']} count={lvl['count']} != 0 after reset"


# ---------------------------------------------------------------------------
# Test 4: non-float16 inputs are rejected with a TypeError.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dtype_rejection() -> None:
    attn = make_attn()
    q_f32  = torch.randn(attn.num_heads, attn.head_dim, dtype=torch.float32, device="cuda")
    k_f16  = rand_token(attn.num_heads, attn.head_dim)
    v_f16  = rand_token(attn.num_heads, attn.head_dim)

    with pytest.raises((TypeError, RuntimeError)):
        attn.attend(q_f32, k_f16, v_f16)

    with pytest.raises((TypeError, RuntimeError)):
        attn.attend(k_f16, q_f32, v_f16)

    with pytest.raises((TypeError, RuntimeError)):
        attn.attend(k_f16, v_f16, q_f32)
