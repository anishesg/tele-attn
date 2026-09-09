"""
tele_attn: Python interface for fused telescoping attention.
"""

from __future__ import annotations

import torch
from typing import List, Dict, Any

try:
    from tele_attn import _C
    _EXTENSION_AVAILABLE = True
except ImportError:
    _EXTENSION_AVAILABLE = False


def _require_extension() -> None:
    if not _EXTENSION_AVAILABLE:
        raise RuntimeError(
            "tele_attn C extension not found. Run `pip install -e .` to build."
        )


class TelescopingAttention:
    """
    Autoregressive attention with a multi-resolution KV cache.

    The cache stores recent tokens at full resolution (level 0) and progressively
    compressed summaries of older tokens at coarser levels. Compression uses
    importance-weighted representative selection and multiplicity-correct online
    softmax for decode.

    Args:
        num_heads:            Number of attention heads.
        head_dim:             Dimension per head.
        level_capacities:     Maximum KV entries per level, from finest to coarsest.
        downsample_ratios:    Window size for compression at each level transition.
                              Length must be len(level_capacities) - 1.
        importance_ema_alpha: EMA coefficient for importance score updates.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        level_capacities: List[int],
        downsample_ratios: List[int],
        importance_ema_alpha: float = 0.1,
    ) -> None:
        _require_extension()

        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.level_capacities  = level_capacities
        self.downsample_ratios = downsample_ratios
        self.importance_ema_alpha = importance_ema_alpha
        self.num_levels = len(level_capacities)
        self._step = 0

        self._handle = _C.create_cache(
            self.num_levels,
            level_capacities,
            downsample_ratios,
            num_heads,
            head_dim,
            importance_ema_alpha,
        )

    def _validate_input(self, t: torch.Tensor, name: str) -> None:
        if t.dtype != torch.float16:
            raise TypeError(f"{name} must be float16, got {t.dtype}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if not t.is_cuda:
            raise ValueError(f"{name} must be on CUDA")
        if t.shape != (self.num_heads, self.head_dim):
            raise ValueError(
                f"{name} expected shape ({self.num_heads}, {self.head_dim}), got {tuple(t.shape)}"
            )

    def attend(self, q: torch.Tensor, new_k: torch.Tensor, new_v: torch.Tensor) -> torch.Tensor:
        """
        Run attention over the current cache with query q, then append new_k/new_v.

        Args:
            q:     [num_heads, head_dim] float16 query.
            new_k: [num_heads, head_dim] float16 key to append after attention.
            new_v: [num_heads, head_dim] float16 value to append after attention.

        Returns:
            output: [num_heads, head_dim] float16 attention output.
        """
        self._validate_input(q,     "q")
        self._validate_input(new_k, "new_k")
        self._validate_input(new_v, "new_v")

        out = _C.fused_telescoping_attention(q, new_k, new_v, self._handle)
        self._step += 1
        return out

    def reference_attend(
        self,
        q:   torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dense reference attention over explicit full KV tensors (no cache).

        Args:
            q:      [num_heads, head_dim] float16.
            full_k: [seq_len, num_heads, head_dim] float16.
            full_v: [seq_len, num_heads, head_dim] float16.

        Returns:
            output: [num_heads, head_dim] float16.
        """
        if q.dtype != torch.float16:
            raise TypeError(f"q must be float16, got {q.dtype}")
        if full_k.dtype != torch.float16:
            raise TypeError(f"full_k must be float16, got {full_k.dtype}")
        if full_v.dtype != torch.float16:
            raise TypeError(f"full_v must be float16, got {full_v.dtype}")
        return _C.reference_attention(q, full_k, full_v)

    def reset(self) -> None:
        """Free and reallocate the cache, resetting all levels to empty."""
        _C.destroy_cache(self._handle)
        self._handle = _C.create_cache(
            self.num_levels,
            self.level_capacities,
            self.downsample_ratios,
            self.num_heads,
            self.head_dim,
            self.importance_ema_alpha,
        )
        self._step = 0

    def cache_stats(self) -> Dict[str, Any]:
        """
        Return per-level cache statistics.

        Returns a dict with:
            - 'levels': list of dicts, each with 'level', 'count', 'capacity',
                        'total_multiplicities', 'memory_mb'
            - 'total_entries': sum of counts across all levels
            - 'total_multiplicities': sum of all multiplicities (should equal step count)
            - 'total_memory_mb': combined peak memory estimate
        """
        raw = _C.cache_stats(self._handle)
        levels = []
        total_entries = 0
        total_mult    = 0

        for row in raw:
            lvl, cnt, cap, mult = row
            # KV arrays for this level.
            kv_bytes = 2 * cap * self.num_heads * self.head_dim * 2  # fp16
            mem_mb   = kv_bytes / (1024 * 1024)
            levels.append({
                "level": lvl,
                "count": cnt,
                "capacity": cap,
                "total_multiplicities": mult,
                "memory_mb": mem_mb,
            })
            total_entries += cnt
            total_mult    += mult

        total_mem_mb = sum(l["memory_mb"] for l in levels)

        return {
            "levels":               levels,
            "total_entries":        total_entries,
            "total_multiplicities": total_mult,
            "total_memory_mb":      total_mem_mb,
        }

    def __del__(self) -> None:
        if hasattr(self, "_handle") and self._handle is not None:
            try:
                if self._handle.item() != 0:
                    _C.destroy_cache(self._handle)
            except Exception:
                pass

    def __repr__(self) -> str:
        return (
            f"TelescopingAttention(num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"level_capacities={self.level_capacities}, "
            f"downsample_ratios={self.downsample_ratios})"
        )
