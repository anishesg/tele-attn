#pragma once
#include "cache.cuh"

// Promote the oldest entries from level `from_level` to `from_level+1` using
// representative selection. Groups `ratio` entries per window. After writing
// representatives to level `from_level+1`, compacts level `from_level` by
// shifting remaining entries to the front. Cascades to higher levels if the
// target level overflows.
//
// Returns the number of levels that actually underwent promotion.
int promote_level(MultiResolutionCache* cache, int from_level, cudaStream_t stream = 0);

// Compact level `level` by removing the first `n_remove` entries and shifting
// the rest to the front. Uses a single warp-cooperative kernel.
void compact_level(MultiResolutionCache* cache, int level, int n_remove, cudaStream_t stream = 0);
