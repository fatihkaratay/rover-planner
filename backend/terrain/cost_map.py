"""
Traversability cost map from DEM data.

Cost represents how expensive it is for a rover to traverse a cell.
  - 1.0  → flat, fully traversable
  - >1.0 → increasing penalty with slope
  - inf  → impassable (slope too steep or nodata)

The cost model is slope-based for Phase 1. Phase 2+ will add:
  - rock density (from HiRISE orthoimage classification)
  - slip risk (regolith type)
  - energy/thermal weighting
"""

import numpy as np
from dataclasses import dataclass

from .dem_loader import DEMData


# Rover-specific slope limits (degrees).
# Based on MER/MSL operational constraints:
#   Hard limit: rover cannot safely traverse above this slope.
#   Soft limit: above this, cost rises steeply (high-risk zone).
HARD_SLOPE_LIMIT_DEG = 25.0
SOFT_SLOPE_LIMIT_DEG = 15.0


@dataclass
class CostMap:
    cost: np.ndarray        # 2D float32, 1.0 = free, inf = impassable
    traversable: np.ndarray # 2D bool mask — True where cost < inf
    resolution_m: float
    transform: object       # rasterio Affine (passthrough from DEMData)


def build_cost_map(
    dem: DEMData,
    hard_limit_deg: float = HARD_SLOPE_LIMIT_DEG,
    soft_limit_deg: float = SOFT_SLOPE_LIMIT_DEG,
) -> CostMap:
    """
    Build a traversability cost map from a loaded DEM.

    Cost function (slope-based):
      - slope <= soft_limit : cost = 1.0 (flat penalty only)
      - soft_limit < slope < hard_limit : exponential rise from 1 to ~10
      - slope >= hard_limit : cost = inf (impassable)
      - NaN elevation : cost = inf

    Args:
        dem: Loaded DEMData from dem_loader.load_dem().
        hard_limit_deg: Slopes at or above this are impassable.
        soft_limit_deg: Slopes above this incur exponentially rising cost.

    Returns:
        CostMap with cost grid and traversable mask.
    """
    slope = dem.slope_deg
    cost = np.ones(slope.shape, dtype=np.float32)

    # Exponential cost rise in the soft-limit zone
    soft_zone = (slope > soft_limit_deg) & (slope < hard_limit_deg)
    if soft_zone.any():
        # Normalise slope within the soft zone to [0, 1]
        t = (slope[soft_zone] - soft_limit_deg) / (hard_limit_deg - soft_limit_deg)
        # e^(3t) maps 0→1, 1→~20 — steep but not discontinuous
        cost[soft_zone] = np.exp(3.0 * t).astype(np.float32)

    # Hard limit and nodata → impassable
    impassable = (slope >= hard_limit_deg) | np.isnan(slope)
    cost[impassable] = np.inf

    traversable = np.isfinite(cost)

    return CostMap(
        cost=cost,
        traversable=traversable,
        resolution_m=dem.resolution_m,
        transform=dem.transform,
    )
