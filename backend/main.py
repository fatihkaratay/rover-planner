"""
FastAPI backend for the rover path planner.

Endpoints:
  GET  /terrain/info         — DEM metadata (shape, resolution, bounds)
  GET  /terrain/cost-map     — cost map as a PNG heatmap (for overlay)
  POST /plan                 — run A* and return path + stats
"""

import io
import base64
import logging

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.terrain.dem_loader import load_dem, DEMData
from backend.terrain.cost_map import build_cost_map, CostMap
from backend.planner.astar import plan, NoPathError

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Rover Planner API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Terrain loaded once at startup
# ---------------------------------------------------------------------------

DEM_PATH = "backend/data/jezero_cr_north.tif"
# Crop to a manageable demo region with good traversable coverage
CROP = (3000, 0, 1000, 1000)

_dem: DEMData | None = None
_cost_map: CostMap | None = None


@app.on_event("startup")
def load_terrain() -> None:
    global _dem, _cost_map
    log.info("Loading DEM …")
    _dem = load_dem(DEM_PATH, crop=CROP)
    _cost_map = build_cost_map(_dem)
    rows, cols = _cost_map.cost.shape
    traversable = int(_cost_map.traversable.sum())
    log.info(f"Terrain ready: {rows}×{cols}, {traversable:,} traversable cells")


def _require_terrain() -> tuple[DEMData, CostMap]:
    if _dem is None or _cost_map is None:
        raise HTTPException(status_code=503, detail="Terrain not loaded yet.")
    return _dem, _cost_map


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PlanRequest(BaseModel):
    start_row: int
    start_col: int
    goal_row: int
    goal_col: int


class PlanResponse(BaseModel):
    path: list[list[int]]      # [[row, col], ...]
    cost: float
    nodes_expanded: int
    length_m: float


class TerrainInfo(BaseModel):
    rows: int
    cols: int
    resolution_m: float
    elevation_min: float
    elevation_max: float
    traversable_pct: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/terrain/info", response_model=TerrainInfo)
def terrain_info() -> TerrainInfo:
    dem, cm = _require_terrain()
    valid = ~np.isnan(dem.elevation)
    return TerrainInfo(
        rows=int(dem.elevation.shape[0]),
        cols=int(dem.elevation.shape[1]),
        resolution_m=float(dem.resolution_m),
        elevation_min=float(dem.elevation[valid].min()),
        elevation_max=float(dem.elevation[valid].max()),
        traversable_pct=float(cm.traversable.mean() * 100),
    )


@app.get("/terrain/cost-map")
def terrain_cost_map() -> dict:
    """
    Return the cost map as a base64-encoded PNG heatmap.
    Green = traversable (low cost), red = high cost, black = impassable.
    """
    _, cm = _require_terrain()
    png_b64 = _cost_map_to_png(cm)
    return {"image_b64": png_b64, "rows": cm.cost.shape[0], "cols": cm.cost.shape[1]}


@app.post("/plan", response_model=PlanResponse)
def plan_route(req: PlanRequest) -> PlanResponse:
    _, cm = _require_terrain()
    start = (req.start_row, req.start_col)
    goal  = (req.goal_row,  req.goal_col)

    try:
        result = plan(cm, start, goal)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NoPathError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PlanResponse(
        path=[[r, c] for r, c in result.path],
        cost=result.cost,
        nodes_expanded=result.nodes_expanded,
        length_m=len(result.path) * cm.resolution_m,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cost_map_to_png(cm: CostMap) -> str:
    """Render cost map to a PNG and return as base64 string."""
    try:
        from PIL import Image
    except ImportError:
        return ""

    cost = cm.cost
    rows, cols = cost.shape
    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)

    # Traversable: map cost 1→20 to green→red
    finite = np.isfinite(cost)
    if finite.any():
        norm = np.clip((cost[finite] - 1.0) / 19.0, 0.0, 1.0)
        rgb[finite, 0] = (norm * 255).astype(np.uint8)         # R
        rgb[finite, 1] = ((1 - norm) * 200).astype(np.uint8)  # G

    # Impassable: black (already 0)

    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
