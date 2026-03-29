"""
DEM ingestion for HiRISE terrain data.

Loads a GeoTIFF DEM, extracts elevation data, and computes slope.
All arrays are row-major (row 0 = north, col 0 = west).
"""

import numpy as np
import rasterio
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DEMData:
    elevation: np.ndarray   # 2D float32 array, meters
    slope_deg: np.ndarray   # 2D float32 array, degrees
    resolution_m: float     # ground sample distance in meters
    transform: object       # rasterio Affine transform
    crs: object             # coordinate reference system
    nodata: float | None    # nodata sentinel value


def load_dem(path: str | Path, crop: tuple[int, int, int, int] | None = None) -> DEMData:
    """
    Load a GeoTIFF DEM and compute slope.

    Args:
        path: Path to .tif file.
        crop: Optional (row_off, col_off, height, width) window to load a
              subset — useful for large files during development.

    Returns:
        DEMData with elevation and slope arrays.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DEM not found: {path}")

    with rasterio.open(path) as src:
        if crop is not None:
            row_off, col_off, height, width = crop
            window = rasterio.windows.Window(col_off, row_off, width, height)
            elevation = src.read(1, window=window).astype(np.float32)
            transform = src.window_transform(window)
        else:
            elevation = src.read(1).astype(np.float32)
            transform = src.transform

        nodata = src.nodata
        crs = src.crs
        res_x, res_y = abs(src.res[0]), abs(src.res[1])
        resolution_m = (res_x + res_y) / 2.0

    # Mask nodata as NaN
    if nodata is not None:
        elevation[elevation == nodata] = np.nan

    slope_deg = _compute_slope(elevation, resolution_m)

    return DEMData(
        elevation=elevation,
        slope_deg=slope_deg,
        resolution_m=resolution_m,
        transform=transform,
        crs=crs,
        nodata=nodata,
    )


def _compute_slope(elevation: np.ndarray, resolution_m: float) -> np.ndarray:
    """
    Compute slope in degrees using central differences (Horn's method).

    Slope = arctan(sqrt((dz/dx)^2 + (dz/dy)^2))
    """
    # Fill NaN with local mean for gradient computation, then mask back
    filled = _fill_nan(elevation)
    dz_dy, dz_dx = np.gradient(filled, resolution_m)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # Re-apply NaN mask
    slope_deg[np.isnan(elevation)] = np.nan
    return slope_deg


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    """Replace NaN with nearest valid neighbor (simple row-fill fallback)."""
    filled = arr.copy()
    mask = np.isnan(filled)
    if not mask.any():
        return filled
    # Forward fill along rows, then columns
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    filled = filled[np.arange(mask.shape[0])[:, None], idx]
    filled = np.where(np.isnan(filled), 0.0, filled)
    return filled
