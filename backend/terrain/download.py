"""
Download HiRISE DEM tiles from the USGS Mars 2020 TRN dataset (Jezero Crater).

Usage:
    python -m backend.terrain.download
    python -m backend.terrain.download --tile CR_NORTH
"""

import argparse
from pathlib import Path

import certifi
import requests

BASE_URL = (
    "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/HiRISE/"
    "DTM_MOLAtopography_DeltaGeoid_Jezero_{tile}_Edited_affine_1m_Eqc_latTs0_lon0.tif"
)

TILES = {
    "CR_NORTH": ("Jezero crater rim north — 68 MB", "68 MB"),
    "CR_SOUTH": ("Jezero crater rim south — 73 MB", "73 MB"),
    "N":        ("Jezero crater interior north — 100 MB", "100 MB"),
    "DL":       ("Delta lobe — 109 MB", "109 MB"),
    "W":        ("Crater interior west — 123 MB", "123 MB"),
    "E":        ("Crater interior east — 126 MB", "126 MB"),
}

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def download_tile(tile: str = "CR_NORTH") -> Path:
    tile = tile.upper()
    if tile not in TILES:
        raise ValueError(f"Unknown tile '{tile}'. Choose from: {list(TILES)}")

    url = BASE_URL.format(tile=tile)
    dest = DATA_DIR / f"jezero_{tile.lower()}.tif"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"Already downloaded: {dest}")
        return dest

    desc, size = TILES[tile]
    print(f"Downloading {desc} ({size})...")
    print(f"  URL : {url}")
    print(f"  Dest: {dest}")

    with requests.get(url, stream=True, timeout=60, verify=certifi.where()) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = min(downloaded / total * 100, 100)
                    bar = "#" * int(pct / 2)
                    print(f"\r  [{bar:<50}] {pct:5.1f}%", end="", flush=True)
    print(f"\nSaved to {dest}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Jezero Crater HiRISE DEM tile.")
    parser.add_argument(
        "--tile",
        default="CR_NORTH",
        choices=list(TILES),
        help="Which tile to download (default: CR_NORTH, ~68 MB)",
    )
    args = parser.parse_args()
    download_tile(args.tile)
