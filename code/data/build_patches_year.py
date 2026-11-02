"""Patchify one year's composite on the WorldCover label grid.

Uses the patch positions already fixed in data/processed/{city}_patches.npz, so
every year has the same patches and labels; only the imagery differs. Also keeps
the 64x64 pixel labels for the segmentation model.

Outputs data/processed/y{year}/{city}_patches.npz (X, Y, y, purity, rows, cols).
"""
import os, sys, argparse
import numpy as np
import rasterio
sys.path.insert(0, os.path.dirname(__file__))
from kerala_config import KERALA_DIR, PROC_DIR, CITIES

PATCH = 64


def build(city, year):
    with rasterio.open(os.path.join(KERALA_DIR, f"{city}_{year}.tif")) as s:
        img = s.read()
    with rasterio.open(os.path.join(KERALA_DIR, f"{city}_label.tif")) as s:
        lab = s.read(1)
    d = np.load(os.path.join(PROC_DIR, f"{city}_patches.npz"))
    rows, cols = d["rows"], d["cols"]
    X = np.stack([img[:, r:r+PATCH, c:c+PATCH] for r, c in zip(rows, cols)]).astype(np.uint16)
    Y = np.stack([lab[r:r+PATCH, c:c+PATCH] for r, c in zip(rows, cols)]).astype(np.uint8)
    out = os.path.join(PROC_DIR, f"y{year}")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, f"{city}_patches.npz"), X=X, Y=Y, y=d["y"], purity=d["purity"],
             rows=rows, cols=cols)
    print(f"  {city} {year}: {len(rows)} patches", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2021, 2024])
    a = ap.parse_args()
    for y in a.years:
        for city in CITIES:
            build(city, y)
    print("DONE")
