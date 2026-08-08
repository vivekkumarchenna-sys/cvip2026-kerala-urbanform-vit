"""Download & cloud-free composite Sentinel-2 L2A for Kerala cities.

For each (city, year): STAC-search dry-season scenes, SCL-mask, take the
per-band temporal median -> a clean 10-band GeoTIFF at 10 m.

Usage:  python download_kerala_s2.py --year 2024 --cities kochi trivandrum kozhikode
"""
import warnings, os, json, argparse, sys
warnings.filterwarnings("ignore")
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["GDAL_HTTP_UNSAFESSL"] = "YES"
os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"
import numpy as np
import rasterio
from rasterio.transform import Affine
from pystac_client import Client
import odc.stac
sys.path.insert(0, os.path.dirname(__file__))
from kerala_config import (STAC_URL, CITIES, BANDS, BAND_CODES, SCL, SCL_KEEP,
                           YEAR_WINDOWS, KERALA_DIR)


def composite_city(city, bbox, year, cloud_lt=20, max_scenes=15):
    win = YEAR_WINDOWS[year]
    cat = Client.open(STAC_URL)
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=win,
                            query={"eo:cloud_cover": {"lt": cloud_lt}}, max_items=100).items())
    if len(items) < 3:
        # widen cloud threshold if too few scenes (rare in dry season)
        items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=win,
                                query={"eo:cloud_cover": {"lt": 40}}, max_items=100).items())
    if not items:
        print(f"  [{city} {year}] NO scenes found"); return None
    # keep the least-cloudy scenes for a fast, clean median composite
    items = sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))[:max_scenes]
    epsg = items[0].properties.get("proj:epsg")
    clouds = [i.properties.get("eo:cloud_cover", -1) for i in items]
    print(f"  [{city} {year}] {len(items)} scenes, cloud {min(clouds):.1f}-{max(clouds):.1f}%, EPSG:{epsg}", flush=True)

    ds = odc.stac.load(items, bands=BANDS + [SCL], crs=f"EPSG:{epsg}", resolution=10,
                       bbox=bbox, groupby="solar_day", chunks={})
    H, W = ds.sizes["y"], ds.sizes["x"]
    scl = ds[SCL].compute()
    keep = scl.isin(SCL_KEEP)
    valid_days = keep.sum("time").compute()             # per-pixel #valid observations
    out = np.zeros((len(BANDS), H, W), dtype=np.uint16)
    for bi, b in enumerate(BANDS):
        med = ds[b].where(keep).median("time").compute().values.astype(np.float32)
        # fill any residual NaN with scene-wide band median
        if np.isnan(med).any():
            fill = np.nanmedian(med)
            med = np.where(np.isnan(med), fill, med)
        out[bi] = np.clip(med, 0, 65535).astype(np.uint16)
        print(f"    band {b:8s} done", flush=True)

    gb = ds.odc.geobox
    transform = gb.transform
    crs = gb.crs
    os.makedirs(KERALA_DIR, exist_ok=True)
    tif = os.path.join(KERALA_DIR, f"{city}_{year}.tif")
    with rasterio.open(tif, "w", driver="GTiff", height=H, width=W, count=len(BANDS),
                       dtype="uint16", crs=str(crs), transform=Affine(*transform[:6]),
                       compress="deflate", tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(out)
        for bi, code in enumerate(BAND_CODES):
            dst.set_band_description(bi + 1, code)
    meta = {"city": city, "year": year, "bbox": bbox, "epsg": int(epsg),
            "n_scenes": len(items), "cloud_min": float(min(clouds)),
            "cloud_max": float(max(clouds)), "height": int(H), "width": int(W),
            "bands": BAND_CODES, "median_valid_days": float(valid_days.values.mean()),
            "window": win, "scene_ids": [i.id for i in items]}
    with open(os.path.join(KERALA_DIR, f"{city}_{year}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [{city} {year}] saved {tif}  ({H}x{W}, mean valid obs/pixel={meta['median_valid_days']:.1f})", flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--cities", nargs="+", default=list(CITIES.keys()))
    ap.add_argument("--cloud_lt", type=int, default=20)
    ap.add_argument("--max_scenes", type=int, default=15)
    args = ap.parse_args()
    for city in args.cities:
        if city not in CITIES:
            print("unknown city", city); continue
        tif = os.path.join(KERALA_DIR, f"{city}_{args.year}.tif")
        if os.path.exists(tif):
            print(f"  [{city} {args.year}] exists, skip"); continue
        composite_city(city, CITIES[city], args.year, args.cloud_lt, args.max_scenes)
    print("DONE")
