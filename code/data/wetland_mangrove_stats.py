"""How much Wetland (90) and Mangrove (95) does the study actually contain?

Re-reads the raw ESA WorldCover 2021 classes on each city grid, reports their
pixel share, and counts the labelled patches that would change under two other
treatments of these classes:
  A  Wetland and Mangrove folded into Water (the paper's rule)
  B  Mangrove -> Tree, Wetland -> Cropland/Grass
  C  patches whose majority raw class is Wetland or Mangrove dropped
Outputs results/metrics/r2/wetland_mangrove.json and data/kerala/{city}_wc_raw.tif.
"""
import warnings, os, sys, json
warnings.filterwarnings("ignore")
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["GDAL_HTTP_UNSAFESSL"] = "YES"
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
sys.path.insert(0, os.path.dirname(__file__))
from kerala_config import KERALA_DIR, PROC_DIR, CITIES, CLASS_NAMES, WC_TO_CLASS, worldcover_tiles

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "results", "metrics", "r2")
os.makedirs(OUT, exist_ok=True)
PATCH, NCLS = 64, len(CLASS_NAMES)
ALT = dict(WC_TO_CLASS); ALT[95] = 1; ALT[90] = 2


def raw_grid(city):
    tif = os.path.join(KERALA_DIR, f"{city}_wc_raw.tif")
    if os.path.exists(tif):
        with rasterio.open(tif) as s:
            return s.read(1)
    with rasterio.open(os.path.join(KERALA_DIR, f"{city}_2024.tif")) as ref:
        crs, tr, H, W = ref.crs, ref.transform, ref.height, ref.width
    wc = np.zeros((H, W), np.uint8)
    for url in worldcover_tiles(CITIES[city]):
        tmp = np.zeros((H, W), np.uint8)
        with rasterio.open(url) as src:
            reproject(source=rasterio.band(src, 1), destination=tmp, src_transform=src.transform,
                      src_crs=src.crs, dst_transform=tr, dst_crs=crs, resampling=Resampling.nearest)
        wc = np.where(wc == 0, tmp, wc)
    with rasterio.open(tif, "w", driver="GTiff", height=H, width=W, count=1, dtype="uint8",
                       crs=crs, transform=tr, compress="deflate") as dst:
        dst.write(wc, 1)
    return wc


def main():
    res = {}
    for city in CITIES:
        wc = raw_grid(city)
        valid = wc > 0
        d = np.load(os.path.join(PROC_DIR, f"{city}_patches.npz"))
        keep = d["purity"] >= 0.5
        rows, cols, y = d["rows"][keep], d["cols"][keep], d["y"][keep]
        lut_alt = np.full(256, 255, np.uint8)
        for k, v in ALT.items():
            lut_alt[k] = v
        changed, majority_wm, water_with_wm = 0, 0, 0
        for r, c, yy in zip(rows, cols, y):
            p = wc[r:r+PATCH, c:c+PATCH]
            p = p[p > 0]
            alt = lut_alt[p]
            cnt = np.bincount(alt[alt != 255], minlength=NCLS)
            changed += int(cnt.argmax() != yy)
            raw = np.bincount(p, minlength=101)
            majority_wm += int(raw.argmax() in (90, 95))
            water_with_wm += int(yy == 3 and (raw[90] + raw[95]) > 0)
        res[city] = {"wetland_pct": round(100*float((wc == 90).sum()/valid.sum()), 3),
                     "mangrove_pct": round(100*float((wc == 95).sum()/valid.sum()), 3),
                     "patches": int(len(y)), "patches_label_changes_under_B": changed,
                     "patches_majority_wetland_or_mangrove": majority_wm,
                     "water_patches_containing_any_wm": water_with_wm}
        print(city, res[city], flush=True)
    json.dump(res, open(os.path.join(OUT, "wetland_mangrove.json"), "w"), indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
