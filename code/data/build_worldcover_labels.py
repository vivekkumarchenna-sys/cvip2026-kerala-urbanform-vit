"""Reproject ESA WorldCover onto each city's Sentinel-2 grid, map to planning
classes. Selects the correct 3-deg WorldCover tile(s) per city bbox.
Output: data/kerala/{city}_label.tif (uint8, 0..N-1, 255=nodata)."""
import warnings, os, sys, glob
warnings.filterwarnings("ignore")
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["GDAL_HTTP_UNSAFESSL"] = "YES"
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
sys.path.insert(0, os.path.dirname(__file__))
from kerala_config import WC_TO_CLASS, KERALA_DIR, CLASS_NAMES, CITIES, worldcover_tiles

lut = np.full(256, 255, dtype=np.uint8)
for raw, cls in WC_TO_CLASS.items():
    lut[raw] = cls


def build_for(city):
    city_tif = os.path.join(KERALA_DIR, f"{city}_2024.tif")
    out = os.path.join(KERALA_DIR, f"{city}_label.tif")
    with rasterio.open(city_tif) as ref:
        dst_crs, dst_t, H, W = ref.crs, ref.transform, ref.height, ref.width
    wc = np.zeros((H, W), dtype=np.uint8)   # 0 = uncovered (WorldCover has no class 0)
    for url in worldcover_tiles(CITIES[city]):
        tmp = np.zeros((H, W), dtype=np.uint8)
        try:
            with rasterio.open(url) as src:
                reproject(source=rasterio.band(src, 1), destination=tmp,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=dst_t, dst_crs=dst_crs, resampling=Resampling.nearest)
        except Exception as e:
            print(f"    tile {url.split('_')[-2]} failed: {repr(e)[:80]}"); continue
        wc = np.where(wc == 0, tmp, wc)
    labels = lut[wc]
    with rasterio.open(out, "w", driver="GTiff", height=H, width=W, count=1,
                       dtype="uint8", crs=dst_crs, transform=dst_t,
                       compress="deflate", nodata=255) as dst:
        dst.write(labels, 1)
    valid = labels[labels != 255]
    dist = {CLASS_NAMES[i]: round(100*float((valid == i).mean()), 1) for i in range(len(CLASS_NAMES))}
    print(f"  {city}: {H}x{W}  valid={100*valid.size/labels.size:.1f}%  class% {dist}", flush=True)


if __name__ == "__main__":
    tifs = sorted(glob.glob(os.path.join(KERALA_DIR, "*_2024.tif")))
    if not tifs:
        print("No *_2024.tif city rasters found. Run download_kerala_s2.py first.")
    for t in tifs:
        build_for(os.path.basename(t).rsplit("_", 1)[0])
    print("DONE")
