"""Patchify each city's 10-band raster + WorldCover label into labeled tiles.

Patch = 64x64 (640 m), stride 32 (overlap). Label = majority planning class in
the tile; purity = fraction of that class. Top-left (row,col) stored so an
in-city spatial-block split can avoid overlap leakage.

Outputs:
  data/processed/{city}_patches.npz  (X:uint16[N,10,64,64], y, purity, rows, cols)
  data/processed/kerala_band_stats.npz  (per-band mean/std over all patches)
"""
import warnings, os, sys, glob
warnings.filterwarnings("ignore")
import numpy as np
import rasterio
sys.path.insert(0, os.path.dirname(__file__))
from kerala_config import KERALA_DIR, PROC_DIR, CLASS_NAMES

PATCH = 64
STRIDE = 32
NODATA_LABEL = 255
NCLS = len(CLASS_NAMES)


def patchify(city):
    img_tif = os.path.join(KERALA_DIR, f"{city}_2024.tif")
    lab_tif = os.path.join(KERALA_DIR, f"{city}_label.tif")
    with rasterio.open(img_tif) as s:
        img = s.read()                    # (10,H,W) uint16
    with rasterio.open(lab_tif) as s:
        lab = s.read(1)                   # (H,W) uint8
    C, H, W = img.shape
    Xs, ys, purs, rows, cols = [], [], [], [], []
    for r in range(0, H - PATCH + 1, STRIDE):
        for c in range(0, W - PATCH + 1, STRIDE):
            lp = lab[r:r+PATCH, c:c+PATCH]
            valid = lp[lp != NODATA_LABEL]
            if valid.size < 0.5 * PATCH * PATCH:      # too much nodata
                continue
            counts = np.bincount(valid, minlength=NCLS)
            cls = int(counts.argmax())
            purity = counts[cls] / valid.size
            Xs.append(img[:, r:r+PATCH, c:c+PATCH])
            ys.append(cls); purs.append(purity); rows.append(r); cols.append(c)
    if not Xs:
        print(f"  {city}: 0 valid patches (check labels!)", flush=True)
        return None, None
    X = np.stack(Xs).astype(np.uint16)
    y = np.array(ys, np.int64); pur = np.array(purs, np.float32)
    rows = np.array(rows, np.int32); cols = np.array(cols, np.int32)
    out = os.path.join(PROC_DIR, f"{city}_patches.npz")
    np.savez_compressed(out, X=X, y=y, purity=pur, rows=rows, cols=cols)
    dist = {CLASS_NAMES[i]: int((y == i).sum()) for i in range(NCLS)}
    print(f"  {city}: {len(y)} patches  dist={dist}  (purity>=0.5: {(pur>=0.5).sum()})", flush=True)
    return X, y


if __name__ == "__main__":
    tifs = sorted(glob.glob(os.path.join(KERALA_DIR, "*_2024.tif")))
    if not tifs:
        print("No city rasters. Run download_kerala_s2.py + build_worldcover_labels.py first."); sys.exit(1)
    # accumulate band stats (mean/std) over all patches, streaming
    ssum = np.zeros(10, np.float64); ssq = np.zeros(10, np.float64); npix = 0
    for t in tifs:
        city = os.path.basename(t).rsplit("_", 1)[0]
        X, y = patchify(city)
        if X is None:
            continue
        Xf = X.astype(np.float64)
        ssum += Xf.sum(axis=(0, 2, 3)); ssq += (Xf**2).sum(axis=(0, 2, 3))
        npix += X.shape[0] * X.shape[2] * X.shape[3]
    mean = ssum / npix
    std = np.sqrt(np.maximum(ssq / npix - mean**2, 1e-6))
    np.savez(os.path.join(PROC_DIR, "kerala_band_stats.npz"), mean=mean.astype(np.float32), std=std.astype(np.float32))
    print("band mean:", np.round(mean, 1))
    print("band std :", np.round(std, 1))
    print("DONE")
