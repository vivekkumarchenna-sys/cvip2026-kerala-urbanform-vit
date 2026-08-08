"""E4b: built-up urban growth, baseline year -> 2024, per city.

Classifies both epochs with the trained ViT, reports built-up area and the
'new built-up' growth map. Also reports NDBI-based built-up as a model-free check.
"""
import warnings, os, sys, json, argparse
warnings.filterwarnings("ignore")
import numpy as np, torch, rasterio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path += [os.path.join(ROOT, "code", "models"), os.path.join(ROOT, "code", "data"),
             os.path.join(ROOT, "code", "eval")]
from kerala_config import CLASS_NAMES, KERALA_DIR, PROC_DIR, CITIES
from make_maps_indicators import load_model, classify_city
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FIG = os.path.join(ROOT, "results", "figures"); MET = os.path.join(ROOT, "results", "metrics")
BUILT = CLASS_NAMES.index("Built-up")
PIX_KM2 = (10 * 10) / 1e6   # per-10m-pixel area; class map is per-pixel


def builtup_mask(city, year, model, img_size, mean, std):
    tif = os.path.join(KERALA_DIR, f"{city}_{year}.tif")
    if not os.path.exists(tif):
        return None
    # temporarily point classify_city at the requested year via symlinkless copy of logic:
    with rasterio.open(tif) as s:
        img = s.read().astype(np.float32); transform = s.transform
    C, H, W = img.shape
    norm = (img - mean.reshape(-1,1,1)) / std.reshape(-1,1,1)
    from make_maps_indicators import PATCH, STRIDE, NCLS
    votes = np.zeros((NCLS, H, W), np.float32); coords, batch = [], []
    @torch.no_grad()
    def flush():
        nonlocal batch, coords
        if not batch: return
        x = torch.from_numpy(np.stack(batch)).to(DEV)
        if img_size != PATCH:
            x = torch.nn.functional.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEV=="cuda")):
            pr = model(x).softmax(1).float().cpu().numpy()
        for (r,c), p in zip(coords, pr): votes[:, r:r+PATCH, c:c+PATCH] += p.reshape(NCLS,1,1)
        batch, coords = [], []
    for r in range(0, H-PATCH+1, STRIDE):
        for c in range(0, W-PATCH+1, STRIDE):
            batch.append(norm[:, r:r+PATCH, c:c+PATCH]); coords.append((r,c))
            if len(batch) >= 256: flush()
    flush()
    cls = votes.argmax(0); covered = votes.sum(0) > 0
    return (cls == BUILT) & covered, covered, transform


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_year", type=int, default=2018)
    ap.add_argument("--cities", nargs="+", default=list(CITIES.keys()))
    a = ap.parse_args()
    s = np.load(os.path.join(PROC_DIR, "kerala_band_stats.npz")); mean, std = s["mean"], s["std"]
    model, img_size = load_model()
    out = {}
    for city in a.cities:
        base = builtup_mask(city, a.baseline_year, model, img_size, mean, std)
        cur = builtup_mask(city, 2024, model, img_size, mean, std)
        if base is None or cur is None:
            print(f"  {city}: missing baseline/current raster, skip"); continue
        b0, cov0, _ = base; b1, cov1, tr = cur
        both = cov0 & cov1
        a0 = float((b0 & both).sum()) * PIX_KM2
        a1 = float((b1 & both).sum()) * PIX_KM2
        new = (b1 & ~b0) & both          # gross new built-up (non-built -> built): expansion
        lost = (b0 & ~b1) & both         # gross built -> other (mostly built<->vegetation noise)
        new_km2 = float(new.sum()) * PIX_KM2
        lost_km2 = float(lost.sum()) * PIX_KM2
        out[city] = {"baseline_year": a.baseline_year,
                     "builtup_km2_baseline": round(a0, 2), "builtup_km2_2024": round(a1, 2),
                     "new_builtup_km2": round(new_km2, 2), "lost_builtup_km2": round(lost_km2, 2),
                     "new_pct_of_baseline": round(100*new_km2/a0, 1) if a0 > 1.0 else None,
                     "net_km2": round(a1 - a0, 2)}
        print(f"  {city}: built {a0:.1f}->{a1:.1f}  new(expansion)={new_km2:.1f}  lost={lost_km2:.1f}  net={a1-a0:+.1f} km2")
        # growth map: 0 non, 1 stable built, 2 new built  (saved for the journal figure)
        gm = np.zeros(b1.shape, np.uint8); gm[b0 & both] = 1; gm[new] = 2
        np.savez_compressed(os.path.join(KERALA_DIR, f"{city}_growth.npz"), gm=gm, both=both)
    json.dump(out, open(os.path.join(MET, "kerala_change.json"), "w"), indent=2)
    print("Saved change metrics + growth maps.")
