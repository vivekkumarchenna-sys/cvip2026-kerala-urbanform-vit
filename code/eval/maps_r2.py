"""Mapping with the deployment ViT trained on the 2021 composite.

For 2021, 2018 and 2024: wall-to-wall classes by overlap-averaged sliding windows.
  2021: pixel agreement with WorldCover 2021, the only year with a same-year reference.
  2024: planning indicators.
  2018 to 2024: gross new built-up, plus a model-free NDBI check.
Outputs in results/metrics/r2/: pixel_agreement.json, indicators.json, change.json.
Rasters in data/kerala/r2/.
"""
import warnings, os, sys, json
warnings.filterwarnings("ignore")
import numpy as np, torch, rasterio
from sklearn.metrics import cohen_kappa_score
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path += [os.path.join(ROOT, "code", "models"), os.path.join(ROOT, "code", "data")]
from backbones import create_model
from kerala_config import CLASS_NAMES, KERALA_DIR, CITIES

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(ROOT, "results", "metrics", "r2"); os.makedirs(OUT, exist_ok=True)
RAS = os.path.join(KERALA_DIR, "r2"); os.makedirs(RAS, exist_ok=True)
CK = os.path.join(ROOT, "results", "checkpoints", "r2_kerala_vit_s_full_y2021_wsqrt.pth")
PATCH, STRIDE, NCLS, BUILT, WATER = 64, 32, len(CLASS_NAMES), 0, 3
PIX_KM2 = 1e-4


def load():
    ck = torch.load(CK, map_location=DEV, weights_only=False)
    m, _ = create_model("vit_s", in_chans=10, num_classes=NCLS, pretrained=False)
    m.load_state_dict(ck["state_dict"]); m.to(DEV).eval()
    return m, np.asarray(ck["mean"], np.float32), np.asarray(ck["std"], np.float32)


@torch.no_grad()
def classify(model, mean, std, city, year):
    with rasterio.open(os.path.join(KERALA_DIR, f"{city}_{year}.tif")) as s:
        img = s.read().astype(np.float32); prof = s.profile.copy()
    _, H, W = img.shape
    norm = (img - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)
    votes = np.zeros((NCLS, H, W), np.float32)
    batch, coords = [], []

    def flush():
        nonlocal batch, coords
        if not batch:
            return
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pr = model(torch.from_numpy(np.stack(batch)).to(DEV)).softmax(1).float().cpu().numpy()
        for (r, c), p in zip(coords, pr):
            votes[:, r:r+PATCH, c:c+PATCH] += p.reshape(NCLS, 1, 1)
        batch, coords = [], []

    for r in range(0, H - PATCH + 1, STRIDE):
        for c in range(0, W - PATCH + 1, STRIDE):
            batch.append(norm[:, r:r+PATCH, c:c+PATCH]); coords.append((r, c))
            if len(batch) >= 128:
                flush()
    flush()
    cls = votes.argmax(0).astype(np.uint8)
    cls[votes.sum(0) <= 0] = 255
    prof.update(count=1, dtype="uint8", nodata=255, compress="deflate")
    with rasterio.open(os.path.join(RAS, f"{city}_{year}_pred.tif"), "w", **prof) as dst:
        dst.write(cls, 1)
    return cls


def fractions(a):
    v = a[a != 255]
    return {CLASS_NAMES[i]: round(100 * float((v == i).mean()), 2) for i in range(NCLS)}


def agreement(pred, ref):
    m = (pred != 255) & (ref != 255)
    p, r = pred[m].astype(int), ref[m].astype(int)
    cm = np.bincount(NCLS * r + p, minlength=NCLS ** 2).reshape(NCLS, NCLS).astype(np.int64)
    f1 = {}
    for i, n in enumerate(CLASS_NAMES):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        f1[n] = round(float(2 * tp / max(2 * tp + fp + fn, 1)), 3)
    s = np.random.RandomState(0).choice(len(p), min(len(p), 500000), replace=False)
    return {"pixel_accuracy": round(float((p == r).mean()), 4), "kappa": round(float(cohen_kappa_score(r[s], p[s])), 4),
            "per_class_f1": f1, "pred_pct": fractions(pred), "worldcover_pct": fractions(ref),
            "confusion": cm.tolist()}


def ndbi_built_km2(city, year, nonwater):
    with rasterio.open(os.path.join(KERALA_DIR, f"{city}_{year}.tif")) as s:
        nir, swir = s.read(7).astype(np.float32), s.read(9).astype(np.float32)
    ndbi = (swir - nir) / np.maximum(swir + nir, 1)
    return float(((ndbi > 0) & nonwater).sum()) * PIX_KM2


def main():
    model, mean, std = load()
    agree, ind, chg = {}, {}, {}
    for city in CITIES:
        with rasterio.open(os.path.join(KERALA_DIR, f"{city}_label.tif")) as s:
            ref = s.read(1)
        c21 = classify(model, mean, std, city, 2021)
        agree[city] = agreement(c21, ref)
        c24 = classify(model, mean, std, city, 2024)
        c18 = classify(model, mean, std, city, 2018)
        f = fractions(c24)
        green = f["Tree"] + f["Cropland/Grass"]
        ind[city] = {"class_pct": f, "built_up_pct": f["Built-up"], "green_pct": round(green, 2),
                     "water_pct": f["Water"], "green_to_builtup": round(green / max(f["Built-up"], 1e-6), 2)}
        both = (c18 != 255) & (c24 != 255)
        b0, b1 = (c18 == BUILT) & both, (c24 == BUILT) & both
        new, lost = b1 & ~b0, b0 & ~b1
        nonwater = ref != WATER
        n18, n24 = ndbi_built_km2(city, 2018, nonwater), ndbi_built_km2(city, 2024, nonwater)
        chg[city] = {"built_2018_km2": round(float(b0.sum()) * PIX_KM2, 2), "built_2024_km2": round(float(b1.sum()) * PIX_KM2, 2),
                     "new_km2": round(float(new.sum()) * PIX_KM2, 2), "lost_km2": round(float(lost.sum()) * PIX_KM2, 2),
                     "new_pct_of_2018": round(100 * float(new.sum()) / max(float(b0.sum()), 1), 1),
                     "net_km2": round(float(b1.sum() - b0.sum()) * PIX_KM2, 2),
                     "ndbi_2018_km2": round(n18, 2), "ndbi_2024_km2": round(n24, 2),
                     "ndbi_change_pct": round(100 * (n24 - n18) / max(n18, 1e-6), 1)}
        gm = np.zeros(c24.shape, np.uint8); gm[b0] = 1; gm[new] = 2
        np.savez_compressed(os.path.join(RAS, f"{city}_growth.npz"), gm=gm, both=both)
        print(city, agree[city]["pixel_accuracy"], agree[city]["kappa"], ind[city]["class_pct"], chg[city], flush=True)
    json.dump(agree, open(os.path.join(OUT, "pixel_agreement.json"), "w"), indent=2)
    json.dump(ind, open(os.path.join(OUT, "indicators.json"), "w"), indent=2)
    json.dump(chg, open(os.path.join(OUT, "change.json"), "w"), indent=2)
    print("MAPS_DONE")


if __name__ == "__main__":
    main()
