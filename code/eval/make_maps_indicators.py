"""E4: wall-to-wall urban-form maps + planning indicators per city.

Slides the trained ViT over each city's 10-band raster, paints a class map,
computes planning indicators, and compares against ESA WorldCover ground truth.
Outputs georeferenced GeoTIFF map, colored PNG, and indicators JSON.
"""
import warnings, os, sys, json, argparse
warnings.filterwarnings("ignore")
import numpy as np, torch, rasterio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path += [os.path.join(ROOT, "code", "models"), os.path.join(ROOT, "code", "data")]
from backbones import create_model
from kerala_config import CLASS_NAMES, CLASS_COLORS, KERALA_DIR, PROC_DIR, CITIES

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FIG = os.path.join(ROOT, "results", "figures"); os.makedirs(FIG, exist_ok=True)
MET = os.path.join(ROOT, "results", "metrics"); os.makedirs(MET, exist_ok=True)
CKPT = os.path.join(ROOT, "results", "checkpoints")
PATCH, STRIDE = 64, 32
NCLS = len(CLASS_NAMES)


def load_model():
    ck = torch.load(os.path.join(CKPT, "kerala_vit_s_full.pth"), map_location=DEV, weights_only=False)
    model, inp = create_model(ck["name"], in_chans=ck["in_chans"], num_classes=NCLS, pretrained=False)
    model.load_state_dict(ck["state_dict"]); model.to(DEV).eval()
    return model, ck.get("img_size", inp)


@torch.no_grad()
def classify_city(city, model, img_size, mean, std):
    with rasterio.open(os.path.join(KERALA_DIR, f"{city}_2024.tif")) as s:
        img = s.read().astype(np.float32); prof = s.profile; transform = s.transform; crs = s.crs
    C, H, W = img.shape
    norm = (img - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)
    votes = np.zeros((NCLS, H, W), np.float32)
    coords, batch = [], []
    def flush():
        nonlocal batch, coords
        if not batch: return
        x = torch.from_numpy(np.stack(batch)).to(DEV)
        if img_size != PATCH:
            x = torch.nn.functional.interpolate(x, size=img_size, mode="bilinear", align_corners=False)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
            pr = model(x).softmax(1).float().cpu().numpy()
        for (r, c), p in zip(coords, pr):
            votes[:, r:r+PATCH, c:c+PATCH] += p.reshape(NCLS, 1, 1)
        batch, coords = [], []
    for r in range(0, H - PATCH + 1, STRIDE):
        for c in range(0, W - PATCH + 1, STRIDE):
            batch.append(norm[:, r:r+PATCH, c:c+PATCH]); coords.append((r, c))
            if len(batch) >= 256: flush()
    flush()
    cls = votes.argmax(0).astype(np.uint8)
    covered = votes.sum(0) > 0
    cls[~covered] = 255
    return cls, transform, crs, prof


def indicators(cls):
    valid = cls[cls != 255]
    frac = {CLASS_NAMES[i]: round(100 * float((valid == i).mean()), 2) for i in range(NCLS)}
    green = frac["Tree"] + frac["Cropland/Grass"]
    return {"class_pct": frac,
            "built_up_pct": frac["Built-up"],
            "impervious_pct": frac["Built-up"],
            "green_pct": round(green, 2),
            "water_pct": frac["Water"],
            "green_to_builtup_ratio": round(green / max(frac["Built-up"], 1e-6), 3)}


def worldcover_dist(city):
    with rasterio.open(os.path.join(KERALA_DIR, f"{city}_label.tif")) as s:
        lab = s.read(1)
    v = lab[lab != 255]
    return {CLASS_NAMES[i]: round(100 * float((v == i).mean()), 2) for i in range(NCLS)}


def save_map(city, cls, transform, crs):
    cmap = ListedColormap(CLASS_COLORS)
    disp = np.ma.masked_equal(cls, 255)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(disp, cmap=cmap, vmin=0, vmax=NCLS-1, interpolation="nearest")
    ax.set_title(f"{city.capitalize()} — predicted urban form (ViT, Sentinel-2 2024)")
    ax.axis("off")
    ax.legend(handles=[Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(NCLS)],
              loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    fig.savefig(os.path.join(FIG, f"map_{city}.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    # georeferenced GeoTIFF
    with rasterio.open(os.path.join(KERALA_DIR, f"{city}_pred.tif"), "w", driver="GTiff",
                       height=cls.shape[0], width=cls.shape[1], count=1, dtype="uint8",
                       crs=crs, transform=transform, nodata=255, compress="deflate") as dst:
        dst.write(cls, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cities", nargs="+", default=list(CITIES.keys()))
    a = ap.parse_args()
    s = np.load(os.path.join(PROC_DIR, "kerala_band_stats.npz")); mean, std = s["mean"], s["std"]
    model, img_size = load_model()
    allind = {}
    for city in a.cities:
        print(f"Classifying {city} ...", flush=True)
        cls, transform, crs, prof = classify_city(city, model, img_size, mean, std)
        save_map(city, cls, transform, crs)
        ind = indicators(cls); ind["worldcover_pct"] = worldcover_dist(city)
        allind[city] = ind
        print(f"  {city}: built-up {ind['built_up_pct']}%  green {ind['green_pct']}%  water {ind['water_pct']}%")
    json.dump(allind, open(os.path.join(MET, "kerala_indicators.json"), "w"), indent=2)
    print("Saved indicators + maps.")
