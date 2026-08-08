"""Generate paper figures from cached data + result JSONs. Robust to missing files."""
import warnings, os, sys, json, glob
warnings.filterwarnings("ignore")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import rasterio
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path += [os.path.join(ROOT, "code", "data")]
from kerala_config import CLASS_NAMES, CLASS_COLORS, KERALA_DIR, PROC_DIR, CITIES
FIG = os.path.join(ROOT, "results", "figures"); os.makedirs(FIG, exist_ok=True)
MET = os.path.join(ROOT, "results", "metrics")
RGB = [2, 1, 0]  # red, green, blue in 10-band stack


def stretch(img, idx=RGB):
    r = img[idx].astype(np.float32)
    for i in range(3):
        lo, hi = np.percentile(r[i], 2), np.percentile(r[i], 98)
        r[i] = np.clip((r[i]-lo)/max(hi-lo, 1e-6), 0, 1)
    return np.transpose(r, (1, 2, 0))


def fig_study_area():
    tifs = [os.path.join(KERALA_DIR, f"{c}_2024.tif") for c in CITIES]
    tifs = [t for t in tifs if os.path.exists(t)]
    if not tifs: return
    fig, axes = plt.subplots(1, len(tifs), figsize=(4.2*len(tifs), 4.2))
    if len(tifs) == 1: axes = [axes]
    for ax, t in zip(axes, tifs):
        with rasterio.open(t) as s: img = s.read()
        ax.imshow(stretch(img)); ax.set_title(os.path.basename(t).rsplit("_",1)[0].capitalize())
        ax.axis("off")
    fig.suptitle("Study areas — Sentinel-2 true-colour dry-season composites (2024)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "study_area.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    print("  study_area.png")


def fig_sample_patches():
    files = glob.glob(os.path.join(PROC_DIR, "*_patches.npz"))
    if not files: return
    per_class = {}
    for f in files:
        d = np.load(f)
        for cls in range(len(CLASS_NAMES)):
            if cls in per_class: continue
            idx = np.where((d["y"] == cls) & (d["purity"] > 0.85))[0]
            if len(idx): per_class[cls] = d["X"][idx[0]]
    classes = sorted(per_class)
    if not classes: return
    fig, axes = plt.subplots(1, len(classes), figsize=(1.7*len(classes), 2.0))
    for ax, cls in zip(np.atleast_1d(axes), classes):
        ax.imshow(stretch(per_class[cls].astype(np.float32)))
        ax.set_title(CLASS_NAMES[cls], fontsize=8); ax.axis("off")
    fig.suptitle("Representative Kerala Sentinel-2 patches by class", y=1.05)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sample_patches.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    print("  sample_patches.png")


def fig_class_distribution():
    rows = {}
    for c in CITIES:
        f = os.path.join(PROC_DIR, f"{c}_patches.npz")
        if not os.path.exists(f): continue
        y = np.load(f)["y"]
        rows[c] = [100*(y==i).mean() for i in range(len(CLASS_NAMES))]
    if not rows: return
    fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(CLASS_NAMES)); w = 0.8/len(rows)
    for k, (c, v) in enumerate(rows.items()):
        ax.bar(x + k*w, v, w, label=c.capitalize())
    ax.set_xticks(x + w*(len(rows)-1)/2); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% of patches"); ax.set_title("Patch-label class distribution per city"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "class_distribution.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    print("  class_distribution.png")


def fig_confusion(json_path, key_path, title, out):
    if not os.path.exists(json_path): return
    data = json.load(open(json_path))
    cm = None; names = CLASS_NAMES
    for k in key_path: data = data[k]
    cm = np.array(data["confusion"], float)
    if "classes" in json.load(open(json_path)): names = json.load(open(json_path))["classes"]
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if cmn[i,j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(os.path.join(FIG, out), dpi=170, bbox_inches="tight"); plt.close(fig); print(f"  {out}")


def fig_eurosat_benchmark():
    f = os.path.join(MET, "eurosat_benchmark.json")
    if not os.path.exists(f): return
    res = json.load(open(f))["results"]
    tags = list(res); acc = [res[t]["accuracy"] for t in tags]; f1 = [res[t]["macro_f1"] for t in tags]
    fig, ax = plt.subplots(figsize=(9, 4)); x = np.arange(len(tags))
    ax.bar(x-0.2, acc, 0.4, label="Accuracy"); ax.bar(x+0.2, f1, 0.4, label="Macro-F1")
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0.8, 1.0); ax.set_ylabel("Score"); ax.set_title("EuroSAT-MSI benchmark"); ax.legend()
    for i, v in enumerate(acc): ax.text(i-0.2, v+0.002, f"{v:.3f}", ha="center", fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "eurosat_benchmark.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    print("  eurosat_benchmark.png")


def fig_indicators():
    f = os.path.join(MET, "kerala_indicators.json")
    if not os.path.exists(f): return
    d = json.load(open(f)); cities = list(d)
    metrics = ["built_up_pct", "green_pct", "water_pct", "impervious_pct"]
    fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(metrics)); w = 0.8/len(cities)
    for k, c in enumerate(cities):
        ax.bar(x+k*w, [d[c][m] for m in metrics], w, label=c.capitalize())
    ax.set_xticks(x+w*(len(cities)-1)/2); ax.set_xticklabels(["Built-up","Green","Water","Impervious"])
    ax.set_ylabel("% area"); ax.set_title("Urban-planning indicators per city (ViT, 2024)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "indicators.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    print("  indicators.png")


if __name__ == "__main__":
    print("Figures:")
    fig_study_area(); fig_sample_patches(); fig_class_distribution()
    fig_eurosat_benchmark(); fig_indicators()
    fig_confusion(os.path.join(MET, "eurosat_benchmark.json"), ["results", "vit_s_MSI"],
                  "EuroSAT ViT-S (MSI) confusion", "cm_eurosat_vit.png")
    for m in ["vit_s", "resnet50"]:
        fig_confusion(os.path.join(MET, "kerala_incity.json"), [m],
                      f"Kerala in-city {m} confusion", f"cm_kerala_{m}.png")
    print("DONE")
