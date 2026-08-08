"""Journal-ready figures: consistent serif styling, 300 dpi, combined map/growth
rows with shared legends, and a readable confusion matrix.

Reads saved artifacts (no GPU): city composites, prediction/growth rasters,
patch npzs and result JSONs.  Run AFTER make_maps_indicators + change_detection.
"""
import warnings, os, sys, json, glob
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, FancyBboxPatch
import rasterio
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path += [os.path.join(ROOT, "code", "data")]
from kerala_config import (CLASS_NAMES, CLASS_COLORS, KERALA_DIR, PROC_DIR,
                           CITIES, CITY_DISP)

FIG = os.path.join(ROOT, "results", "figures"); os.makedirs(FIG, exist_ok=True)
MET = os.path.join(ROOT, "results", "metrics")
RGB = [2, 1, 0]                      # red, green, blue indices in the 10-band stack
CITY_ORDER = ["kochi", "trivandrum", "kozhikode"]
PANEL = ["(a)", "(b)", "(c)"]

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "figure.dpi": 300, "savefig.dpi": 300,
})


def _stretch(img, idx=RGB):
    r = img[idx].astype(np.float32)
    for i in range(3):
        lo, hi = np.percentile(r[i], 2), np.percentile(r[i], 98)
        r[i] = np.clip((r[i] - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.transpose(r, (1, 2, 0))


def _save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); print("  ", name)


def fig_study_area():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.3))
    for ax, c, p in zip(axes, CITY_ORDER, PANEL):
        with rasterio.open(os.path.join(KERALA_DIR, f"{c}_2024.tif")) as s:
            img = s.read()
        ax.imshow(_stretch(img)); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{p} {CITY_DISP[c]}")
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.6)
    fig.tight_layout(w_pad=0.6); _save(fig, "study_area.png")


def fig_sample_patches():
    per = {}
    for c in CITY_ORDER:
        f = os.path.join(PROC_DIR, f"{c}_patches.npz")
        if not os.path.exists(f): continue
        d = np.load(f)
        for cls in range(len(CLASS_NAMES)):
            if cls in per: continue
            idx = np.where((d["y"] == cls) & (d["purity"] > 0.9))[0]
            if len(idx): per[cls] = d["X"][idx[0]]
    classes = sorted(per)
    fig, axes = plt.subplots(1, len(classes), figsize=(1.6*len(classes), 1.9))
    for ax, cls in zip(np.atleast_1d(axes), classes):
        ax.imshow(_stretch(per[cls].astype(np.float32)))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(CLASS_NAMES[cls], fontsize=8.5)
        for sp in ax.spines.values(): sp.set_linewidth(0.6)
    fig.tight_layout(w_pad=0.4); _save(fig, "sample_patches.png")


def fig_class_distribution():
    rows = {}
    for c in CITY_ORDER:
        f = os.path.join(PROC_DIR, f"{c}_patches.npz")
        if not os.path.exists(f): continue
        y = np.load(f)["y"]
        rows[c] = [100*(y == i).mean() for i in range(len(CLASS_NAMES))]
    fig, ax = plt.subplots(figsize=(6.2, 2.8)); x = np.arange(len(CLASS_NAMES)); w = 0.8/len(rows)
    palette = ["#4C72B0", "#DD8452", "#55A868"]
    for k, (c, v) in enumerate(rows.items()):
        ax.bar(x + k*w, v, w, label=CITY_DISP[c], color=palette[k % len(palette)], edgecolor="white", linewidth=0.4)
    ax.set_xticks(x + w*(len(rows)-1)/2); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Patches (%)"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); _save(fig, "class_distribution.png")


def fig_eurosat_benchmark():
    f = os.path.join(MET, "eurosat_benchmark.json")
    if not os.path.exists(f): return
    res = json.load(open(f))["results"]
    order = ["vit_s_MSI", "resnet50_MSI", "effb0_MSI", "vit_s_RGB", "resnet50_RGB"]
    disp = {"vit_s": "ViT-S", "resnet50": "ResNet-50", "effb0": "EfficientNet-B0"}
    tags = [k for k in order if k in res]
    labels = [f"{disp[k.rsplit('_',1)[0]]}\n({k.rsplit('_',1)[1]})" for k in tags]
    acc = [res[k]["accuracy"] for k in tags]; f1 = [res[k]["macro_f1"] for k in tags]
    fig, ax = plt.subplots(figsize=(6.4, 3.0)); x = np.arange(len(tags))
    ax.bar(x-0.2, acc, 0.4, label="Accuracy", color="#4C72B0")
    ax.bar(x+0.2, f1, 0.4, label="Macro-F1", color="#DD8452")
    for i, v in enumerate(acc): ax.text(i-0.2, v+0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    for i, v in enumerate(f1): ax.text(i+0.2, v+0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0.90, 1.0); ax.set_ylabel("Score"); ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); _save(fig, "eurosat_benchmark.png")


def fig_confusion(model="vit_s"):
    f = os.path.join(MET, "kerala_incity.json")
    if not os.path.exists(f): return
    cm = np.array(json.load(open(f))[model]["confusion"], float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(CLASS_NAMES)
    for i in range(n):
        for j in range(n):
            val = cmn[i, j]
            if val < 0.005: continue
            ax.text(j, i, f"{val*100:.1f}%\n({int(cm[i,j])})", ha="center", va="center",
                    fontsize=7, color="white" if val > 0.55 else "black")
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    cb = fig.colorbar(im, fraction=0.046, pad=0.03); cb.set_label("Row-normalised proportion", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout(); _save(fig, f"cm_kerala_{model}.png")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.axis("off")
    ax.set_xlim(0, 3); ax.set_ylim(0, 2); ax.set_aspect("auto")
    boxes = {(0, 1): "Sentinel-2 L2A\n(open AWS STAC)",
             (1, 1): "Cloud-free\nmedian composite",
             (2, 1): "Patchify +\nESA WorldCover labels",
             (2, 0): "ViT-S / CNN\ntraining",
             (1, 0): "Wall-to-wall\ninference",
             (0, 0): "Maps, indicators &\n2018-2024 change"}
    W, H = 0.74, 0.5
    ctr = lambda g: (g[0] + 0.5, g[1] + 0.5)
    for g, label in boxes.items():
        cx, cy = ctr(g)
        ax.add_patch(FancyBboxPatch((cx-W/2, cy-H/2), W, H,
                     boxstyle="round,pad=0.015,rounding_size=0.05",
                     linewidth=1.0, edgecolor="#33455a", facecolor="#eef2f7"))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=8.2)
    seq = [(0, 1), (1, 1), (2, 1), (2, 0), (1, 0), (0, 0)]
    for a, b in zip(seq, seq[1:]):
        (ax0, ay0), (bx0, by0) = ctr(a), ctr(b)
        if ay0 == by0:                                   # horizontal: edge to edge
            if bx0 > ax0: start, end = (ax0 + W/2, ay0), (bx0 - W/2, by0)
            else:          start, end = (ax0 - W/2, ay0), (bx0 + W/2, by0)
        else:                                            # vertical (downward)
            start, end = (ax0, ay0 - H/2), (bx0, by0 + H/2)
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color="#33455a", lw=1.4,
                                    shrinkA=1, shrinkB=1))
    _save(fig, "pipeline.png")


def fig_confusion_compare():
    f = os.path.join(MET, "kerala_incity.json")
    if not os.path.exists(f): return
    data = json.load(open(f)); n = len(CLASS_NAMES)
    models = [("vit_s", "(a) ViT-S (10-band)"), ("resnet50", "(b) ResNet-50 (10-band)")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6)); im = None
    for ax, (mk, title) in zip(axes, models):
        if mk not in data:
            ax.set_visible(False); continue
        cm = np.array(data[mk]["confusion"], float)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(CLASS_NAMES, fontsize=7)
        for i in range(n):
            for j in range(n):
                v = cmn[i, j]
                if v < 0.005: continue
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.55 else "black")
        ax.set_title(title, fontsize=9); ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes, fraction=0.024, pad=0.02)
    cb.set_label("Row-normalised proportion", fontsize=8); cb.ax.tick_params(labelsize=7)
    _save(fig, "cm_compare.png")


def fig_maps_row():
    cmap = ListedColormap(CLASS_COLORS)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6))
    for ax, c, p in zip(axes, CITY_ORDER, PANEL):
        tif = os.path.join(KERALA_DIR, f"{c}_pred.tif")
        if not os.path.exists(tif):
            ax.set_visible(False); continue
        with rasterio.open(tif) as s: cls = s.read(1)
        ax.imshow(np.ma.masked_equal(cls, 255), cmap=cmap, vmin=0, vmax=len(CLASS_NAMES)-1,
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{p} {CITY_DISP[c]}")
        for sp in ax.spines.values(): sp.set_linewidth(0.6)
    handles = [Patch(facecolor=CLASS_COLORS[i], edgecolor="0.3", label=CLASS_NAMES[i]) for i in range(len(CLASS_NAMES))]
    fig.legend(handles=handles, ncol=len(CLASS_NAMES), loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(w_pad=0.6, rect=(0, 0.03, 1, 1)); _save(fig, "map_all.png")


def fig_growth_row():
    gcmap = ListedColormap(["#eef0f2", "#7a7a7a", "#e31a1c"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6))
    for ax, c, p in zip(axes, CITY_ORDER, PANEL):
        f = os.path.join(KERALA_DIR, f"{c}_growth.npz")
        if not os.path.exists(f):
            ax.set_visible(False); continue
        d = np.load(f); gm = d["gm"]; both = d["both"]
        ax.imshow(np.ma.masked_where(~both, gm), cmap=gcmap, vmin=0, vmax=2, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{p} {CITY_DISP[c]}")
        for sp in ax.spines.values(): sp.set_linewidth(0.6)
    handles = [Patch(facecolor="#7a7a7a", edgecolor="0.3", label="Stable built-up"),
               Patch(facecolor="#e31a1c", edgecolor="0.3", label="New built-up (2018→2024)")]
    fig.legend(handles=handles, ncol=2, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(w_pad=0.6, rect=(0, 0.04, 1, 1)); _save(fig, "growth_all.png")


if __name__ == "__main__":
    print("Journal figures:")
    fig_pipeline(); fig_study_area(); fig_sample_patches(); fig_class_distribution()
    fig_eurosat_benchmark(); fig_confusion("vit_s"); fig_confusion("resnet50"); fig_confusion_compare()
    fig_maps_row(); fig_growth_row()
    print("DONE")
