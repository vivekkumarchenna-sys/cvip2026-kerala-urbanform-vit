"""Figures. Plots are vector PDF, rasters are RGB PNG with no alpha
channel, sizes match the 122 mm LNCS text width so lettering prints at 7 pt or more,
and colours keep distinct lightness so they survive grayscale printing.

Output: paper/latex/figures/
"""
import warnings, os, sys, json, glob, io
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, FancyBboxPatch
from PIL import Image
import rasterio
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path += [os.path.join(ROOT, "code", "data")]
from kerala_config import CLASS_NAMES, KERALA_DIR, PROC_DIR, CITY_DISP

FIG = os.path.join(ROOT, "paper", "latex", "figures")
MET = os.path.join(ROOT, "results", "metrics")
R2 = os.path.join(MET, "r2")
RAS = os.path.join(KERALA_DIR, "r2")
CITY_ORDER = ["kochi", "trivandrum", "kozhikode"]
PANEL = ["(a)", "(b)", "(c)"]
W = 4.8                                    # inches, LNCS text width
# lightness order: built-up darkest, tree, water, cropland lightest
COLORS = ["#8c1c13", "#3f8f3a", "#8ec6e6", "#e6ef9a"]
CLASS_COLORS = [COLORS[0], COLORS[1], COLORS[3], COLORS[2]]   # class order: built, tree, crop, water

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42, "hatch.linewidth": 0.5,
})


def save_pdf(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); print("  ", name)


def save_png(fig, name, dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig); buf.seek(0)
    Image.open(buf).convert("RGB").save(os.path.join(FIG, name), dpi=(dpi, dpi), optimize=True)
    print("  ", name)


def stretch(img):
    r = img[[2, 1, 0]].astype(np.float32)
    for i in range(3):
        lo, hi = np.percentile(r[i], 2), np.percentile(r[i], 98)
        r[i] = np.clip((r[i] - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.transpose(r, (1, 2, 0))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(W, 1.75)); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 3); ax.set_ylim(0, 2)
    boxes = {(0, 1): "Sentinel-2 L2A\n(open AWS STAC)", (1, 1): "Cloud-free 2021\nmedian composite",
             (2, 1): "Patches with ESA\nWorldCover 2021 labels", (2, 0): "ViT, CNN, SegFormer:\n5-fold and leave-one-city-out",
             (1, 0): "Deployment ViT:\n2018, 2021, 2024 maps", (0, 0): "Land-cover indicators,\n2018 to 2024 built-up change"}
    bw, bh = 0.86, 0.66
    ctr = lambda g: (g[0] + 0.5, g[1] + 0.5)
    for g, label in boxes.items():
        cx, cy = ctr(g)
        ax.add_patch(FancyBboxPatch((cx - bw/2, cy - bh/2), bw, bh, boxstyle="round,pad=0.01,rounding_size=0.04",
                                    linewidth=0.8, edgecolor="#33455a", facecolor="#eef2f7"))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=7.8)
    seq = [(0, 1), (1, 1), (2, 1), (2, 0), (1, 0), (0, 0)]
    for a, b in zip(seq, seq[1:]):
        (x0, y0), (x1, y1) = ctr(a), ctr(b)
        if y0 == y1:
            s = bw/2 if x1 > x0 else -bw/2
            start, end = (x0 + s, y0), (x1 - s, y1)
        else:
            start, end = (x0, y0 - bh/2), (x1, y1 + bh/2)
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color="#33455a", lw=1.0, shrinkA=1, shrinkB=1))
    save_pdf(fig, "pipeline.pdf")


def fig_study_area():
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.75))
    for ax, c, p in zip(axes, CITY_ORDER, PANEL):
        with rasterio.open(os.path.join(KERALA_DIR, f"{c}_2021.tif")) as s:
            ax.imshow(stretch(s.read()))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{p} {CITY_DISP[c]}", fontsize=7.5)
    fig.tight_layout(w_pad=0.4); save_png(fig, "study-area.png")


def fig_patches_and_distribution():
    per, rows = {}, {}
    for c in CITY_ORDER:
        d = np.load(os.path.join(PROC_DIR, "y2021", f"{c}_patches.npz"))
        keep = d["purity"] >= 0.5
        rows[c] = [int((d["y"][keep] == i).sum()) for i in range(4)]
        for cls in range(4):
            idx = np.where((d["y"] == cls) & (d["purity"] > 0.9))[0]
            if cls not in per and len(idx):
                per[cls] = d["X"][idx[0]]
    fig, axes = plt.subplots(1, 5, figsize=(W, 1.1), gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.02]})
    for ax, cls in zip(axes, sorted(per)):
        ax.imshow(stretch(per[cls].astype(np.float32))); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(CLASS_NAMES[cls], fontsize=7.5)
    axes[-1].axis("off")
    fig.tight_layout(w_pad=0.3); save_png(fig, "sample-patches.png")
    fig, ax = plt.subplots(figsize=(W * 0.72, 1.7)); x = np.arange(4); w = 0.8 / 3
    fc, ht = ["#333333", "#9a9a9a", "#ffffff"], ["", "///", "..."]
    for k, c in enumerate(CITY_ORDER):
        v = 100 * np.array(rows[c]) / sum(rows[c])
        ax.bar(x + k * w, v, w, label=CITY_DISP[c], color=fc[k], hatch=ht[k], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x + w); ax.set_xticklabels(CLASS_NAMES); ax.set_ylabel("Patches (%)")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); save_pdf(fig, "class-distribution.pdf")


def fig_eurosat():
    res = json.load(open(os.path.join(MET, "eurosat_benchmark.json")))["results"]
    order = ["vit_s_MSI", "resnet50_MSI", "effb0_MSI", "vit_s_RGB", "resnet50_RGB"]
    disp = {"vit_s": "ViT-S", "resnet50": "ResNet-50", "effb0": "EffNet-B0"}
    labels = [f"{disp[k.rsplit('_',1)[0]]}\n{k.rsplit('_',1)[1]}" for k in order]
    acc, f1 = [res[k]["accuracy"] for k in order], [res[k]["macro_f1"] for k in order]
    fig, ax = plt.subplots(figsize=(W * 0.8, 1.7)); x = np.arange(5)
    ax.bar(x - 0.2, acc, 0.4, label="Accuracy", color="#444444", edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.2, f1, 0.4, label="Macro-F1", color="#ffffff", hatch="///", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.96, 0.99); ax.set_ylabel("Score")
    ax.legend(frameon=False, ncol=2, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); save_pdf(fig, "eurosat-benchmark.pdf")


def pooled(tag):
    cm = np.zeros((4, 4), int)
    for f in glob.glob(os.path.join(R2, f"cv_y2021_{tag}_wsqrt_f*.json")):
        cm += np.array(json.load(open(f))["confusion"])
    return cm


def fig_confusion():
    models = [("vit_s", "(a) ViT-S"), ("segformer_b0", "(b) SegFormer-B0"), ("resnet50", "(c) ResNet-50")]
    short = ["Built", "Tree", "Crop", "Water"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.8), sharey=True)
    for ax, (mk, title) in zip(axes, models):
        cm = pooled(mk).astype(float); cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1.3)
        ax.set_xticks(range(4)); ax.set_xticklabels(short, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(4)); ax.set_yticklabels(short, fontsize=7)
        for i in range(4):
            for j in range(4):
                if cmn[i, j] >= 0.005:
                    ax.text(j, i, f"{cmn[i,j]*100:.0f}", ha="center", va="center", fontsize=7.5,
                            color="white" if cmn[i, j] > 0.6 else "black")
        ax.set_title(title, fontsize=7.5); ax.set_xlabel("Predicted", fontsize=7.5)
    axes[0].set_ylabel("WorldCover 2021", fontsize=7.5)
    fig.tight_layout(w_pad=0.5); save_pdf(fig, "cm-compare.pdf")


def fig_maps():
    cmap = ListedColormap(CLASS_COLORS)
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.0))
    for ax, c, p in zip(axes, CITY_ORDER, PANEL):
        with rasterio.open(os.path.join(RAS, f"{c}_2024_pred.tif")) as s:
            cls = s.read(1)
        ax.imshow(np.ma.masked_equal(cls, 255), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{p} {CITY_DISP[c]}", fontsize=7.5)
    h = [Patch(facecolor=CLASS_COLORS[i], edgecolor="0.3", label=CLASS_NAMES[i]) for i in range(4)]
    fig.legend(handles=h, ncol=4, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(w_pad=0.4, rect=(0, 0.05, 1, 1)); save_png(fig, "map-all.png")


def fig_growth():
    cm = ListedColormap(["#f2f2f2", "#a6a6a6", "#7a0010"])
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.0))
    for ax, c, p in zip(axes, CITY_ORDER, PANEL):
        d = np.load(os.path.join(RAS, f"{c}_growth.npz"))
        ax.imshow(np.ma.masked_where(~d["both"], d["gm"]), cmap=cm, vmin=0, vmax=2, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{p} {CITY_DISP[c]}", fontsize=7.5)
    h = [Patch(facecolor="#a6a6a6", edgecolor="0.3", label="Built-up in 2018"),
         Patch(facecolor="#7a0010", edgecolor="0.3", label="Detected new built-up, 2018 to 2024")]
    fig.legend(handles=h, ncol=2, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(w_pad=0.4, rect=(0, 0.05, 1, 1)); save_png(fig, "growth-all.png")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_pipeline(); fig_study_area(); fig_patches_and_distribution(); fig_eurosat()
    if glob.glob(os.path.join(R2, "cv_y2021_resnet50_wsqrt_f*.json")):
        fig_confusion()
    if os.path.exists(os.path.join(RAS, "kochi_growth.npz")):
        fig_maps(); fig_growth()
    print("DONE")
