"""E6: ViT attention-rollout visualisation on representative Kerala patches.

Disables fused attention so timm's manual path exposes softmax weights via a
hook on each block's attn_drop, then computes attention rollout (Abnar & Zuidema).
"""
import warnings, os, sys
warnings.filterwarnings("ignore")
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9, "figure.dpi": 300, "savefig.dpi": 300})
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path += [os.path.join(ROOT, "code", "models"), os.path.join(ROOT, "code", "data")]
from backbones import create_model
from kerala_config import CLASS_NAMES, PROC_DIR, CITIES
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FIG = os.path.join(ROOT, "results", "figures"); os.makedirs(FIG, exist_ok=True)
CKPT = os.path.join(ROOT, "results", "checkpoints")
RGB = [2, 1, 0]  # red,green,blue indices in the 10-band Kerala stack


def load():
    ck = torch.load(os.path.join(CKPT, "kerala_vit_s_full.pth"), map_location=DEV, weights_only=False)
    model, inp = create_model(ck["name"], in_chans=ck["in_chans"], num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(ck["state_dict"]); model.to(DEV).eval()
    return model, ck.get("img_size", inp)


def attach(model):
    maps = []
    for blk in model.blocks:
        blk.attn.fused_attn = False
        blk.attn.attn_drop.register_forward_hook(lambda m, i, o: maps.append(i[0].detach()))
    return maps


def rollout(maps):
    result = None
    for a in maps:                      # a: [1, heads, N, N]
        a = a.mean(1)[0]                # avg heads -> [N,N]
        a = a + torch.eye(a.size(0), device=a.device)
        a = a / a.sum(-1, keepdim=True)
        result = a if result is None else a @ result
    cls_to_patch = result[0, 1:]       # CLS attention to patch tokens
    g = int(cls_to_patch.numel() ** 0.5)
    return cls_to_patch.reshape(g, g).cpu().numpy()


def rgb_stretch(x):
    r = x[RGB].astype(np.float32)
    for i in range(3):
        lo, hi = np.percentile(r[i], 2), np.percentile(r[i], 98)
        r[i] = np.clip((r[i]-lo)/max(hi-lo, 1e-6), 0, 1)
    return np.transpose(r, (1, 2, 0))


if __name__ == "__main__":
    s = np.load(os.path.join(PROC_DIR, "kerala_band_stats.npz")); mean, std = s["mean"], s["std"]
    model, img_size = load(); maps = attach(model)
    # pick one clear example per class from kochi/trivandrum/kozhikode
    pool = {}
    for city in CITIES:
        d = np.load(os.path.join(PROC_DIR, f"{city}_patches.npz"))
        for cls in range(len(CLASS_NAMES)):
            if cls in pool: continue
            idx = np.where((d["y"] == cls) & (d["purity"] > 0.8))[0]
            if len(idx): pool[cls] = d["X"][idx[0]]
    classes = sorted(pool.keys())
    fig, axes = plt.subplots(2, len(classes), figsize=(1.7*len(classes), 3.7))
    im = None
    for j, cls in enumerate(classes):
        x = pool[cls].astype(np.float32)
        xn = (x - mean.reshape(-1,1,1)) / std.reshape(-1,1,1)
        t = torch.from_numpy(xn[None]).to(DEV)
        if img_size != 64:
            t = torch.nn.functional.interpolate(t, size=img_size, mode="bilinear", align_corners=False)
        maps.clear()
        with torch.no_grad(): model(t)
        att = rollout(maps)
        att = (att - att.min()) / (att.max() - att.min() + 1e-8)
        rgb = rgb_stretch(pool[cls])
        axes[0, j].imshow(rgb, extent=(0, 64, 64, 0)); axes[0, j].set_title(CLASS_NAMES[cls], fontsize=8.5)
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        axes[1, j].imshow(rgb, extent=(0, 64, 64, 0))
        im = axes[1, j].imshow(att, cmap="inferno", alpha=0.55, extent=(0, 64, 64, 0),
                               interpolation="bilinear", vmin=0, vmax=1)
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
    axes[0, 0].set_ylabel("True colour", fontsize=8.5)
    axes[1, 0].set_ylabel("Attention", fontsize=8.5)
    fig.tight_layout(w_pad=0.3, h_pad=0.3)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Normalised attention", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.savefig(os.path.join(FIG, "attention_rollout.png"), bbox_inches="tight", pad_inches=0.02)
    print("Saved attention_rollout.png")
