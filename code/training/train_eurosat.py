"""E1 + E5(RGB): ViT-vs-CNN benchmark on EuroSAT-MSI (13-band Sentinel-2)."""
import warnings, os, sys, json, time, argparse
warnings.filterwarnings("ignore")
import numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path += [os.path.join(ROOT, "code", "training"), os.path.join(ROOT, "code", "models")]
from common import loaders_idx, train_model, evaluate, full_metrics, band_stats_chunked, PatchDataset
from backbones import create_model, count_params

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "results", "metrics"); os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(ROOT, "results", "checkpoints"); os.makedirs(CKPT, exist_ok=True)
# EuroSAT MSI 13-band order: B01,B02,B03,B04,B05,B06,B07,B08,B09,B10,B11,B12,B8A
RGB_IDX = [3, 2, 1]   # B04,B03,B02


@torch.no_grad()
def latency_ms(model, in_chans, img_size, n=100):
    model.eval(); x = torch.randn(1, in_chans, img_size, img_size, device=DEV)
    for _ in range(10): model(x)
    torch.cuda.synchronize() if DEV == "cuda" else None
    t0 = time.time()
    for _ in range(n): model(x)
    torch.cuda.synchronize() if DEV == "cuda" else None
    return 1000 * (time.time() - t0) / n


def run(models, epochs, bs):
    d = np.load(os.path.join(PROC, "eurosat_msi.npz"), allow_pickle=True)
    X, y = d["X"], d["y"]; classes = [str(c) for c in d["classes"]]
    tr, va, te = d["train_idx"], d["val_idx"], d["test_idx"]
    mean, std = band_stats_chunked(X, tr, X.shape[1])
    print(f"EuroSAT: {len(y)} imgs, {len(classes)} classes, dev={DEV}")
    results = {}
    configs = []
    for m in models:
        configs.append((m, "MSI", None, 13))
        if m in ("vit_s", "resnet50"):
            configs.append((m, "RGB", RGB_IDX, 3))
    for name, mode, bands, ic in configs:
        tag = f"{name}_{mode}"
        print(f"\n=== {tag} (in_chans={ic}) ===")
        model, inp = create_model(name, in_chans=ic, num_classes=len(classes), pretrained=True)
        ltr, lva = loaders_idx(X, y, tr, va, mean, std, bands=bands, bs=bs, img_size=inp, workers=0)
        model, info = train_model(model, ltr, lva, DEV, epochs=epochs, lr=3e-4)
        # test
        from torch.utils.data import DataLoader
        dte = PatchDataset(X, y, mean, std, bands=bands, augment=False, img_size=inp, idx=te)
        lte = DataLoader(dte, batch_size=bs, shuffle=False)
        yt, pt, _ = evaluate(model, lte, DEV)
        met = full_metrics(yt, pt, classes)
        met.update({"params_M": round(count_params(model), 2),
                    "latency_ms": round(latency_ms(model, ic, inp), 3),
                    "best_val_f1": info["best_f1"], "mode": mode, "in_chans": ic})
        results[tag] = met
        torch.save(model.state_dict(), os.path.join(CKPT, f"eurosat_{tag}.pth"))
        print(f"  TEST acc {met['accuracy']:.4f}  macroF1 {met['macro_f1']:.4f}  "
              f"params {met['params_M']}M  lat {met['latency_ms']}ms")
        with open(os.path.join(OUT, "eurosat_benchmark.json"), "w") as f:
            json.dump({"classes": classes, "results": results}, f, indent=2)
    print("\nSaved:", os.path.join(OUT, "eurosat_benchmark.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vit_s", "resnet50", "effb0"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    run(a.models, a.epochs, a.bs)
