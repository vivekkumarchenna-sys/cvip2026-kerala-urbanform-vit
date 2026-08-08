"""E2 (in-city spatial holdout), E3 (leave-one-city-out), and a 'full' model
trained on all cities for wall-to-wall mapping.  10-band Sentinel-2, ViT vs CNN."""
import warnings, os, sys, json, argparse
warnings.filterwarnings("ignore")
import numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path += [os.path.join(ROOT, "code", "training"), os.path.join(ROOT, "code", "models"),
             os.path.join(ROOT, "code", "data")]
from common import loaders, train_model, evaluate, full_metrics, PatchDataset
from backbones import create_model, count_params
from kerala_config import CLASS_NAMES, CITIES
from torch.utils.data import DataLoader

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "results", "metrics"); os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(ROOT, "results", "checkpoints"); os.makedirs(CKPT, exist_ok=True)
NCLS = len(CLASS_NAMES)


def load_city(city, purity_min):
    d = np.load(os.path.join(PROC, f"{city}_patches.npz"))
    keep = d["purity"] >= purity_min
    return {k: d[k][keep] for k in ("X", "y", "purity", "rows", "cols")}


def band_stats():
    s = np.load(os.path.join(PROC, "kerala_band_stats.npz"))
    return s["mean"], s["std"]


def class_weights(y):
    c = np.bincount(y, minlength=NCLS).astype(np.float64)
    w = c.sum() / (NCLS * np.maximum(c, 1))
    return (w / w.mean()).astype(np.float32)


def block_split(city_data, block_px=256, test_frac=0.2, seed=0):
    """Randomized spatial-block holdout: tile the city into ~256 m blocks and assign
    whole blocks to train/test at random. Overlapping (stride-32) patches within a block
    stay together, controlling leakage, while both splits sample the full class mix."""
    rng = np.random.RandomState(seed)
    r = (city_data["rows"] // block_px).astype(np.int64)
    c = (city_data["cols"] // block_px).astype(np.int64)
    blk = r * 100000 + c
    ub = np.unique(blk); rng.shuffle(ub)
    n_test = max(1, int(round(test_frac * len(ub))))
    test_blocks = set(ub[:n_test].tolist())
    test_m = np.isin(blk, list(test_blocks))
    return ~test_m, test_m


def subset(d, m):
    return d["X"][m], d["y"][m]


def run_incity(models, cities, epochs, bs, purity_min):
    mean, std = band_stats()
    data = {c: load_city(c, purity_min) for c in cities}
    Xtr_l, ytr_l, Xte_l, yte_l = [], [], [], []
    for c in cities:
        tr_m, te_m = block_split(data[c])
        xt, yt = subset(data[c], tr_m); xe, ye = subset(data[c], te_m)
        Xtr_l.append(xt); ytr_l.append(yt); Xte_l.append(xe); yte_l.append(ye)
        print(f"  {c}: train {len(yt)}  test {len(ye)}")
    Xtr = np.concatenate(Xtr_l); ytr = np.concatenate(ytr_l)
    Xte = np.concatenate(Xte_l); yte = np.concatenate(yte_l)
    cw = class_weights(ytr)
    res = {}
    # 10-band for all models, plus an RGB ablation for vit_s and resnet50
    configs = [(m, None, 10, m) for m in models]
    configs += [(m, [2, 1, 0], 3, f"{m}_rgb") for m in models if m in ("vit_s", "resnet50")]
    for name, bands, ic, tag in configs:
        print(f"\n=== in-city {tag} (in_chans={ic}) ===")
        model, inp = create_model(name, in_chans=ic, num_classes=NCLS, pretrained=True)
        ltr, lva = loaders(Xtr, ytr, Xte, yte, mean, std, bands=bands, bs=bs, img_size=inp)
        model, info = train_model(model, ltr, lva, DEV, epochs=epochs, lr=3e-4, class_weights=cw)
        y, p, _ = evaluate(model, lva, DEV)
        met = full_metrics(y, p, CLASS_NAMES); met["params_M"] = round(count_params(model), 2)
        met["input"] = "RGB" if bands else "MSI(10)"
        res[tag] = met
        print(f"  TEST acc {met['accuracy']:.4f}  macroF1 {met['macro_f1']:.4f}")
        json.dump(res, open(os.path.join(OUT, "kerala_incity.json"), "w"), indent=2)


def run_loco(models, cities, epochs, bs, purity_min):
    mean, std = band_stats()
    data = {c: load_city(c, purity_min) for c in cities}
    res = {}
    for name in models:
        res[name] = {"folds": {}}
        for held in cities:
            tr_cities = [c for c in cities if c != held]
            Xtr = np.concatenate([data[c]["X"] for c in tr_cities])
            ytr = np.concatenate([data[c]["y"] for c in tr_cities])
            Xte, yte = data[held]["X"], data[held]["y"]
            cw = class_weights(ytr)
            print(f"\n=== LOCO {name}: train {tr_cities} -> test {held} ({len(ytr)}/{len(yte)}) ===")
            model, inp = create_model(name, in_chans=10, num_classes=NCLS, pretrained=True)
            ltr, lva = loaders(Xtr, ytr, Xte, yte, mean, std, bs=bs, img_size=inp)
            model, info = train_model(model, ltr, lva, DEV, epochs=epochs, lr=3e-4, class_weights=cw)
            y, p, _ = evaluate(model, lva, DEV)
            met = full_metrics(y, p, CLASS_NAMES)
            res[name]["folds"][held] = met
            print(f"  {held}: acc {met['accuracy']:.4f}  macroF1 {met['macro_f1']:.4f}")
        accs = [res[name]["folds"][c]["accuracy"] for c in cities]
        f1s = [res[name]["folds"][c]["macro_f1"] for c in cities]
        res[name]["mean_accuracy"] = float(np.mean(accs)); res[name]["mean_macro_f1"] = float(np.mean(f1s))
        print(f"  {name} LOCO mean acc {np.mean(accs):.4f}  macroF1 {np.mean(f1s):.4f}")
        json.dump(res, open(os.path.join(OUT, "kerala_loco.json"), "w"), indent=2)


def run_full(models, cities, epochs, bs, purity_min):
    """Train on ALL patches (all cities) -> deployment model for mapping."""
    mean, std = band_stats()
    data = {c: load_city(c, purity_min) for c in cities}
    X = np.concatenate([data[c]["X"] for c in cities]); y = np.concatenate([data[c]["y"] for c in cities])
    idx = np.random.RandomState(0).permutation(len(y)); n_va = int(0.1 * len(y))
    va, tr = idx[:n_va], idx[n_va:]
    cw = class_weights(y[tr])
    for name in models:
        print(f"\n=== full {name} (deployment) ===")
        model, inp = create_model(name, in_chans=10, num_classes=NCLS, pretrained=True)
        ltr, lva = loaders(X[tr], y[tr], X[va], y[va], mean, std, bs=bs, img_size=inp)
        model, info = train_model(model, ltr, lva, DEV, epochs=epochs, lr=3e-4, class_weights=cw)
        torch.save({"state_dict": model.state_dict(), "name": name, "in_chans": 10,
                    "img_size": inp, "classes": CLASS_NAMES},
                   os.path.join(CKPT, f"kerala_{name}_full.pth"))
        print(f"  saved deployment model kerala_{name}_full.pth (val_f1 {info['best_f1']:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", nargs="+", default=["incity", "loco", "full"])
    ap.add_argument("--models", nargs="+", default=["vit_s", "resnet50", "effb0"])
    ap.add_argument("--cities", nargs="+", default=list(CITIES.keys()))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--purity_min", type=float, default=0.5)
    a = ap.parse_args()
    if "incity" in a.mode: run_incity(a.models, a.cities, a.epochs, a.bs, a.purity_min)
    if "loco" in a.mode:   run_loco(a.models, a.cities, a.epochs, a.bs, a.purity_min)
    if "full" in a.mode:   run_full(["vit_s"], a.cities, a.epochs, a.bs, a.purity_min)
    print("DONE")
