"""Kerala experiments.

Changes from train_kerala.py, each answering a design weakness:
  - imagery year matches the labels (2021 composite with WorldCover 2021)
  - the best epoch is chosen on a validation split, never on the test data
  - 5-fold spatial-block cross-validation: every patch is tested exactly once
  - a guard band removes train/val patches whose footprint touches a test patch
  - band statistics come from the training patches of each run only

Modes: cv (in-city, pooled over cities), loco (leave one city out), full (deployment).
Each run writes results/metrics/r2/<mode>_<tag>_<fold>.json and is skipped if present.
"""
import warnings, os, sys, json, argparse, time
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path += [os.path.join(ROOT, "code", "training"), os.path.join(ROOT, "code", "models"),
             os.path.join(ROOT, "code", "data")]
from common import train_model, evaluate, full_metrics, PatchDataset, band_stats_chunked
from backbones import create_model, count_params
from kerala_config import CLASS_NAMES, CITIES

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "results", "metrics", "r2"); os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(ROOT, "results", "checkpoints"); os.makedirs(CKPT, exist_ok=True)
NCLS, PATCH, BLOCK, KFOLD = len(CLASS_NAMES), 64, 256, 5
FOLD_SEED = 2026
RGB = [2, 1, 0]
ALT_LABELS = False


def load_city(city, year):
    d = np.load(os.path.join(PROC, f"y{year}", f"{city}_patches.npz"))
    keep = d["purity"] >= 0.5
    out = {k: d[k][keep] for k in ("X", "Y", "y", "rows", "cols")}
    if ALT_LABELS:      # sensitivity: Mangrove -> Tree, Wetland -> Cropland/Grass
        out["y"] = np.load(os.path.join(PROC, "alt_labels", f"{city}.npy"))[keep]
    out["blk"] = (out["rows"] // BLOCK).astype(np.int64) * 100000 + out["cols"] // BLOCK
    return out


def touches(cd, idx, ref):
    """Mask over idx: the patch footprint overlaps the footprint of any patch in ref."""
    H, W = cd["rows"].max() + PATCH, cd["cols"].max() + PATCH
    m = np.zeros((H, W), bool)
    for i in ref:
        r, c = cd["rows"][i], cd["cols"][i]
        m[r:r+PATCH, c:c+PATCH] = True
    return np.array([m[cd["rows"][i]:cd["rows"][i]+PATCH, cd["cols"][i]:cd["cols"][i]+PATCH].any()
                     for i in idx], bool) if len(idx) else np.zeros(0, bool)


def city_split(cd, city, fold):
    """Train/val/test indices of one city: test is block-fold `fold`; val is half of
    the next fold; the guard band drops train/val patches touching test, and train
    patches touching val."""
    ub = np.unique(cd["blk"])
    np.random.RandomState(FOLD_SEED + sum(map(ord, city))).shuffle(ub)
    b2f = {b: i % KFOLD for i, b in enumerate(ub)}
    pf = np.array([b2f[b] for b in cd["blk"]])
    vf_blocks = [b for b in ub if b2f[b] == (fold + 1) % KFOLD][::2]
    te = np.where(pf == fold)[0]
    va = np.where(np.isin(cd["blk"], vf_blocks))[0]
    tr = np.where((pf != fold) & ~np.isin(cd["blk"], vf_blocks))[0]
    tr, va = tr[~touches(cd, tr, te)], va[~touches(cd, va, te)]
    tr = tr[~touches(cd, tr, va)]
    return tr, va, te


def class_weights(y, mode):
    c = np.bincount(y, minlength=NCLS).astype(np.float64)
    w = c.sum() / (NCLS * np.maximum(c, 1))
    if mode == "none":
        return np.ones(NCLS, np.float32)
    if mode == "sqrt":
        w = np.sqrt(w)
    return (w / w.mean()).astype(np.float32)


class SegDS(Dataset):
    def __init__(self, X, Y, mean, std, augment):
        self.X, self.Y, self.augment = X, Y, augment
        self.mean = np.asarray(mean, np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(std, np.float32).reshape(-1, 1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = (self.X[i].astype(np.float32) - self.mean) / self.std
        y = self.Y[i]
        if self.augment:
            if np.random.rand() < 0.5: x, y = x[:, :, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5: x, y = x[:, ::-1, :], y[::-1, :]
            k = np.random.randint(4)
            if k: x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y).astype(np.int64))


@torch.no_grad()
def seg_predict(model, loader):
    """Patch label = the most frequent predicted pixel class."""
    model.eval(); out, inter, total = [], 0, 0
    for x, y in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            p = model(x.to(DEV, non_blocking=True)).argmax(1).cpu()
        out.append(torch.stack([(p == c).sum((1, 2)) for c in range(NCLS)], 1).argmax(1).numpy())
        v = y != 255
        total += int(v.sum()); inter += int((p[v] == y[v]).sum())
    return np.concatenate(out), inter / max(total, 1)


def train_seg(X, Y, ytr_patch, tr, va, te, mean, std, cw, seed, epochs, bs, patience=8):
    import segmentation_models_pytorch as smp
    torch.manual_seed(seed); np.random.seed(seed)
    model = smp.Segformer("mit_b0", encoder_weights="imagenet", in_channels=10, classes=NCLS).to(DEV)
    ltr = DataLoader(SegDS(X[tr], Y[tr], mean, std, True), batch_size=bs, shuffle=True, pin_memory=True)
    lva = DataLoader(SegDS(X[va], Y[va], mean, std, False), batch_size=bs, pin_memory=True)
    lte = DataLoader(SegDS(X[te], Y[te], mean, std, False), batch_size=bs, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, device=DEV), ignore_index=255)
    scaler = torch.amp.GradScaler("cuda")
    best, best_ep, state = -1, -1, None
    yva = ytr_patch[va]
    for ep in range(epochs):
        model.train()
        for x, y in ltr:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = crit(model(x.to(DEV, non_blocking=True)), y.to(DEV, non_blocking=True))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        pv, _ = seg_predict(model, lva)
        f1 = f1_score(yva, pv, labels=sorted(set(yva.tolist())), average="macro")
        if f1 > best:
            best, best_ep = f1, ep
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= patience:
            break
    model.load_state_dict(state)
    p, pix_acc = seg_predict(model, lte)
    return model, p, {"best_val_f1": best, "best_epoch": best_ep, "pixel_accuracy": pix_acc}


def run_one(tag, name, bands, X, Y, y, tr, va, te, seed, weight, epochs, bs, extra):
    Xtr_stats = X[tr]
    mean, std = band_stats_chunked(Xtr_stats, np.arange(len(tr)), X.shape[1])
    if name == "segformer_b0":
        cnt = np.bincount(Y[tr][Y[tr] != 255].ravel(), minlength=NCLS).astype(np.float64)
        w = cnt.sum() / (NCLS * np.maximum(cnt, 1))
        w = np.sqrt(w) if weight == "sqrt" else w
        cw = (w / w.mean()).astype(np.float32)
        model, p, info = train_seg(X, Y, y, tr, va, te, mean, std, cw, seed, epochs, bs)
        yt = y[te]
    else:
        torch.manual_seed(seed); np.random.seed(seed)
        ic = 10 if bands is None else len(bands)
        model, inp = create_model(name, in_chans=ic, num_classes=NCLS, pretrained=True)
        dtr = PatchDataset(X, y, mean, std, bands, augment=True, img_size=inp, idx=tr)
        dva = PatchDataset(X, y, mean, std, bands, augment=False, img_size=inp, idx=va)
        dte = PatchDataset(X, y, mean, std, bands, augment=False, img_size=inp, idx=te)
        ltr = DataLoader(dtr, batch_size=bs, shuffle=True, pin_memory=True)
        lva = DataLoader(dva, batch_size=bs, pin_memory=True)
        lte = DataLoader(dte, batch_size=bs, pin_memory=True)
        model, info = train_model(model, ltr, lva, DEV, epochs=epochs, lr=3e-4,
                                  class_weights=class_weights(y[tr], weight), log=lambda s: None)
        yt, p, _ = evaluate(model, lte, DEV)
        info = {"best_val_f1": info["best_f1"], "best_epoch": info["best_epoch"]}
    met = full_metrics(yt, p, CLASS_NAMES)
    met.update(info); met.update(extra)
    met.update({"y_true": yt.tolist(), "y_pred": np.asarray(p).tolist(), "n_train": int(len(tr)),
                "n_val": int(len(va)), "n_test": int(len(te)), "params_M": round(count_params(model), 2)})
    return model, met, (mean, std)


def configs(models):
    cfg = [(m, None, m) for m in models]
    cfg += [(m, RGB, f"{m}_rgb") for m in models if m in ("vit_s", "resnet50")]
    return cfg


def run_cv(a):
    data = {c: load_city(c, a.year) for c in CITIES}
    keys = ("X", "Y", "y")
    all_ = {k: np.concatenate([data[c][k] for c in CITIES]) for k in keys}
    offs = np.cumsum([0] + [len(data[c]["y"]) for c in CITIES])
    city_id = np.concatenate([np.full(len(data[c]["y"]), i) for i, c in enumerate(CITIES)])
    for name, bands, tag in (configs(a.models) if not a.no_rgb else [(m, None, m) for m in a.models]):
        for fold in a.folds:
            f = os.path.join(OUT, f"cv_y{a.year}{'_altB' if ALT_LABELS else ''}_{tag}{'' if a.weight == 'inverse' else '_w' + a.weight}_f{fold}.json")
            if os.path.exists(f):
                continue
            tr, va, te = [], [], []
            for i, c in enumerate(CITIES):
                t, v, e = city_split(data[c], c, fold)
                tr.append(t + offs[i]); va.append(v + offs[i]); te.append(e + offs[i])
            tr, va, te = map(np.concatenate, (tr, va, te))
            t0 = time.time()
            _, met, _ = run_one(tag, name, bands, all_["X"], all_["Y"], all_["y"], tr, va, te, fold,
                                a.weight, a.epochs, a.bs, {"fold": fold, "year": a.year, "tag": tag})
            met["city_test"] = city_id[te].tolist()
            json.dump(met, open(f, "w"))
            print(f"[{time.strftime('%H:%M')}] cv {tag} f{fold}: acc {met['accuracy']:.4f} mF1 {met['macro_f1']:.4f} "
                  f"crop {met['per_class_f1']['Cropland/Grass']:.3f} ({time.time()-t0:.0f}s)", flush=True)
            torch.cuda.empty_cache()


def run_loco(a):
    data = {c: load_city(c, a.year) for c in CITIES}
    suffix = "" if a.weight == "inverse" else f"_w{a.weight}"
    for name in a.models:
        for held in a.held:
            for seed in a.seeds:
                f = os.path.join(OUT, f"loco_y{a.year}_{name}{suffix}_{held}_s{seed}.json")
                if os.path.exists(f):
                    continue
                trc = [c for c in CITIES if c != held]
                tr, va, offs = [], [], np.cumsum([0] + [len(data[c]["y"]) for c in trc])
                for i, c in enumerate(trc):
                    t, v, e = city_split(data[c], c, seed % KFOLD)
                    t = np.concatenate([t, e])            # no test split inside training cities
                    t = t[~touches(data[c], t, v)]
                    tr.append(t + offs[i]); va.append(v + offs[i])
                n = offs[-1]
                X = np.concatenate([data[c]["X"] for c in trc] + [data[held]["X"]])
                Y = np.concatenate([data[c]["Y"] for c in trc] + [data[held]["Y"]])
                y = np.concatenate([data[c]["y"] for c in trc] + [data[held]["y"]])
                te = np.arange(n, len(y))
                t0 = time.time()
                _, met, _ = run_one(name, name, None, X, Y, y, np.concatenate(tr), np.concatenate(va), te,
                                    seed, a.weight, a.epochs, a.bs, {"held": held, "seed": seed})
                json.dump(met, open(f, "w"))
                print(f"[{time.strftime('%H:%M')}] loco {name}{suffix} ->{held} s{seed}: acc {met['accuracy']:.4f} "
                      f"mF1 {met['macro_f1']:.4f} ({time.time()-t0:.0f}s)", flush=True)
                del X, Y, y
                torch.cuda.empty_cache()


def run_full(a):
    data = {c: load_city(c, a.year) for c in CITIES}
    tr, va, offs = [], [], np.cumsum([0] + [len(data[c]["y"]) for c in CITIES])
    for i, c in enumerate(CITIES):
        t, v, e = city_split(data[c], c, 0)
        t = np.concatenate([t, e]); t = t[~touches(data[c], t, v)]
        tr.append(t + offs[i]); va.append(v + offs[i])
    tr, va = np.concatenate(tr), np.concatenate(va)
    X = np.concatenate([data[c]["X"] for c in CITIES])
    Y = np.concatenate([data[c]["Y"] for c in CITIES])
    y = np.concatenate([data[c]["y"] for c in CITIES])
    model, met, (mean, std) = run_one("vit_s", "vit_s", None, X, Y, y, tr, va, va, 0, a.weight, a.epochs, a.bs, {})
    torch.save({"state_dict": model.state_dict(), "name": "vit_s", "in_chans": 10, "img_size": 64,
                "classes": CLASS_NAMES, "mean": np.asarray(mean), "std": np.asarray(std), "year": a.year},
               os.path.join(CKPT, f"r2_kerala_vit_s_full_y{a.year}{'' if a.weight == 'inverse' else '_w' + a.weight}.pth"))
    print(f"full model saved: val mF1 {met['macro_f1']:.4f} n_train {met['n_train']}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", nargs="+", default=["cv"])
    ap.add_argument("--models", nargs="+", default=["vit_s", "resnet50", "effb0", "segformer_b0"])
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(KFOLD)))
    ap.add_argument("--held", nargs="+", default=list(CITIES.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--weight", default="inverse", choices=["inverse", "sqrt", "none"])
    ap.add_argument("--no_rgb", action="store_true")
    ap.add_argument("--alt_labels", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    ALT_LABELS = a.alt_labels
    if "cv" in a.mode: run_cv(a)
    if "loco" in a.mode: run_loco(a)
    if "full" in a.mode: run_full(a)
    print("DONE", flush=True)
