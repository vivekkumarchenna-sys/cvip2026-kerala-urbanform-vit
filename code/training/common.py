"""Shared training utilities: dataset, AMP train loop, metrics. 8GB-friendly."""
import os, time, json, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


class PatchDataset(Dataset):
    """Multispectral patches. X uint16 [N,C,H,W]. Normalizes with band mean/std.
    Optional band subset (e.g. RGB). Geometric augmentation only (spectrally safe).
    `idx` selects a subset of rows WITHOUT copying the big X array (memory-safe)."""
    def __init__(self, X, y, mean, std, bands=None, augment=False, img_size=None, idx=None):
        self.X = X; self.y = y.astype(np.int64)
        self.idx = None if idx is None else np.asarray(idx)
        self.bands = bands
        m = np.asarray(mean, np.float32); s = np.asarray(std, np.float32)
        if bands is not None:
            m = m[bands]; s = s[bands]
        self.mean = m.reshape(-1, 1, 1); self.std = s.reshape(-1, 1, 1)
        self.augment = augment; self.img_size = img_size

    def __len__(self): return len(self.idx) if self.idx is not None else len(self.y)

    def __getitem__(self, i):
        j = int(self.idx[i]) if self.idx is not None else i
        x = self.X[j]
        if self.bands is not None:
            x = x[self.bands]
        x = x.astype(np.float32)
        x = (x - self.mean) / self.std
        if self.augment:
            if np.random.rand() < 0.5: x = x[:, :, ::-1]
            if np.random.rand() < 0.5: x = x[:, ::-1, :]
            k = np.random.randint(4)
            if k: x = np.rot90(x, k, axes=(1, 2))
        x = np.ascontiguousarray(x)
        t = torch.from_numpy(x)
        if self.img_size and self.img_size != t.shape[-1]:
            t = torch.nn.functional.interpolate(t[None], size=self.img_size,
                                                mode="bilinear", align_corners=False)[0]
        return t, int(self.y[j])


def loaders(Xtr, ytr, Xva, yva, mean, std, bands=None, bs=64, img_size=None, workers=0):
    dtr = PatchDataset(Xtr, ytr, mean, std, bands, augment=True, img_size=img_size)
    dva = PatchDataset(Xva, yva, mean, std, bands, augment=False, img_size=img_size)
    ltr = DataLoader(dtr, batch_size=bs, shuffle=True, num_workers=workers,
                     pin_memory=True, drop_last=False)
    lva = DataLoader(dva, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    return ltr, lva


def loaders_idx(X, y, tr_idx, va_idx, mean, std, bands=None, bs=64, img_size=None, workers=0):
    """Memory-safe loaders that index into a shared X (no big row-copies)."""
    dtr = PatchDataset(X, y, mean, std, bands, augment=True, img_size=img_size, idx=tr_idx)
    dva = PatchDataset(X, y, mean, std, bands, augment=False, img_size=img_size, idx=va_idx)
    ltr = DataLoader(dtr, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)
    lva = DataLoader(dva, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    return ltr, lva


def band_stats_chunked(X, idx, n_bands, chunk=1000):
    """Per-band mean/std over X[idx] computed in chunks (no 2 GB copy)."""
    ssum = np.zeros(n_bands, np.float64); ssq = np.zeros(n_bands, np.float64); npix = 0
    for k in range(0, len(idx), chunk):
        xb = X[idx[k:k+chunk]].astype(np.float64)
        ssum += xb.sum((0, 2, 3)); ssq += (xb**2).sum((0, 2, 3))
        npix += xb.shape[0] * xb.shape[2] * xb.shape[3]
    mean = ssum / npix
    std = np.sqrt(np.maximum(ssq / npix - mean**2, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys, ps, probs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            out = model(x)
        p = out.softmax(1).float().cpu().numpy()
        probs.append(p); ps.append(p.argmax(1)); ys.append(y.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return y, p, np.concatenate(probs)


def train_model(model, ltr, lva, device, epochs=30, lr=3e-4, wd=0.05,
                class_weights=None, patience=8, log=print, label_smoothing=0.05):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    cw = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    best_f1, best_state, best_ep, hist = -1, None, -1, []
    for ep in range(epochs):
        model.train(); t0 = time.time(); tl = 0.0
        for x, y in ltr:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * x.size(0)
        sched.step()
        yv, pv, _ = evaluate(model, lva, device)
        present = sorted(set(int(v) for v in np.unique(yv)))  # macro over classes present in truth
        f1 = f1_score(yv, pv, labels=present, average="macro"); acc = accuracy_score(yv, pv)
        hist.append({"epoch": ep, "train_loss": tl/len(ltr.dataset), "val_acc": acc, "val_f1": f1})
        log(f"  ep{ep:02d} loss {tl/len(ltr.dataset):.3f}  val_acc {acc:.4f}  val_f1 {f1:.4f}  ({time.time()-t0:.0f}s)")
        if f1 > best_f1:
            best_f1, best_ep = f1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep - best_ep >= patience:
            log(f"  early stop at ep{ep} (best ep{best_ep} f1 {best_f1:.4f})"); break
    model.load_state_dict(best_state)
    return model, {"best_f1": best_f1, "best_epoch": best_ep, "history": hist}


def full_metrics(y, p, class_names):
    present = sorted(set(int(v) for v in np.unique(y)))  # fair macro over classes in test
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=present, average="macro")),
        "weighted_f1": float(f1_score(y, p, average="weighted")),
        "per_class_f1": {class_names[i]: float(v) for i, v in
                         enumerate(f1_score(y, p, average=None, labels=list(range(len(class_names)))))},
        "confusion": confusion_matrix(y, p, labels=list(range(len(class_names)))).tolist(),
        "report": classification_report(y, p, labels=list(range(len(class_names))),
                                         target_names=class_names, zero_division=0),
    }
