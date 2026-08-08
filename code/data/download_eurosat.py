"""Download EuroSAT-MSI (13-band Sentinel-2) and cache as a compact npz.

Output: data/processed/eurosat_msi.npz  (X:uint16 [N,13,64,64], y:int [N], classes)
Also makes a fixed stratified 70/15/15 split (indices saved in the same npz).
"""
import os, sys, warnings
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore")
import numpy as np
from datasets import load_dataset, concatenate_datasets

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)
outfile = os.path.join(OUT, "eurosat_msi.npz")

if os.path.exists(outfile):
    print("Already cached:", outfile); sys.exit(0)

print("Loading blanchon/EuroSAT_MSI (all splits) ...", flush=True)
dsd = load_dataset("blanchon/EuroSAT_MSI")
print("Splits:", {k: len(v) for k, v in dsd.items()}, flush=True)
ds = concatenate_datasets(list(dsd.values()))
n = len(ds)
print("Total samples:", n, flush=True)

# Detect shape from first example
ex0 = ds[0]
a0 = np.asarray(ex0["image"])
print("Raw image array shape:", a0.shape, "dtype:", a0.dtype, flush=True)

def to_chw(a):
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[0] in (13, 12, 10) and a.shape[1] == a.shape[2]:
        return a                      # already C,H,W
    if a.ndim == 3 and a.shape[2] in (13, 12, 10):
        return np.transpose(a, (2, 0, 1))  # H,W,C -> C,H,W
    raise ValueError(f"Unexpected shape {a.shape}")

C, H, W = to_chw(a0).shape
print(f"Standardized to C={C}, H={H}, W={W}", flush=True)

X = np.zeros((n, C, H, W), dtype=np.uint16)
y = np.zeros((n,), dtype=np.int64)
labelnames = ds.features["label"].names if hasattr(ds.features["label"], "names") else None
print("Classes:", labelnames, flush=True)

from tqdm import tqdm
for i in tqdm(range(n)):
    ex = ds[i]
    X[i] = to_chw(ex["image"]).astype(np.uint16)
    y[i] = int(ex["label"])

# Stratified 70/15/15 split, fixed seed
rng = np.random.default_rng(42)
train_idx, val_idx, test_idx = [], [], []
for c in np.unique(y):
    idx = np.where(y == c)[0]
    rng.shuffle(idx)
    n_tr = int(0.70 * len(idx)); n_va = int(0.15 * len(idx))
    train_idx += idx[:n_tr].tolist()
    val_idx   += idx[n_tr:n_tr+n_va].tolist()
    test_idx  += idx[n_tr+n_va:].tolist()
train_idx = np.array(sorted(train_idx)); val_idx = np.array(sorted(val_idx)); test_idx = np.array(sorted(test_idx))
print(f"Split -> train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}", flush=True)

np.savez_compressed(
    outfile, X=X, y=y,
    classes=np.array(labelnames if labelnames else [str(i) for i in range(int(y.max())+1)]),
    train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
)
print("Saved:", outfile, f"({os.path.getsize(outfile)/1e6:.0f} MB)", flush=True)
