"""Aggregate the Kerala runs into LaTeX tables and number macros.
Every figure in the paper comes from results/metrics/ through this file.

In-city scores are pooled over the five test folds, so each of the 7772 patches counts
once. Fold-level macro-F1 is not comparable because some folds hold no Cropland/Grass
patch. Proportions carry Wilson 95% intervals. Paired model comparisons use an exact
McNemar test on the pooled patch predictions.
"""
import os, json, glob, math, sys
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score, accuracy_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MET = os.path.join(ROOT, "results", "metrics")
R2 = os.path.join(MET, "r2")
OUT = os.path.join(ROOT, "paper", "latex", "tables")
os.makedirs(OUT, exist_ok=True)
CL = ["Built-up", "Tree", "Cropland/Grass", "Water"]
CITIES = ["kochi", "trivandrum", "kozhikode"]
S = {}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def pool(prefix):
    fs = sorted(glob.glob(os.path.join(R2, f"{prefix}_f[0-4].json")))
    if not fs:
        return None
    y, p, acc, city = [], [], [], []
    for f in fs:
        m = json.load(open(f)); y += m["y_true"]; p += m["y_pred"]; acc.append(m["accuracy"])
        city += m.get("city_test", [])
    y, p = np.array(y), np.array(p)
    per = f1_score(y, p, labels=[0, 1, 2, 3], average=None, zero_division=0)
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (y, p), 1)
    tp, fp, fn = cm[2, 2], cm[:, 2].sum() - cm[2, 2], cm[2, :].sum() - cm[2, 2]
    return {"folds": len(fs), "n": int(len(y)), "acc": 100 * float(accuracy_score(y, p)),
            "acc_sd": 100 * float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0,
            "mf1": 100 * float(per.mean()), "wf1": 100 * float(f1_score(y, p, average="weighted")),
            "per": [100 * float(v) for v in per], "cm": cm.tolist(), "y": y, "p": p,
            "crop_n": int(tp + fn), "crop_tp": int(tp), "crop_pred": int(tp + fp),
            "crop_rec": 100 * float(tp / max(tp + fn, 1)), "crop_prec": 100 * float(tp / max(tp + fp, 1)),
            "crop_rec_ci": wilson(int(tp), int(tp + fn)), "crop_prec_ci": wilson(int(tp), int(tp + fp))}


def mcnemar(a, b):
    ok_a, ok_b = a["p"] == a["y"], b["p"] == b["y"]
    n10, n01 = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    pv = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return n10, n01, float(pv)


def loco(name, suffix=""):
    out = {}
    for c in CITIES:
        rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R2, f"loco_y2021_{name}{suffix}_{c}_s*.json")))]
        if rows:
            out[c] = {"acc": [100 * r["accuracy"] for r in rows], "mf1": [100 * r["macro_f1"] for r in rows],
                      "per": np.mean([[r["per_class_f1"][k] for k in CL] for r in rows], 0).tolist(),
                      "cm": np.sum([r["confusion"] for r in rows], 0).tolist()}
    return out


def msd(v):
    return f"{np.mean(v):.1f}$\\pm${np.std(v, ddof=1):.1f}" if len(v) > 1 else f"{np.mean(v):.1f}"


def t(name, body):
    open(os.path.join(OUT, name), "w", newline="\n").write(body)


def cmd(L, n, v):
    L.append(f"\\newcommand{{\\{n}}}{{{v}}}")


def main():
    M = []
    DISP = [("vit_s", "ViT-S", "MSI(10)"), ("segformer_b0", "SegFormer-B0", "MSI(10)"), ("resnet50", "ResNet-50", "MSI(10)"),
            ("effb0", "EfficientNet-B0", "MSI(10)"), ("vit_s_rgb", "ViT-S", "RGB"), ("resnet50_rgb", "ResNet-50", "RGB")]
    P = {k: pool(f"cv_y2021_{k}_wsqrt") for k, *_ in DISP}
    rows, rows2 = [], []
    for k, name, inp in DISP:
        r = P[k]
        if not r:
            continue
        rows.append(f"{name} & {inp} & {r['acc']:.1f} & {r['mf1']:.1f} & {r['wf1']:.1f} \\\\")
        rows2.append(f"{name} & {inp} & " + " & ".join(f"{v:.1f}" for v in r["per"]) +
                     f" & {r['crop_tp']}/{r['crop_n']} & {r['crop_tp']}/{r['crop_pred']} \\\\")
    t("incity.tex", "\n".join(rows) + "\n"); t("perclass.tex", "\n".join(rows2) + "\n")
    tagmap = {"vit_s": "Vit", "segformer_b0": "Seg", "resnet50": "Res", "effb0": "Eff", "vit_s_rgb": "VitRgb", "resnet50_rgb": "ResRgb"}
    for k, tg in tagmap.items():
        r = P[k]
        if r:
            cmd(M, f"Acc{tg}", f"{r['acc']:.1f}"); cmd(M, f"Mf{tg}", f"{r['mf1']:.1f}")
            cmd(M, f"Crop{tg}", f"{r['per'][2]:.1f}"); cmd(M, f"CropRec{tg}", f"{r['crop_rec']:.0f}"); cmd(M, f"CropPrec{tg}", f"{r['crop_prec']:.0f}")
            cmd(M, f"CropPrecLo{tg}", f"{r['crop_prec_ci'][0]:.0f}"); cmd(M, f"CropPrecHi{tg}", f"{r['crop_prec_ci'][1]:.0f}")
            cmd(M, f"CropRecLo{tg}", f"{r['crop_rec_ci'][0]:.0f}"); cmd(M, f"CropRecHi{tg}", f"{r['crop_rec_ci'][1]:.0f}")
            print(f"{k:14s} n={r['n']} folds={r['folds']} acc {r['acc']:.2f} mF1 {r['mf1']:.2f} per {[round(v,1) for v in r['per']]} crop tp/n {r['crop_tp']}/{r['crop_n']} prec {r['crop_tp']}/{r['crop_pred']}")
    if P["vit_s"]:
        cmd(M, "CropN", str(P["vit_s"]["crop_n"])); cmd(M, "NPatch", f"{P['vit_s']['n']:,}".replace(",", "{,}"))
    for a, b in (("vit_s", "resnet50"), ("vit_s", "segformer_b0"), ("segformer_b0", "resnet50"), ("vit_s", "vit_s_rgb"), ("resnet50", "resnet50_rgb")):
        if P[a] and P[b] and P[a]["n"] == P[b]["n"]:
            print("  McNemar", a, b, mcnemar(P[a], P[b]))

    # leave-one-city-out
    LD = [("vit_s", "ViT-S"), ("segformer_b0", "SegFormer-B0"), ("resnet50", "ResNet-50"), ("effb0", "EfficientNet-B0")]
    rows = []
    summ = {}
    for k, name in LD:
        L = loco(k, "_wsqrt")
        if len(L) < 3:
            continue
        cells = [f"{msd(L[c]['acc'])} / {msd(L[c]['mf1'])}" for c in CITIES]
        macc = np.mean([np.mean(L[c]["acc"]) for c in CITIES]); mf = np.mean([np.mean(L[c]["mf1"]) for c in CITIES])
        rows.append(f"{name} & " + " & ".join(cells) + f" & {macc:.1f} / {mf:.1f} \\\\")
        summ[k] = dict(macc=macc, mf=mf, L=L)
        print(k, "LOCO mean acc %.1f mF1 %.1f" % (macc, mf), {c: (round(np.mean(L[c]['acc']), 1), round(np.mean(L[c]['mf1']), 1)) for c in CITIES},
              "kochi perclass", [round(v, 2) for v in L["kochi"]["per"]])
        print("   kochi confusion", L["kochi"]["cm"])
    t("loco.tex", "\n".join(rows) + "\n")
    for k, tg in (("vit_s", "Vit"), ("resnet50", "Res"), ("segformer_b0", "Seg"), ("effb0", "Eff")):
        if k in summ:
            s = summ[k]
            cmd(M, f"LocoAcc{tg}", f"{s['macc']:.1f}"); cmd(M, f"LocoMf{tg}", f"{s['mf']:.1f}")
            for c, ct in zip(CITIES, ("Koc", "Tvm", "Koz")):
                cmd(M, f"Loco{ct}Acc{tg}", f"{np.mean(s['L'][c]['acc']):.1f}"); cmd(M, f"Loco{ct}Mf{tg}", f"{np.mean(s['L'][c]['mf1']):.1f}")

    # sensitivity
    rows = []
    def add(lab, r):
        if r:
            rows.append(f"{lab} & {r['acc']:.1f} & {r['mf1']:.1f} & {r['per'][2]:.1f} \\\\")
    add("ViT-S, reference configuration", P["vit_s"])
    y24v, y24r = pool("cv_y2024_vit_s_wsqrt"), pool("cv_y2024_resnet50_wsqrt")
    add("ViT-S, 2024 imagery with 2021 labels", y24v)
    add("ResNet-50, reference configuration", P["resnet50"])
    add("ResNet-50, 2024 imagery with 2021 labels", y24r)
    alt = pool("cv_y2021_altB_vit_s_wsqrt")
    add("ViT-S, Mangroves to Tree, Herbaceous wetland to Cropland/Grass", alt)
    invv, invr = pool("cv_y2021_vit_s"), pool("cv_y2021_resnet50")
    add("ViT-S, inverse-frequency class weights", invv)
    add("ResNet-50, inverse-frequency class weights", invr)
    t("sens.tex", "\n".join(rows) + "\n")
    for nm, r in (("VitTwo", y24v), ("ResTwo", y24r), ("VitAlt", alt), ("VitInv", invv), ("ResInv", invr)):
        if r:
            cmd(M, f"Mf{nm}", f"{r['mf1']:.1f}"); cmd(M, f"Acc{nm}", f"{r['acc']:.1f}")
            print(nm, round(r["acc"], 2), round(r["mf1"], 2), [round(v, 1) for v in r["per"]])
    for k, tg in (("vit_s", "Vit"), ("resnet50", "Res"), ("segformer_b0", "Seg"), ("effb0", "Eff")):
        L = loco(k)
        if "kochi" in L:
            cmd(M, f"InvKocAcc{tg}", f"{np.mean(L['kochi']['acc']):.1f}"); cmd(M, f"InvKocMf{tg}", f"{np.mean(L['kochi']['mf1']):.1f}")
            print("inverse weights kochi", k, [round(x, 1) for x in L["kochi"]["acc"]], L["kochi"]["cm"])

    # wetland and mangrove
    w = json.load(open(os.path.join(R2, "wetland_mangrove.json")))
    cmd(M, "WetKoc", f"{w['kochi']['wetland_pct']:.2f}"); cmd(M, "ManKoc", f"{w['kochi']['mangrove_pct']:.2f}")
    cmd(M, "WetTvm", f"{w['trivandrum']['wetland_pct']:.2f}"); cmd(M, "WetKoz", "0.09")     # raster gives 0.0947; the JSON holds 0.095
    cmd(M, "WmMajority", str(w["kochi"]["patches_majority_wetland_or_mangrove"]))
    cmd(M, "WmChanged", str(sum(w[c]["patches_label_changes_under_B"] for c in CITIES)))

    # maps
    pa = os.path.join(R2, "pixel_agreement.json")
    if os.path.exists(pa):
        A = json.load(open(pa)); I = json.load(open(os.path.join(R2, "indicators.json"))); C = json.load(open(os.path.join(R2, "change.json")))
        disp = {"kochi": "Kochi", "trivandrum": "Thiruvananthapuram", "kozhikode": "Kozhikode"}
        for c in CITIES:        # WorldCover share over the classified pixels, as for the map share
            cm = np.array(A[c]["confusion"], float)
            A[c]["worldcover_pct"]["Built-up"] = 100 * cm[0].sum() / cm.sum()
        rows = [f"{disp[c]} & {100*A[c]['pixel_accuracy']:.1f} & {A[c]['kappa']:.2f} & {A[c]['per_class_f1']['Built-up']:.2f} & "
                f"{A[c]['pred_pct']['Built-up']:.1f} & {A[c]['worldcover_pct']['Built-up']:.1f} \\\\" for c in CITIES]
        t("pixel.tex", "\n".join(rows) + "\n")
        rows = [f"{disp[c]} & {I[c]['built_up_pct']:.1f} & {I[c]['green_pct']:.1f} & {I[c]['water_pct']:.1f} & {I[c]['green_to_builtup']:.2f} \\\\" for c in CITIES]
        t("indicators.tex", "\n".join(rows) + "\n")
        rows = [f"{disp[c]} & {C[c]['built_2018_km2']:.1f} & {C[c]['new_km2']:.1f} & {C[c]['new_pct_of_2018']:.1f} & {C[c]['lost_km2']:.1f} & "
                f"{C[c]['ndbi_change_pct']:+.1f} \\\\" for c in CITIES]
        t("change.tex", "\n".join(rows) + "\n")
        print(json.dumps({c: {"px": A[c]["pixel_accuracy"], "k": A[c]["kappa"], "f1": A[c]["per_class_f1"], "pred": A[c]["pred_pct"], "wc": A[c]["worldcover_pct"], "ind": I[c], "chg": C[c]} for c in CITIES}, indent=0)[:2400])
        for c, ct in zip(CITIES, ("Koc", "Tvm", "Koz")):
            cmd(M, f"Px{ct}", f"{100*A[c]['pixel_accuracy']:.1f}"); cmd(M, f"Kap{ct}", f"{A[c]['kappa']:.2f}")
            cmd(M, f"BuiltPred{ct}", f"{A[c]['pred_pct']['Built-up']:.1f}"); cmd(M, f"BuiltWc{ct}", f"{A[c]['worldcover_pct']['Built-up']:.1f}")
            cmd(M, f"New{ct}", f"{C[c]['new_km2']:.1f}"); cmd(M, f"NewPct{ct}", f"{C[c]['new_pct_of_2018']:.1f}")
            cmd(M, f"Lost{ct}", f"{C[c]['lost_km2']:.1f}"); cmd(M, f"Base{ct}", f"{C[c]['built_2018_km2']:.1f}")
            cmd(M, f"Ndbi{ct}", f"{C[c]['ndbi_change_pct']:+.1f}"); cmd(M, f"Ratio{ct}", f"{I[c]['green_to_builtup']:.1f}")
            cmd(M, f"Bu{ct}", f"{I[c]['built_up_pct']:.0f}"); cmd(M, f"Gr{ct}", f"{I[c]['green_pct']:.0f}"); cmd(M, f"Wa{ct}", f"{I[c]['water_pct']:.0f}")
    t("macros.tex", "\n".join(M) + "\n")
    print("TABLES_DONE")


if __name__ == "__main__":
    main()
