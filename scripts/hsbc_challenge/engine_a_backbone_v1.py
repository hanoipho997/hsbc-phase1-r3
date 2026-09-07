"""Engine-A backbone tuning (preregistration section 3 + amendment A2).

Frozen protocol: XGBoost and LightGBM, 36 hyperparameter cells x
scale_pos_weight in {1, neg/pos} = 72 fits per library; n_estimators <= 4000
with early stopping 100 on the validation metric; selection by sklearn
average_precision_score on validation predictions; backbone = best single
model or the two-model mean, whichever has higher validation AUPRC.
Train/validation only — the sealed test partition is never read here.

Feature recipe (frozen + A2): label-encode object columns (train+val
categories, unseen -> -1 later); frequency-encode card1, addr1,
P_emaildomain; TransactionAmt decimal part; hour-of-day and day-of-week from
TransactionDT; drop TransactionID, raw TransactionDT, isFraud, split; NaN
native; float32.

Crash-resumable: per-config results are appended to the results JSON and
completed configs are skipped on restart.

Artifacts -> runs/hsbc_challenge/engine_a_v1/backbone/
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTD = "runs/hsbc_challenge/engine_a_v1/backbone"
RES = os.path.join(OUTD, "backbone_results_v1.json")
os.makedirs(OUTD, exist_ok=True)
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# features (frozen recipe; encoders saved for the unblinding run)
# ---------------------------------------------------------------------------
df = pd.read_parquet("runs/hsbc_challenge/data/ieee/trainval.parquet")
y = df["isFraud"].to_numpy()
split = df["split"].to_numpy()

FREQ_COLS = ["card1", "addr1", "P_emaildomain"]
enc = {"label_categories": {}, "freq_maps": {}, "feature_columns": None}

feat = df.drop(columns=["TransactionID", "TransactionDT", "isFraud", "split"])
feat["hour"] = ((df["TransactionDT"] // 3600) % 24).astype(np.float32)
feat["dow"] = ((df["TransactionDT"] // 86400) % 7).astype(np.float32)
feat["amt_decimal"] = (df["TransactionAmt"]
                       - np.floor(df["TransactionAmt"])).astype(np.float32)
for c in FREQ_COLS:
    vc = df[c].value_counts()
    enc["freq_maps"][c] = {str(k): int(v) for k, v in vc.items()}
    feat[c + "_freq"] = df[c].map(vc).astype(np.float32)
for c in feat.columns[feat.dtypes == object]:
    codes, cats = pd.factorize(feat[c], use_na_sentinel=True)
    enc["label_categories"][c] = [str(x) for x in cats]
    feat[c] = codes.astype(np.float32)         # NaN -> -1 sentinel
feat = feat.astype(np.float32)
enc["feature_columns"] = list(feat.columns)
joblib.dump(enc, os.path.join(OUTD, "preprocess_v1.joblib"))

X_tr, y_tr = feat[split == "train"].to_numpy(), y[split == "train"]
X_va, y_va = feat[split == "val"].to_numpy(), y[split == "val"]
SPW = float((y_tr == 0).sum() / (y_tr == 1).sum())
log(f"features: {feat.shape[1]} cols; train {X_tr.shape}, val {X_va.shape}; "
    f"neg/pos = {SPW:.2f}")
del df, feat

# ---------------------------------------------------------------------------
# grid (frozen)
# ---------------------------------------------------------------------------
GRID = [(d, lr, sub, col, spw)
        for d in (6, 8, 10) for lr in (0.02, 0.05, 0.1)
        for sub in (0.7, 0.9) for col in (0.7, 0.9)
        for spw in (1.0, SPW)]

results = json.load(open(RES)) if os.path.exists(RES) else {}


def save():
    with open(RES, "w") as fh:
        json.dump(results, fh, indent=1)


def fit_xgb(d, lr, sub, col, spw):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=4000, max_depth=d, learning_rate=lr, subsample=sub,
        colsample_bytree=col, scale_pos_weight=spw, tree_method="hist",
        eval_metric="aucpr", early_stopping_rounds=100, random_state=0,
        n_jobs=os.cpu_count())
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    p = clf.predict_proba(X_va)[:, 1]
    return clf, float(average_precision_score(y_va, p)), int(clf.best_iteration)


def fit_lgb(d, lr, sub, col, spw):
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(
        n_estimators=4000, max_depth=d, num_leaves=min(2 ** d, 255),
        learning_rate=lr, subsample=sub, subsample_freq=1,
        colsample_bytree=col, scale_pos_weight=spw, random_state=0,
        n_jobs=os.cpu_count(), verbosity=-1)
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100, verbose=False)])
    p = clf.predict_proba(X_va)[:, 1]
    return clf, float(average_precision_score(y_va, p)), int(clf.best_iteration_)


best = {"xgb": (-1.0, None, None), "lgb": (-1.0, None, None)}
for lib, fitter in (("xgb", fit_xgb), ("lgb", fit_lgb)):
    for i, (d, lr, sub, col, spw) in enumerate(GRID):
        key = f"{lib}_d{d}_lr{lr}_s{sub}_c{col}_w{'bal' if spw > 1 else '1'}"
        if key in results:
            ap = results[key]["val_auprc"]
        else:
            t0 = time.time()
            clf, ap, best_it = fitter(d, lr, sub, col, spw)
            results[key] = {"lib": lib, "depth": d, "lr": lr, "subsample": sub,
                            "colsample": col, "spw": spw, "val_auprc": ap,
                            "best_iteration": best_it,
                            "wall_s": round(time.time() - t0, 1)}
            save()
            if ap > best[lib][0]:
                path = os.path.join(OUTD, f"best_{lib}.model")
                (clf.save_model(path) if lib == "xgb"
                 else clf.booster_.save_model(path))
                np.save(os.path.join(OUTD, f"best_{lib}_val_pred.npy"),
                        clf.predict_proba(X_va)[:, 1])
                results[f"_best_{lib}"] = key
                save()
        if ap > best[lib][0]:
            best[lib] = (ap, key, None)
        log(f"{lib} {i+1:02d}/72 {key}: val AUPRC {ap:.5f} "
            f"(best {lib} {best[lib][0]:.5f})")

# ---------------------------------------------------------------------------
# backbone choice (frozen): best single vs two-model mean, by val AUPRC
# ---------------------------------------------------------------------------
p_x = np.load(os.path.join(OUTD, "best_xgb_val_pred.npy"))
p_l = np.load(os.path.join(OUTD, "best_lgb_val_pred.npy"))
ap_mean = float(average_precision_score(y_va, 0.5 * (p_x + p_l)))
choice = max([("xgb", best["xgb"][0]), ("lgb", best["lgb"][0]),
              ("mean", ap_mean)], key=lambda t: t[1])
results["_backbone"] = {
    "choice": choice[0], "val_auprc": choice[1],
    "best_xgb": {"key": results["_best_xgb"], "val_auprc": best["xgb"][0]},
    "best_lgb": {"key": results["_best_lgb"], "val_auprc": best["lgb"][0]},
    "mean_val_auprc": ap_mean,
}
save()
log(f"BACKBONE FROZEN: {choice[0]} (val AUPRC {choice[1]:.5f}; "
    f"xgb {best['xgb'][0]:.5f}, lgb {best['lgb'][0]:.5f}, mean {ap_mean:.5f})")
log(f"done -> {RES}")
