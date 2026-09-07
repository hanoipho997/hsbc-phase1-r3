"""Engine-A out-of-fold backbone scores (preregistration section 4 + A4).

5-fold OOF predictions on the TRAIN partition for both frozen backbone
configurations, run on the VM (A3 environment). Frozen details (A4):
KFold(n_splits=5, shuffle=True, random_state=0) over train rows;
n_estimators fixed at the canonical refits' best iterations (no fold-level
early stopping, so no held-fold leakage); scale_pos_weight uses the
train-global value where the frozen config is 'balanced'.

Self-contained; needs trainval.parquet beside it.
Artifacts -> ./out/oof_{xgb,lgb,backbone}.npy, ./out/oof_folds.npy,
             ./out/oof_manifest_v1.json
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "out")
N_THREADS = 20
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


# ---- identical frozen feature recipe --------------------------------------
df = pd.read_parquet(os.path.join(HERE, "trainval.parquet"))
y = df["isFraud"].to_numpy()
split = df["split"].to_numpy()
feat = df.drop(columns=["TransactionID", "TransactionDT", "isFraud", "split"])
feat["hour"] = ((df["TransactionDT"] // 3600) % 24).astype(np.float32)
feat["dow"] = ((df["TransactionDT"] // 86400) % 7).astype(np.float32)
feat["amt_decimal"] = (df["TransactionAmt"]
                       - np.floor(df["TransactionAmt"])).astype(np.float32)
for c in ["card1", "addr1", "P_emaildomain"]:
    feat[c + "_freq"] = df[c].map(df[c].value_counts()).astype(np.float32)
for c in feat.columns[feat.dtypes == object]:
    feat[c] = pd.factorize(feat[c], use_na_sentinel=True)[0].astype(np.float32)
feat = feat.astype(np.float32)

X_tr = feat[split == "train"].to_numpy()
y_tr = y[split == "train"]
SPW = float((y_tr == 0).sum() / (y_tr == 1).sum())
log(f"train {X_tr.shape}, neg/pos {SPW:.2f}")
del df, feat

FROZEN = {
    "xgb": dict(depth=10, lr=0.05, subsample=0.9, colsample=0.9, spw=SPW,
                n_estimators=3932),
    "lgb": dict(depth=10, lr=0.02, subsample=0.7, colsample=0.9, spw=1.0,
                n_estimators=534),
}

kf = KFold(n_splits=5, shuffle=True, random_state=0)
FOLDS = list(kf.split(X_tr))
fold_of = np.full(len(y_tr), -1, np.int8)
for i, (_, te) in enumerate(FOLDS):
    fold_of[te] = i


def fit_fold(lib, fold_idx):
    tr_idx, te_idx = FOLDS[fold_idx]
    c = FROZEN[lib]
    t0 = time.time()
    if lib == "xgb":
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=c["n_estimators"], max_depth=c["depth"],
            learning_rate=c["lr"], subsample=c["subsample"],
            colsample_bytree=c["colsample"], scale_pos_weight=c["spw"],
            tree_method="hist", random_state=0, n_jobs=N_THREADS)
        clf.fit(X_tr[tr_idx], y_tr[tr_idx], verbose=False)
    else:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=c["n_estimators"], max_depth=c["depth"],
            num_leaves=min(2 ** c["depth"], 255), learning_rate=c["lr"],
            subsample=c["subsample"], subsample_freq=1,
            colsample_bytree=c["colsample"], scale_pos_weight=c["spw"],
            random_state=0, n_jobs=N_THREADS, verbosity=-1)
        clf.fit(X_tr[tr_idx], y_tr[tr_idx])
    pred = clf.predict_proba(X_tr[te_idx])[:, 1]
    return lib, fold_idx, te_idx, pred, round(time.time() - t0, 1)


def main():
    oof = {"xgb": np.full(len(y_tr), np.nan), "lgb": np.full(len(y_tr), np.nan)}
    jobs = [(lib, f) for lib in ("xgb", "lgb") for f in range(5)]
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(fit_fold, *j) for j in jobs]
        for fu in as_completed(futs):
            lib, f, te_idx, pred, wall = fu.result()
            oof[lib][te_idx] = pred
            log(f"{lib} fold {f}: done ({wall}s)")
    assert not np.isnan(oof["xgb"]).any() and not np.isnan(oof["lgb"]).any()
    oof_b = 0.5 * (oof["xgb"] + oof["lgb"])
    np.save(os.path.join(OUTD, "oof_xgb.npy"), oof["xgb"])
    np.save(os.path.join(OUTD, "oof_lgb.npy"), oof["lgb"])
    np.save(os.path.join(OUTD, "oof_backbone.npy"), oof_b)
    np.save(os.path.join(OUTD, "oof_folds.npy"), fold_of)
    man = {"frozen": {k: {kk: (float(vv) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()} for k, v in FROZEN.items()},
           "kfold": "KFold(5, shuffle=True, random_state=0)",
           "train_oof_auprc": {
               "xgb": float(average_precision_score(y_tr, oof["xgb"])),
               "lgb": float(average_precision_score(y_tr, oof["lgb"])),
               "backbone_mean": float(average_precision_score(y_tr, oof_b))}}
    json.dump(man, open(os.path.join(OUTD, "oof_manifest_v1.json"), "w"),
              indent=2)
    log(f"OOF train AUPRC: {man['train_oof_auprc']}")
    log("DONE_MARKER")


if __name__ == "__main__":
    main()
