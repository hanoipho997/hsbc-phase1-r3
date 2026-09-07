"""Engine-A backbone grid, VM edition (preregistration section 3 + A2 + A3).

Identical frozen grid/recipe/seeds to engine_a_backbone_v1.py, executed as
12 concurrent fits x 20 threads each (A3: thread count fixed for
determinism; this run is CANONICAL for backbone selection). Self-contained:
no repo imports; needs only trainval.parquet beside it.

After the grid: the best configuration per library is refit once (same
seed/threads) to serialize the model and validation predictions, and the
backbone choice (best single vs two-model mean, by validation AUPRC) is
recorded.

Usage on the VM:   python3 engine_a_backbone_vm_v1.py
Artifacts:         ./out/backbone_results_vm_v1.json, ./out/best_*.model,
                   ./out/best_*_val_pred.npy, ./out/preprocess_v1.joblib
Resumable: completed configs are skipped on restart.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "out")
RES = os.path.join(OUTD, "backbone_results_vm_v1.json")
os.makedirs(OUTD, exist_ok=True)
N_WORKERS, N_THREADS = 12, 20
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# features (byte-identical recipe to engine_a_backbone_v1.py)
# ---------------------------------------------------------------------------
df = pd.read_parquet(os.path.join(HERE, "trainval.parquet"))
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
    feat[c] = codes.astype(np.float32)
feat = feat.astype(np.float32)
enc["feature_columns"] = list(feat.columns)
joblib.dump(enc, os.path.join(OUTD, "preprocess_v1.joblib"))

X_tr, y_tr = feat[split == "train"].to_numpy(), y[split == "train"]
X_va, y_va = feat[split == "val"].to_numpy(), y[split == "val"]
SPW = float((y_tr == 0).sum() / (y_tr == 1).sum())
log(f"features: {feat.shape[1]} cols; train {X_tr.shape}, val {X_va.shape}; "
    f"neg/pos = {SPW:.2f}")
del df, feat

GRID = [(d, lr, sub, col, spw)
        for d in (6, 8, 10) for lr in (0.02, 0.05, 0.1)
        for sub in (0.7, 0.9) for col in (0.7, 0.9)
        for spw in (1.0, SPW)]


def cfg_key(lib, d, lr, sub, col, spw):
    return f"{lib}_d{d}_lr{lr}_s{sub}_c{col}_w{'bal' if spw > 1 else '1'}"


def fit_one(lib, d, lr, sub, col, spw, return_model=False):
    t0 = time.time()
    if lib == "xgb":
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=4000, max_depth=d, learning_rate=lr, subsample=sub,
            colsample_bytree=col, scale_pos_weight=spw, tree_method="hist",
            eval_metric="aucpr", early_stopping_rounds=100, random_state=0,
            n_jobs=N_THREADS)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        best_it = int(clf.best_iteration)
    else:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=4000, max_depth=d, num_leaves=min(2 ** d, 255),
            learning_rate=lr, subsample=sub, subsample_freq=1,
            colsample_bytree=col, scale_pos_weight=spw, random_state=0,
            n_jobs=N_THREADS, verbosity=-1)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(100, verbose=False)])
        best_it = int(clf.best_iteration_)
    p = clf.predict_proba(X_va)[:, 1]
    ap = float(average_precision_score(y_va, p))
    row = {"lib": lib, "depth": d, "lr": lr, "subsample": sub,
           "colsample": col, "spw": spw, "val_auprc": ap,
           "best_iteration": best_it, "wall_s": round(time.time() - t0, 1)}
    if return_model:
        return clf, p, row
    return cfg_key(lib, d, lr, sub, col, spw), row


def main():
    results = json.load(open(RES)) if os.path.exists(RES) else {}
    import lightgbm
    import sklearn
    import xgboost
    results["_env"] = {"xgboost": xgboost.__version__,
                       "lightgbm": lightgbm.__version__,
                       "sklearn": sklearn.__version__,
                       "pandas": pd.__version__, "numpy": np.__version__,
                       "workers": N_WORKERS, "threads_per_fit": N_THREADS}
    todo = [(lib, *g) for lib in ("xgb", "lgb") for g in GRID
            if cfg_key(lib, *g) not in results]
    log(f"grid: {len(todo)} of 144 configs to run "
        f"({N_WORKERS} workers x {N_THREADS} threads)")
    done = 144 - len(todo)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(fit_one, *t) for t in todo]
        for f in as_completed(futs):
            key, row = f.result()
            results[key] = row
            done += 1
            with open(RES, "w") as fh:
                json.dump(results, fh, indent=1)
            log(f"{done:3d}/144 {key}: val AUPRC {row['val_auprc']:.5f} "
                f"({row['wall_s']:.0f}s)")

    # refit best per library once, serialize
    summary = {}
    preds = {}
    for lib in ("xgb", "lgb"):
        rows = {k: v for k, v in results.items()
                if isinstance(v, dict) and v.get("lib") == lib}
        bk = max(rows, key=lambda k: rows[k]["val_auprc"])
        r = rows[bk]
        log(f"refitting best {lib}: {bk} (val AUPRC {r['val_auprc']:.5f})")
        clf, p, row = fit_one(lib, r["depth"], r["lr"], r["subsample"],
                              r["colsample"], r["spw"], return_model=True)
        assert abs(row["val_auprc"] - r["val_auprc"]) < 1e-9, \
            (row["val_auprc"], r["val_auprc"])
        path = os.path.join(OUTD, f"best_{lib}.model")
        (clf.save_model(path) if lib == "xgb"
         else clf.booster_.save_model(path))
        np.save(os.path.join(OUTD, f"best_{lib}_val_pred.npy"), p)
        preds[lib] = p
        summary[f"best_{lib}"] = {"key": bk, "val_auprc": r["val_auprc"]}

    ap_mean = float(average_precision_score(
        y_va, 0.5 * (preds["xgb"] + preds["lgb"])))
    choice = max([("xgb", summary["best_xgb"]["val_auprc"]),
                  ("lgb", summary["best_lgb"]["val_auprc"]),
                  ("mean", ap_mean)], key=lambda t: t[1])
    results["_backbone"] = {**summary, "mean_val_auprc": ap_mean,
                            "choice": choice[0], "val_auprc": choice[1]}
    with open(RES, "w") as fh:
        json.dump(results, fh, indent=1)
    log(f"BACKBONE FROZEN: {choice[0]} (val AUPRC {choice[1]:.5f}; "
        f"xgb {summary['best_xgb']['val_auprc']:.5f}, "
        f"lgb {summary['best_lgb']['val_auprc']:.5f}, mean {ap_mean:.5f})")
    log("DONE_MARKER")


if __name__ == "__main__":
    main()
