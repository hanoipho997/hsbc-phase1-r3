"""Engine-A view selection + per-view Ising fits (prereg section 5 + A4.v).

Train-only: MI-based top-4 selection within each semantic candidate list,
frequency-encoding for categorical candidates, train-CDF quantile
transforms, median-split bits, per-view pseudo-likelihood Ising fit on
TRAIN LEGITIMATE rows only (supervised screening is the MI step and is
declared as such — no 'label-free' claim anywhere).

Artifacts -> runs/hsbc_challenge/engine_a_v1/views_v1.joblib (+ summary json)
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsbc_common import fit_ising_pseudolikelihood

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


df = pd.read_parquet("runs/hsbc_challenge/data/ieee/trainval.parquet")
df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype(float)
tr = df[df["split"] == "train"].reset_index(drop=True)
y_tr = tr["isFraud"].to_numpy()

CANDS = {
    "A": ["TransactionAmt", "ProductCD", "hour", "dist1"]
         + [f"C{i}" for i in range(1, 15)],
    "B": [f"card{i}" for i in range(1, 7)] + ["addr1", "addr2"],
    "C": ["P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]
         + [f"id_{i:02d}" for i in range(1, 39)],
}
CANDS = {v: [c for c in cols if c in tr.columns] for v, cols in CANDS.items()}

# encode candidates: categoricals -> train-frequency; numeric as-is; NaN -> -1
freq_maps = {}
enc_cols = {}
for v, cols in CANDS.items():
    for c in cols:
        s = tr[c]
        if s.dtype == object:
            vc = s.value_counts()
            freq_maps[c] = {str(k): int(x) for k, x in vc.items()}
            enc_cols[c] = s.map(vc).astype(float)
        else:
            enc_cols[c] = s.astype(float)

# MI on a 120k stratified seed-0 subsample (A4.v)
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

idx_sub, _ = train_test_split(np.arange(len(tr)), train_size=120_000,
                              stratify=y_tr, random_state=0)
mi_table = {}
selected = {}
for v, cols in CANDS.items():
    X = np.column_stack([enc_cols[c].to_numpy()[idx_sub] for c in cols])
    X = np.nan_to_num(X, nan=-1.0)
    discrete = np.array([tr[c].dtype == object for c in cols])
    mi = mutual_info_classif(X, y_tr[idx_sub], discrete_features=discrete,
                             random_state=0)
    order = np.argsort(-mi)
    mi_table[v] = {cols[i]: float(mi[i]) for i in order}
    selected[v] = [cols[i] for i in order[:4]]
    log(f"view {v}: selected {selected[v]} "
        f"(MI {[round(mi[i],4) for i in order[:4]]})")

FEATURES = selected["A"] + selected["B"] + selected["C"]   # qubits 0..11

# train-CDF quantile transform material (midrank, as in hsbc_common)
sorted_cols = {c: np.sort(enc_cols[c].to_numpy()[~np.isnan(enc_cols[c].to_numpy())])
               for c in FEATURES}


def to_u(values, c):
    col = sorted_cols[c]
    x = np.nan_to_num(np.asarray(values, float), nan=-1.0)
    lo = np.searchsorted(col, x, side="left")
    hi = np.searchsorted(col, x, side="right")
    return np.clip(0.5 * (lo + hi) / len(col), 0.0, 1.0)


U_tr = np.column_stack([to_u(enc_cols[c].to_numpy(), c) for c in FEATURES])
bits_tr = (U_tr > 0.5).astype(int)

# per-view Ising fits on TRAIN LEGIT rows only; [0,1] span-normalized (A1)
hp_tables = {}
legit = y_tr == 0
for i, v in enumerate(("A", "B", "C")):
    b = bits_tr[legit][:, 4 * i: 4 * i + 4]
    model = fit_ising_pseudolikelihood(b, seed=0)
    hp_tables[v] = model.hp.tolist()          # 16 energies in [0,1]
    log(f"view {v}: Ising fitted, hp span-normalized "
        f"(coupling l1 {np.abs(model.b).sum()/2:.3f})")

art = {
    "features": FEATURES, "selected": selected, "mi_table": mi_table,
    "freq_maps": {c: freq_maps[c] for c in FEATURES if c in freq_maps},
    "sorted_cols": sorted_cols, "hp_tables": hp_tables,
    "view_qubits": {"A": [0, 1, 2, 3], "B": [4, 5, 6, 7],
                    "C": [8, 9, 10, 11]},
    "layer_pairs": [["A", "B"], ["B", "C"], ["A", "C"]],
}
joblib.dump(art, "runs/hsbc_challenge/engine_a_v1/views_v1.joblib")
json.dump({k: art[k] for k in ("features", "selected", "mi_table")},
          open("runs/hsbc_challenge/engine_a_v1/views_summary_v1.json", "w"),
          indent=2)
log("views artifact -> runs/hsbc_challenge/engine_a_v1/views_v1.joblib")
