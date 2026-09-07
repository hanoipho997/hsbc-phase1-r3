"""Engine-A arms trainer (preregistration sections 5-6 + A4/A5).

Trains, on the OOF-routed TRAIN pool with validation-pool tuning only:
  Q    : CrossViewSLCU circuit (18 params, seeds 900-904; pairwise ranking
         loss on the flag; per-seed logistic head on [backbone logit,
         8 syndrome log-probs]; deployed score = mean of the 5 heads)
  C-a  : logistic [backbone logit, 12 raw quantile features]
  C-b  : MLP(8 hidden) on the same inputs as C-a
  C-c  : logistic [backbone logit, PCA(k=4 on all-train-legit) recon error]
  C-f  : logistic [backbone logit, 8 iid N(0,1) noise features]

The sealed test partition is never read. All arms + encoders + val-pool
metrics -> runs/hsbc_challenge/engine_a_v1/arms_v1/
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsbc_common import (
    CrossViewConfig, CrossViewSLCU, broadcast_view_diagonal,
    pairwise_loss_and_grad, product_state_batch, save_json,
)

BASE = "runs/hsbc_challenge/engine_a_v1"
OUTD = os.path.join(BASE, "arms_v1")
os.makedirs(OUTD, exist_ok=True)
SEEDS = (900, 901, 902, 903, 904)
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# data, pools, quantile features
# ---------------------------------------------------------------------------
views = joblib.load(os.path.join(BASE, "views_v1.joblib"))
routing = json.load(open(os.path.join(BASE, "routing_freeze_v1.json")))
t_lo = routing["routing_thresholds"]["t_lo_val_q988"]
t_hi = routing["routing_thresholds"]["t_hi_val_q998"]

df = pd.read_parquet("runs/hsbc_challenge/data/ieee/trainval.parquet")
df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype(float)
y_all = df["isFraud"].to_numpy()
is_tr = (df["split"] == "train").to_numpy()

oof = np.load(os.path.join(BASE, "backbone_vm", "oof_backbone.npy"))
pxv = np.load(os.path.join(BASE, "backbone_vm", "best_xgb_val_pred.npy"))
plv = np.load(os.path.join(BASE, "backbone_vm", "best_lgb_val_pred.npy"))
pval = 0.5 * (pxv.astype(np.float64) + plv.astype(np.float64))


def to_u_matrix(frame):
    cols = []
    for c in views["features"]:
        s = frame[c]
        if c in views["freq_maps"]:
            s = s.map({k: v for k, v in views["freq_maps"][c].items()}).astype(float) \
                if s.dtype == object else s.astype(float)
        x = np.nan_to_num(s.to_numpy(dtype=float), nan=-1.0)
        col = views["sorted_cols"][c]
        lo = np.searchsorted(col, x, side="left")
        hi = np.searchsorted(col, x, side="right")
        cols.append(np.clip(0.5 * (lo + hi) / len(col), 0.0, 1.0))
    return np.column_stack(cols)


tr_pool_mask = is_tr.copy()
tr_pool_mask[is_tr] = (oof > t_lo) & (oof <= t_hi)
va_mask = ~is_tr
va_pool_local = (pval > t_lo) & (pval <= t_hi)

pool_tr = df[tr_pool_mask]
y_pool = y_all[tr_pool_mask]
s_pool = oof[(oof > t_lo) & (oof <= t_hi)]
pool_va = df[va_mask][va_pool_local]
y_vp = y_all[va_mask][va_pool_local]
s_vp = pval[va_pool_local]
log(f"train pool: {len(pool_tr)} rows / {int(y_pool.sum())} frauds "
    f"({100*y_pool.mean():.1f}%); val pool: {len(pool_va)} rows / "
    f"{int(y_vp.sum())} frauds ({100*y_vp.mean():.1f}%)")

U_pool = to_u_matrix(pool_tr)
U_vp = to_u_matrix(pool_va)

# ---------------------------------------------------------------------------
# quantum arm
# ---------------------------------------------------------------------------
VQ = views["view_qubits"]
name2idx = {"A": 0, "B": 1, "C": 2}
cfg = CrossViewConfig(
    n_qubits=12,
    views=tuple(tuple(VQ[v]) for v in ("A", "B", "C")),
    layer_pairs=tuple((name2idx[a], name2idx[b])
                      for a, b in views["layer_pairs"]),
    hp_views=tuple(broadcast_view_diagonal(
        np.array(views["hp_tables"][v]), VQ[v], 12) for v in ("A", "B", "C")))
sim = CrossViewSLCU(cfg)

states_pool = product_state_batch(U_pool).astype(complex)
states_vp = product_state_batch(U_vp).astype(complex)
fraud_states = states_pool[y_pool == 1]
legit_states = states_pool[y_pool == 0]
minor, major = ((legit_states, fraud_states)
                if len(legit_states) <= len(fraud_states)
                else (fraud_states, legit_states))
log(f"A5 batching: minority {len(minor)} (all) + 256 of majority {len(major)}")


def train_circuit(seed, iters=300):
    rng = np.random.default_rng(seed)
    params = np.zeros(cfg.n_params)
    for l in range(cfg.n_layers):
        params[6 * l: 6 * l + 2] = rng.normal(scale=0.15, size=2)
        params[6 * l + 2] = rng.uniform(0.5, 2.5)
        params[6 * l + 3] = rng.normal(scale=0.15)
        params[6 * l + 4] = rng.uniform(0.5, 2.5)
        params[6 * l + 5] = rng.normal(scale=0.15)
    m1 = np.zeros(cfg.n_params)
    m2 = np.zeros(cfg.n_params)
    best = (-1.0, params.copy())
    minority_is_legit = len(legit_states) <= len(fraud_states)
    for it in range(iters):
        sel = rng.choice(len(major), min(256, len(major)), replace=False)
        leg, fra = ((minor, major[sel]) if minority_is_legit
                    else (major[sel], minor))
        _, grad, _, _ = pairwise_loss_and_grad(sim, params, leg, fra)
        m1 = 0.9 * m1 + 0.1 * grad
        m2 = 0.999 * m2 + 0.001 * grad * grad
        params = params - 0.05 * (m1 / (1 - 0.9 ** (it + 1))) / (
            np.sqrt(m2 / (1 - 0.999 ** (it + 1))) + 1e-8)
        if (it + 1) % 20 == 0 or it == iters - 1:
            v = sim.forward(params, states_vp)
            ps = np.einsum("md,md->m", v.conj(), v).real
            ap = average_precision_score(y_vp, 1.0 - ps)
            if ap > best[0]:
                best = (ap, params.copy())
    return best


def fit_head(feats_tr, feats_va, tag):
    """Standardized logistic head on [backbone logit, feats]; C by val-pool AP."""
    A_tr = np.column_stack([logit(s_pool), feats_tr])
    A_va = np.column_stack([logit(s_vp), feats_va])
    sc = StandardScaler().fit(A_tr)
    best = (-1.0, None, None)
    for C in (0.1, 1.0, 10.0):
        clf = LogisticRegression(C=C, max_iter=5000)
        clf.fit(sc.transform(A_tr), y_pool)
        ap = average_precision_score(y_vp, clf.predict_proba(sc.transform(A_va))[:, 1])
        if ap > best[0]:
            best = (ap, clf, C)
    log(f"  head[{tag}]: val-pool AUPRC {best[0]:.5f} (C={best[2]})")
    return {"scaler": sc, "clf": best[1], "C": best[2], "val_pool_auprc": best[0]}


arms = {}
qa = {"seeds": {}}
head_va_preds = []
for seed in SEEDS:
    t0 = time.time()
    ap_circ, params = train_circuit(seed)
    feats_tr = sim.syndrome_features(params, states_pool)
    feats_va = sim.syndrome_features(params, states_vp)
    head = fit_head(feats_tr, feats_va, f"Q seed {seed}")
    A_va = np.column_stack([logit(s_vp), feats_va])
    head_va_preds.append(head["clf"].predict_proba(
        head["scaler"].transform(A_va))[:, 1])
    qa["seeds"][seed] = {"circuit_val_pool_auprc_flag": float(ap_circ),
                         "params": params.tolist(),
                         "head_val_pool_auprc": head["val_pool_auprc"],
                         "head_C": head["C"], "wall_s": round(time.time() - t0, 1)}
    joblib.dump(head, os.path.join(OUTD, f"q_head_seed{seed}.joblib"))
    log(f"Q seed {seed}: circuit-flag AP {ap_circ:.5f}, "
        f"head AP {head['val_pool_auprc']:.5f} ({time.time()-t0:.0f}s)")
q_ens = np.mean(head_va_preds, axis=0)
qa["ensemble_val_pool_auprc"] = float(average_precision_score(y_vp, q_ens))
arms["Q"] = qa
log(f"Q ensemble (mean of 5 heads): val-pool AUPRC {qa['ensemble_val_pool_auprc']:.5f}")

# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------
arms["C_a"] = {"head": None}
h = fit_head(U_pool, U_vp, "C-a logistic raw-12")
joblib.dump(h, os.path.join(OUTD, "head_C_a.joblib"))
arms["C_a"] = {"val_pool_auprc": h["val_pool_auprc"], "C": h["C"]}

A_tr = np.column_stack([logit(s_pool), U_pool])
A_va = np.column_stack([logit(s_vp), U_vp])
sc_b = StandardScaler().fit(A_tr)
mlp = MLPClassifier(hidden_layer_sizes=(8,), random_state=0, max_iter=2000)
mlp.fit(sc_b.transform(A_tr), y_pool)
ap_b = float(average_precision_score(
    y_vp, mlp.predict_proba(sc_b.transform(A_va))[:, 1]))
joblib.dump({"scaler": sc_b, "clf": mlp}, os.path.join(OUTD, "head_C_b.joblib"))
arms["C_b"] = {"val_pool_auprc": ap_b}
log(f"  head[C-b MLP(8)]: val-pool AUPRC {ap_b:.5f}")

from sklearn.decomposition import PCA
U_train_legit = to_u_matrix(df[is_tr][y_all[is_tr] == 0])
pca = PCA(n_components=4, random_state=0).fit(U_train_legit)


def pca_err(U):
    Z = pca.inverse_transform(pca.transform(U))
    return np.sqrt(((U - Z) ** 2).sum(axis=1))[:, None]


h = fit_head(pca_err(U_pool), pca_err(U_vp), "C-c PCA-residual")
joblib.dump({"head": h, "pca": pca}, os.path.join(OUTD, "head_C_c.joblib"))
arms["C_c"] = {"val_pool_auprc": h["val_pool_auprc"], "C": h["C"]}

from hsbc_common import keyed_noise
h = fit_head(keyed_noise(pool_tr["TransactionID"].to_numpy()),
             keyed_noise(pool_va["TransactionID"].to_numpy()), "C-f noise")
joblib.dump(h, os.path.join(OUTD, "head_C_f.joblib"))
arms["C_f"] = {"val_pool_auprc": h["val_pool_auprc"], "C": h["C"]}

arms["_pools"] = {"train_pool_rows": int(len(pool_tr)),
                  "train_pool_frauds": int(y_pool.sum()),
                  "val_pool_rows": int(len(pool_va)),
                  "val_pool_frauds": int(y_vp.sum())}
save_json(os.path.join(OUTD, "arms_summary_v1.json"), arms)
log("ARMS DONE -> runs/hsbc_challenge/engine_a_v1/arms_v1/arms_summary_v1.json")
