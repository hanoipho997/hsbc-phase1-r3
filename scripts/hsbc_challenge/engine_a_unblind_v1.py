"""Engine-A single unblinding run (preregistration v1 sections 7-9).

Prints the preregistration file's SHA-256 first. Executes every arm and
endpoint in one pass and applies gates G1-G4. The sealed test partition is
read by THIS script only, in --real mode, ONCE.

--rehearsal mode: runs the identical pipeline with the VALIDATION partition
standing in for test (no seal broken). Doubles as a scoring-path referee:
serialized-model predictions must reproduce the stored validation
predictions, and head outputs must reproduce arms_summary_v1.json.

Usage:
    python engine_a_unblind_v1.py --rehearsal
    python engine_a_unblind_v1.py --real

Artifacts -> runs/hsbc_challenge/engine_a_v1/unblind_{mode}_v1.json
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsbc_common import (
    CrossViewConfig, CrossViewSLCU, broadcast_view_diagonal,
    product_state_batch,
)

BASE = "runs/hsbc_challenge/engine_a_v1"
MODE = "real" if "--real" in sys.argv else (
    "rehearsal" if "--rehearsal" in sys.argv else None)
assert MODE, "pass --rehearsal or --real"
T0 = time.time()
BOOT_N, BOOT_SEED, K_BUDGET = 10_000, 1234, 400


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


prereg = "docs/hsbc_engine_a_preregistration_v1.md"
sha = hashlib.sha256(open(prereg, "rb").read()).hexdigest()
log(f"PREREGISTRATION {prereg} SHA-256 = {sha}")
log(f"MODE = {MODE}")

# ---------------------------------------------------------------------------
# frozen artifacts
# ---------------------------------------------------------------------------
enc = joblib.load(os.path.join(BASE, "backbone_vm", "preprocess_v1.joblib"))
views = joblib.load(os.path.join(BASE, "views_v1.joblib"))
routing = json.load(open(os.path.join(BASE, "routing_freeze_v1.json")))
arms_sum = json.load(open(os.path.join(BASE, "arms_v1", "arms_summary_v1.json")))
t_lo = routing["routing_thresholds"]["t_lo_val_q988"]
t_hi = routing["routing_thresholds"]["t_hi_val_q998"]

import lightgbm as lgb
import xgboost as xgb
xgb_m = xgb.XGBClassifier()
xgb_m.load_model(os.path.join(BASE, "backbone_vm", "best_xgb.model"))
lgb_m = lgb.Booster(model_file=os.path.join(BASE, "backbone_vm", "best_lgb.model"))

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
if MODE == "real":
    frame = pd.read_parquet("runs/hsbc_challenge/data/ieee/test_SEALED.parquet")
else:
    tv = pd.read_parquet("runs/hsbc_challenge/data/ieee/trainval.parquet")
    frame = tv[tv["split"] == "val"].drop(columns=["split"]).reset_index(drop=True)
    del tv
y = frame["isFraud"].to_numpy()
log(f"rows: {len(frame)}, frauds: {int(y.sum())} ({100*y.mean():.3f}%)")


def backbone_features(fr):
    feat = fr.drop(columns=["TransactionID", "TransactionDT", "isFraud"])
    feat["hour"] = ((fr["TransactionDT"] // 3600) % 24).astype(np.float32)
    feat["dow"] = ((fr["TransactionDT"] // 86400) % 7).astype(np.float32)
    feat["amt_decimal"] = (fr["TransactionAmt"]
                           - np.floor(fr["TransactionAmt"])).astype(np.float32)
    for c, m in enc["freq_maps"].items():
        feat[c + "_freq"] = fr[c].astype(str).map(
            {k: float(v) for k, v in m.items()}).astype(np.float32)
    for c, cats in enc["label_categories"].items():
        feat[c] = pd.Categorical(fr[c], categories=cats).codes.astype(np.float32)
    feat = feat[enc["feature_columns"]].astype(np.float32)
    return feat.to_numpy()


X = backbone_features(frame)
p_backbone = 0.5 * (xgb_m.predict_proba(X)[:, 1].astype(np.float64)
                    + lgb_m.predict(X).astype(np.float64))
log(f"backbone scored ({X.shape})")

if MODE == "rehearsal":
    ref = 0.5 * (np.load(os.path.join(BASE, "backbone_vm", "best_xgb_val_pred.npy")).astype(np.float64)
                 + np.load(os.path.join(BASE, "backbone_vm", "best_lgb_val_pred.npy")).astype(np.float64))
    dev = float(np.abs(p_backbone - ref).max())
    log(f"REHEARSAL scoring-path referee: max |pred - stored val pred| = {dev:.3e}")
    assert dev < 1e-5, "serialized-model scoring path does not reproduce val preds"

# ---------------------------------------------------------------------------
# regions and arm scores
# ---------------------------------------------------------------------------
pool = (p_backbone > t_lo) & (p_backbone <= t_hi)
above = p_backbone > t_hi
log(f"auto-flag: {int(above.sum())} rows / {int(y[above].sum())} frauds; "
    f"pool: {int(pool.sum())} rows / {int(y[pool].sum())} frauds; "
    f"below: {int((~pool & ~above).sum())} rows / {int(y[~pool & ~above].sum())} frauds")

pf = frame[pool].copy()
pf["hour"] = ((pf["TransactionDT"] // 3600) % 24).astype(float)


def views_u(fr):
    cols = []
    for c in views["features"]:
        s = fr[c]
        if c in views["freq_maps"]:
            s = s.astype(str).map({k: float(v) for k, v in views["freq_maps"][c].items()}) \
                if s.dtype == object else s.astype(float)
        x = np.nan_to_num(s.to_numpy(dtype=float), nan=-1.0)
        col = views["sorted_cols"][c]
        lo = np.searchsorted(col, x, side="left")
        hi = np.searchsorted(col, x, side="right")
        cols.append(np.clip(0.5 * (lo + hi) / len(col), 0.0, 1.0))
    return np.column_stack(cols)


U_pool = views_u(pf)
states_pool = product_state_batch(U_pool).astype(complex)

name2idx = {"A": 0, "B": 1, "C": 2}
VQ = views["view_qubits"]
cfg = CrossViewConfig(
    n_qubits=12, views=tuple(tuple(VQ[v]) for v in ("A", "B", "C")),
    layer_pairs=tuple((name2idx[a], name2idx[b]) for a, b in views["layer_pairs"]),
    hp_views=tuple(broadcast_view_diagonal(np.array(views["hp_tables"][v]),
                                           VQ[v], 12) for v in ("A", "B", "C")))
sim = CrossViewSLCU(cfg)


def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def head_out(head, feats):
    A = np.column_stack([logit(p_backbone[pool]), feats])
    return head["clf"].predict_proba(head["scaler"].transform(A))[:, 1]


arm_pool_scores = {}
q_outs = []
for seed, rec in arms_sum["Q"]["seeds"].items():
    params = np.array(rec["params"])
    feats = sim.syndrome_features(params, states_pool)
    head = joblib.load(os.path.join(BASE, "arms_v1", f"q_head_seed{seed}.joblib"))
    q_outs.append(head_out(head, feats))
arm_pool_scores["Q"] = np.mean(q_outs, axis=0)

arm_pool_scores["C_a"] = head_out(
    joblib.load(os.path.join(BASE, "arms_v1", "head_C_a.joblib")), U_pool)
cb = joblib.load(os.path.join(BASE, "arms_v1", "head_C_b.joblib"))
A = np.column_stack([logit(p_backbone[pool]), U_pool])
arm_pool_scores["C_b"] = cb["clf"].predict_proba(cb["scaler"].transform(A))[:, 1]
cc = joblib.load(os.path.join(BASE, "arms_v1", "head_C_c.joblib"))
Z = cc["pca"].inverse_transform(cc["pca"].transform(U_pool))
err = np.sqrt(((U_pool - Z) ** 2).sum(axis=1))[:, None]
arm_pool_scores["C_c"] = head_out(cc["head"], err)
from hsbc_common import keyed_noise
arm_pool_scores["C_f"] = head_out(
    joblib.load(os.path.join(BASE, "arms_v1", "head_C_f.joblib")),
    keyed_noise(pf["TransactionID"].to_numpy()))
log("all arm pool scores computed")

if MODE == "rehearsal":
    from sklearn.metrics import average_precision_score
    for k in ("Q", "C_a", "C_b", "C_c", "C_f"):
        ap = float(average_precision_score(y[pool], arm_pool_scores[k]))
        stored = (arms_sum[k]["val_pool_auprc"] if k != "Q"
                  else arms_sum["Q"]["ensemble_val_pool_auprc"])
        log(f"REHEARSAL {k}: pool AUPRC {ap:.5f} (stored {stored:.5f}, "
            f"dev {abs(ap-stored):.2e})")
        assert abs(ap - stored) < 5e-4


def full_scores(arm):
    s = p_backbone.copy()
    if arm != "backbone":
        s[pool] = arm_pool_scores[arm]
    return s


ARMS = ["backbone", "Q", "C_a", "C_b", "C_c", "C_f"]
S = {a: full_scores(a) for a in ARMS}
assert all(np.array_equal(S[a][~pool], p_backbone[~pool]) for a in ARMS[1:])

# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------
def auprc(yv, sv):
    order = np.argsort(-sv, kind="mergesort")
    yy = yv[order]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / max(yy.sum(), 1))


def recall_at_budget(yv, sv, poolmask, abovemask):
    pool_idx = np.flatnonzero(poolmask)
    top = pool_idx[np.argsort(-sv[pool_idx], kind="mergesort")[:K_BUDGET]]
    caught = int(yv[abovemask].sum()) + int(yv[top].sum())
    return caught / max(int(yv.sum()), 1)


point = {}
for a in ARMS:
    point[a] = {"auprc": auprc(y, S[a]),
                "recall_at_budget": recall_at_budget(y, S[a], pool, above)}
    log(f"{a:9s} AUPRC {point[a]['auprc']:.5f}  "
        f"recall@budget {point[a]['recall_at_budget']:.5f}")

log(f"paired bootstrap ({BOOT_N} resamples)...")
rng = np.random.default_rng(BOOT_SEED)
n = len(y)
deltas = {f"{a}-backbone": {"auprc": [], "recall": []} for a in ARMS[1:]}
best_ctrl = max(("C_a", "C_b", "C_c", "C_f"), key=lambda a: point[a]["auprc"])
deltas[f"Q-{best_ctrl}"] = {"auprc": [], "recall": []}
for b in range(BOOT_N):
    idx = rng.integers(0, n, n)
    yb = y[idx]
    if yb.sum() == 0:
        continue
    poolb, aboveb = pool[idx], above[idx]
    vals = {}
    for a in ARMS:
        sb = S[a][idx]
        vals[a] = (auprc(yb, sb), recall_at_budget(yb, sb, poolb, aboveb))
    for a in ARMS[1:]:
        deltas[f"{a}-backbone"]["auprc"].append(vals[a][0] - vals["backbone"][0])
        deltas[f"{a}-backbone"]["recall"].append(vals[a][1] - vals["backbone"][1])
    deltas[f"Q-{best_ctrl}"]["auprc"].append(vals["Q"][0] - vals[best_ctrl][0])
    deltas[f"Q-{best_ctrl}"]["recall"].append(vals["Q"][1] - vals[best_ctrl][1])
    if (b + 1) % 2000 == 0:
        log(f"  bootstrap {b+1}/{BOOT_N}")

ci = {}
for k, d in deltas.items():
    ci[k] = {m: [float(np.percentile(d[m], 2.5)), float(np.percentile(d[m], 97.5)),
                 float(np.mean(d[m]))] for m in ("auprc", "recall")}

# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
dA = point["Q"]["auprc"] - point["backbone"]["auprc"]
dR = point["Q"]["recall_at_budget"] - point["backbone"]["recall_at_budget"]
qb = ci["Q-backbone"]
g1 = bool((qb["auprc"][0] > 0) or (qb["recall"][0] > 0))
g2 = bool(dA >= 0.005 or dR >= 0.020)
g3_point = all(point["Q"]["auprc"] > point[a]["auprc"]
               for a in ("C_a", "C_b", "C_c", "C_f"))
g3 = bool(g3_point and ci[f"Q-{best_ctrl}"]["auprc"][0] > 0)
g4 = bool(point["Q"]["auprc"] >= point["backbone"]["auprc"] - 0.001)
verdict = "PASS" if (g1 and g2 and g3 and g4) else "FAIL"

out = {"mode": MODE, "prereg_sha256": sha,
       "regions": {"above": int(above.sum()), "pool": int(pool.sum()),
                   "pool_frauds": int(y[pool].sum()),
                   "below_frauds": int(y[~pool & ~above].sum())},
       "points": point, "deltas_ci95": ci, "best_control": best_ctrl,
       "gates": {"G1_ci_excludes_0": g1, "G2_practical": g2,
                 "G3_beats_all_controls": g3, "G4_no_harm": g4},
       "verdict": verdict}
path = os.path.join(BASE, f"unblind_{MODE}_v1.json")
json.dump(out, open(path, "w"), indent=2)
log(f"GATES: G1={g1} G2={g2} G3={g3} G4={g4}  ->  "
    f"{'REHEARSAL (non-binding): ' if MODE == 'rehearsal' else ''}{verdict}")
log(f"artifact -> {path}")
