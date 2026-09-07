"""EXP-F: does the S-LCU scorer address the challenge's 3.4 bottlenecks?

  F1  Distribution shift (bottleneck: adversarial adaptation / robustness ask):
      temporal-PROXY split by row order (ULB rows are time-ordered in the
      source CSV; OpenML preserves row order, the Time column is dropped) --
      first 60% train / next 20% val / last 20% test.  Degradation of each
      method vs the stratified split.
  F2  Novel fraud modes (bottleneck 4): k-means clusters of the 492 frauds;
      hold out one cluster entirely from every supervised signal; evaluate on
      test legit + held-out-cluster frauds.  One-class scorers should degrade
      less than supervised XGBoost.
  F3  Detection-experience tradeoff (bottleneck 1) + the headroom hunt:
      review-band analysis of XGBoost-full (band = test scores between the
      99.0th and 99.9th percentiles), re-ranked by the hybrid; plus
      complementarity statistics: rank correlation of the quantum score with
      XGBoost, and the "missed-fraud AUC" -- how well p_s ranks the frauds
      XGBoost leaves below the review band.  Requires exp_b_scorer.json.

Artifacts -> runs/hsbc_challenge/fit_v0/exp_f_bottlenecks.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hsbc_common import (
    BatchedSLCU, SLCUStackConfig, fit_ising_pseudolikelihood, hard_bits,
    pairwise_loss_and_grad, prepare_hsbc, product_probs_batch,
    product_state_batch, ranking_metrics, save_json,
)

SEED = 0
N = 8
L = 2
CHUNK = 8192
T0 = time.time()
OUT = {}


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def ps_chunked(sim, params, U):
    outs = []
    for i in range(0, len(U), CHUNK):
        v = sim.forward(params, product_state_batch(U[i:i + CHUNK]).astype(complex))
        outs.append(np.einsum("md,md->m", v.conj(), v).real)
    return np.concatenate(outs)


def probs_matmul(U, G):
    return np.concatenate([product_probs_batch(U[i:i + CHUNK]) @ G
                           for i in range(0, len(U), CHUNK)])


def fit_xgb(A_tr, y_tr, A_va, y_va, seed=SEED):
    import xgboost as xgb
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.1, subsample=0.9,
        colsample_bytree=0.9, eval_metric="aucpr", tree_method="hist",
        early_stopping_rounds=50, scale_pos_weight=spw, n_jobs=4,
        random_state=seed)
    clf.fit(A_tr, y_tr, eval_set=[(A_va, y_va)], verbose=False)
    return clf


def train_s2(model_hp, U_tr, y_tr, U_va_sel, y_va_sel, fraud_mask=None,
             seeds=(0, 1, 2), iters=300):
    """Train the mixer stack; fraud_mask restricts which train frauds are used."""
    from sklearn.metrics import average_precision_score
    cfg = SLCUStackConfig(n_qubits=N, n_layers=L, hp=model_hp)
    sim = BatchedSLCU(cfg)
    fr = (y_tr == 1) if fraud_mask is None else ((y_tr == 1) & fraud_mask)
    fraud_states = product_state_batch(U_tr[fr]).astype(complex)
    legit_idx = np.flatnonzero(y_tr == 0)
    best_all = (-1.0, None)
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        params = np.zeros(cfg.n_params)
        for l in range(L):
            params[4 * l: 4 * l + 2] = rng.normal(scale=0.15, size=2)
            params[4 * l + 2] = rng.uniform(0.5, 2.5)
            params[4 * l + 3] = rng.normal(scale=0.15)
        m1 = np.zeros(cfg.n_params)
        m2 = np.zeros(cfg.n_params)
        best = (-1.0, params.copy())
        for it in range(iters):
            batch = rng.choice(legit_idx, 384, replace=False)
            legit_states = product_state_batch(U_tr[batch]).astype(complex)
            _, grad, _, _ = pairwise_loss_and_grad(sim, params, legit_states,
                                                   fraud_states)
            m1 = 0.9 * m1 + 0.1 * grad
            m2 = 0.999 * m2 + 0.001 * grad * grad
            params = params - 0.05 * (m1 / (1 - 0.9 ** (it + 1))) / (
                np.sqrt(m2 / (1 - 0.999 ** (it + 1))) + 1e-8)
            if (it + 1) % 20 == 0 or it == iters - 1:
                ap = average_precision_score(
                    y_va_sel, -ps_chunked(sim, params, U_va_sel))
                if ap > best[0]:
                    best = (ap, params.copy())
        if best[0] > best_all[0]:
            best_all = best
    return sim, best_all[1], best_all[0]


def recall_at_fpr(y, s, fpr=1e-3):
    thr = np.quantile(s[y == 0], 1.0 - fpr)
    return float((s[y == 1] > thr).mean())


# ===========================================================================
# F1: temporal-proxy split
# ===========================================================================
from hsbc_common import load_ulb
X, y, cols = load_ulb()
m = len(y)
b1, b2 = int(0.6 * m), int(0.8 * m)
splits = (np.arange(0, b1), np.arange(b1, b2), np.arange(b2, m))
OUT["F1_split_fraud_counts"] = {k: int(y[i].sum()) for k, i in
                                zip(("train", "val", "test"), splits)}
log(f"F1 temporal-proxy split fraud counts: {OUT['F1_split_fraud_counts']}")

pipe_t = prepare_hsbc(N, seed=SEED, splits=splits)
model_t = pipe_t["model"]
U_tr, U_va, U_te = pipe_t["U"]["tr"], pipe_t["U"]["va"], pipe_t["U"]["te"]
y_tr, y_va, y_te = (pipe_t["y_split"][k] for k in ("tr", "va", "te"))
OUT["F1_features"] = pipe_t["feature_names"]

from sklearn.metrics import average_precision_score

rows = {}
# S1: analytic ITE, tau on val
tau_grid = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
G = np.column_stack([np.exp(-2.0 * t * model_t.hp) for t in tau_grid])
E_va = probs_matmul(U_va, G)
E_te = probs_matmul(U_te, G)
j_star = int(np.argmax([average_precision_score(y_va, -E_va[:, j])
                        for j in range(len(tau_grid))]))
rows["S1_soft_ite"] = ranking_metrics(y_te, -E_te[:, j_star])
rows["S1_soft_ite"]["tau_star"] = tau_grid[j_star]

# S2 trained
rng_v = np.random.default_rng(SEED)
legit_va = np.flatnonzero(y_va == 0)
val_sel = np.sort(np.concatenate(
    [np.flatnonzero(y_va == 1),
     rng_v.choice(legit_va, min(8000, len(legit_va)), replace=False)]))
sim_t, params_t, _ = train_s2(model_t.hp, U_tr, y_tr, U_va[val_sel], y_va[val_sel])
s2_te = -ps_chunked(sim_t, params_t, U_te)
rows["S2_trained_mixer"] = ranking_metrics(y_te, s2_te)

# classical
feat = pipe_t["feat"]
idx_tr, idx_va, idx_te = splits
clf_sub = fit_xgb(X[np.ix_(idx_tr, feat)], y_tr, X[np.ix_(idx_va, feat)], y_va)
rows["C_xgb_subset_raw"] = ranking_metrics(
    y_te, clf_sub.predict_proba(X[np.ix_(idx_te, feat)])[:, 1])
clf_full = fit_xgb(X[idx_tr], y_tr, X[idx_va], y_va)
xgb_te = clf_full.predict_proba(X[idx_te])[:, 1]
rows["C_xgb_full"] = ranking_metrics(y_te, xgb_te)

# hybrid
s2_tr = -ps_chunked(sim_t, params_t, U_tr)
s2_va = -ps_chunked(sim_t, params_t, U_va)
clf_hy = fit_xgb(np.column_stack([X[idx_tr], s2_tr]), y_tr,
                 np.column_stack([X[idx_va], s2_va]), y_va)
rows["HY_xgb_full_plus_S2"] = ranking_metrics(
    y_te, clf_hy.predict_proba(np.column_stack([X[idx_te], s2_te]))[:, 1])

OUT["F1_temporal_proxy_test_metrics"] = rows
for k, v in rows.items():
    log(f"F1 {k}: AUPRC {v['auprc']:.4f}  AUC {v['auc_roc']:.4f}")

# ===========================================================================
# F2: fraud-cluster holdout (novel fraud modes)
# ===========================================================================
pipe = prepare_hsbc(N, seed=SEED)
model = pipe["model"]
U_tr, U_va, U_te = pipe["U"]["tr"], pipe["U"]["va"], pipe["U"]["te"]
y_tr, y_va, y_te = (pipe["y_split"][k] for k in ("tr", "va", "te"))
idx_tr, idx_va, idx_te = (pipe["idx"][k] for k in ("tr", "va", "te"))

from sklearn.cluster import KMeans

U_fr_all = np.concatenate([U_tr[y_tr == 1], U_va[y_va == 1], U_te[y_te == 1]])
km = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit(U_fr_all)
lab_tr = km.predict(U_tr[y_tr == 1])
lab_va = km.predict(U_va[y_va == 1])
lab_te = km.predict(U_te[y_te == 1])
OUT["F2_cluster_sizes"] = np.bincount(km.labels_, minlength=4).tolist()
log(f"F2 fraud cluster sizes: {OUT['F2_cluster_sizes']}")

f2 = {}
for c in range(4):
    # supervised signal excludes cluster c everywhere
    keep_tr_fr = lab_tr != c
    keep_va_fr = lab_va != c

    # rows used for xgb training/val: legit + non-c frauds
    tr_keep = np.flatnonzero((y_tr == 0) | ((y_tr == 1) & np.isin(
        np.arange(len(y_tr)), np.flatnonzero(y_tr == 1)[keep_tr_fr])))
    va_keep = np.flatnonzero((y_va == 0) | ((y_va == 1) & np.isin(
        np.arange(len(y_va)), np.flatnonzero(y_va == 1)[keep_va_fr])))

    # eval rows: all test legit + ALL cluster-c frauds (never seen in training)
    U_eval = np.concatenate([U_te[y_te == 0], U_fr_all[km.labels_ == c]])
    y_eval = np.concatenate([np.zeros((y_te == 0).sum(), int),
                             np.ones(int((km.labels_ == c).sum()), int)])
    X_eval = np.vstack([
        X[idx_te[y_te == 0]],
        np.vstack([X[idx_tr[y_tr == 1]][lab_tr == c],
                   X[idx_va[y_va == 1]][lab_va == c],
                   X[idx_te[y_te == 1]][lab_te == c]])])

    row = {"held_out_frauds": int(y_eval.sum())}

    # S1 label-free (tau preset 4.0) and tau-on-remaining-frauds
    g4 = np.exp(-2.0 * 4.0 * model.hp)
    s = -probs_matmul(U_eval, g4[:, None])[:, 0]
    row["S1_fixed_tau4"] = {**ranking_metrics(y_eval, s),
                            "recall_at_fpr1e-3": recall_at_fpr(y_eval, s)}

    # S2 trained without cluster-c frauds
    rng_v = np.random.default_rng(SEED)
    va_fr_keep = np.flatnonzero(y_va == 1)[keep_va_fr]
    val_sel = np.sort(np.concatenate(
        [va_fr_keep, rng_v.choice(np.flatnonzero(y_va == 0), 8000, replace=False)]))
    mask = np.zeros(len(y_tr), bool)
    mask[np.flatnonzero(y_tr == 1)[keep_tr_fr]] = True
    sim2, params2, _ = train_s2(model.hp, U_tr, y_tr, U_va[val_sel],
                                y_va[val_sel], fraud_mask=mask)
    s = -ps_chunked(sim2, params2, U_eval)
    row["S2_wo_cluster"] = {**ranking_metrics(y_eval, s),
                           "recall_at_fpr1e-3": recall_at_fpr(y_eval, s)}

    # XGBoost-full without cluster-c frauds
    clf = fit_xgb(X[idx_tr[tr_keep]], y_tr[tr_keep],
                  X[idx_va[va_keep]], y_va[va_keep])
    s = clf.predict_proba(X_eval)[:, 1]
    row["XGB_full_wo_cluster"] = {**ranking_metrics(y_eval, s),
                                  "recall_at_fpr1e-3": recall_at_fpr(y_eval, s)}

    # ceiling: XGBoost-full trained with everything (in-distribution reference)
    clf_ref = fit_xgb(X[idx_tr], y_tr, X[idx_va], y_va)
    s = clf_ref.predict_proba(X_eval)[:, 1]
    row["XGB_full_with_cluster_ref"] = {**ranking_metrics(y_eval, s),
                                        "recall_at_fpr1e-3": recall_at_fpr(y_eval, s)}
    f2[f"cluster_{c}"] = row
    log(f"F2 cluster {c} ({row['held_out_frauds']} frauds): " + ", ".join(
        f"{k}: AUPRC {v['auprc']:.3f}/R@ {v['recall_at_fpr1e-3']:.3f}"
        for k, v in row.items() if isinstance(v, dict)))

OUT["F2_cluster_holdout"] = f2

# ===========================================================================
# F3: review-band re-ranking + headroom hunt (needs exp_b artifacts)
# ===========================================================================
with open("runs/hsbc_challenge/fit_v0/exp_b_scorer.json") as fh:
    expb = json.load(fh)
params_b = np.array(expb["n8_L2"]["S2_trained_mixer_params"])
assert expb["n8_L2"]["features"] == pipe["feature_names"], "pipeline drift"

cfg = SLCUStackConfig(n_qubits=N, n_layers=L, hp=model.hp)
sim = BatchedSLCU(cfg)
s2_te = -ps_chunked(sim, params_b, U_te)
s2_tr = -ps_chunked(sim, params_b, U_tr)
s2_va = -ps_chunked(sim, params_b, U_va)

clf_full = fit_xgb(X[idx_tr], y_tr, X[idx_va], y_va)
xgb_te = clf_full.predict_proba(X[idx_te])[:, 1]
clf_hy = fit_xgb(np.column_stack([X[idx_tr], s2_tr]), y_tr,
                 np.column_stack([X[idx_va], s2_va]), y_va)
hy_te = clf_hy.predict_proba(np.column_stack([X[idx_te], s2_te]))[:, 1]

t_hi, t_lo = np.quantile(xgb_te, 0.999), np.quantile(xgb_te, 0.99)
band = (xgb_te > t_lo) & (xgb_te <= t_hi)
above = xgb_te > t_hi
below = xgb_te <= t_lo
f3 = {
    "band_def": "xgb_full test scores in (P99.0, P99.9]",
    "auto_flag_region": {"rows": int(above.sum()), "frauds": int(y_te[above].sum())},
    "review_band": {"rows": int(band.sum()), "frauds": int(y_te[band].sum())},
    "below_band": {"rows": int(below.sum()), "frauds": int(y_te[below].sum())},
}
for k_cap in (25, 50, 100):
    def caught(order_scores):
        b_idx = np.flatnonzero(band)
        top = b_idx[np.argsort(-order_scores[b_idx])[:k_cap]]
        return int(y_te[top].sum())
    f3[f"band_frauds_caught_top{k_cap}"] = {
        "xgb_order": caught(xgb_te),
        "hybrid_order": caught(hy_te),
        "s2_order": caught(s2_te),
    }

# complementarity / headroom statistics
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
legit_te = y_te == 0
fraud_te = y_te == 1
f3["spearman_s2_vs_xgb_legit"] = float(spearmanr(s2_te[legit_te][:20000],
                                                 xgb_te[legit_te][:20000]).statistic)
f3["spearman_s2_vs_xgb_fraud"] = float(spearmanr(s2_te[fraud_te],
                                                 xgb_te[fraud_te]).statistic)
missed = fraud_te & below
f3["missed_frauds_below_band"] = int(missed.sum())
if missed.sum() >= 3:
    yy = np.concatenate([np.zeros(int(legit_te.sum()), int),
                         np.ones(int(missed.sum()), int)])
    f3["missed_fraud_auc_S2"] = float(roc_auc_score(
        yy, np.concatenate([s2_te[legit_te], s2_te[missed]])))
    g4 = np.exp(-2.0 * 4.0 * model.hp)
    s1_te = -probs_matmul(U_te, g4[:, None])[:, 0]
    f3["missed_fraud_auc_S1"] = float(roc_auc_score(
        yy, np.concatenate([s1_te[legit_te], s1_te[missed]])))
OUT["F3_review_band"] = f3
log(f"F3: {json.dumps(f3, indent=1)[:600]}")

OUT["wall_seconds"] = time.time() - T0
save_json("runs/hsbc_challenge/fit_v0/exp_f_bottlenecks.json", OUT)
log("EXP-F done -> runs/hsbc_challenge/fit_v0/exp_f_bottlenecks.json")
