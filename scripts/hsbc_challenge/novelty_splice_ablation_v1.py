"""Candidate H runner: Engine-A splice-mechanism ablation, VALIDATION ONLY.

Protocol: docs/hsbc_challenge_splice_ablation_preregistration_v1.md (SHA-256
printed at start).  Reads trainval.parquet only; the sealed test parquet is
never opened (guarded).  Frozen Engine-A heads and routing are reused; one
new logit-only head C_0 is fitted on the train pool.

Artifacts -> runs/hsbc_challenge/novelty_v1/splice_ablation_v1.json
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
from hsbc_common import (
    CrossViewConfig, CrossViewSLCU, broadcast_view_diagonal, keyed_noise,
    product_state_batch,
)
from novelty_common_v1 import save_json_no_overwrite, sha256_file

T0 = time.time()
PROTOCOL = "docs/hsbc_challenge_splice_ablation_preregistration_v1.md"
BASE = "runs/hsbc_challenge/engine_a_v1"
OUTD = "runs/hsbc_challenge/novelty_v1"
os.makedirs(OUTD, exist_ok=True)
OUT_PATH = os.path.join(OUTD, "splice_ablation_v1.json")
if os.path.exists(OUT_PATH):
    raise FileExistsError(OUT_PATH)
BOOT_N, BOOT_SEED, K_BUDGET = 10_000, 20260904, 400
OUT = {"schema": "hsbc-novelty-splice-ablation-v1",
       "protocol": {"path": PROTOCOL, "sha256": sha256_file(PROTOCOL)}, "sources": {}}


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


log(f"PROTOCOL {PROTOCOL} SHA-256 = {OUT['protocol']['sha256']}")

_orig_read_parquet = pd.read_parquet


def guarded_read_parquet(path, *a, **k):
    assert "SEALED" not in str(path), "sealed test partition must never be read"
    return _orig_read_parquet(path, *a, **k)


pd.read_parquet = guarded_read_parquet

# ---------------------------------------------------------------------------
# frozen artifacts
# ---------------------------------------------------------------------------
views = joblib.load(os.path.join(BASE, "views_v1.joblib"))
routing = json.load(open(os.path.join(BASE, "routing_freeze_v1.json")))
arms_sum = json.load(open(os.path.join(BASE, "arms_v1", "arms_summary_v1.json")))
for f in ("views_v1.joblib", "routing_freeze_v1.json", "arms_v1/arms_summary_v1.json",
          "arms_v1/head_C_a.joblib", "arms_v1/head_C_b.joblib", "arms_v1/head_C_c.joblib",
          "arms_v1/head_C_f.joblib"):
    OUT["sources"][f] = sha256_file(os.path.join(BASE, f))
t_lo = routing["routing_thresholds"]["t_lo_val_q988"]
t_hi = routing["routing_thresholds"]["t_hi_val_q998"]

cols = list(dict.fromkeys(["TransactionID", "TransactionDT", "isFraud", "split"] + views["features"]))
df = pd.read_parquet("runs/hsbc_challenge/data/ieee/trainval.parquet", columns=cols)
is_tr = (df["split"] == "train").to_numpy()
va = df[~is_tr].reset_index(drop=True)
tr = df[is_tr].reset_index(drop=True)
del df
y = va["isFraud"].to_numpy()
pval = 0.5 * (np.load(os.path.join(BASE, "backbone_vm", "best_xgb_val_pred.npy")).astype(np.float64)
              + np.load(os.path.join(BASE, "backbone_vm", "best_lgb_val_pred.npy")).astype(np.float64))
oof = np.load(os.path.join(BASE, "backbone_vm", "oof_backbone.npy"))
assert len(pval) == len(va) and len(oof) == len(tr)
log(f"validation rows {len(va)}, frauds {int(y.sum())}; train rows {len(tr)}")


def views_u(fr):
    out = []
    for c in views["features"]:
        s = fr[c]
        if c in views["freq_maps"]:
            s = s.astype(str).map({k: float(v) for k, v in views["freq_maps"][c].items()}) \
                if s.dtype == object else s.astype(float)
        x = np.nan_to_num(s.to_numpy(dtype=float), nan=-1.0)
        col = views["sorted_cols"][c]
        lo = np.searchsorted(col, x, side="left")
        hi = np.searchsorted(col, x, side="right")
        out.append(np.clip(0.5 * (lo + hi) / len(col), 0.0, 1.0))
    return np.column_stack(out)


def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


VQ = views["view_qubits"]
name2idx = {"A": 0, "B": 1, "C": 2}
cfg = CrossViewConfig(
    n_qubits=12, views=tuple(tuple(VQ[v]) for v in ("A", "B", "C")),
    layer_pairs=tuple((name2idx[a], name2idx[b]) for a, b in views["layer_pairs"]),
    hp_views=tuple(broadcast_view_diagonal(np.array(views["hp_tables"][v]), VQ[v], 12)
                   for v in ("A", "B", "C")))
sim = CrossViewSLCU(cfg)

# new C_0 logit-only head on the train OOF pool (no selection, C=1)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
tr_pool = (oof > t_lo) & (oof <= t_hi)
y_tr = tr["isFraud"].to_numpy()
sc0 = StandardScaler().fit(logit(oof[tr_pool])[:, None])
c0 = LogisticRegression(C=1.0, max_iter=5000).fit(sc0.transform(logit(oof[tr_pool])[:, None]), y_tr[tr_pool])
OUT["C_0_logit_only_head"] = {"train_pool_rows": int(tr_pool.sum()), "coef": float(c0.coef_[0, 0]),
                              "intercept": float(c0.intercept_[0])}


def head_out(head, pb, feats):
    A = np.column_stack([logit(pb), feats])
    return head["clf"].predict_proba(head["scaler"].transform(A))[:, 1]


def pool_scores(frame, pb):
    """Head outputs for every arm on the given band rows."""
    U = views_u(frame)
    st = product_state_batch(U).astype(complex)
    out = {}
    qs = []
    for seed, rec in arms_sum["Q"]["seeds"].items():
        feats = sim.syndrome_features(np.array(rec["params"]), st)
        head = joblib.load(os.path.join(BASE, "arms_v1", f"q_head_seed{seed}.joblib"))
        qs.append(head_out(head, pb, feats))
    out["Q"] = np.mean(qs, axis=0)
    out["C_a"] = head_out(joblib.load(os.path.join(BASE, "arms_v1", "head_C_a.joblib")), pb, U)
    cb = joblib.load(os.path.join(BASE, "arms_v1", "head_C_b.joblib"))
    out["C_b"] = cb["clf"].predict_proba(cb["scaler"].transform(np.column_stack([logit(pb), U])))[:, 1]
    cc = joblib.load(os.path.join(BASE, "arms_v1", "head_C_c.joblib"))
    Z = cc["pca"].inverse_transform(cc["pca"].transform(U))
    out["C_c"] = head_out(cc["head"], pb, np.sqrt(((U - Z) ** 2).sum(axis=1))[:, None])
    out["C_f"] = head_out(joblib.load(os.path.join(BASE, "arms_v1", "head_C_f.joblib")), pb,
                          keyed_noise(frame["TransactionID"].to_numpy()))
    out["C_0"] = c0.predict_proba(sc0.transform(logit(pb)[:, None]))[:, 1]
    return out


def auprc_w(yv, sv, w):
    order = np.argsort(-sv, kind="mergesort")
    yy, ww = yv[order], w[order]
    tp = np.cumsum(ww * yy)
    tot = np.cumsum(ww)
    prec = np.where(tot > 0, tp / np.maximum(tot, 1e-12), 0.0)
    denom = (ww * yy).sum()
    return float((prec * ww * yy).sum() / denom) if denom > 0 else 0.0


def recall_w(yv, sv, w, poolmask, abovemask):
    pool_idx = np.flatnonzero(poolmask)
    o = pool_idx[np.argsort(-sv[pool_idx], kind="mergesort")]
    cw = np.cumsum(w[o])
    top = o[cw <= K_BUDGET]
    caught = (w[abovemask] * yv[abovemask]).sum() + (w[top] * yv[top]).sum()
    denom = (w * yv).sum()
    return float(caught / denom) if denom > 0 else 0.0


def build_period(frame, pb, yv, lo, hi, tag):
    pool = (pb > lo) & (pb <= hi)
    above = pb > hi
    heads = pool_scores(frame[pool], pb[pool])
    S = {"backbone": pb.copy()}
    band_sorted = np.sort(pb[pool])
    interleave = {}
    for arm, h in heads.items():
        s1 = pb.copy()
        s1[pool] = h
        S[f"{arm}|M1"] = s1
        s2 = pb.copy()
        rank = np.argsort(np.argsort(h, kind="mergesort"), kind="mergesort")
        s2[pool] = band_sorted[rank]
        S[f"{arm}|M2"] = s2
        interleave[arm] = {"frac_band_rows_above_t_hi_under_M1": float((h > hi).mean()),
                           "frac_band_rows_below_t_lo_under_M1": float((h <= lo).mean())}
    log(f"[{tag}] rows {len(pb)}, frauds {int(yv.sum())}, band {int(pool.sum())} "
        f"(frauds {int(yv[pool].sum())}), above {int(above.sum())}")
    return S, pool, above, interleave


periods = {}
# in-period: full validation, frozen thresholds
periods["in_period_full_validation"] = (va, pval, y, t_lo, t_hi)
# rolling origin: thresholds from V1, evaluate on V2
order = np.argsort(va["TransactionDT"].to_numpy(), kind="mergesort")
half = len(order) // 2
v1, v2 = order[:half], order[half:]
lo2, hi2 = float(np.percentile(pval[v1], 98.8)), float(np.percentile(pval[v1], 99.8))
periods["rolling_origin_V2"] = (va.iloc[v2].reset_index(drop=True), pval[v2], y[v2], lo2, hi2)
OUT["rolling_origin_thresholds_from_V1"] = {"t_lo": lo2, "t_hi": hi2, "V1_rows": int(half),
                                            "V2_rows": int(len(v2)), "V2_frauds": int(y[v2].sum())}

results = {}
rng = np.random.default_rng(BOOT_SEED)
for tag, (frame, pb, yv, lo, hi) in periods.items():
    S, pool, above, interleave = build_period(frame, pb, yv, lo, hi, tag)
    n = len(yv)
    keys = list(S.keys())
    point = {k: {"auprc": auprc_w(yv, S[k], np.ones(n)),
                 "recall_at_budget": recall_w(yv, S[k], np.ones(n), pool, above)} for k in keys}
    # paired Poisson(1) bootstrap
    boots = {k: {"auprc": np.empty(BOOT_N), "recall": np.empty(BOOT_N)} for k in keys}
    orders = {k: np.argsort(-S[k], kind="mergesort") for k in keys}
    t0 = time.time()
    for b in range(BOOT_N):
        w = rng.poisson(1.0, n).astype(float)
        if (w * yv).sum() == 0:
            w[:] = 1.0
        for k in keys:
            o = orders[k]
            yy, ww = yv[o], w[o]
            tp = np.cumsum(ww * yy)
            tot = np.cumsum(ww)
            prec = np.where(tot > 0, tp / np.maximum(tot, 1e-12), 0.0)
            boots[k]["auprc"][b] = (prec * ww * yy).sum() / (ww * yy).sum()
            boots[k]["recall"][b] = recall_w(yv, S[k], w, pool, above)
        if (b + 1) % 2000 == 0:
            log(f"  [{tag}] bootstrap {b+1}/{BOOT_N} ({time.time()-t0:.0f}s)")
    deltas = {}
    for k in keys:
        if k == "backbone":
            continue
        for metric, key in (("auprc", "auprc"), ("recall", "recall")):
            d = boots[k][key] - boots["backbone"][key]
            deltas.setdefault(k, {})[metric] = {
                "point": point[k]["auprc" if metric == "auprc" else "recall_at_budget"]
                         - point["backbone"]["auprc" if metric == "auprc" else "recall_at_budget"],
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "ci97.5": [float(np.percentile(d, 1.25)), float(np.percentile(d, 98.75))]}
    contrasts = {}
    for arm in ("C_f", "Q", "C_a", "C_b", "C_c", "C_0"):
        d = (boots[f"{arm}|M1"]["auprc"] - boots[f"{arm}|M2"]["auprc"])
        contrasts[f"{arm}: M1 minus M2 (interleaving component)"] = {
            "point": point[f"{arm}|M1"]["auprc"] - point[f"{arm}|M2"]["auprc"],
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "ci97.5": [float(np.percentile(d, 1.25)), float(np.percentile(d, 98.75))]}
    results[tag] = {"points": point, "deltas_vs_backbone": deltas, "M1_minus_M2": contrasts,
                    "interleaving_diagnostic": interleave,
                    "regions": {"rows": int(n), "frauds": int(yv.sum()), "band": int(pool.sum()),
                                "band_frauds": int(yv[pool].sum()), "above": int(above.sum()),
                                "above_frauds": int(yv[above].sum())}}
    for k in keys:
        log(f"  [{tag}] {k:10s} AUPRC {point[k]['auprc']:.5f} recall@K {point[k]['recall_at_budget']:.5f}")

OUT["results"] = results
ip = results["in_period_full_validation"]["M1_minus_M2"]
OUT["primary"] = {
    "Delta_1_C_f_interleaving_component_97.5": ip["C_f: M1 minus M2 (interleaving component)"],
    "Delta_2_Q_interleaving_component_97.5": ip["Q: M1 minus M2 (interleaving component)"],
    "C_f_M2_delta_vs_backbone_95": results["in_period_full_validation"]["deltas_vs_backbone"]["C_f|M2"]["auprc"],
    "Q_M2_delta_vs_backbone_95": results["in_period_full_validation"]["deltas_vs_backbone"]["Q|M2"]["auprc"],
}
OUT["wall_seconds"] = time.time() - T0
save_json_no_overwrite(OUT_PATH, OUT)
log(f"DONE -> {OUT_PATH}")
