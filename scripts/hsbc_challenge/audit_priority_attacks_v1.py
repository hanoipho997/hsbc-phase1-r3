"""Preregistered HSBC audit priority attacks (C1, C5, C6, C10).

The protocol was frozen in docs/hsbc_challenge_audit_report_v1.md before this
script was run.  This script never writes into fit_v0.  It produces a JSON
summary and a compressed score bundle so every reported paired comparison can
be independently recomputed.

Run from the repository root:

  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
    scripts/hsbc_challenge/audit_priority_attacks_v1.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hsbc_common import (  # noqa: E402
    BatchedSLCU,
    QuantileTransform,
    SLCUStackConfig,
    fit_ising_pseudolikelihood,
    hard_bits,
    load_ulb,
    pairwise_loss_and_grad,
    product_probs_batch,
    product_state_batch,
    ranking_metrics,
    select_features,
    stratified_split,
)


AUDIT_SEED = 20260830
BOOT_SEED = 20260831
CHUNK = 8192
N_BOOT = 2000
T0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {message}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def probs_matmul(U: np.ndarray, G: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [product_probs_batch(U[i : i + CHUNK]) @ G for i in range(0, len(U), CHUNK)]
    )


def ps_chunked(sim: BatchedSLCU, params: np.ndarray, U: np.ndarray) -> np.ndarray:
    rows = []
    for i in range(0, len(U), CHUNK):
        v = sim.forward(params, product_state_batch(U[i : i + CHUNK]).astype(complex))
        rows.append(np.einsum("md,md->m", v.conj(), v).real)
    return np.concatenate(rows)


def recall_at_fpr(y: np.ndarray, scores: np.ndarray, fpr: float = 1e-3) -> float:
    threshold = np.quantile(scores[y == 0], 1.0 - fpr)
    return float(np.mean(scores[y == 1] > threshold))


def _weighted_ap_sorted(
    y_sorted: np.ndarray,
    weights_sorted: np.ndarray,
    group_ends: np.ndarray,
) -> np.ndarray:
    """Tie-aware weighted AP; score groups end at ``group_ends``."""

    pos_w = weights_sorted * y_sorted[None, :]
    cum_pos = np.cumsum(pos_w, axis=1, dtype=float)
    cum_all = np.cumsum(weights_sorted, axis=1, dtype=float)
    n_pos = cum_pos[:, -1]
    cum_pos_group = cum_pos[:, group_ends]
    cum_all_group = cum_all[:, group_ends]
    precision_group = np.divide(
        cum_pos_group,
        cum_all_group,
        out=np.zeros_like(cum_pos_group),
        where=cum_all_group > 0,
    )
    group_pos = np.diff(
        np.column_stack([np.zeros(len(weights_sorted)), cum_pos_group]), axis=1
    )
    return np.divide(
        np.sum(precision_group * group_pos, axis=1),
        n_pos,
        out=np.full(len(weights_sorted), np.nan),
        where=n_pos > 0,
    )


def paired_poisson_bootstrap(
    y: np.ndarray,
    score_map: dict[str, np.ndarray],
    reference: str,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """Paired Poisson(1) row bootstrap for AP CIs and deltas.

    Poisson bootstrap weights keep every method on identical test rows and are
    the scalable row-bootstrap implementation used by this audit.
    """

    from sklearn.metrics import average_precision_score

    y = np.asarray(y, int)
    orders = {name: np.argsort(-np.asarray(s), kind="mergesort") for name, s in score_map.items()}
    group_ends = {}
    for name, order in orders.items():
        sorted_score = np.asarray(score_map[name])[order]
        group_ends[name] = np.flatnonzero(
            np.r_[sorted_score[1:] != sorted_score[:-1], True]
        )
    point = {name: float(average_precision_score(y, s)) for name, s in score_map.items()}
    draws = {name: [] for name in score_map}
    rng = np.random.default_rng(seed)
    made = 0
    batch = 24
    while made < n_boot:
        take = min(batch, n_boot - made)
        weights = rng.poisson(1.0, size=(take, len(y))).astype(np.float32)
        valid = (weights[:, y == 1].sum(axis=1) > 0) & (weights[:, y == 0].sum(axis=1) > 0)
        if not np.any(valid):
            continue
        weights = weights[valid]
        for name, order in orders.items():
            vals = _weighted_ap_sorted(y[order], weights[:, order], group_ends[name])
            draws[name].extend(vals[np.isfinite(vals)].tolist())
        made = min(len(v) for v in draws.values())
    arrays = {name: np.asarray(vals[:n_boot]) for name, vals in draws.items()}
    result = {
        "bootstrap": "paired Poisson(1) row weights",
        "n_boot": n_boot,
        "point": point,
        "ci95": {
            name: [float(v) for v in np.quantile(vals, [0.025, 0.975])]
            for name, vals in arrays.items()
        },
        "delta_vs_reference": {},
    }
    for name, vals in arrays.items():
        delta = vals - arrays[reference]
        result["delta_vs_reference"][name] = {
            "point": float(point[name] - point[reference]),
            "ci95": [float(v) for v in np.quantile(delta, [0.025, 0.975])],
        }
    return result


def multi_seed_mean_delta_bootstrap(
    y: np.ndarray,
    seed_scores: list[np.ndarray],
    reference_scores: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    from sklearn.metrics import average_precision_score

    y = np.asarray(y, int)
    orders = [np.argsort(-s, kind="mergesort") for s in seed_scores]
    ref_order = np.argsort(-reference_scores, kind="mergesort")
    seed_groups = [
        np.flatnonzero(np.r_[s[order][1:] != s[order][:-1], True])
        for s, order in zip(seed_scores, orders)
    ]
    ref_sorted = reference_scores[ref_order]
    ref_groups = np.flatnonzero(np.r_[ref_sorted[1:] != ref_sorted[:-1], True])
    seed_point = np.array([average_precision_score(y, s) for s in seed_scores], float)
    ref_point = float(average_precision_score(y, reference_scores))
    delta_draws = []
    rng = np.random.default_rng(seed)
    batch = 16
    while len(delta_draws) < n_boot:
        take = min(batch, n_boot - len(delta_draws))
        weights = rng.poisson(1.0, size=(take, len(y))).astype(np.float32)
        valid = (weights[:, y == 1].sum(axis=1) > 0) & (weights[:, y == 0].sum(axis=1) > 0)
        weights = weights[valid]
        if len(weights) == 0:
            continue
        ref_ap = _weighted_ap_sorted(y[ref_order], weights[:, ref_order], ref_groups)
        aps = np.column_stack(
            [
                _weighted_ap_sorted(y[order], weights[:, order], groups)
                for order, groups in zip(orders, seed_groups)
            ]
        )
        delta_draws.extend((np.nanmean(aps, axis=1) - ref_ap).tolist())
    delta_draws = np.asarray(delta_draws[:n_boot])
    deltas = seed_point - ref_point
    return {
        "seed_auprc": seed_point.tolist(),
        "reference_auprc": ref_point,
        "per_seed_delta": deltas.tolist(),
        "positive_seed_count": int(np.sum(deltas > 0)),
        "mean_delta": float(np.mean(deltas)),
        "mean_delta_ci95": [float(v) for v in np.quantile(delta_draws, [0.025, 0.975])],
        "bootstrap": "paired Poisson(1) row weights; seed AP averaged within draw",
        "n_boot": n_boot,
    }


def fit_xgb(A_tr, y_tr, A_va, y_va, seed: int = AUDIT_SEED):
    import xgboost as xgb

    spw = float(np.sum(y_tr == 0) / max(np.sum(y_tr == 1), 1))
    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="aucpr",
        tree_method="hist",
        early_stopping_rounds=50,
        scale_pos_weight=spw,
        n_jobs=4,
        random_state=seed,
    )
    model.fit(A_tr, y_tr, eval_set=[(A_va, y_va)], verbose=False)
    return model


def original_pipeline() -> dict:
    X, y, cols = load_ulb()
    idx_tr, idx_va, idx_te = stratified_split(y, seed=0)
    feat = select_features(X[idx_tr], y[idx_tr], 8)
    qt = QuantileTransform(X[np.ix_(idx_tr, feat)])
    U_tr, U_va, U_te = (qt(X[np.ix_(idx, feat)]) for idx in (idx_tr, idx_va, idx_te))
    y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]
    model = fit_ising_pseudolikelihood(hard_bits(U_tr)[y_tr == 0], seed=0)
    return {
        "X": X,
        "y": y,
        "cols": cols,
        "idx": {"tr": idx_tr, "va": idx_va, "te": idx_te},
        "feat": feat,
        "features": [cols[j] for j in feat],
        "U": {"tr": U_tr, "va": U_va, "te": U_te},
        "ys": {"tr": y_tr, "va": y_va, "te": y_te},
        "model": model,
    }


def make_s1_scores(pipe: dict, U: np.ndarray) -> np.ndarray:
    g4 = np.exp(-8.0 * pipe["model"].hp)
    return -probs_matmul(U, g4[:, None])[:, 0]


def run_c5(pipe: dict, score_bundle: dict[str, np.ndarray]) -> dict:
    from sklearn.cluster import KMeans

    X, y = pipe["X"], pipe["y"]
    idx_tr, idx_va, idx_te = (pipe["idx"][k] for k in ("tr", "va", "te"))
    U_tr, U_va, U_te = (pipe["U"][k] for k in ("tr", "va", "te"))
    y_tr, y_va, y_te = (pipe["ys"][k] for k in ("tr", "va", "te"))
    fit_fraud = np.concatenate([U_tr[y_tr == 1], U_va[y_va == 1]])
    result = {
        "protocol": "KMeans fit train+val fraud only; evaluation strictly test legit + test held-mode fraud",
        "features": pipe["features"],
        "feature_screen_uses_train_labels": True,
        "tau_uses_prior_label_informed_choice": True,
        "k": {},
    }

    for k in (3, 4, 6):
        km = KMeans(n_clusters=k, n_init=20, random_state=AUDIT_SEED).fit(fit_fraud)
        lab_tr = km.predict(U_tr[y_tr == 1])
        lab_va = km.predict(U_va[y_va == 1])
        lab_te = km.predict(U_te[y_te == 1])
        counts_fit = np.bincount(np.concatenate([lab_tr, lab_va]), minlength=k)
        dominant = int(np.argmax(counts_fit))
        rows = {}
        for cluster in range(k):
            tr_fraud_rows = np.flatnonzero(y_tr == 1)
            va_fraud_rows = np.flatnonzero(y_va == 1)
            te_fraud_rows = np.flatnonzero(y_te == 1)
            held_tr = tr_fraud_rows[lab_tr == cluster]
            held_va = va_fraud_rows[lab_va == cluster]
            held_te = te_fraud_rows[lab_te == cluster]
            tr_keep = np.ones(len(y_tr), bool)
            va_keep = np.ones(len(y_va), bool)
            tr_keep[held_tr] = False
            va_keep[held_va] = False
            eval_local = np.concatenate([np.flatnonzero(y_te == 0), held_te])
            y_eval = y_te[eval_local]
            if np.sum(y_eval == 1) == 0:
                rows[f"cluster_{cluster}"] = {
                    "fit_count": int(counts_fit[cluster]),
                    "train_frauds": int(len(held_tr)),
                    "val_frauds": int(len(held_va)),
                    "test_frauds": 0,
                    "status": "NOT_EVALUABLE_NO_TEST_FRAUD",
                }
                continue
            clf = fit_xgb(
                X[idx_tr[tr_keep]],
                y_tr[tr_keep],
                X[idx_va[va_keep]],
                y_va[va_keep],
                seed=AUDIT_SEED + 10 * k + cluster,
            )
            xgb_score = clf.predict_proba(X[idx_te[eval_local]])[:, 1]
            s1_score = make_s1_scores(pipe, U_te[eval_local])
            boots = paired_poisson_bootstrap(
                y_eval,
                {"S1": s1_score, "XGB_without_mode": xgb_score},
                reference="XGB_without_mode",
                seed=BOOT_SEED + 10 * k + cluster,
            )
            row = {
                "fit_count": int(counts_fit[cluster]),
                "train_frauds": int(len(held_tr)),
                "val_frauds": int(len(held_va)),
                "test_frauds": int(len(held_te)),
                "test_legit": int(np.sum(y_eval == 0)),
                "test_prevalence": float(np.mean(y_eval)),
                "dominant_by_train_val": cluster == dominant,
                "metrics": {
                    "S1": {
                        **ranking_metrics(y_eval, s1_score),
                        "recall_at_fpr1e-3": recall_at_fpr(y_eval, s1_score),
                        "auprc_ci95": boots["ci95"]["S1"],
                    },
                    "XGB_without_mode": {
                        **ranking_metrics(y_eval, xgb_score),
                        "recall_at_fpr1e-3": recall_at_fpr(y_eval, xgb_score),
                        "auprc_ci95": boots["ci95"]["XGB_without_mode"],
                    },
                },
                "S1_minus_XGB": boots["delta_vs_reference"]["S1"],
            }
            rows[f"cluster_{cluster}"] = row
            score_bundle[f"c5_k{k}_c{cluster}_y"] = y_eval
            score_bundle[f"c5_k{k}_c{cluster}_s1"] = s1_score
            score_bundle[f"c5_k{k}_c{cluster}_xgb"] = xgb_score
            log(
                f"C5 k={k} c={cluster}{'*' if cluster == dominant else ''}: "
                f"test fraud={len(held_te)}, S1={row['metrics']['S1']['auprc']:.4f}, "
                f"XGB={row['metrics']['XGB_without_mode']['auprc']:.4f}"
            )
        result["k"][str(k)] = {
            "dominant_cluster_by_train_val": dominant,
            "fit_cluster_counts": counts_fit.tolist(),
            "rows": rows,
        }

    # Label-free feature-screen sensitivity, k=4 only.  Robust scaling by
    # legitimate-train IQR makes the frozen variance ranking identifiable.
    X_tr = X[idx_tr]
    legit = X_tr[y_tr == 0]
    med = np.median(legit, axis=0)
    q25, q75 = np.quantile(legit, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, 1e-12)
    robust_var = np.var((legit - med) / scale, axis=0)
    feat_u = np.argsort(-robust_var)[:8]
    qt_u = QuantileTransform(X[np.ix_(idx_tr, feat_u)])
    Uu_tr, Uu_va, Uu_te = (qt_u(X[np.ix_(idx, feat_u)]) for idx in (idx_tr, idx_va, idx_te))
    model_u = fit_ising_pseudolikelihood(hard_bits(Uu_tr)[y_tr == 0], seed=0)
    km_u = KMeans(n_clusters=4, n_init=20, random_state=AUDIT_SEED).fit(
        np.concatenate([Uu_tr[y_tr == 1], Uu_va[y_va == 1]])
    )
    labels_fit_u = km_u.labels_
    dominant_u = int(np.argmax(np.bincount(labels_fit_u, minlength=4)))
    labels_te_u = km_u.predict(Uu_te[y_te == 1])
    held_te_u = np.flatnonzero(y_te == 1)[labels_te_u == dominant_u]
    eval_u = np.concatenate([np.flatnonzero(y_te == 0), held_te_u])
    g_u = np.exp(-8.0 * model_u.hp)
    s_u = -probs_matmul(Uu_te[eval_u], g_u[:, None])[:, 0]
    result["unsupervised_screen_sensitivity_k4_dominant"] = {
        "screen": "legitimate-train robust-IQR-scaled variance",
        "features": [pipe["cols"][j] for j in feat_u],
        "dominant_cluster": dominant_u,
        "test_frauds": int(len(held_te_u)),
        "test_legit": int(np.sum(y_te == 0)),
        "metrics": ranking_metrics(y_te[eval_u], s_u),
    }
    s_u_full = -probs_matmul(Uu_te, g_u[:, None])[:, 0]
    result["unsupervised_screen_sensitivity_full_test"] = {
        "screen": "legitimate-train robust-IQR-scaled variance",
        "features": [pipe["cols"][j] for j in feat_u],
        "tau": 4.0,
        "test_rows": int(len(y_te)),
        "test_frauds": int(np.sum(y_te)),
        "metrics": ranking_metrics(y_te, s_u_full),
    }
    score_bundle["c5_unsup_y"] = y_te[eval_u]
    score_bundle["c5_unsup_s1"] = s_u
    score_bundle["c5_unsup_full_y"] = y_te
    score_bundle["c5_unsup_full_s1"] = s_u_full
    return result


def run_tau_stability(pipe: dict) -> dict:
    """Paired validation-row bootstrap over the already-frozen S1 tau grid."""

    from sklearn.metrics import average_precision_score

    tau_grid = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
    y = pipe["ys"]["va"]
    score_map = {str(tau): make_s1_scores(pipe, pipe["U"]["va"])
                 if tau == 4.0 else
                 -probs_matmul(
                     pipe["U"]["va"],
                     np.exp(-2.0 * tau * pipe["model"].hp)[:, None],
                 )[:, 0]
                 for tau in tau_grid}
    orders = {name: np.argsort(-score, kind="mergesort") for name, score in score_map.items()}
    group_ends = {}
    for name, order in orders.items():
        sorted_score = score_map[name][order]
        group_ends[name] = np.flatnonzero(
            np.r_[sorted_score[1:] != sorted_score[:-1], True]
        )
    point = {name: float(average_precision_score(y, score)) for name, score in score_map.items()}
    selected = {name: 0 for name in score_map}
    delta_from_tau4 = {name: [] for name in score_map}
    rng = np.random.default_rng(BOOT_SEED + 40)
    made = 0
    while made < N_BOOT:
        take = min(24, N_BOOT - made)
        weights = rng.poisson(1.0, size=(take, len(y))).astype(np.float32)
        valid = (weights[:, y == 1].sum(axis=1) > 0) & (weights[:, y == 0].sum(axis=1) > 0)
        weights = weights[valid]
        if len(weights) == 0:
            continue
        draw_aps = {}
        for name, order in orders.items():
            draw_aps[name] = _weighted_ap_sorted(
                y[order], weights[:, order], group_ends[name]
            )
        matrix = np.column_stack([draw_aps[str(tau)] for tau in tau_grid])
        winners = np.argmax(matrix, axis=1)
        for idx, tau in enumerate(tau_grid):
            selected[str(tau)] += int(np.sum(winners == idx))
            delta_from_tau4[str(tau)].extend(
                (draw_aps[str(tau)] - draw_aps["4.0"]).tolist()
            )
        made += len(weights)
    return {
        "validation_rows": int(len(y)),
        "validation_frauds": int(np.sum(y)),
        "n_boot": N_BOOT,
        "bootstrap": "paired Poisson(1) validation-row weights",
        "point_auprc": point,
        "selection_frequency": {
            name: count / N_BOOT for name, count in selected.items()
        },
        "delta_vs_tau4_ci95": {
            name: [float(value) for value in np.quantile(values[:N_BOOT], [0.025, 0.975])]
            for name, values in delta_from_tau4.items()
        },
    }


def train_l1_replication(pipe: dict, train_seed: int):
    from sklearn.metrics import average_precision_score

    U_tr, U_va = pipe["U"]["tr"], pipe["U"]["va"]
    y_tr, y_va = pipe["ys"]["tr"], pipe["ys"]["va"]
    cfg = SLCUStackConfig(n_qubits=8, n_layers=1, hp=pipe["model"].hp)
    sim = BatchedSLCU(cfg)
    fraud_states = product_state_batch(U_tr[y_tr == 1]).astype(complex)
    legit_idx = np.flatnonzero(y_tr == 0)
    rng_v = np.random.default_rng(0)
    val_sel = np.sort(
        np.concatenate(
            [
                np.flatnonzero(y_va == 1),
                rng_v.choice(np.flatnonzero(y_va == 0), 8000, replace=False),
            ]
        )
    )
    rng = np.random.default_rng(train_seed)
    params = np.array(
        [*rng.normal(scale=0.15, size=2), rng.uniform(0.5, 2.5), rng.normal(scale=0.15)]
    )
    m1, m2 = np.zeros(4), np.zeros(4)
    best = (-np.inf, params.copy(), -1)
    for iteration in range(300):
        batch = rng.choice(legit_idx, 384, replace=False)
        legit_states = product_state_batch(U_tr[batch]).astype(complex)
        _, grad, _, _ = pairwise_loss_and_grad(sim, params, legit_states, fraud_states)
        m1 = 0.9 * m1 + 0.1 * grad
        m2 = 0.999 * m2 + 0.001 * grad * grad
        params = params - 0.05 * (m1 / (1 - 0.9 ** (iteration + 1))) / (
            np.sqrt(m2 / (1 - 0.999 ** (iteration + 1))) + 1e-8
        )
        if (iteration + 1) % 20 == 0:
            val_score = -ps_chunked(sim, params, U_va[val_sel])
            ap = average_precision_score(y_va[val_sel], val_score)
            if ap > best[0]:
                best = (float(ap), params.copy(), iteration + 1)
    return sim, best


def run_c1(pipe: dict, score_bundle: dict[str, np.ndarray]) -> tuple[dict, list[np.ndarray]]:
    y_te = pipe["ys"]["te"]
    s1 = make_s1_scores(pipe, pipe["U"]["te"])
    seed_scores = []
    rows = []
    for train_seed in range(500, 510):
        sim, (val_ap, params, checkpoint) = train_l1_replication(pipe, train_seed)
        test_score = -ps_chunked(sim, params, pipe["U"]["te"])
        seed_scores.append(test_score)
        rows.append(
            {
                "seed": train_seed,
                "val_auprc": val_ap,
                "checkpoint": checkpoint,
                "params": params.tolist(),
                "test_metrics": ranking_metrics(y_te, test_score),
            }
        )
        score_bundle[f"c1_seed_{train_seed}"] = test_score
        log(f"C1 seed={train_seed}: val={val_ap:.4f}, test={rows[-1]['test_metrics']['auprc']:.4f}")
    score_bundle["c1_y"] = y_te
    score_bundle["c1_s1"] = s1
    summary = multi_seed_mean_delta_bootstrap(y_te, seed_scores, s1)
    return {"protocol_seeds": list(range(500, 510)), "per_seed": rows, "summary": summary}, seed_scores


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def run_c6(pipe: dict, score_bundle: dict[str, np.ndarray]) -> dict:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y = pipe["X"], pipe["y"]
    idx_tr, idx_va, idx_te = (pipe["idx"][k] for k in ("tr", "va", "te"))
    y_tr, y_va, y_te = (pipe["ys"][k] for k in ("tr", "va", "te"))
    base = fit_xgb(X[idx_tr], y_tr, X[idx_va], y_va, seed=AUDIT_SEED)
    xgb_va = base.predict_proba(X[idx_va])[:, 1]
    xgb_te = base.predict_proba(X[idx_te])[:, 1]

    expb = json.loads(Path("runs/hsbc_challenge/fit_v0/exp_b_scorer.json").read_text())
    params = np.asarray(expb["n8_L2"]["S2_trained_mixer_params"], float)
    sim = BatchedSLCU(SLCUStackConfig(n_qubits=8, n_layers=2, hp=pipe["model"].hp))
    s2_va = -ps_chunked(sim, params, pipe["U"]["va"])
    s2_te = -ps_chunked(sim, params, pipe["U"]["te"])

    scaler = StandardScaler().fit(s2_va[:, None])
    Z_va = np.column_stack([_logit(xgb_va), scaler.transform(s2_va[:, None])[:, 0]])
    Z_te = np.column_stack([_logit(xgb_te), scaler.transform(s2_te[:, None])[:, 0]])
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    lr.fit(Z_va, y_va)
    logit_te = lr.predict_proba(Z_te)[:, 1]

    Zi_va = np.column_stack([Z_va, Z_va[:, 0] * Z_va[:, 1]])
    Zi_te = np.column_stack([Z_te, Z_te[:, 0] * Z_te[:, 1]])
    lr_i = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    lr_i.fit(Zi_va, y_va)
    interact_te = lr_i.predict_proba(Zi_te)[:, 1]

    iso_x = IsotonicRegression(out_of_bounds="clip").fit(xgb_va, y_va)
    iso_s = IsotonicRegression(out_of_bounds="clip").fit(s2_va, y_va)
    px, ps = iso_x.predict(xgb_te), iso_s.predict(s2_te)
    iso_mean = 0.5 * (px + ps)
    iso_product = 1.0 - (1.0 - px) * (1.0 - ps)

    lo, hi = np.quantile(xgb_va, [0.99, 0.999])
    band_va = (xgb_va > lo) & (xgb_va <= hi)
    band_te = (xgb_te > lo) & (xgb_te <= hi)
    band_score = xgb_te.copy()
    band_status = "NOT_RUN"
    if np.unique(y_va[band_va]).size == 2 and np.sum(band_te) > 0:
        lr_b = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
        lr_b.fit(Z_va[band_va], y_va[band_va])
        meta = lr_b.predict_proba(Z_te[band_te])[:, 1]
        original_values = np.sort(xgb_te[band_te])
        destination = np.flatnonzero(band_te)[np.argsort(meta, kind="mergesort")]
        band_score[destination] = original_values
        band_status = "OK"

    methods = {
        "XGB_full": xgb_te,
        "logistic_stack": logit_te,
        "logistic_interaction": interact_te,
        "isotonic_mean": iso_mean,
        "isotonic_product": iso_product,
        "band_conditional": band_score,
    }
    boots = paired_poisson_bootstrap(y_te, methods, reference="XGB_full", seed=BOOT_SEED + 600)
    rows = {}
    for name, score in methods.items():
        rows[name] = {
            **ranking_metrics(y_te, score),
            "auprc_ci95": boots["ci95"][name],
            "delta_vs_xgb": boots["delta_vs_reference"][name],
        }
        score_bundle[f"c6_{name}"] = score
        log(f"C6 {name}: AUPRC={rows[name]['auprc']:.4f}, delta={rows[name]['delta_vs_xgb']['point']:+.4f}")
    score_bundle["c6_y"] = y_te
    return {
        "s2_source": "fit_v0 n8 L2 frozen params",
        "meta_fit_rows": int(len(y_va)),
        "meta_fit_frauds": int(np.sum(y_va)),
        "band_thresholds_from_validation": [float(lo), float(hi)],
        "band_validation_rows_frauds": [int(np.sum(band_va)), int(np.sum(y_va[band_va]))],
        "band_test_rows_frauds": [int(np.sum(band_te)), int(np.sum(y_te[band_te]))],
        "band_status": band_status,
        "rows": rows,
    }


def run_c10(pipe: dict, score_bundle: dict[str, np.ndarray]) -> dict:
    from sklearn.metrics import average_precision_score

    expb = json.loads(Path("runs/hsbc_challenge/fit_v0/exp_b_scorer.json").read_text())
    params = np.asarray(expb["n8_L2"]["S2_trained_mixer_params"], float)
    sim = BatchedSLCU(SLCUStackConfig(n_qubits=8, n_layers=2, hp=pipe["model"].hp))
    ps = ps_chunked(sim, params, pipe["U"]["te"])
    y_te = pipe["ys"]["te"]
    exact = float(average_precision_score(y_te, -ps))
    rows = {}
    for shots in (128, 1024, 8192):
        values = []
        for draw_seed in range(700, 730):
            rng = np.random.default_rng(draw_seed)
            noisy = rng.binomial(shots, np.clip(ps, 0, 1)) / shots
            values.append(float(average_precision_score(y_te, -noisy)))
        a = np.asarray(values)
        rows[str(shots)] = {
            "n_draws": len(a),
            "seeds": [700, 729],
            "auprc_mean": float(np.mean(a)),
            "auprc_sd": float(np.std(a, ddof=1)),
            "auprc_median": float(np.median(a)),
            "auprc_q025_q975": [float(v) for v in np.quantile(a, [0.025, 0.975])],
            "delta_vs_exact_mean": float(np.mean(a - exact)),
            "delta_vs_exact_q025_q975": [float(v) for v in np.quantile(a - exact, [0.025, 0.975])],
            "all_draws": values,
        }
        log(f"C10 shots={shots}: mean={np.mean(a):.4f}, sd={np.std(a, ddof=1):.4f}")
    score_bundle["c10_y"] = y_te
    score_bundle["c10_ps_exact"] = ps
    return {"exact_auprc": exact, "rows": rows}


def verify_time_column(npz_y: np.ndarray) -> dict:
    path = Path(
        "runs/hsbc_challenge/data/openml/openml.org/data/v1/download/1673544/creditcard.arff.gz"
    )
    times, labels, amounts = [], [], []
    in_data = False
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if not in_data:
                if line.strip().lower() == "@data":
                    in_data = True
                continue
            parts = line.rstrip().split(",")
            if len(parts) < 31:
                continue
            times.append(float(parts[0]))
            amounts.append(float(parts[-2]))
            labels.append(int(parts[-1].strip("'")))
    t = np.asarray(times)
    labels = np.asarray(labels, int)
    npz = np.load("runs/hsbc_challenge/data/ulb_creditcard.npz", allow_pickle=True)
    return {
        "source": str(path),
        "rows": int(len(t)),
        "labels_match_npz": bool(np.array_equal(labels, npz_y)),
        "amount_matches_npz": bool(np.allclose(np.asarray(amounts), np.asarray(npz["X"])[:, -1])),
        "time_nondecreasing": bool(np.all(np.diff(t) >= 0)),
        "time_min_max_seconds": [float(t.min()), float(t.max())],
        "time_unique": int(len(np.unique(t))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="runs/hsbc_challenge/audit_v1/priority_attacks_v1.json",
    )
    parser.add_argument(
        "--scores-output",
        default="runs/hsbc_challenge/audit_v1/priority_scores_v1.npz",
    )
    args = parser.parse_args()
    output = Path(args.output)
    scores_output = Path(args.scores_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores_output.parent.mkdir(parents=True, exist_ok=True)

    pipe = original_pipeline()
    result = {
        "schema": "hsbc-audit-priority-v1",
        "protocol_freeze": "docs/hsbc_challenge_audit_report_v1.md section 1",
        "audit_seed": AUDIT_SEED,
        "bootstrap_seed": BOOT_SEED,
        "n_boot": N_BOOT,
        "source_hashes": {
            "data": sha256(Path("runs/hsbc_challenge/data/ulb_creditcard.npz")),
            "exp_b": sha256(Path("runs/hsbc_challenge/fit_v0/exp_b_scorer.json")),
            "handoff": sha256(Path("docs/hsbc_challenge_codex_audit_handoff.md")),
        },
        "split_counts": {
            key: {"rows": int(len(idx)), "frauds": int(np.sum(pipe["y"][idx]))}
            for key, idx in pipe["idx"].items()
        },
        "features": pipe["features"],
    }
    score_bundle: dict[str, np.ndarray] = {}

    result["time_column_verification"] = verify_time_column(pipe["y"])
    result["S1_tau_validation_stability"] = run_tau_stability(pipe)
    result["C5_clean_mode_holdout"] = run_c5(pipe, score_bundle)
    result["C1_mixer_replication"], _ = run_c1(pipe, score_bundle)
    result["C6_calibrated_integration"] = run_c6(pipe, score_bundle)
    result["C10_shot_noise_replication"] = run_c10(pipe, score_bundle)
    result["wall_seconds"] = time.time() - T0

    np.savez_compressed(scores_output, **score_bundle)
    result["scores_artifact"] = str(scores_output)
    result["scores_sha256"] = sha256(scores_output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    log(f"wrote {output}")
    log(f"wrote {scores_output}")


if __name__ == "__main__":
    main()
