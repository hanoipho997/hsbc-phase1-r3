"""Preregistered OCC-parity gate and matched-feature tournament for HSBC v1.

The protocol is frozen in docs/hsbc_challenge_audit_report_v1.md section 1.5.
This script writes only under runs/hsbc_challenge/audit_v1/.

Run after audit_priority_attacks_v1.py so the fixed seed-500 L=1 replication
score is available as the S2(L=1) reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit_priority_attacks_v1 import (  # noqa: E402
    BOOT_SEED,
    paired_poisson_bootstrap,
    probs_matmul,
    ps_chunked,
)
from hsbc_common import (  # noqa: E402
    BatchedSLCU,
    SLCUStackConfig,
    _apply_rx_all,
    hard_bits,
    prepare_hsbc,
    product_probs_batch,
    product_state_batch,
    ranking_metrics,
)


SEED = 20260830
CHUNK = 4096
T0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {message}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def count_tree_nodes(model) -> int:
    if hasattr(model, "estimators_"):
        return int(sum(tree.tree_.node_count for tree in np.ravel(model.estimators_)))
    return 0


def fit_xgb_subset(pipe: dict):
    import xgboost as xgb

    X = pipe["X"]
    feat = pipe["feat"]
    idx_tr, idx_va, idx_te = (pipe["idx"][k] for k in ("tr", "va", "te"))
    y_tr, y_va = pipe["y_split"]["tr"], pipe["y_split"]["va"]
    spw = float(np.sum(y_tr == 0) / np.sum(y_tr == 1))
    clf = xgb.XGBClassifier(
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
        random_state=SEED,
    )
    clf.fit(
        X[np.ix_(idx_tr, feat)],
        y_tr,
        eval_set=[(X[np.ix_(idx_va, feat)], y_va)],
        verbose=False,
    )
    score = clf.predict_proba(X[np.ix_(idx_te, feat)])[:, 1]
    params = int(len(clf.get_booster().trees_to_dataframe()))
    return score, params


def standardize_legit(pipe: dict):
    from sklearn.preprocessing import StandardScaler

    U_tr, U_va, U_te = (pipe["U"][k] for k in ("tr", "va", "te"))
    y_tr, y_va = pipe["y_split"]["tr"], pipe["y_split"]["va"]
    scaler = StandardScaler().fit(U_tr[y_tr == 0])
    return (
        scaler.transform(U_tr),
        scaler.transform(U_va),
        scaler.transform(U_te),
        y_tr,
        y_va,
        pipe["y_split"]["te"],
    )


def empirical_tail_score(train_legit: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values))
    for j in range(values.shape[1]):
        col = np.sort(train_legit[:, j])
        left = np.searchsorted(col, values[:, j], side="left")
        right = np.searchsorted(col, values[:, j], side="right")
        cdf = (0.5 * (left + right) + 0.5) / (len(col) + 1.0)
        tail = np.maximum(2.0 * np.minimum(cdf, 1.0 - cdf), 1.0 / (len(col) + 1.0))
        out += -np.log(tail)
    return out


def fit_deep_svdd_numpy(train_legit: np.ndarray, test: np.ndarray):
    """Bias-free two-layer Deep-SVDD network with a fixed initialized center."""

    rng = np.random.default_rng(SEED)
    d = train_legit.shape[1]
    w1 = rng.normal(scale=np.sqrt(2.0 / d), size=(d, 32))
    w2 = rng.normal(scale=np.sqrt(2.0 / 32), size=(32, 16))

    def forward(values):
        h_pre = values @ w1
        h = np.maximum(h_pre, 0.0)
        z_pre = h @ w2
        z = np.maximum(z_pre, 0.0)
        return h_pre, h, z_pre, z

    center = forward(train_legit[: min(40000, len(train_legit))])[-1].mean(axis=0)
    center[np.abs(center) < 1e-3] = np.where(center[np.abs(center) < 1e-3] < 0, -1e-3, 1e-3)
    m1a = np.zeros_like(w1)
    m1b = np.zeros_like(w2)
    m2a = np.zeros_like(w1)
    m2b = np.zeros_like(w2)
    step_number = 0
    weight_decay = 1e-6
    for _ in range(30):
        order = rng.permutation(len(train_legit))
        for start in range(0, len(order), 1024):
            batch = train_legit[order[start : start + 1024]]
            h_pre, h, z_pre, z = forward(batch)
            dz = 2.0 * (z - center) / len(batch)
            dz_pre = dz * (z_pre > 0)
            gw2 = h.T @ dz_pre + 2.0 * weight_decay * w2
            dh_pre = (dz_pre @ w2.T) * (h_pre > 0)
            gw1 = batch.T @ dh_pre + 2.0 * weight_decay * w1
            step_number += 1
            for grad, first, second, weight in (
                (gw1, m1a, m2a, w1),
                (gw2, m1b, m2b, w2),
            ):
                first *= 0.9
                first += 0.1 * grad
                second *= 0.999
                second += 0.001 * grad * grad
                first_hat = first / (1 - 0.9**step_number)
                second_hat = second / (1 - 0.999**step_number)
                weight -= 1e-3 * first_hat / (np.sqrt(second_hat) + 1e-8)
    score_parts = []
    for start in range(0, len(test), 8192):
        z = forward(test[start : start + 8192])[-1]
        score_parts.append(np.sum((z - center) ** 2, axis=1))
    return np.concatenate(score_parts), {
        "params": int(w1.size + w2.size),
        "train_rows": int(len(train_legit)),
        "epochs": 30,
        "batch_size": 1024,
        "weight_decay": weight_decay,
        "shots_tx": "N/A",
        "note": "bias-free NumPy Deep-SVDD; fixed initialized center; no autoencoder substitution",
    }


def fit_occ_bracket(pipe: dict) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    from sklearn.covariance import EmpiricalCovariance
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import KernelDensity, LocalOutlierFactor
    from sklearn.svm import OneClassSVM

    A_tr, A_va, A_te, y_tr, y_va, _ = standardize_legit(pipe)
    legit_tr = A_tr[y_tr == 0]
    legit_va = A_va[y_va == 0]
    scores: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}

    t = time.time()
    iso = IsolationForest(
        n_estimators=300,
        max_samples=8192,
        contamination="auto",
        random_state=SEED,
        n_jobs=4,
    ).fit(legit_tr)
    scores["IsolationForest"] = -iso.score_samples(A_te)
    meta["IsolationForest"] = {
        "params": count_tree_nodes(iso),
        "train_rows": int(len(legit_tr)),
        "train_seconds": time.time() - t,
        "shots_tx": "N/A",
    }
    log("OCC IsolationForest done")

    rng = np.random.default_rng(SEED)
    cap20 = np.sort(rng.choice(len(legit_tr), min(20000, len(legit_tr)), replace=False))
    t = time.time()
    ocsvm = OneClassSVM(kernel="rbf", nu=0.001727, gamma="scale").fit(legit_tr[cap20])
    scores["OneClassSVM"] = -ocsvm.decision_function(A_te).ravel()
    meta["OneClassSVM"] = {
        "params": int(ocsvm.support_vectors_.size + len(ocsvm.dual_coef_.ravel())),
        "train_rows": int(len(cap20)),
        "support_vectors": int(len(ocsvm.support_)),
        "train_seconds": time.time() - t,
        "shots_tx": "N/A",
    }
    log("OCC OneClassSVM done")

    gmms = []
    t = time.time()
    for components in (1, 4, 8):
        gmm = GaussianMixture(
            n_components=components,
            covariance_type="diag",
            reg_covar=1e-6,
            random_state=SEED,
            max_iter=300,
        ).fit(legit_tr)
        gmms.append((float(gmm.score(legit_va)), gmm))
    _, gmm = max(gmms, key=lambda row: row[0])
    scores["GMM_diag"] = -gmm.score_samples(A_te)
    d, c = A_tr.shape[1], gmm.n_components
    meta["GMM_diag"] = {
        "params": int(c * (2 * d + 1) - 1),
        "components": int(c),
        "train_rows": int(len(legit_tr)),
        "train_seconds": time.time() - t,
        "total_em_iterations_over_candidates": int(
            sum(getattr(m, "n_iter_", 0) for _, m in gmms)
        ),
        "shots_tx": "N/A",
    }
    log(f"OCC GMM done (components={c})")

    cap5 = np.sort(rng.choice(len(legit_tr), min(5000, len(legit_tr)), replace=False))
    kde_rows = []
    t = time.time()
    for bandwidth in (0.2, 0.5, 1.0):
        kde = KernelDensity(
            kernel="gaussian",
            bandwidth=bandwidth,
            algorithm="ball_tree",
            atol=1e-5,
            rtol=1e-5,
        ).fit(legit_tr[cap5])
        kde_rows.append((float(kde.score(legit_va)), kde))
    _, kde = max(kde_rows, key=lambda row: row[0])
    scores["KDE"] = -kde.score_samples(A_te)
    meta["KDE"] = {
        "params": int(len(cap5) * A_tr.shape[1]),
        "bandwidth": float(kde.bandwidth),
        "train_rows": int(len(cap5)),
        "train_seconds": time.time() - t,
        "shots_tx": "N/A",
    }
    log(f"OCC KDE done (bandwidth={kde.bandwidth})")

    cap40 = np.sort(rng.choice(len(legit_tr), min(40000, len(legit_tr)), replace=False))
    t = time.time()
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=4).fit(legit_tr[cap40])
    scores["LOF"] = -lof.decision_function(A_te)
    meta["LOF"] = {
        "params": int(len(cap40) * A_tr.shape[1]),
        "train_rows": int(len(cap40)),
        "train_seconds": time.time() - t,
        "shots_tx": "N/A",
    }
    log("OCC LOF done")

    t = time.time()
    scores["ECOD_like"] = empirical_tail_score(legit_tr, A_te)
    meta["ECOD_like"] = {
        "params": int(legit_tr.size),
        "train_rows": int(len(legit_tr)),
        "train_seconds": time.time() - t,
        "shots_tx": "N/A",
        "note": "independent two-sided empirical tail score; not PyOD ECOD",
    }

    t = time.time()
    pca = PCA(n_components=0.90, svd_solver="full").fit(legit_tr)
    rec = pca.inverse_transform(pca.transform(A_te))
    scores["PCA_error"] = np.mean((A_te - rec) ** 2, axis=1)
    meta["PCA_error"] = {
        "params": int(pca.components_.size + pca.mean_.size),
        "components": int(pca.n_components_),
        "train_rows": int(len(legit_tr)),
        "train_seconds": time.time() - t,
        "shots_tx": "N/A",
    }

    # A robust Mahalanobis positive control is useful but is not counted as one
    # of the handoff's mandatory OCC rows.
    cov = EmpiricalCovariance().fit(legit_tr)
    scores["Mahalanobis_control"] = cov.mahalanobis(A_te)
    meta["Mahalanobis_control"] = {
        "params": int(A_tr.shape[1] ** 2 + A_tr.shape[1]),
        "train_rows": int(len(legit_tr)),
        "train_seconds": 0.0,
        "shots_tx": "N/A",
        "note": "additional preregistered-family diagnostic",
    }

    t = time.time()
    scores["deep_SVDD"], meta["deep_SVDD"] = fit_deep_svdd_numpy(legit_tr, A_te)
    meta["deep_SVDD"]["train_seconds"] = time.time() - t
    log("OCC deep-SVDD done")
    return scores, meta


def fidelity_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    out = np.ones((len(A), len(B)), dtype=np.float32)
    for j in range(A.shape[1]):
        out *= np.cos(0.5 * np.pi * (A[:, j, None] - B[None, :, j])) ** 2
    return out


def fidelity_kernel_scores(U_tr, y_tr, U_te):
    from sklearn.svm import SVC

    rng = np.random.default_rng(SEED)
    legit = np.flatnonzero(y_tr == 0)
    fraud = np.flatnonzero(y_tr == 1)
    chosen = np.sort(
        np.concatenate([rng.choice(legit, min(2000, len(legit)), replace=False), fraud])
    )
    train = U_tr[chosen]
    yy = y_tr[chosen]
    K = fidelity_kernel(train, train)
    svc = SVC(C=1.0, kernel="precomputed", class_weight="balanced").fit(K, yy)
    rows = []
    for start in range(0, len(U_te), 2048):
        rows.append(svc.decision_function(fidelity_kernel(U_te[start : start + 2048], train)))
    score = np.concatenate(rows)
    return score, {
        "params": int(len(svc.support_) * train.shape[1]),
        "train_rows": int(len(chosen)),
        "support_vectors": int(len(svc.support_)),
        "shots_tx": "exact product-state kernel",
    }


def projected_features(U: np.ndarray) -> np.ndarray:
    z = np.cos(np.pi * U)
    pairs = [z[:, i] * z[:, j] for i in range(z.shape[1]) for j in range(i + 1, z.shape[1])]
    return np.column_stack([z, *pairs])


def projected_kernel_scores(U_tr, y_tr, U_te):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    rng = np.random.default_rng(SEED)
    legit = np.flatnonzero(y_tr == 0)
    fraud = np.flatnonzero(y_tr == 1)
    chosen = np.sort(
        np.concatenate([rng.choice(legit, min(2000, len(legit)), replace=False), fraud])
    )
    F_tr = projected_features(U_tr[chosen])
    F_te = projected_features(U_te)
    scaler = StandardScaler().fit(F_tr)
    svc = SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced").fit(
        scaler.transform(F_tr), y_tr[chosen]
    )
    return svc.decision_function(scaler.transform(F_te)), {
        "params": int(len(svc.support_) * F_tr.shape[1]),
        "train_rows": int(len(chosen)),
        "support_vectors": int(len(svc.support_)),
        "projected_features": int(F_tr.shape[1]),
        "shots_tx": "exact projected Z/ZZ kernel",
    }


def _ry_apply(states: np.ndarray, n: int, qubit: int, angle) -> np.ndarray:
    m = len(states)
    angle = np.asarray(angle)
    if angle.ndim == 0:
        c, s = np.cos(angle / 2), np.sin(angle / 2)
    else:
        c, s = np.cos(angle / 2)[:, None, None], np.sin(angle / 2)[:, None, None]
    v = states.reshape(m, -1, 2, 2**qubit)
    v0, v1 = v[:, :, 0, :].copy(), v[:, :, 1, :].copy()
    out = np.empty_like(v)
    out[:, :, 0, :] = c * v0 - s * v1
    out[:, :, 1, :] = s * v0 + c * v1
    return out.reshape(m, -1)


def _ring_cz_phase(n: int) -> np.ndarray:
    idx = np.arange(2**n)
    phase = np.ones(2**n)
    for q in range(n):
        r = (q + 1) % n
        phase[((idx >> q) & 1).astype(bool) & ((idx >> r) & 1).astype(bool)] *= -1
    return phase


def vqc_forward(U: np.ndarray, theta: np.ndarray) -> np.ndarray:
    n = U.shape[1]
    state = product_state_batch(U).astype(float)
    state *= _ring_cz_phase(n)[None, :]
    for q in range(n):
        state = _ry_apply(state, n, q, theta[q])
    for q in range(n):
        state = _ry_apply(state, n, q, np.pi * U[:, q])
    state *= _ring_cz_phase(n)[None, :]
    probs = state**2
    idx = np.arange(2**n)
    p1 = np.column_stack([probs[:, ((idx >> q) & 1).astype(bool)].sum(axis=1) for q in range(n)])
    return np.mean(p1, axis=1)


def train_vqc(U_tr, y_tr, U_te):
    rng = np.random.default_rng(SEED)
    theta = rng.normal(scale=0.1, size=U_tr.shape[1])
    legit = np.flatnonzero(y_tr == 0)
    fraud = np.flatnonzero(y_tr == 1)
    m1, m2 = np.zeros_like(theta), np.zeros_like(theta)
    for iteration in range(200):
        chosen_legit = rng.choice(legit, 384, replace=False)
        batch = np.concatenate([chosen_legit, fraud])
        Ub, yb = U_tr[batch], y_tr[batch]
        p = np.clip(vqc_forward(Ub, theta), 1e-5, 1 - 1e-5)
        weight = np.where(yb == 1, 0.5 / len(fraud), 0.5 / len(chosen_legit))
        dloss_dp = weight * (p - yb) / (p * (1 - p))
        grad = np.zeros_like(theta)
        for j in range(len(theta)):
            tp, tm = theta.copy(), theta.copy()
            tp[j] += np.pi / 2
            tm[j] -= np.pi / 2
            dp = 0.5 * (vqc_forward(Ub, tp) - vqc_forward(Ub, tm))
            grad[j] = np.sum(dloss_dp * dp)
        m1 = 0.9 * m1 + 0.1 * grad
        m2 = 0.999 * m2 + 0.001 * grad * grad
        theta -= 0.05 * (m1 / (1 - 0.9 ** (iteration + 1))) / (
            np.sqrt(m2 / (1 - 0.999 ** (iteration + 1))) + 1e-8
        )
    score = np.concatenate(
        [vqc_forward(U_te[i : i + 2048], theta) for i in range(0, len(U_te), 2048)]
    )
    return score, {
        "params": int(len(theta)),
        "raw_params": theta.tolist(),
        "train_rows_per_step": int(384 + len(fraud)),
        "iterations": 200,
        "shots_tx": "exact dense simulation",
        "note": "faithful shallow RY/CZ data-reuploading circuit; no claim of optimized VQC SOTA",
    }


def train_qae(U_tr, y_tr, U_te):
    rng = np.random.default_rng(SEED + 1)
    n = U_tr.shape[1]
    theta = rng.normal(scale=0.1, size=n)
    legit = np.flatnonzero(y_tr == 0)
    m1, m2 = np.zeros_like(theta), np.zeros_like(theta)

    def trash_score(U, th):
        state = product_state_batch(U).astype(float) * _ring_cz_phase(n)[None, :]
        for q in range(n):
            state = _ry_apply(state, n, q, th[q])
        probs = state**2
        idx = np.arange(2**n)
        trash = range(n // 2, n)
        p1 = [probs[:, ((idx >> q) & 1).astype(bool)].sum(axis=1) for q in trash]
        return np.mean(np.column_stack(p1), axis=1)

    for iteration in range(200):
        batch = rng.choice(legit, 384, replace=False)
        Ub = U_tr[batch]
        grad = np.zeros_like(theta)
        for j in range(n):
            tp, tm = theta.copy(), theta.copy()
            tp[j] += np.pi / 2
            tm[j] -= np.pi / 2
            grad[j] = np.mean(0.5 * (trash_score(Ub, tp) - trash_score(Ub, tm)))
        m1 = 0.9 * m1 + 0.1 * grad
        m2 = 0.999 * m2 + 0.001 * grad * grad
        theta -= 0.05 * (m1 / (1 - 0.9 ** (iteration + 1))) / (
            np.sqrt(m2 / (1 - 0.999 ** (iteration + 1))) + 1e-8
        )
    score = np.concatenate(
        [trash_score(U_te[i : i + 2048], theta) for i in range(0, len(U_te), 2048)]
    )
    return score, {
        "params": int(n),
        "raw_params": theta.tolist(),
        "train_rows_per_step": 384,
        "iterations": 200,
        "shots_tx": "exact dense simulation",
        "note": "faithful shallow trash-qubit QAE objective; deliberately small matched-parameter ansatz",
    }


def shadow_scores(U_tr, y_tr, U_te):
    from sklearn.ensemble import IsolationForest

    rng = np.random.default_rng(SEED)
    n, n_obs = U_tr.shape[1], 64
    observables = []
    for _ in range(n_obs):
        support = np.sort(rng.choice(n, rng.integers(1, min(3, n) + 1), replace=False))
        axes = rng.choice(["X", "Z"], size=len(support))
        observables.append((support, axes))

    def features(U, draw_seed):
        exact = np.ones((len(U), n_obs))
        x, z = np.sin(np.pi * U), np.cos(np.pi * U)
        for k, (support, axes) in enumerate(observables):
            for q, axis in zip(support, axes):
                exact[:, k] *= x[:, q] if axis == "X" else z[:, q]
        rr = np.random.default_rng(draw_seed)
        return 2.0 * rr.binomial(2, np.clip((1 + exact) / 2, 0, 1)) / 2.0 - 1.0

    legit = np.flatnonzero(y_tr == 0)
    cap = np.sort(rng.choice(legit, min(40000, len(legit)), replace=False))
    F_tr = features(U_tr[cap], SEED + 10)
    clf = IsolationForest(
        n_estimators=300,
        max_samples=8192,
        random_state=SEED,
        n_jobs=4,
    ).fit(F_tr)
    score = -clf.score_samples(features(U_te, SEED + 11))
    return score, {
        "params": count_tree_nodes(clf),
        "train_rows": int(len(cap)),
        "observables": n_obs,
        "shots_tx": 128,
        "note": "64 random X/Z Pauli products, two shots each; product encoding is classically simulable",
    }


def tt_svd_reconstruct(tensor: np.ndarray, max_rank: int = 4):
    shape = tensor.shape
    cores = []
    work = tensor.copy()
    r_prev = 1
    for mode in range(len(shape) - 1):
        work = work.reshape(r_prev * shape[mode], -1)
        u, s, vh = np.linalg.svd(work, full_matrices=False)
        rank = min(max_rank, len(s))
        u, s, vh = u[:, :rank], s[:rank], vh[:rank]
        cores.append(u.reshape(r_prev, shape[mode], rank))
        work = s[:, None] * vh
        r_prev = rank
    cores.append(work.reshape(r_prev, shape[-1], 1))
    rec = cores[0]
    for core in cores[1:]:
        rec = np.tensordot(rec, core, axes=([-1], [0]))
    rec = np.squeeze(rec, axis=(0, -1))
    return rec, cores


def mps_density_scores(U_tr, y_tr, U_te):
    n = U_tr.shape[1]
    density = np.zeros(2**n)
    legit = U_tr[y_tr == 0]
    for start in range(0, len(legit), 512):
        density += product_probs_batch(legit[start : start + 512]).sum(axis=0)
    density /= len(legit)
    rec, cores = tt_svd_reconstruct(density.reshape((2,) * n), max_rank=4)
    rec = np.clip(rec.reshape(-1), 1e-12, None)
    rec /= rec.sum()
    energy = -np.log(rec)
    score = np.concatenate(
        [probs_matmul(U_te[i : i + CHUNK], energy[:, None])[:, 0] for i in range(0, len(U_te), CHUNK)]
    )
    return score, {
        "params": int(sum(core.size for core in cores)),
        "train_rows": int(len(legit)),
        "tt_rank": 4,
        "shots_tx": "N/A",
        "note": "legitimate product-density compressed by TT-SVD; quantum-inspired MPS normality model",
    }


def rbm_scores(U_tr, y_tr, U_te):
    from sklearn.neural_network import BernoulliRBM

    bits_tr = hard_bits(U_tr)
    bits_te = hard_bits(U_te)
    rbm = BernoulliRBM(
        n_components=4,
        learning_rate=0.01,
        batch_size=1024,
        n_iter=30,
        random_state=SEED,
        verbose=0,
    ).fit(bits_tr[y_tr == 0])
    score = rbm._free_energy(bits_te)  # lower free energy is more normal
    params = rbm.components_.size + rbm.intercept_hidden_.size + rbm.intercept_visible_.size
    return score, {
        "params": int(params),
        "train_rows": int(np.sum(y_tr == 0)),
        "shots_tx": "N/A",
        "note": "legitimate-fitted Bernoulli RBM on the same median bits",
    }


def lcu_reference_scores(pipe: dict, priority_scores: dict, n: int):
    fit = json.loads(Path("runs/hsbc_challenge/fit_v0/exp_b_scorer.json").read_text())
    row = fit[f"n{n}_L2"]
    tau = float(row["tau_star"])
    s1 = -probs_matmul(pipe["U"]["te"], np.exp(-2 * tau * pipe["model"].hp)[:, None])[:, 0]
    params = np.asarray(row["S2_trained_mixer_params"])
    sim = BatchedSLCU(SLCUStackConfig(n_qubits=n, n_layers=2, hp=pipe["model"].hp))
    s2_l2 = -ps_chunked(sim, params, pipe["U"]["te"])
    softmax = lambda v: np.exp(v - np.max(v)) / np.exp(v - np.max(v)).sum()
    dephased_ps = 1.0
    for layer in range(2):
        a = softmax(params[4 * layer : 4 * layer + 2])
        dephased_ps *= float(np.sum(a**2))
    dephased = np.full(len(s1), -dephased_ps)
    if n == 8:
        s2_l1 = np.asarray(priority_scores["c1_seed_500"])
    else:
        s2_l1 = np.full(len(s1), np.nan)
    return s1, s2_l1, s2_l2, dephased, params


def explicit_four_path_scores(pipe: dict, params: np.ndarray) -> tuple[np.ndarray, float]:
    """Expand the two-layer coherent stack into its four ordered unitary paths."""

    n = len(pipe["feat"])
    hp = pipe["model"].hp
    simulator = BatchedSLCU(SLCUStackConfig(n_qubits=n, n_layers=2, hp=hp))
    out_scores = []
    max_state_deviation = 0.0
    softmax = lambda v: np.exp(v - np.max(v)) / np.exp(v - np.max(v)).sum()
    weights = [softmax(params[4 * layer : 4 * layer + 2]) for layer in range(2)]

    for start in range(0, len(pipe["U"]["te"]), CHUNK):
        inputs = product_state_batch(pipe["U"]["te"][start : start + CHUNK]).astype(complex)
        expanded = np.zeros_like(inputs)
        for branch0 in (0, 1):
            if branch0 == 0:
                state0 = inputs * np.exp(-1j * params[2] * hp)[None, :]
            else:
                state0 = _apply_rx_all(inputs, n, params[3])
            for branch1 in (0, 1):
                if branch1 == 0:
                    state1 = state0 * np.exp(-1j * params[6] * hp)[None, :]
                else:
                    state1 = _apply_rx_all(state0, n, params[7])
                expanded += weights[0][branch0] * weights[1][branch1] * state1
        structured = simulator.forward(params, inputs)
        max_state_deviation = max(
            max_state_deviation, float(np.max(np.abs(expanded - structured)))
        )
        out_scores.append(-np.einsum("md,md->m", expanded.conj(), expanded).real)
    return np.concatenate(out_scores), max_state_deviation


def benchmark_inference_n8(pipe: dict, params: np.ndarray) -> dict:
    """Frozen full-test CPU timing; raw-to-quantile transformation is excluded."""

    U = pipe["U"]["te"]
    sim = BatchedSLCU(SLCUStackConfig(n_qubits=8, n_layers=2, hp=pipe["model"].hp))
    g = np.exp(-8.0 * pipe["model"].hp)[:, None]

    probs_matmul(U, g)
    ps_chunked(sim, params, U)
    rows = {"S1_analytic": [], "S2_L2_structured": []}
    for _ in range(5):
        start = time.perf_counter()
        probs_matmul(U, g)
        rows["S1_analytic"].append(1e6 * (time.perf_counter() - start) / len(U))
        start = time.perf_counter()
        ps_chunked(sim, params, U)
        rows["S2_L2_structured"].append(1e6 * (time.perf_counter() - start) / len(U))
    return {
        "unit": "microseconds_per_transaction",
        "scope": "transformed feature angles through exact score; raw feature transform excluded",
        "test_rows": int(len(U)),
        "repetitions": 5,
        "chunk": CHUNK,
        "machine": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rows": {
            name: {
                "all": values,
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, values in rows.items()
        },
    }


def run_size(n: int, priority_scores: dict, score_bundle: dict) -> dict:
    pipe = prepare_hsbc(n, seed=0)
    y_te = pipe["y_split"]["te"]
    scores, meta = fit_occ_bracket(pipe)

    s1, s2_l1, s2_l2, dephased, params_l2 = lcu_reference_scores(pipe, priority_scores, n)
    scores["S1_analytic"] = s1
    scores["S2_L2_exact"] = s2_l2
    scores["dephased_LCU"] = dephased
    meta["S1_analytic"] = {
        "params": int(n + n * (n - 1) // 2),
        "train_rows": int(np.sum(pipe["y_split"]["tr"] == 0)),
        "shots_tx": "exact closed form",
        "note": "H fit legitimate-only, but feature screen and tau use labels",
    }
    meta["S2_L2_exact"] = {
        "params": int(len(params_l2)),
        "effective_params": int(3 * 2),
        "shots_tx": "exact dense simulation",
        "note": "two softmax logits per layer contain one exact gauge redundancy",
    }
    meta["dephased_LCU"] = {
        "params": int(len(params_l2)),
        "shots_tx": "exact",
        "note": "removing all path cross-terms makes success probability input-independent for unitary branches",
    }
    if n == 8:
        scores["S2_L1_seed500"] = s2_l1
        meta["S2_L1_seed500"] = {
            "params": 4,
            "effective_params": 3,
            "shots_tx": "exact dense simulation",
            "note": "new preregistered seed; no best-seed selection",
        }

    xgb, xgb_params = fit_xgb_subset(pipe)
    scores["XGB_subset"] = xgb
    meta["XGB_subset"] = {
        "params": xgb_params,
        "train_rows": int(len(pipe["idx"]["tr"])),
        "shots_tx": "N/A",
    }

    fk, fk_meta = fidelity_kernel_scores(pipe["U"]["tr"], pipe["y_split"]["tr"], pipe["U"]["te"])
    scores["fidelity_QSVC"] = fk
    meta["fidelity_QSVC"] = fk_meta
    pk, pk_meta = projected_kernel_scores(pipe["U"]["tr"], pipe["y_split"]["tr"], pipe["U"]["te"])
    scores["projected_ZZ_QSVC"] = pk
    meta["projected_ZZ_QSVC"] = pk_meta

    shadow, shadow_meta = shadow_scores(pipe["U"]["tr"], pipe["y_split"]["tr"], pipe["U"]["te"])
    scores["shadow_features_IF"] = shadow
    meta["shadow_features_IF"] = shadow_meta
    mps, mps_meta = mps_density_scores(pipe["U"]["tr"], pipe["y_split"]["tr"], pipe["U"]["te"])
    scores["MPS_TT_density"] = mps
    meta["MPS_TT_density"] = mps_meta
    rbm, rbm_meta = rbm_scores(pipe["U"]["tr"], pipe["y_split"]["tr"], pipe["U"]["te"])
    scores["RBM_legit"] = rbm
    meta["RBM_legit"] = rbm_meta

    if n == 8:
        vqc, vqc_meta = train_vqc(pipe["U"]["tr"], pipe["y_split"]["tr"], pipe["U"]["te"])
        scores["VQC_reupload_8p"] = vqc
        meta["VQC_reupload_8p"] = vqc_meta
        qae, qae_meta = train_qae(pipe["U"]["tr"], pipe["y_split"]["tr"], pipe["U"]["te"])
        scores["QAE_legit_8p"] = qae
        meta["QAE_legit_8p"] = qae_meta
    else:
        meta["VQC_reupload_12p"] = {
            "status": "NOT_RUN",
            "reason": "dense n=12 parameter-shift budget exceeded frozen local audit budget",
        }
        meta["QAE_legit_12p"] = {
            "status": "NOT_RUN",
            "reason": "dense n=12 parameter-shift budget exceeded frozen local audit budget",
        }

    # Materialize every ordered branch sequence, rather than aliasing the
    # structured-simulator array.  This is an independent algebraic referee.
    expanded_score, path_state_deviation = explicit_four_path_scores(pipe, params_l2)
    path_score_deviation = float(np.max(np.abs(expanded_score - s2_l2)))
    scores["exact_2powL_path_expansion"] = expanded_score
    meta["exact_2powL_path_expansion"] = {
        "params": int(len(params_l2)),
        "paths": 4,
        "shots_tx": "exact classical",
        "max_state_difference_vs_structured_S2": path_state_deviation,
        "max_score_difference_vs_S2": path_score_deviation,
    }

    reference = "S2_L1_seed500" if n == 8 else "S2_L2_exact"
    boots = paired_poisson_bootstrap(y_te, scores, reference=reference, seed=BOOT_SEED + n)
    xgb_boot = paired_poisson_bootstrap(y_te, scores, reference="XGB_subset", seed=BOOT_SEED + 100 + n)
    s1_boot = paired_poisson_bootstrap(y_te, scores, reference="S1_analytic", seed=BOOT_SEED + 200 + n)
    rows = {}
    for name, score in scores.items():
        point = ranking_metrics(y_te, score)
        rows[name] = {
            **point,
            "auprc_ci95": boots["ci95"][name],
            "delta_vs_S2_L1_or_L2": boots["delta_vs_reference"][name],
            "delta_vs_XGB_subset": xgb_boot["delta_vs_reference"][name],
            "delta_vs_S1": s1_boot["delta_vs_reference"][name],
            **meta[name],
        }
        score_bundle[f"n{n}_{name}"] = score
        log(f"n={n} {name}: AUPRC={point['auprc']:.4f}")
    score_bundle[f"n{n}_y"] = y_te

    classical_occ_names = [
        "IsolationForest",
        "OneClassSVM",
        "GMM_diag",
        "KDE",
        "LOF",
        "ECOD_like",
        "PCA_error",
        "deep_SVDD",
    ]
    best_occ = max(classical_occ_names, key=lambda name: rows[name]["auprc"])
    gate_delta = rows[best_occ]["auprc"] - rows["S1_analytic"]["auprc"]
    return {
        "features": pipe["feature_names"],
        "test_rows": int(len(y_te)),
        "test_frauds": int(np.sum(y_te)),
        "reference_for_deltas": reference,
        "rows": rows,
        "not_run": {name: value for name, value in meta.items() if value.get("status") == "NOT_RUN"},
        "occ_parity_gate": {
            "best_classical_occ": best_occ,
            "best_occ_auprc": rows[best_occ]["auprc"],
            "S1_auprc": rows["S1_analytic"]["auprc"],
            "point_delta": gate_delta,
            "matches_or_beats_rule": bool(
                gate_delta >= 0
                or abs(gate_delta) <= 0.01
                or (
                    rows[best_occ]["delta_vs_S1"]["ci95"][0] <= 0
                    <= rows[best_occ]["delta_vs_S1"]["ci95"][1]
                )
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--priority-scores",
        default="runs/hsbc_challenge/audit_v1/priority_scores_v1.npz",
    )
    parser.add_argument(
        "--output",
        default="runs/hsbc_challenge/audit_v1/occ_tournament_v1.json",
    )
    parser.add_argument(
        "--scores-output",
        default="runs/hsbc_challenge/audit_v1/occ_tournament_scores_v1.npz",
    )
    parser.add_argument("--sizes", default="8,12")
    args = parser.parse_args()
    priority_path = Path(args.priority_scores)
    if not priority_path.exists():
        raise FileNotFoundError(
            f"{priority_path} is required; run audit_priority_attacks_v1.py first"
        )
    priority = np.load(priority_path)
    output = Path(args.output)
    scores_output = Path(args.scores_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores_output.parent.mkdir(parents=True, exist_ok=True)
    score_bundle: dict[str, np.ndarray] = {}
    result = {
        "schema": "hsbc-occ-tournament-v1",
        "protocol_freeze": "docs/hsbc_challenge_audit_report_v1.md section 1.5",
        "seed": SEED,
        "sizes": {},
    }
    sizes = [int(v) for v in args.sizes.split(",") if v]
    for n in sizes:
        result["sizes"][str(n)] = run_size(n, priority, score_bundle)
    if 8 in sizes:
        pipe8 = prepare_hsbc(8, seed=0)
        fit = json.loads(Path("runs/hsbc_challenge/fit_v0/exp_b_scorer.json").read_text())
        params8 = np.asarray(fit["n8_L2"]["S2_trained_mixer_params"], float)
        result["n8_inference_benchmark"] = benchmark_inference_n8(pipe8, params8)
    np.savez_compressed(scores_output, **score_bundle)
    result["scores_artifact"] = str(scores_output)
    result["scores_sha256"] = sha256(scores_output)
    result["wall_seconds"] = time.time() - T0
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
