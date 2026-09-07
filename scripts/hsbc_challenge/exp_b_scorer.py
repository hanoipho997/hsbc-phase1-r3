"""EXP-B: one-class S-LCU spectral-filter scorer on the ULB fraud dataset.

Ablation ladder (each rung isolates one ingredient):

  S0   hard-bits Ising energy            (classical reference; H's raw info)
  S0.5 soft mean energy E_q[E(z)]        (adds quantile-angle smoothing, linear)
  S1   soft ITE filter p_s = E_q[e^{-2 tau E'}]  (adds nonlinear spectral filter;
       classical closed form by Prop 1)
  S1L  truncated-Cauchy LCHS realization of S1 (adds LCU implementability)
  S2d  trained diagonal-only S-LCU stack (adds training; still Prop-1 classical)
  S2   trained S-LCU stack with RX mixers (adds the non-commuting quantum layer)

Classical baselines: logistic (subset), XGBoost (subset raw / subset bits /
all features), plus the hybrid XGBoost-full + S2-score re-ranking test with a
noise-feature control.

Primary size n=8 (repo-verified machinery); scale check n=12 (chunked, the
batched simulator is verified against DifferentiableStackedLCU at n<=8).
Everything seeded; artifacts -> runs/hsbc_challenge/fit_v0/exp_b_scorer.json
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hsbc_common import (
    BatchedSLCU, SLCUStackConfig, QuantileTransform,
    bootstrap_auprc_ci, bootstrap_delta_auprc, f1_at_best_threshold,
    fit_ising_pseudolikelihood, hard_bits, lchs_cauchy_filter, load_ulb,
    pairwise_loss_and_grad, product_probs_batch, product_state_batch,
    ranking_metrics, save_json, select_features, shot_noise_scores,
    stratified_split, verify_against_repo,
)

SEED = 0
CHUNK = 8192
OUT = {"seed": SEED}
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
X, y, cols = load_ulb()
idx_tr, idx_va, idx_te = stratified_split(y, seed=SEED)
OUT["splits"] = {k: {"rows": int(len(i)), "frauds": int(y[i].sum())}
                 for k, i in (("train", idx_tr), ("val", idx_va), ("test", idx_te))}
log(f"ULB loaded: {X.shape}, splits {OUT['splits']}")


def probs_matmul(U: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Chunked E_{z~q_x}[G(z)] = product_probs(U) @ G without materializing probs."""
    outs = []
    for i in range(0, len(U), CHUNK):
        outs.append(product_probs_batch(U[i:i + CHUNK]) @ G)
    return np.concatenate(outs)


def ps_chunked(sim: BatchedSLCU, params: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Chunked p_s(x) = ||A |phi(x)>||^2 without materializing all states."""
    outs = []
    for i in range(0, len(U), CHUNK):
        v = sim.forward(params, product_state_batch(U[i:i + CHUNK]).astype(complex))
        outs.append(np.einsum("md,md->m", v.conj(), v).real)
    return np.concatenate(outs)


def run_size(n_feat: int, n_layers: int, train_iters: int = 300,
             legit_batch: int = 384, seeds=(0, 1, 2, 3, 4),
             val_sub_legit: int | None = None):
    res = {}
    feat = select_features(X[idx_tr], y[idx_tr], n_feat)
    res["features"] = [cols[j] for j in feat]
    log(f"n={n_feat}: selected features {res['features']}")

    qt = QuantileTransform(X[np.ix_(idx_tr, feat)])
    U_tr, U_va, U_te = (qt(X[np.ix_(i, feat)]) for i in (idx_tr, idx_va, idx_te))
    y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]

    # ---- Ising fit on legit-only training rows -----------------------------
    bits_tr = hard_bits(U_tr)
    model = fit_ising_pseudolikelihood(bits_tr[y_tr == 0], seed=SEED)
    span = float(model.energies.max() - model.energies.min())
    res["ising"] = {
        "coupling_l1": float(np.abs(model.b).sum() / 2),
        "field_l1": float(np.abs(model.a).sum()),
        "significant_couplings_gt_0.01": int((np.abs(model.b[np.triu_indices(n_feat, 1)]) > 0.01).sum()),
        "energy_span": span,
    }
    log(f"n={n_feat}: Ising fitted, {res['ising']}")

    scores = {}  # name -> (fraud scores on val, test)

    # ---- S0 / S0.5 / S1 / S1L via one chunked pass -------------------------
    tau_grid = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]
    from sklearn.metrics import average_precision_score

    G_cols = {"energy": model.energies}
    for t in tau_grid:
        G_cols[f"ite_{t}"] = np.exp(-2.0 * t * model.hp)
    G = np.column_stack(list(G_cols.values()))
    E_va = probs_matmul(U_va, G)
    E_te = probs_matmul(U_te, G)
    col = {k: j for j, k in enumerate(G_cols)}

    scores["S0_hard_energy"] = (model.energy_bits(hard_bits(U_va)),
                                model.energy_bits(hard_bits(U_te)))
    scores["S0.5_soft_energy"] = (E_va[:, col["energy"]], E_te[:, col["energy"]])

    res["S1_tau_grid_val_auprc"] = {
        str(t): float(average_precision_score(y_va, -E_va[:, col[f"ite_{t}"]]))
        for t in tau_grid}
    tau_star = max(tau_grid, key=lambda t: res["S1_tau_grid_val_auprc"][str(t)])
    res["tau_star"] = tau_star
    scores["S1_soft_ite"] = (-E_va[:, col[f"ite_{tau_star}"]],
                             -E_te[:, col[f"ite_{tau_star}"]])
    log(f"n={n_feat}: S1 tau*={tau_star} "
        f"(val AUPRC {res['S1_tau_grid_val_auprc'][str(tau_star)]:.4f})")

    G_l = []
    lchs_specs = ((8, 6.0), (16, 12.0))
    for M, K in lchs_specs:
        F, w1 = lchs_cauchy_filter(model.hp, tau_star, M, K)
        G_l.append((np.abs(F) ** 2) / (w1 ** 2))
    L_va = probs_matmul(U_va, np.column_stack(G_l))
    L_te = probs_matmul(U_te, np.column_stack(G_l))
    for j, (M, K) in enumerate(lchs_specs):
        scores[f"S1L_lchs_M{M}"] = (-L_va[:, j], -L_te[:, j])

    # ---- S2 / S2d: trained stacks ------------------------------------------
    cfg = SLCUStackConfig(n_qubits=n_feat, n_layers=n_layers, hp=model.hp)
    if n_feat <= 8:
        res["repo_verification_max_dev"] = verify_against_repo(cfg, seed=SEED)
    sim = BatchedSLCU(cfg)

    fraud_tr_states = product_state_batch(U_tr[y_tr == 1]).astype(complex)
    legit_tr_idx = np.flatnonzero(y_tr == 0)

    # model-selection subset of validation (all frauds + subsampled legit)
    rng_v = np.random.default_rng(SEED)
    if val_sub_legit is None:
        val_sel = np.arange(len(y_va))
    else:
        legit_va = np.flatnonzero(y_va == 0)
        val_sel = np.sort(np.concatenate(
            [np.flatnonzero(y_va == 1),
             rng_v.choice(legit_va, val_sub_legit, replace=False)]))
    U_va_sel, y_va_sel = U_va[val_sel], y_va[val_sel]

    def train_stack(diagonal_only: bool, seed: int):
        rng = np.random.default_rng(seed)
        params = np.zeros(cfg.n_params)
        for l in range(n_layers):
            params[4 * l: 4 * l + 2] = rng.normal(scale=0.15, size=2)
            params[4 * l + 2] = rng.uniform(0.5, 2.5)
            params[4 * l + 3] = 0.0 if diagonal_only else rng.normal(scale=0.15)
        m1 = np.zeros(cfg.n_params)
        m2 = np.zeros(cfg.n_params)
        best_val = (-1.0, params.copy())
        for it in range(train_iters):
            batch = rng.choice(legit_tr_idx, legit_batch, replace=False)
            legit_states = product_state_batch(U_tr[batch]).astype(complex)
            loss, grad, ps_l, ps_f = pairwise_loss_and_grad(
                sim, params, legit_states, fraud_tr_states)
            if diagonal_only:
                grad[3::4] = 0.0
            m1 = 0.9 * m1 + 0.1 * grad
            m2 = 0.999 * m2 + 0.001 * grad * grad
            step = 0.05 * (m1 / (1 - 0.9 ** (it + 1))) / (
                np.sqrt(m2 / (1 - 0.999 ** (it + 1))) + 1e-8)
            params = params - step
            if diagonal_only:
                params[3::4] = 0.0
            if (it + 1) % 20 == 0 or it == train_iters - 1:
                ap = average_precision_score(
                    y_va_sel, -ps_chunked(sim, params, U_va_sel))
                if ap > best_val[0]:
                    best_val = (ap, params.copy())
        return best_val

    for tag, diag_only in (("S2d_trained_diagonal", True), ("S2_trained_mixer", False)):
        best_overall = (-1.0, None, None)
        per_seed = []
        for s in seeds:
            ap, p = train_stack(diag_only, 1000 + s)
            per_seed.append(float(ap))
            if ap > best_overall[0]:
                best_overall = (ap, p, s)
        params = best_overall[1]
        scores[tag] = (-ps_chunked(sim, params, U_va), -ps_chunked(sim, params, U_te))
        res[f"{tag}_val_auprc_per_seed"] = per_seed
        res[f"{tag}_best_seed"] = best_overall[2]
        res[f"{tag}_params"] = [float(v) for v in params]
        if tag == "S2_trained_mixer":
            st = -scores[tag][1]
            res["S2_test_ps_range"] = [float(st.min()), float(st.max()),
                                       float(np.median(st))]
        log(f"n={n_feat}: {tag} val AUPRC per seed {np.round(per_seed,4).tolist()}")

    # ---- classical baselines ----------------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    Xs_tr, Xs_va, Xs_te = (X[np.ix_(i, feat)] for i in (idx_tr, idx_va, idx_te))
    sc = StandardScaler().fit(Xs_tr)
    lr = LogisticRegression(class_weight="balanced", max_iter=3000)
    lr.fit(sc.transform(Xs_tr), y_tr)
    scores["C_logistic_subset"] = (lr.predict_proba(sc.transform(Xs_va))[:, 1],
                                   lr.predict_proba(sc.transform(Xs_te))[:, 1])

    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    def fit_xgb(A_tr, A_va, A_te, seed=SEED):
        clf = xgb.XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, eval_metric="aucpr", tree_method="hist",
            early_stopping_rounds=50, scale_pos_weight=spw, n_jobs=4,
            random_state=seed,
        )
        clf.fit(A_tr, y_tr, eval_set=[(A_va, y_va)], verbose=False)
        return clf.predict_proba(A_va)[:, 1], clf.predict_proba(A_te)[:, 1]

    scores["C_xgb_subset_raw"] = fit_xgb(Xs_tr, Xs_va, Xs_te)
    scores["C_xgb_subset_bits"] = fit_xgb(bits_tr, hard_bits(U_va), hard_bits(U_te))
    Xf_tr, Xf_va, Xf_te = X[idx_tr], X[idx_va], X[idx_te]
    scores["C_xgb_full"] = fit_xgb(Xf_tr, Xf_va, Xf_te)
    log(f"n={n_feat}: classical baselines fitted")

    # ---- hybrid re-ranking: XGBoost-full + S2 score as extra feature -------
    params = np.array(res["S2_trained_mixer_params"])
    s2_tr = -ps_chunked(sim, params, U_tr)
    H_tr = np.column_stack([Xf_tr, s2_tr])
    H_va = np.column_stack([Xf_va, scores["S2_trained_mixer"][0]])
    H_te = np.column_stack([Xf_te, scores["S2_trained_mixer"][1]])
    scores["HY_xgb_full_plus_S2"] = fit_xgb(H_tr, H_va, H_te)
    rngh = np.random.default_rng(7)
    scores["HY_xgb_full_plus_noise"] = fit_xgb(
        np.column_stack([Xf_tr, rngh.normal(size=len(Xf_tr))]),
        np.column_stack([Xf_va, rngh.normal(size=len(Xf_va))]),
        np.column_stack([Xf_te, rngh.normal(size=len(Xf_te))]))
    log(f"n={n_feat}: hybrid fitted")

    # ---- metrics table -----------------------------------------------------
    table = {}
    for name, (s_va, s_te) in scores.items():
        m = ranking_metrics(y_te, s_te)
        m.update(f1_at_best_threshold(y_va, s_va, y_te, s_te))
        m["auprc_ci"] = bootstrap_auprc_ci(y_te, s_te, seed=SEED)
        table[name] = m
    res["test_metrics"] = table

    deltas = {}
    for a, b in (("S2_trained_mixer", "S2d_trained_diagonal"),
                 ("S2d_trained_diagonal", "S1_soft_ite"),
                 ("S1_soft_ite", "S0.5_soft_energy"),
                 ("S0.5_soft_energy", "S0_hard_energy"),
                 ("S2_trained_mixer", "C_logistic_subset"),
                 ("S2_trained_mixer", "C_xgb_subset_raw"),
                 ("S2_trained_mixer", "C_xgb_subset_bits"),
                 ("HY_xgb_full_plus_S2", "C_xgb_full"),
                 ("HY_xgb_full_plus_noise", "C_xgb_full")):
        if a in scores and b in scores:
            mean, lo, hi = bootstrap_delta_auprc(
                y_te, scores[a][1], scores[b][1], seed=SEED)
            deltas[f"{a} - {b}"] = {"mean": mean, "ci95": [lo, hi]}
    res["paired_delta_auprc"] = deltas

    # ---- shot-noise readout of the S2 score --------------------------------
    ps_te = -scores["S2_trained_mixer"][1]
    rowsn = {}
    for shots in (128, 1024, 8192):
        noisy = shot_noise_scores(ps_te, shots, seed=SEED)
        rowsn[str(shots)] = ranking_metrics(y_te, -noisy)
    rowsn["exact"] = ranking_metrics(y_te, -ps_te)
    res["S2_shot_noise_auprc"] = rowsn

    # ---- per-transaction attribution demo (top-5 scored test frauds) ------
    fraud_rows = np.flatnonzero(y_te == 1)
    order = fraud_rows[np.argsort(scores["S2_trained_mixer"][1][fraud_rows])[::-1][:5]]
    attr = []
    for r in order:
        u = U_te[r:r + 1].copy()

        def fval(uu):
            v = sim.forward(params, product_state_batch(uu).astype(complex))
            return float(np.einsum("md,md->m", v.conj(), v).real[0])

        base = fval(u)
        sal = []
        for j in range(n_feat):
            up = u.copy(); up[0, j] = min(1.0, up[0, j] + 0.02)
            dn = u.copy(); dn[0, j] = max(0.0, dn[0, j] - 0.02)
            sal.append((fval(up) - fval(dn)) / 0.04)
        z = 1.0 - 2.0 * hard_bits(u)[0]
        eterm = -(0.5 * model.b * np.outer(z, z))
        np.fill_diagonal(eterm, 0.0)
        top_pairs = np.dstack(np.unravel_index(
            np.argsort(-np.abs(eterm), axis=None)[:6], eterm.shape))[0]
        attr.append({
            "test_row": int(r),
            "p_s": base,
            "saliency_dps_du": [float(v) for v in sal],
            "top_energy_terms": [
                {"pair": [res["features"][i], res["features"][j]],
                 "contribution": float(eterm[i, j])}
                for i, j in top_pairs if i < j][:3],
        })
    res["attribution_demo"] = attr
    return res


# primary and scale-check runs -----------------------------------------------
OUT["n8_L2"] = run_size(8, 2)
save_json("runs/hsbc_challenge/fit_v0/exp_b_scorer.json", OUT)
OUT["n12_L2"] = run_size(12, 2, train_iters=250, legit_batch=256,
                         seeds=(0, 1, 2), val_sub_legit=8000)
OUT["wall_seconds"] = time.time() - T0
save_json("runs/hsbc_challenge/fit_v0/exp_b_scorer.json", OUT)
log("EXP-B done -> runs/hsbc_challenge/fit_v0/exp_b_scorer.json")

for tag in ("n8_L2", "n12_L2"):
    print(f"\n=== {tag} test metrics (AUPRC [CI] | AUC-ROC | F1) ===")
    for name, m in OUT[tag]["test_metrics"].items():
        print(f"  {name:28s} {m['auprc']:.4f} [{m['auprc_ci'][0]:.3f},{m['auprc_ci'][1]:.3f}]"
              f" | {m['auc_roc']:.4f} | {m['f1']:.3f}")
    print("  paired dAUPRC:")
    for k, v in OUT[tag]["paired_delta_auprc"].items():
        print(f"    {k:55s} {v['mean']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]")
