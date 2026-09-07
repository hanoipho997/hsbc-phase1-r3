"""Preregistered HSBC method improvements and ULB regime hunt.

Protocols are frozen in docs/hsbc_challenge_audit_report_v1.md sections
1.6 and 1.8.  Outputs are new audit artifacts; fit_v0 is read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit_priority_attacks_v1 import (  # noqa: E402
    BOOT_SEED,
    fit_xgb,
    paired_poisson_bootstrap,
    probs_matmul,
    ps_chunked,
)
from hsbc_common import (  # noqa: E402
    BatchedSLCU,
    QuantileTransform,
    SLCUStackConfig,
    batch_fs_metric,
    fit_ising_pseudolikelihood,
    hard_bits,
    pairwise_loss_and_grad,
    prepare_hsbc,
    product_state_batch,
    ranking_metrics,
    select_features,
)


SEED = 20260830
T0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {message}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def s1_score(pipe: dict, U: np.ndarray, tau: float) -> np.ndarray:
    return -probs_matmul(U, np.exp(-2.0 * tau * pipe["model"].hp)[:, None])[:, 0]


def run_cross_view(pipe: dict, score_bundle: dict) -> dict:
    from sklearn.cross_decomposition import CCA

    U_tr, U_va, U_te = (pipe["U"][k] for k in ("tr", "va", "te"))
    y_tr, y_te = pipe["y_split"]["tr"], pipe["y_split"]["te"]
    view_a, view_b = np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7])
    cca = CCA(n_components=4, scale=True, max_iter=2000)
    cca.fit(U_tr[y_tr == 0][:, view_a], U_tr[y_tr == 0][:, view_b])
    A_legit, B_legit = cca.transform(
        U_tr[y_tr == 0][:, view_a], U_tr[y_tr == 0][:, view_b]
    )
    qa, qb = QuantileTransform(A_legit), QuantileTransform(B_legit)

    def score(U):
        A, B = cca.transform(U[:, view_a], U[:, view_b])
        ua, ub = qa(A), qb(B)
        overlap = np.prod(np.cos(0.5 * np.pi * (ua - ub)), axis=1)
        p_success = 0.5 * (1.0 + overlap)
        return 1.0 - p_success

    cross_te = score(U_te)
    ref = s1_score(pipe, U_te, tau=4.0)
    boots = paired_poisson_bootstrap(
        y_te,
        {"cross_view_CCA_LCU": cross_te, "S1": ref},
        reference="S1",
        seed=BOOT_SEED + 1000,
    )
    score_bundle["cross_view"] = cross_te
    score_bundle["cross_view_y"] = y_te
    return {
        "view_A": [pipe["feature_names"][j] for j in view_a],
        "view_B": [pipe["feature_names"][j] for j in view_b],
        "fit_rows": int(np.sum(y_tr == 0)),
        "fraud_labels_used": 0,
        "params": int(cca.x_weights_.size + cca.y_weights_.size),
        "metrics": ranking_metrics(y_te, cross_te),
        "auprc_ci95": boots["ci95"]["cross_view_CCA_LCU"],
        "delta_vs_S1": boots["delta_vs_reference"]["cross_view_CCA_LCU"],
        "scope": "first circuit-realizable equal-weight branch-overlap implementation; not learned branch unitaries",
    }


def flag_fisher(sim, params, legit_states, fraud_states):
    vl, dl = sim.forward_and_derivatives(params, legit_states)
    vf, df = sim.forward_and_derivatives(params, fraud_states)
    sl = np.einsum("md,md->m", vl.conj(), vl).real
    sf = np.einsum("md,md->m", vf.conj(), vf).real
    dsl = 2.0 * np.einsum("pmd,md->pm", dl.conj(), vl).real
    dsf = 2.0 * np.einsum("pmd,md->pm", df.conj(), vf).real
    p = np.concatenate([sl, sf])
    jac = np.concatenate([dsl, dsf], axis=1)
    weights = np.concatenate(
        [np.full(len(sl), 0.5 / len(sl)), np.full(len(sf), 0.5 / len(sf))]
    )
    denom = np.clip(p, 1e-4, 1 - 1e-4) * np.clip(1 - p, 1e-4, 1 - 1e-4)
    fisher = (jac * (weights / denom)[None, :]) @ jac.T
    return 0.5 * (fisher + fisher.T), p


def train_flag_method(pipe: dict, method: str, train_seed: int):
    U_tr, U_te = pipe["U"]["tr"], pipe["U"]["te"]
    y_tr, y_te = pipe["y_split"]["tr"], pipe["y_split"]["te"]
    sim = BatchedSLCU(SLCUStackConfig(n_qubits=8, n_layers=1, hp=pipe["model"].hp))
    fraud_states = product_state_batch(U_tr[y_tr == 1]).astype(complex)
    legit_idx = np.flatnonzero(y_tr == 0)
    rng = np.random.default_rng(train_seed)
    params = np.array(
        [*rng.normal(scale=0.15, size=2), rng.uniform(0.5, 2.5), rng.normal(scale=0.15)]
    )
    m1, m2 = np.zeros(4), np.zeros(4)
    diagnostics = {"max_step": 0.0, "fisher_condition_median_source": [], "clip_fraction": []}
    for iteration in range(300):
        chosen = rng.choice(legit_idx, 384, replace=False)
        legit_states = product_state_batch(U_tr[chosen]).astype(complex)
        _, grad, _, _ = pairwise_loss_and_grad(sim, params, legit_states, fraud_states)
        if method == "flag_ng":
            fisher, p = flag_fisher(sim, params, legit_states, fraud_states)
            eig = np.linalg.eigvalsh(fisher)
            pos = eig[eig > 1e-12]
            diagnostics["fisher_condition_median_source"].append(
                float(pos.max() / pos.min()) if len(pos) else float("inf")
            )
            diagnostics["clip_fraction"].append(float(np.mean((p < 1e-4) | (p > 1 - 1e-4))))
            step = 0.05 * np.linalg.solve(fisher + 1e-2 * np.eye(4), grad)
            norm = np.linalg.norm(step)
            if norm > 0.1:
                step *= 0.1 / norm
        else:
            m1 = 0.9 * m1 + 0.1 * grad
            m2 = 0.999 * m2 + 0.001 * grad * grad
            step = 0.05 * (m1 / (1 - 0.9 ** (iteration + 1))) / (
                np.sqrt(m2 / (1 - 0.999 ** (iteration + 1))) + 1e-8
            )
        diagnostics["max_step"] = max(diagnostics["max_step"], float(np.linalg.norm(step)))
        params -= step
    score = -ps_chunked(sim, params, U_te)
    return score, params, {
        "test_metrics": ranking_metrics(y_te, score),
        "max_step": diagnostics["max_step"],
        "median_positive_fisher_condition": (
            float(np.median(diagnostics["fisher_condition_median_source"]))
            if diagnostics["fisher_condition_median_source"]
            else None
        ),
        "mean_clip_fraction": (
            float(np.mean(diagnostics["clip_fraction"])) if diagnostics["clip_fraction"] else None
        ),
    }


def run_flag_ng(pipe: dict, score_bundle: dict) -> dict:
    rows = {"Adam": [], "flag_NG": []}
    scores = {"Adam": [], "flag_NG": []}
    for seed in range(610, 615):
        for label, method in (("Adam", "adam"), ("flag_NG", "flag_ng")):
            score, params, diag = train_flag_method(pipe, method, seed)
            rows[label].append({"seed": seed, "params": params.tolist(), **diag})
            scores[label].append(score)
            score_bundle[f"{label}_seed{seed}"] = score
            log(f"improvement {label} seed={seed}: AUPRC={diag['test_metrics']['auprc']:.4f}")
    y_te = pipe["y_split"]["te"]
    mean_score = {name: np.mean(np.vstack(value), axis=0) for name, value in scores.items()}
    boots = paired_poisson_bootstrap(
        y_te,
        mean_score,
        reference="Adam",
        seed=BOOT_SEED + 1100,
    )
    score_bundle["flag_methods_y"] = y_te
    return {
        "seeds": list(range(610, 615)),
        "per_seed": rows,
        "mean_per_seed_auprc": {
            name: float(np.mean([row["test_metrics"]["auprc"] for row in values]))
            for name, values in rows.items()
        },
        "ensemble_score_metrics": {
            name: ranking_metrics(y_te, value) for name, value in mean_score.items()
        },
        "flag_NG_minus_Adam_ensemble": boots["delta_vs_reference"]["flag_NG"],
        "note": "ensemble is a secondary summary; all five unselected seed rows are primary",
    }


def run_metric_redundancy(pipe: dict) -> dict:
    rng = np.random.default_rng(SEED)
    cfg = SLCUStackConfig(n_qubits=8, n_layers=2, hp=pipe["model"].hp)
    sim = BatchedSLCU(cfg)
    params = rng.normal(scale=0.25, size=8)
    states = product_state_batch(pipe["U"]["tr"][:128]).astype(complex)
    _, derivatives = sim.forward_and_derivatives(params, states)
    gauge_residuals = []
    for layer in range(2):
        gauge_residuals.append(float(np.max(np.abs(derivatives[4 * layer] + derivatives[4 * layer + 1]))))
    g = batch_fs_metric(sim, params, states)
    eig = np.linalg.eigvalsh(g)
    J = np.zeros((8, 6))
    for layer in range(2):
        J[4 * layer, 3 * layer] = 0.5
        J[4 * layer + 1, 3 * layer] = -0.5
        J[4 * layer + 2, 3 * layer + 1] = 1.0
        J[4 * layer + 3, 3 * layer + 2] = 1.0
    reduced = J.T @ g @ J
    reig = np.linalg.eigvalsh(reduced)
    pos_full = eig[eig > 1e-12]
    pos_reduced = reig[reig > 1e-12]
    return {
        "raw_parameter_count": 8,
        "identifiable_parameter_count": 6,
        "per_layer_logit_gauge_derivative_residual": gauge_residuals,
        "full_metric_eigenvalues": eig.tolist(),
        "reduced_metric_eigenvalues": reig.tolist(),
        "full_positive_subspace_condition": float(pos_full.max() / pos_full.min()),
        "reduced_condition": float(pos_reduced.max() / pos_reduced.min()),
        "finding": "the published floor-clipped condition number includes two exact softmax gauge nulls",
    }


def fit_xgb_given_rows(X_tr, y_tr, X_va, y_va, X_te, seed):
    model = fit_xgb(X_tr, y_tr, X_va, y_va, seed=seed)
    return model.predict_proba(X_te)[:, 1]


def run_label_scarcity(pipe: dict) -> dict:
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler

    X, y = pipe["X"], pipe["y"]
    idx_tr, idx_va, idx_te = (pipe["idx"][k] for k in ("tr", "va", "te"))
    y_tr, y_va, y_te = (pipe["y_split"][k] for k in ("tr", "va", "te"))
    tr_legit, tr_fraud = np.flatnonzero(y_tr == 0), np.flatnonzero(y_tr == 1)
    va_legit, va_fraud = np.flatnonzero(y_va == 0), np.flatnonzero(y_va == 1)
    result = {}
    for budget in (5, 20, 50, 200, "all"):
        rows = []
        for replicate in range(5):
            if budget == "all":
                chosen_tr, chosen_va = tr_fraud, va_fraud
            else:
                n_tr = min(len(tr_fraud), max(1, int(round(0.8 * budget))))
                n_va = min(len(va_fraud), max(1, int(budget) - n_tr))
                rng = np.random.default_rng(SEED + 100 * int(budget) + replicate)
                chosen_tr = np.sort(rng.choice(tr_fraud, n_tr, replace=False))
                chosen_va = np.sort(rng.choice(va_fraud, n_va, replace=False))
            use_tr = np.sort(np.concatenate([tr_legit, chosen_tr]))
            use_va = np.sort(np.concatenate([va_legit, chosen_va]))

            # Every representation choice obeys the same fraud-label budget.
            # The first audit implementation held the all-label S1 screen/tau
            # fixed; that curve is explicitly excluded by report section 1.10.
            feat_budget = select_features(X[idx_tr[use_tr]], y_tr[use_tr], 8)
            qt = QuantileTransform(X[np.ix_(idx_tr[use_tr], feat_budget)])
            U_legit = qt(X[np.ix_(idx_tr[tr_legit], feat_budget)])
            U_va_budget = qt(X[np.ix_(idx_va[use_va], feat_budget)])
            U_te_budget = qt(X[np.ix_(idx_te, feat_budget)])
            ising = fit_ising_pseudolikelihood(
                hard_bits(U_legit), seed=SEED + replicate
            )
            tau_grid = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
            val_scores = {
                tau: -probs_matmul(
                    U_va_budget,
                    np.exp(-2.0 * tau * ising.hp)[:, None],
                )[:, 0]
                for tau in tau_grid
            }
            tau = max(
                tau_grid,
                key=lambda value: average_precision_score(
                    y_va[use_va], val_scores[value]
                ),
            )
            s1 = -probs_matmul(
                U_te_budget,
                np.exp(-2.0 * tau * ising.hp)[:, None],
            )[:, 0]
            scaler = StandardScaler().fit(U_legit)
            iso = IsolationForest(
                n_estimators=300,
                max_samples=8192,
                random_state=SEED + replicate,
                n_jobs=4,
            ).fit(scaler.transform(U_legit))
            occ = -iso.score_samples(scaler.transform(U_te_budget))

            sub = fit_xgb_given_rows(
                X[np.ix_(idx_tr[use_tr], feat_budget)],
                y_tr[use_tr],
                X[np.ix_(idx_va[use_va], feat_budget)],
                y_va[use_va],
                X[np.ix_(idx_te, feat_budget)],
                SEED + replicate,
            )
            full = fit_xgb_given_rows(
                X[idx_tr[use_tr]],
                y_tr[use_tr],
                X[idx_va[use_va]],
                y_va[use_va],
                X[idx_te],
                SEED + replicate,
            )
            rows.append(
                {
                    "replicate": replicate,
                    "train_fraud_labels": int(len(chosen_tr)),
                    "val_fraud_labels": int(len(chosen_va)),
                    "features": [pipe["cols"][j] for j in feat_budget],
                    "tau": float(tau),
                    "S1_budgeted": ranking_metrics(y_te, s1),
                    "IsolationForest_budgeted": ranking_metrics(y_te, occ),
                    "XGB_subset": ranking_metrics(y_te, sub),
                    "XGB_full": ranking_metrics(y_te, full),
                }
            )
        result[str(budget)] = {
            "rows": rows,
            "mean_auprc": {
                name: float(np.mean([row[name]["auprc"] for row in rows]))
                for name in (
                    "S1_budgeted",
                    "IsolationForest_budgeted",
                    "XGB_subset",
                    "XGB_full",
                )
            },
            "range_auprc": {
                name: [
                    float(np.min([row[name]["auprc"] for row in rows])),
                    float(np.max([row[name]["auprc"] for row in rows])),
                ]
                for name in (
                    "S1_budgeted",
                    "IsolationForest_budgeted",
                    "XGB_subset",
                    "XGB_full",
                )
            },
        }
        log(f"label scarcity {budget}: {result[str(budget)]['mean_auprc']}")
    return result


def run_prevalence(tournament_scores: dict) -> dict:
    from sklearn.metrics import average_precision_score

    y = np.asarray(tournament_scores["n8_y"])
    methods = {
        "S1": np.asarray(tournament_scores["n8_S1_analytic"]),
        "IsolationForest": np.asarray(tournament_scores["n8_IsolationForest"]),
        "XGB_subset": np.asarray(tournament_scores["n8_XGB_subset"]),
    }
    n0, n1 = np.sum(y == 0), np.sum(y == 1)
    rows = {}
    for percent in (0.02, 0.05, 0.10, 0.1727, 0.50):
        prevalence = percent / 100.0
        fraud_weight = prevalence * n0 / ((1 - prevalence) * n1)
        weights = np.where(y == 1, fraud_weight, 1.0)
        rows[str(percent)] = {
            name: float(average_precision_score(y, score, sample_weight=weights))
            for name, score in methods.items()
        }
        rows[str(percent)]["effective_prevalence"] = float(
            np.sum(weights[y == 1]) / np.sum(weights)
        )
    return {
        "protocol": "exact class weighting on every test row; no row subsampling",
        "rows": rows,
    }


def run_feature_poverty() -> dict:
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler

    result = {}
    for n in (2, 4, 6, 8, 12):
        pipe = prepare_hsbc(n, seed=0)
        y_tr, y_va, y_te = (pipe["y_split"][k] for k in ("tr", "va", "te"))
        tau_grid = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)
        val_scores = {tau: s1_score(pipe, pipe["U"]["va"], tau) for tau in tau_grid}
        tau = max(tau_grid, key=lambda t: average_precision_score(y_va, val_scores[t]))
        s1 = s1_score(pipe, pipe["U"]["te"], tau)

        scaler = StandardScaler().fit(pipe["U"]["tr"][y_tr == 0])
        iso = IsolationForest(
            n_estimators=300,
            max_samples=8192,
            random_state=SEED,
            n_jobs=4,
        ).fit(scaler.transform(pipe["U"]["tr"][y_tr == 0]))
        occ = -iso.score_samples(scaler.transform(pipe["U"]["te"]))

        X, feat = pipe["X"], pipe["feat"]
        idx_tr, idx_va, idx_te = (pipe["idx"][k] for k in ("tr", "va", "te"))
        xgb = fit_xgb_given_rows(
            X[np.ix_(idx_tr, feat)],
            y_tr,
            X[np.ix_(idx_va, feat)],
            y_va,
            X[np.ix_(idx_te, feat)],
            SEED + n,
        )
        result[str(n)] = {
            "features": pipe["feature_names"],
            "tau": tau,
            "S1": ranking_metrics(y_te, s1),
            "IsolationForest": ranking_metrics(y_te, occ),
            "XGB_subset": ranking_metrics(y_te, xgb),
        }
        log(f"feature poverty n={n}: " + json.dumps({k: v['auprc'] for k, v in result[str(n)].items() if isinstance(v, dict)}))
    return result


def run_bounded_shift(pipe: dict) -> dict:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    U_tr, U_va, U_te = (pipe["U"][k] for k in ("tr", "va", "te"))
    y_tr, y_va, y_te = (pipe["y_split"][k] for k in ("tr", "va", "te"))
    scaler = StandardScaler().fit(U_tr[y_tr == 0])
    iso = IsolationForest(
        n_estimators=300,
        max_samples=8192,
        random_state=SEED,
        n_jobs=4,
    ).fit(scaler.transform(U_tr[y_tr == 0]))
    xgb = fit_xgb(U_tr, y_tr, U_va, y_va, seed=SEED)
    rows = {}
    for epsilon in (0.0, 0.05, 0.10, 0.20):
        shifted = U_te.copy()
        shifted[y_te == 1] = (1 - epsilon) * shifted[y_te == 1] + 0.5 * epsilon
        methods = {
            "S1": s1_score(pipe, shifted, 4.0),
            "IsolationForest": -iso.score_samples(scaler.transform(shifted)),
            "XGB_quantiles": xgb.predict_proba(shifted)[:, 1],
        }
        rows[str(epsilon)] = {name: ranking_metrics(y_te, value) for name, value in methods.items()}
    return {
        "definition": "fraud u'=(1-epsilon)u+0.5epsilon; legitimate rows unchanged",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tournament-scores",
        default="runs/hsbc_challenge/audit_v1/occ_tournament_scores_v1.npz",
    )
    parser.add_argument(
        "--output",
        default="runs/hsbc_challenge/audit_v1/improvements_regimes_v1.json",
    )
    parser.add_argument(
        "--scores-output",
        default="runs/hsbc_challenge/audit_v1/improvement_scores_v1.npz",
    )
    parser.add_argument(
        "--skip-label-scarcity",
        action="store_true",
        help="diagnostic-only option; a final audit run must omit this flag",
    )
    args = parser.parse_args()
    tournament_path = Path(args.tournament_scores)
    if not tournament_path.exists():
        raise FileNotFoundError(f"run audit_occ_tournament_v1.py first: {tournament_path}")
    tournament = np.load(tournament_path)
    pipe = prepare_hsbc(8, seed=0)
    score_bundle: dict[str, np.ndarray] = {}
    result = {
        "schema": "hsbc-improvements-regimes-v1",
        "protocol_freeze": "docs/hsbc_challenge_audit_report_v1.md sections 1.6 and 1.8",
        "cross_view": run_cross_view(pipe, score_bundle),
        "flag_NG_vs_Adam": run_flag_ng(pipe, score_bundle),
        "metric_gauge_diagnostic": run_metric_redundancy(pipe),
        "prevalence_stress": run_prevalence(tournament),
        "feature_poverty": run_feature_poverty(),
        "bounded_shift": run_bounded_shift(pipe),
    }
    if args.skip_label_scarcity:
        result["label_scarcity"] = {"status": "SKIPPED_NONFINAL_DIAGNOSTIC"}
    else:
        result["label_scarcity"] = run_label_scarcity(pipe)
    output = Path(args.output)
    scores_output = Path(args.scores_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scores_output, **score_bundle)
    result["scores_artifact"] = str(scores_output)
    result["scores_sha256"] = sha256(scores_output)
    result["wall_seconds"] = time.time() - T0
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    log(f"wrote {output}")


if __name__ == "__main__":
    main()
