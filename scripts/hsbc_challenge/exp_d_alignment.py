"""EXP-D: objective-architecture alignment head-to-head (E.ON vs HSBC).

Same ansatz family (L=2 stacked LCU of [e^{-i theta H'}, RX(phi)^n], n=8),
same optimizers (regularized QNG vs Adam), two objectives:

  E.ON side : conditional energy  <psi|H|psi>  of the normalized S-LCU output
              on the n=8 congestion QUBO -- p_s is a NUISANCE (the ratio
              objective is p_s-blind; docs/stacked_lcu_qng_implementation_plan
              hazard).  Variants: plain, and with the success-floor penalty
              mu*max(0, p_min - p_s)^2 (the E.ON mitigation).
  HSBC side : pairwise separation loss on p_s(x) = ||A|phi(x)>||^2 with the
              legit-only fitted Ising H' -- p_s IS the readout (aligned).

Tracked per iteration: p_s (E.ON: of the optimized state; HSBC: batch means
per class), Fubini-Study metric condition number, gradient norm, deliverable
quality (E.ON: E_cond gap + P(x*); HSBC: validation AUPRC), failure events
(LinAlgError, nonfinite step, p_s floor).

The local normalized-state gradient/metric formulas are asserted against the
repo's DifferentiableStackedLCU / fubini_study_metric at iteration 0.
Artifacts -> runs/hsbc_challenge/fit_v0/exp_d_alignment.json
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hsbc_common import (
    BatchedSLCU, SLCUStackConfig, batch_fs_metric, make_eon_qubo,
    pairwise_loss_and_grad, prepare_hsbc, product_state_batch, save_json,
    verify_against_repo,
)
from fqe_ising.variational import regularized_qng_step

SEED = 0
N = 8
L = 2
ITERS = 500
SEEDS = (0, 1, 2, 3, 4)
PS_FLOOR = 1e-4
T0 = time.time()
OUT = {"n": N, "layers": L, "iters": ITERS, "seeds": list(SEEDS)}


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def init_params(rng):
    p = np.empty(4 * L)
    for l in range(L):
        p[4 * l: 4 * l + 2] = rng.normal(scale=0.15, size=2)
        p[4 * l + 2: 4 * l + 4] = rng.uniform(-0.55, 0.55, size=2)
    return p


def normalized_quantities(v, dv):
    """psi, dpsi, p_s, dp_s from unnormalized (v, dv) -- repo formulas."""
    ps = float(np.vdot(v, v).real)
    dps = 2.0 * np.real(dv.conj() @ v)
    norm = np.sqrt(ps)
    ndv = np.real(dv.conj() @ v) / norm
    psi = v / norm
    dpsi = dv / norm - np.outer(ndv, v) / ps
    return psi, dpsi, ps, dps


def fs_metric_from(psi, dpsi):
    overlaps = dpsi.conj() @ dpsi.T
    so = dpsi.conj() @ psi
    g = np.real(overlaps - np.outer(so, so.conj()))
    return 0.5 * (g + g.T)


def metric_cond(g):
    ev = np.linalg.eigvalsh(g)
    lo = max(float(ev.min()), 1e-16)
    return float(ev.max() / lo)


# ---------------------------------------------------------------------------
# E.ON side
# ---------------------------------------------------------------------------
inst = make_eon_qubo(n_cand=N, seed=3)
cfg_e = SLCUStackConfig(n_qubits=N, n_layers=L, hp=inst["norm"])
verify_against_repo(cfg_e, seed=SEED)
sim_e = BatchedSLCU(cfg_e)
PSI0 = np.full(2 ** N, 1.0 / np.sqrt(2 ** N), complex)[None, :]
H_DIAG = inst["norm"]
OUT["eon_instance"] = {"e_opt": inst["e_opt"], "gap": inst["gap"],
                       "idx_opt": inst["idx_opt"]}


def eon_run(method: str, penalty: bool, seed: int):
    rng = np.random.default_rng(seed)
    params = init_params(rng)
    m1 = np.zeros(4 * L)
    m2 = np.zeros(4 * L)
    traj = {"e_cond": [], "ps": [], "cond": [], "gnorm": []}
    events = {"linalg": 0, "nonfinite": 0, "floor_hits": 0}
    best = (np.inf, params.copy())
    mu, p_min = 5.0, 0.1
    for it in range(ITERS):
        v, dv = sim_e.forward_and_derivatives(params, PSI0)
        v, dv = v[0], dv[:, 0, :]
        psi, dpsi, ps, dps = normalized_quantities(v, dv)
        e_cond = float(np.real(np.vdot(psi, H_DIAG * psi)))
        grad = 2.0 * np.real(dpsi.conj() @ (H_DIAG * psi))
        if penalty and ps < p_min:
            grad = grad - 2.0 * mu * (p_min - ps) * dps
        g = fs_metric_from(psi, dpsi)
        if it == 0 and seed == SEEDS[0]:
            _assert_repo_match(cfg_e, params, PSI0[0], H_DIAG, e_cond, grad_raw=2.0 * np.real(dpsi.conj() @ (H_DIAG * psi)), metric=g)
        traj["e_cond"].append(e_cond)
        traj["ps"].append(ps)
        traj["cond"].append(metric_cond(g))
        traj["gnorm"].append(float(np.linalg.norm(grad)))
        if ps < PS_FLOOR:
            events["floor_hits"] += 1
        if e_cond < best[0] and ps > PS_FLOOR:
            best = (e_cond, params.copy())
        try:
            if method == "qng":
                step = regularized_qng_step(g, grad, 0.25, 1e-3)
            else:
                m1 = 0.9 * m1 + 0.1 * grad
                m2 = 0.999 * m2 + 0.001 * grad * grad
                step = 0.08 * (m1 / (1 - 0.9 ** (it + 1))) / (
                    np.sqrt(m2 / (1 - 0.999 ** (it + 1))) + 1e-8)
        except np.linalg.LinAlgError:
            events["linalg"] += 1
            break
        if not np.all(np.isfinite(step)):
            events["nonfinite"] += 1
            break
        params = params - step
    # deliverables at the best accepted iterate
    v = sim_e.forward(best[1], PSI0)[0]
    ps_b = float(np.vdot(v, v).real)
    psi_b = v / np.sqrt(ps_b)
    p_opt = float(np.abs(psi_b[inst["idx_opt"]]) ** 2)
    e_raw = float(np.real(np.vdot(psi_b, H_DIAG * psi_b))) * inst["scale"] + inst["shift"]
    rho = spearmanr(np.arange(len(traj["ps"])), traj["ps"]).statistic
    return {
        "final_ps": traj["ps"][-1], "best_ps": ps_b,
        "ps_spearman_vs_iter": float(rho),
        "median_cond": float(np.median(traj["cond"])),
        "max_cond": float(np.max(traj["cond"])),
        "e_cond_gap_raw": e_raw - inst["e_opt"],
        "P_opt": p_opt, "events": events,
        "iters_run": len(traj["ps"]),
    }


def _assert_repo_match(cfg, params, psi0, h_diag, e_cond, grad_raw, metric):
    from hsbc_common import repo_stack
    from fqe_ising.stacked_lcu import StackedLCUAnsatz
    from fqe_ising.variational import fubini_study_metric
    stack = repo_stack(cfg)
    h_mat = np.diag(h_diag).astype(complex)
    e_ref = stack.conditional_expectation(params, psi0, h_mat)
    g_ref = stack.conditional_expectation_gradient(params, psi0, h_mat)
    ans = StackedLCUAnsatz(stack, psi0)
    m_ref = fubini_study_metric(ans, params)
    assert abs(e_ref - e_cond) < 1e-9, (e_ref, e_cond)
    assert np.allclose(g_ref, grad_raw, atol=1e-9)
    assert np.allclose(m_ref, metric, atol=1e-9)
    log("iteration-0 gradient/metric verified against repo API")


# ---------------------------------------------------------------------------
# HSBC side
# ---------------------------------------------------------------------------
pipe = prepare_hsbc(N, seed=SEED)
cfg_h = SLCUStackConfig(n_qubits=N, n_layers=L, hp=pipe["model"].hp)
sim_h = BatchedSLCU(cfg_h)
U_tr, y_tr = pipe["U"]["tr"], pipe["y_split"]["tr"]
U_va, y_va = pipe["U"]["va"], pipe["y_split"]["va"]
fraud_states = product_state_batch(U_tr[y_tr == 1]).astype(complex)
legit_idx = np.flatnonzero(y_tr == 0)
rng_v = np.random.default_rng(SEED)
val_sel = np.sort(np.concatenate(
    [np.flatnonzero(y_va == 1),
     rng_v.choice(np.flatnonzero(y_va == 0), 8000, replace=False)]))
U_va_sel, y_va_sel = U_va[val_sel], y_va[val_sel]
va_states = product_state_batch(U_va_sel).astype(complex)


def hsbc_run(method: str, seed: int):
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    params = init_params(rng)
    m1 = np.zeros(4 * L)
    m2 = np.zeros(4 * L)
    traj = {"loss": [], "ps_legit": [], "ps_fraud": [], "cond": [], "gnorm": []}
    events = {"linalg": 0, "nonfinite": 0, "floor_hits": 0}
    best = (-1.0, params.copy())
    for it in range(ITERS):
        batch = rng.choice(legit_idx, 384, replace=False)
        legit_states = product_state_batch(U_tr[batch]).astype(complex)
        loss, grad, ps_l, ps_f = pairwise_loss_and_grad(
            sim_h, params, legit_states, fraud_states)
        g = batch_fs_metric(
            sim_h, params, np.concatenate([legit_states[:128], fraud_states]))
        traj["loss"].append(loss)
        traj["ps_legit"].append(ps_l)
        traj["ps_fraud"].append(ps_f)
        traj["cond"].append(metric_cond(g))
        traj["gnorm"].append(float(np.linalg.norm(grad)))
        if ps_l < PS_FLOOR:
            events["floor_hits"] += 1
        try:
            if method == "qng":
                step = regularized_qng_step(g, grad, 0.25, 1e-3)
            else:
                m1 = 0.9 * m1 + 0.1 * grad
                m2 = 0.999 * m2 + 0.001 * grad * grad
                step = 0.05 * (m1 / (1 - 0.9 ** (it + 1))) / (
                    np.sqrt(m2 / (1 - 0.999 ** (it + 1))) + 1e-8)
        except np.linalg.LinAlgError:
            events["linalg"] += 1
            break
        if not np.all(np.isfinite(step)):
            events["nonfinite"] += 1
            break
        params = params - step
        if (it + 1) % 25 == 0 or it == ITERS - 1:
            v = sim_h.forward(params, va_states)
            s = np.einsum("md,md->m", v.conj(), v).real
            ap = average_precision_score(y_va_sel, -s)
            if ap > best[0]:
                best = (ap, params.copy())
    rho = spearmanr(np.arange(len(traj["ps_legit"])), traj["ps_legit"]).statistic
    return {
        "final_ps_legit": traj["ps_legit"][-1],
        "final_ps_fraud": traj["ps_fraud"][-1],
        "ps_legit_spearman_vs_iter": float(rho),
        "median_cond": float(np.median(traj["cond"])),
        "max_cond": float(np.max(traj["cond"])),
        "best_val_auprc": best[0], "events": events,
        "iters_run": len(traj["ps_legit"]),
    }


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
def agg(rows, keys):
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "per_seed": [float(r[k]) for r in rows]} for k in keys}


for method in ("qng", "adam"):
    for penalty in (False, True):
        rows = [eon_run(method, penalty, 100 + s) for s in SEEDS]
        tag = f"eon_{method}{'_penalty' if penalty else ''}"
        OUT[tag] = agg(rows, ("final_ps", "best_ps", "ps_spearman_vs_iter",
                              "median_cond", "max_cond", "e_cond_gap_raw",
                              "P_opt"))
        OUT[tag]["events"] = {k: int(sum(r["events"][k] for r in rows))
                              for k in rows[0]["events"]}
        log(f"{tag}: ps={OUT[tag]['final_ps']['per_seed']}, "
            f"cond_med={OUT[tag]['median_cond']['mean']:.2e}, "
            f"events={OUT[tag]['events']}")

for method in ("qng", "adam"):
    rows = [hsbc_run(method, 100 + s) for s in SEEDS]
    tag = f"hsbc_{method}"
    OUT[tag] = agg(rows, ("final_ps_legit", "final_ps_fraud",
                          "ps_legit_spearman_vs_iter", "median_cond",
                          "max_cond", "best_val_auprc"))
    OUT[tag]["events"] = {k: int(sum(r["events"][k] for r in rows))
                          for k in rows[0]["events"]}
    log(f"{tag}: ps_legit={np.round(OUT[tag]['final_ps_legit']['per_seed'],3).tolist()}, "
        f"cond_med={OUT[tag]['median_cond']['mean']:.2e}, "
        f"val_auprc={np.round(OUT[tag]['best_val_auprc']['per_seed'],4).tolist()}, "
        f"events={OUT[tag]['events']}")

OUT["wall_seconds"] = time.time() - T0
save_json("runs/hsbc_challenge/fit_v0/exp_d_alignment.json", OUT)
log("EXP-D done -> runs/hsbc_challenge/fit_v0/exp_d_alignment.json")
