"""EXP-E: deliverable-vs-depth asymmetry + circuit resources (E.ON vs HSBC).

Same filter family on both problems, two figures of merit:

  E.ON (mode concentration): P(x*) of the postselected state -- the deliverable
      is the argmax, which fights the density of states near the optimum.
      Exact-ITE tau sweep + trained stacks at L = 1..4.
  HSBC (ranking): test AUPRC of the p_s score -- the deliverable is score
      separation.  Exact-ITE tau sweep + trained stacks at L = 1..3.
      Sanity: with HARD encoding the ITE score is a monotone transform of the
      energy, so its ranking metrics are exactly tau-independent.

Plus transpiled circuit resources (cz/rz/sx/x, O3) for both n=8 Hamiltonians
at L=2, following the eon_slcu_feasibility.py synthesis (c-RZZ = CX.CRZ.CX).

Artifacts -> runs/hsbc_challenge/fit_v0/exp_e_depth.json
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hsbc_common import (
    BatchedSLCU, SLCUStackConfig, make_eon_qubo, pairwise_loss_and_grad,
    prepare_hsbc, product_probs_batch, product_state_batch, save_json,
)

SEED = 0
T0 = time.time()
OUT = {}
CHUNK = 8192


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def probs_matmul(U, G):
    return np.concatenate([product_probs_batch(U[i:i + CHUNK]) @ G
                           for i in range(0, len(U), CHUNK)])


def ps_chunked(sim, params, U):
    outs = []
    for i in range(0, len(U), CHUNK):
        v = sim.forward(params, product_state_batch(U[i:i + CHUNK]).astype(complex))
        outs.append(np.einsum("md,md->m", v.conj(), v).real)
    return np.concatenate(outs)


# ---------------------------------------------------------------------------
# E.ON side: concentration needs depth
# ---------------------------------------------------------------------------
inst = make_eon_qubo(n_cand=8, seed=3)
D = 2 ** 8
uni = np.full(D, 1.0 / D)
rows_tau = {}
for tau in (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0):
    gg = np.exp(-2.0 * tau * inst["hp"])
    p_s = float(gg.mean())
    cond = gg / gg.sum()
    rows_tau[str(tau)] = {"P_opt": float(cond[inst["idx_opt"]]), "p_s": p_s}
OUT["eon_ite_tau_sweep"] = rows_tau
log("E.ON ITE sweep: " + ", ".join(
    f"tau={t}: P*={v['P_opt']:.3f}/ps={v['p_s']:.3f}" for t, v in rows_tau.items()))

PSI0 = np.full(D, 1.0 / np.sqrt(D), complex)[None, :]
H_DIAG = inst["norm"]


def train_eon(L, seed, iters=400):
    cfg = SLCUStackConfig(n_qubits=8, n_layers=L, hp=inst["norm"])
    sim = BatchedSLCU(cfg)
    rng = np.random.default_rng(seed)
    params = np.empty(4 * L)
    for l in range(L):
        params[4 * l: 4 * l + 2] = rng.normal(scale=0.15, size=2)
        params[4 * l + 2: 4 * l + 4] = rng.uniform(-0.55, 0.55, size=2)
    m1 = np.zeros(4 * L)
    m2 = np.zeros(4 * L)
    best = (np.inf, params.copy())
    for it in range(iters):
        v, dv = sim.forward_and_derivatives(params, PSI0)
        v, dv = v[0], dv[:, 0, :]
        ps = float(np.vdot(v, v).real)
        psi = v / np.sqrt(ps)
        dpsi = dv / np.sqrt(ps) - np.outer(np.real(dv.conj() @ v) / ps ** 1.5, v)
        e = float(np.real(np.vdot(psi, H_DIAG * psi)))
        if e < best[0] and ps > 1e-4:
            best = (e, params.copy())
        grad = 2.0 * np.real(dpsi.conj() @ (H_DIAG * psi))
        m1 = 0.9 * m1 + 0.1 * grad
        m2 = 0.999 * m2 + 0.001 * grad * grad
        params = params - 0.08 * (m1 / (1 - 0.9 ** (it + 1))) / (
            np.sqrt(m2 / (1 - 0.999 ** (it + 1))) + 1e-8)
    v = sim.forward(best[1], PSI0)[0]
    ps = float(np.vdot(v, v).real)
    psi = v / np.sqrt(ps)
    return {"P_opt": float(np.abs(psi[inst["idx_opt"]]) ** 2), "p_s": ps,
            "gap_raw": float(np.real(np.vdot(psi, H_DIAG * psi))) * inst["scale"]
            + inst["shift"] - inst["e_opt"]}


rows_L = {}
for L in (1, 2, 3, 4):
    per = [train_eon(L, 200 + s) for s in range(3)]
    best = max(per, key=lambda r: r["P_opt"])
    rows_L[str(L)] = {"best": best,
                      "P_opt_per_seed": [r["P_opt"] for r in per],
                      "p_s_per_seed": [r["p_s"] for r in per]}
    log(f"E.ON trained L={L}: P(x*) per seed "
        f"{np.round([r['P_opt'] for r in per], 4).tolist()}")
OUT["eon_trained_depth"] = rows_L

# ---------------------------------------------------------------------------
# HSBC side: ranking saturates early
# ---------------------------------------------------------------------------
from sklearn.metrics import average_precision_score

pipe = prepare_hsbc(8, seed=SEED)
model = pipe["model"]
U_tr, U_va, U_te = pipe["U"]["tr"], pipe["U"]["va"], pipe["U"]["te"]
y_tr, y_va, y_te = (pipe["y_split"][k] for k in ("tr", "va", "te"))

tau_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0]
G = np.column_stack([np.exp(-2.0 * t * model.hp) for t in tau_grid])
E_te = probs_matmul(U_te, G)
rows_tau_h = {str(t): {"test_auprc": float(average_precision_score(y_te, -E_te[:, j]))}
              for j, t in enumerate(tau_grid)}
OUT["hsbc_ite_tau_sweep"] = rows_tau_h
log("HSBC ITE sweep AUPRC: " + ", ".join(
    f"{t}:{v['test_auprc']:.4f}" for t, v in rows_tau_h.items()))

# hard-encoding sanity: ranking exactly tau-independent (monotone transform)
from hsbc_common import hard_bits
e_hard = model.energy_bits(hard_bits(U_te))
ap_hard = [float(average_precision_score(y_te, -np.exp(-2.0 * t * (e_hard - e_hard.min())
                                                       / (e_hard.max() - e_hard.min()))))
           for t in (0.5, 4.0, 16.0)]
assert max(ap_hard) - min(ap_hard) < 1e-12, ap_hard
OUT["hsbc_hard_encoding_tau_invariance"] = {"auprc_values": ap_hard}
log(f"HSBC hard-encoding tau-invariance verified: {ap_hard}")

# trained stacks vs depth
fraud_states = product_state_batch(U_tr[y_tr == 1]).astype(complex)
legit_idx = np.flatnonzero(y_tr == 0)
rng_v = np.random.default_rng(SEED)
val_sel = np.sort(np.concatenate(
    [np.flatnonzero(y_va == 1),
     rng_v.choice(np.flatnonzero(y_va == 0), 8000, replace=False)]))
U_va_sel, y_va_sel = U_va[val_sel], y_va[val_sel]


def train_hsbc(L, seed, iters=300):
    cfg = SLCUStackConfig(n_qubits=8, n_layers=L, hp=model.hp)
    sim = BatchedSLCU(cfg)
    rng = np.random.default_rng(seed)
    params = np.zeros(4 * L)
    for l in range(L):
        params[4 * l: 4 * l + 2] = rng.normal(scale=0.15, size=2)
        params[4 * l + 2] = rng.uniform(0.5, 2.5)
        params[4 * l + 3] = rng.normal(scale=0.15)
    m1 = np.zeros(4 * L)
    m2 = np.zeros(4 * L)
    best = (-1.0, params.copy())
    for it in range(iters):
        batch = rng.choice(legit_idx, 384, replace=False)
        legit_states = product_state_batch(U_tr[batch]).astype(complex)
        _, grad, _, _ = pairwise_loss_and_grad(sim, params, legit_states, fraud_states)
        m1 = 0.9 * m1 + 0.1 * grad
        m2 = 0.999 * m2 + 0.001 * grad * grad
        params = params - 0.05 * (m1 / (1 - 0.9 ** (it + 1))) / (
            np.sqrt(m2 / (1 - 0.999 ** (it + 1))) + 1e-8)
        if (it + 1) % 20 == 0 or it == iters - 1:
            ap = average_precision_score(y_va_sel, -ps_chunked(sim, params, U_va_sel))
            if ap > best[0]:
                best = (ap, params.copy())
    return best[0], best[1], sim


rows_Lh = {}
score_by_L = {}
for L in (1, 2, 3):
    per = []
    best = (-1.0, None, None)
    for s in range(3):
        ap, p, sim = train_hsbc(L, 300 + s)
        per.append(float(ap))
        if ap > best[0]:
            best = (ap, p, sim)
    s_te = -ps_chunked(best[2], best[1], U_te)
    score_by_L[L] = s_te
    rows_Lh[str(L)] = {
        "val_auprc_per_seed": per,
        "test_auprc": float(average_precision_score(y_te, s_te)),
    }
    log(f"HSBC trained L={L}: test AUPRC {rows_Lh[str(L)]['test_auprc']:.4f}")

from hsbc_common import bootstrap_delta_auprc
for a, b in ((2, 1), (3, 2)):
    mean, lo, hi = bootstrap_delta_auprc(y_te, score_by_L[a], score_by_L[b], seed=SEED)
    rows_Lh[f"delta_L{a}_minus_L{b}"] = {"mean": mean, "ci95": [lo, hi]}
OUT["hsbc_trained_depth"] = rows_Lh

# ---------------------------------------------------------------------------
# transpiled circuit resources at n=8, L=2 (both Hamiltonians)
# ---------------------------------------------------------------------------
from qiskit import QuantumCircuit, QuantumRegister, transpile


def ising_from_energies_eon(inst):
    """Exact h, J of the congestion QUBO via the closed-form mapping."""
    R, f0, cb, w, lam = inst["R"], inst["f0"], inst["cb"], np.ones(len(inst["f0"])), 1.0
    n = inst["n"]
    lin = lam * cb - 2.0 * (w * f0) @ R + (w @ (R ** 2))
    quad = 2.0 * np.einsum("l,le,lf->ef", w, R, R)
    h = np.zeros(n)
    for e in range(n):
        h[e] = -0.5 * lin[e] - 0.25 * sum(
            quad[min(e, f), max(e, f)] for f in range(n) if f != e)
    J = {(e, f): 0.25 * quad[e, f] for e in range(n) for f in range(e + 1, n)}
    return h, J


def ising_from_model(model, thresh=0.01):
    h = -model.a
    J = {}
    n = model.n
    for i in range(n):
        for j in range(i + 1, n):
            if abs(model.b[i, j]) > thresh:
                J[(i, j)] = -model.b[i, j]
    return h, J


def build_circ(n, L, h, J, scale):
    sys_q = QuantumRegister(n, "s")
    anc = QuantumRegister(L, "a")
    circ = QuantumCircuit(sys_q, anc)
    circ.h(sys_q)
    theta, phi, prep = 0.9, 0.4, 1.1
    for l in range(L):
        aq = n + l
        circ.ry(prep, aq)
        circ.x(aq)
        for e in range(n):
            if abs(h[e]) > 1e-12:
                circ.crz(2.0 * theta * h[e] / scale, aq, e)
        for (e, f), j in J.items():
            circ.cx(e, f)
            circ.crz(2.0 * theta * j / scale, aq, f)
            circ.cx(e, f)
        circ.x(aq)
        for e in range(n):
            circ.crx(2.0 * phi, aq, e)
        circ.ry(-prep, aq)
    return circ


res_rows = {}
h_e, J_e = ising_from_energies_eon(inst)
pipe8 = pipe
h_h, J_h = ising_from_model(pipe8["model"], thresh=0.01)
for tag, (h, J, scale) in (
        ("eon_n8_dense", (h_e, J_e, inst["scale"])),
        ("hsbc_n8_fitted_thr0.01", (h_h, J_h,
                                    float(pipe8["model"].energies.std())))):
    J_sig = {k: v for k, v in J.items() if abs(v) > 1e-12}
    circ = build_circ(8, 2, h, J_sig, scale)
    tq = transpile(circ, basis_gates=["cz", "rz", "sx", "x"],
                   optimization_level=3, seed_transpiler=7)
    res_rows[tag] = {
        "couplings": len(J_sig),
        "qubits": tq.num_qubits,
        "cz": int(sum(v for k, v in tq.count_ops().items() if k == "cz")),
        "depth": tq.depth(),
        "two_qubit_depth": tq.depth(lambda inst_: inst_.operation.num_qubits == 2),
    }
    log(f"resources {tag}: {res_rows[tag]}")
OUT["circuit_resources_L2"] = res_rows

OUT["wall_seconds"] = time.time() - T0
save_json("runs/hsbc_challenge/fit_v0/exp_e_depth.json", OUT)
log("EXP-E done -> runs/hsbc_challenge/fit_v0/exp_e_depth.json")
