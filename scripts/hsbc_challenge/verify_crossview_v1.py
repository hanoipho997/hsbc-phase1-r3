"""M1 gate: verification of the cross-view stacked-LCU class (Engine-A
preregistration section 5) — the adapted V1–V5 suite of verify_syndrome_v1.py
plus a repo-dense derivative twin and a training-gradient finite-difference
check.

  W1  Completeness + flag consistency (sum_a P(a|x)=1; P(0..0)=||forward||^2).
  W2  Repo-dense twin: forward AND all 6L parameter derivatives against
      DifferentiableStackedLCU built from dense cross-view branch terms.
  W3  Qiskit circuit-level ground truth: per-layer-ancilla statevector
      blocks vs syndrome_sector_states (vectors and probabilities).
  W4  Failure-sector = derivative identity in logit coordinates
      (sector(only layer j fails) = -deriv[6j]/sqrt(a0*a1)).
  W5  Dephased-twin theorem for cross-view branches: numeric instrument
      simulation vs the product-Bernoulli(2 a0 a1) formula; exact data
      independence.
  W6  Full Engine-A spec smoke (n=12, 3 views x 4 qubits, layer pairs
      (A,B),(B,C),(A,C)): completeness, timing, pairwise_loss_and_grad
      compatibility, and a finite-difference check of the training-loss
      gradient at n=6.

Artifacts -> runs/hsbc_challenge/engine_a_v1/verify_crossview_v1.json
"""

from __future__ import annotations

import os
import sys
import time
from functools import reduce

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hsbc_common import (
    CrossViewConfig, CrossViewSLCU, _softmax, broadcast_view_diagonal,
    crossview_repo_stack, dephased_syndrome_distribution,
    pairwise_loss_and_grad, save_json,
)

T0 = time.time()
OUT = {}


def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def make_config(n, views, pairs, seed):
    r = np.random.default_rng(seed)
    hp_views = tuple(
        broadcast_view_diagonal(r.uniform(0, 1, size=2 ** len(v)), v, n)
        for v in views)
    cfg = CrossViewConfig(n_qubits=n, views=tuple(tuple(v) for v in views),
                          layer_pairs=tuple(pairs), hp_views=hp_views)
    params = r.normal(scale=0.5, size=cfg.n_params)
    states = r.normal(size=(3, 2 ** n)) + 1j * r.normal(size=(3, 2 ** n))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    return cfg, params, states


SETUPS = [
    (6, [(0, 1), (2, 3), (4, 5)], [(0, 1), (1, 2), (0, 2)], 1),
    (8, [(0, 1, 2), (3, 4, 5), (6, 7)], [(0, 1), (1, 2)], 2),
]

# ---------------------------------------------------------------------------
# W1 + W2 + W4
# ---------------------------------------------------------------------------
w_sum = w_flag = w_repo_v = w_repo_d = w_id = 0.0
for n, views, pairs, seed in SETUPS:
    cfg, params, states = make_config(n, views, pairs, seed)
    sim = CrossViewSLCU(cfg)
    P = sim.syndrome_distribution(params, states)
    w_sum = max(w_sum, float(np.abs(P.sum(axis=1) - 1.0).max()))
    v, derivs = sim.forward_and_derivatives(params, states)
    ps = np.einsum("md,md->m", v.conj(), v).real
    w_flag = max(w_flag, float(np.abs(P[:, 0] - ps).max()))

    stack = crossview_repo_stack(cfg)
    for i in range(states.shape[0]):
        v_ref, d_ref = stack.unnormalized_state_and_derivatives(params, states[i])
        w_repo_v = max(w_repo_v, float(np.abs(v[i] - v_ref).max()))
        w_repo_d = max(w_repo_d, float(np.abs(derivs[:, i, :] - d_ref).max()))

    sec = sim.syndrome_sector_states(params, states)
    for j in range(cfg.n_layers):
        a = _softmax(params[6 * j: 6 * j + 2])
        pred = -derivs[6 * j] / np.sqrt(a[0] * a[1])
        w_id = max(w_id, float(np.abs(sec[1 << j] - pred).max()))

OUT["W1_completeness_max_dev"] = w_sum
OUT["W1_flag_consistency_max_dev"] = w_flag
OUT["W2_repo_forward_max_dev"] = w_repo_v
OUT["W2_repo_derivatives_max_dev"] = w_repo_d
OUT["W4_sector_derivative_identity_max_dev"] = w_id
assert w_sum < 1e-12 and w_flag < 1e-12
assert w_repo_v < 1e-9 and w_repo_d < 1e-9
assert w_id < 1e-9
log(f"W1 completeness {w_sum:.2e}, flag {w_flag:.2e}")
log(f"W2 repo twin: forward {w_repo_v:.2e}, derivatives {w_repo_d:.2e}")
log(f"W4 sector=derivative identity {w_id:.2e}")

# ---------------------------------------------------------------------------
# W3: Qiskit circuit ground truth (dense branch matrices built from scratch)
# ---------------------------------------------------------------------------
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

I2 = np.eye(2)


def dense_branch(cfg, view, theta, phi):
    rx = np.array([[np.cos(phi), -1j * np.sin(phi)],
                   [-1j * np.sin(phi), np.cos(phi)]])
    factors = [rx if q in cfg.views[view] else I2
               for q in reversed(range(cfg.n_qubits))]
    mix = reduce(np.kron, factors).astype(complex)
    return np.diag(np.exp(-1j * theta * cfg.hp_views[view])) @ mix


def build_circuit(cfg, params, init_state):
    n, L = cfg.n_qubits, cfg.n_layers
    circ = QuantumCircuit(n + L)
    circ.initialize(init_state, list(range(n)))
    for l in range(L):
        p = params[6 * l: 6 * l + 6]
        a = _softmax(np.asarray(p[:2], float))
        prep = 2.0 * np.arctan2(np.sqrt(a[1]), np.sqrt(a[0]))
        vpair = cfg.layer_pairs[l]
        u0 = dense_branch(cfg, vpair[0], p[2], p[3])
        u1 = dense_branch(cfg, vpair[1], p[4], p[5])
        aq = n + l
        circ.ry(prep, aq)
        circ.x(aq)
        circ.append(UnitaryGate(u0).control(1), [aq] + list(range(n)))
        circ.x(aq)
        circ.append(UnitaryGate(u1).control(1), [aq] + list(range(n)))
        circ.ry(-prep, aq)
    return circ


w_vec = w_prob = 0.0
for n, views, pairs, seed in (SETUPS[0],):
    for L_use in (2, 3):
        cfg, params, states = make_config(n, views, pairs[:L_use], seed + 10)
        sim = CrossViewSLCU(cfg)
        sec = sim.syndrome_sector_states(params, states)
        P = sim.syndrome_distribution(params, states)
        for i in range(2):
            sv = Statevector(build_circuit(cfg, params, states[i])).data
            blocks = sv.reshape(2 ** cfg.n_layers, 2 ** n)
            w_vec = max(w_vec, float(np.abs(blocks - sec[:, i, :]).max()))
            w_prob = max(w_prob, float(
                np.abs((np.abs(blocks) ** 2).sum(axis=1) - P[i]).max()))
OUT["W3_circuit_block_vector_max_dev"] = w_vec
OUT["W3_circuit_pattern_prob_max_dev"] = w_prob
assert w_vec < 1e-9 and w_prob < 1e-12
log(f"W3 circuit ground truth: vectors {w_vec:.2e}, probs {w_prob:.2e}")

# ---------------------------------------------------------------------------
# W5: dephased-twin theorem for cross-view branches
# ---------------------------------------------------------------------------
def dephased_numeric(cfg, params, psi):
    branches = [(np.array(psi, complex), 1.0, ())]
    for l in range(cfg.n_layers):
        p = params[6 * l: 6 * l + 6]
        a = _softmax(np.asarray(p[:2], float))
        u0 = dense_branch(cfg, cfg.layer_pairs[l][0], p[2], p[3])
        u1 = dense_branch(cfg, cfg.layer_pairs[l][1], p[4], p[5])
        new = []
        for st, pr, pat in branches:
            for u, w, p_keep in ((u0, a[0], a[0]), (u1, a[1], a[1])):
                phi_b = u @ st
                new.append((phi_b, pr * w * p_keep, pat + (0,)))
                new.append((phi_b, pr * w * (1 - p_keep), pat + (1,)))
        branches = new
    D = 2 ** cfg.n_layers
    out = np.zeros(D)
    for st, pr, pat in branches:
        idx = sum(b << i for i, b in enumerate(pat))
        out[idx] += pr * float(np.vdot(st, st).real)
    return out


cfg, params, states = make_config(*SETUPS[0][:3], seed=31)
ref = dephased_syndrome_distribution(params, cfg.n_layers, states.shape[0],
                                     stride=6)
w_dep = 0.0
for i in range(states.shape[0]):
    num = dephased_numeric(cfg, params, states[i])
    w_dep = max(w_dep, float(np.abs(num - ref[i]).max()))
spread = float(np.abs(ref - ref[0]).max())
OUT["W5_dephased_numeric_vs_formula_max_dev"] = w_dep
OUT["W5_dephased_data_independence_spread"] = spread
assert w_dep < 1e-12 and spread == 0.0
log(f"W5 dephased twin: dev {w_dep:.2e}, data-independent (spread {spread})")

# ---------------------------------------------------------------------------
# W6: full Engine-A spec smoke + training-gradient finite-difference check
# ---------------------------------------------------------------------------
# finite-difference check of the pairwise training loss gradient (n=6)
cfg6, params6, _ = make_config(*SETUPS[0][:3], seed=41)
sim6 = CrossViewSLCU(cfg6)
r = np.random.default_rng(42)
legit = r.normal(size=(24, 2 ** 6)) + 1j * r.normal(size=(24, 2 ** 6))
legit /= np.linalg.norm(legit, axis=1, keepdims=True)
fraud = r.normal(size=(8, 2 ** 6)) + 1j * r.normal(size=(8, 2 ** 6))
fraud /= np.linalg.norm(fraud, axis=1, keepdims=True)
loss0, grad, _, _ = pairwise_loss_and_grad(sim6, params6, legit, fraud)
fd = np.zeros_like(grad)
h = 1e-6
for k in range(cfg6.n_params):
    pp = params6.copy(); pp[k] += h
    lp, *_ = pairwise_loss_and_grad(sim6, pp, legit, fraud)
    pm = params6.copy(); pm[k] -= h
    lm, *_ = pairwise_loss_and_grad(sim6, pm, legit, fraud)
    fd[k] = (lp - lm) / (2 * h)
w_fd = float(np.abs(grad - fd).max() / max(np.abs(grad).max(), 1e-12))
OUT["W6_loss_gradient_fd_rel_dev"] = w_fd
assert w_fd < 1e-5
log(f"W6 training-loss gradient vs finite differences: rel dev {w_fd:.2e}")

# full spec: n=12, 3 views x 4 qubits, pairs (A,B),(B,C),(A,C)
views12 = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)]
cfg12, params12, _ = make_config(12, views12, [(0, 1), (1, 2), (0, 2)], 51)
sim12 = CrossViewSLCU(cfg12)
m = 512
st = np.random.default_rng(52).normal(size=(m, 2 ** 12)) \
    + 1j * np.random.default_rng(53).normal(size=(m, 2 ** 12))
st /= np.linalg.norm(st, axis=1, keepdims=True)
t0 = time.time()
feats = sim12.syndrome_features(params12, st)
t_syn = time.time() - t0
t0 = time.time()
_, d12 = sim12.forward_and_derivatives(params12, st)
t_grad = time.time() - t0
P12 = np.exp(feats) - 1e-6
OUT["W6_fullspec"] = {
    "n_qubits": 12, "layers": 3, "params": int(cfg12.n_params),
    "syndrome_row_sum_max_dev": float(np.abs(P12.sum(axis=1) - 1.0).max()),
    "syndrome_us_per_row": float(1e6 * t_syn / m),
    "forward_and_derivs_us_per_row": float(1e6 * t_grad / m),
}
assert OUT["W6_fullspec"]["syndrome_row_sum_max_dev"] < 1e-9
log(f"W6 full spec n=12/L=3/18 params: syndrome {OUT['W6_fullspec']['syndrome_us_per_row']:.0f} us/row, "
    f"forward+derivs {OUT['W6_fullspec']['forward_and_derivs_us_per_row']:.0f} us/row, "
    f"row-sum dev {OUT['W6_fullspec']['syndrome_row_sum_max_dev']:.2e}")

OUT["wall_seconds"] = time.time() - T0
save_json("runs/hsbc_challenge/engine_a_v1/verify_crossview_v1.json", OUT)
log("ALL CROSS-VIEW VERIFICATIONS PASSED -> "
    "runs/hsbc_challenge/engine_a_v1/verify_crossview_v1.json")
