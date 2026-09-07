"""Verification of the ancilla-syndrome machinery (Engine-A prerequisite).

Checks, in order of strength:

  V1  Completeness: sum_a P(a|x) = 1 (instrument property K^dag K + F^dag F = I).
  V2  Flag consistency: P(00...0|x) equals ||forward(params)|phi>||^2 --- the
      path already verified against DifferentiableStackedLCU to ~1e-15.
  V3  Circuit-level ground truth: full Qiskit statevector of the
      per-layer-ancilla PREP-SELECT-PREP^dag circuit; ALL 2^L ancilla-pattern
      block vectors and probabilities must match syndrome_sector_states /
      syndrome_distribution.
  V4  Failure-sector = derivative identity in implementation (logit)
      coordinates: sector(only layer j fails) = -(1/sqrt(a_j0 a_j1)) *
      d psi~ / d logit_{j,0}, tying the syndrome to AIG-I1 with the C9
      coordinate bridge made explicit.
  V5  Dephased-twin theorem: with the layer ancilla dephased between SELECT
      and PREP^dag, the pattern distribution is data-INDEPENDENT and equals
      the product-Bernoulli(2 a0 a1) formula --- verified by explicit
      density-matrix simulation on random states.
  V6  Pipeline smoke test: syndrome features on real ULB rows with the
      stored EXP-B parameters (row sums, ranges, timing).

Artifacts -> runs/hsbc_challenge/engine_a_v1/verify_syndrome_v1.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hsbc_common import (
    BatchedSLCU, SLCUStackConfig, _softmax, dephased_syndrome_distribution,
    prepare_hsbc, product_state_batch, save_json,
)

T0 = time.time()
OUT = {}
rng = np.random.default_rng(0)


def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def random_setup(n, L, seed, m=3):
    r = np.random.default_rng(seed)
    hp = r.uniform(0, 1, size=2 ** n)
    cfg = SLCUStackConfig(n_qubits=n, n_layers=L, hp=hp)
    params = r.normal(scale=0.5, size=cfg.n_params)
    states = r.normal(size=(m, 2 ** n)) + 1j * r.normal(size=(m, 2 ** n))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    return cfg, params, states


# ---------------------------------------------------------------------------
# V1 + V2
# ---------------------------------------------------------------------------
worst_sum, worst_flag = 0.0, 0.0
for n, L, seed in ((4, 2, 1), (4, 3, 2), (6, 2, 3), (8, 3, 4)):
    cfg, params, states = random_setup(n, L, seed)
    sim = BatchedSLCU(cfg)
    P = sim.syndrome_distribution(params, states)
    worst_sum = max(worst_sum, float(np.abs(P.sum(axis=1) - 1.0).max()))
    v = sim.forward(params, states)
    ps = np.einsum("md,md->m", v.conj(), v).real
    worst_flag = max(worst_flag, float(np.abs(P[:, 0] - ps).max()))
OUT["V1_completeness_max_dev"] = worst_sum
OUT["V2_flag_consistency_max_dev"] = worst_flag
assert worst_sum < 1e-12 and worst_flag < 1e-12
log(f"V1 completeness dev {worst_sum:.2e}; V2 flag dev {worst_flag:.2e}")

# ---------------------------------------------------------------------------
# V3: Qiskit circuit-level ground truth
# ---------------------------------------------------------------------------
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from functools import reduce


def branch_matrices(cfg, theta, phi):
    cost = np.diag(np.exp(-1j * theta * cfg.hp))
    rx = np.array([[np.cos(phi), -1j * np.sin(phi)],
                   [-1j * np.sin(phi), np.cos(phi)]])
    mix = reduce(np.kron, [rx] * cfg.n_qubits).astype(complex)
    return cost, mix


def build_syndrome_circuit(cfg, params, init_state):
    n, L = cfg.n_qubits, cfg.n_layers
    circ = QuantumCircuit(n + L)
    circ.initialize(init_state, list(range(n)))
    for l in range(L):
        logits = params[4 * l: 4 * l + 2]
        theta, phi = params[4 * l + 2], params[4 * l + 3]
        a = _softmax(np.asarray(logits, float))
        prep = 2.0 * np.arctan2(np.sqrt(a[1]), np.sqrt(a[0]))
        cost, mix = branch_matrices(cfg, theta, phi)
        aq = n + l
        circ.ry(prep, aq)
        circ.x(aq)
        circ.append(UnitaryGate(cost).control(1), [aq] + list(range(n)))
        circ.x(aq)
        circ.append(UnitaryGate(mix).control(1), [aq] + list(range(n)))
        circ.ry(-prep, aq)
    return circ


worst_vec, worst_prob = 0.0, 0.0
for n, L, seed in ((4, 2, 11), (4, 3, 12), (5, 2, 13)):
    cfg, params, states = random_setup(n, L, seed, m=2)
    sim = BatchedSLCU(cfg)
    sec = sim.syndrome_sector_states(params, states)          # (2^L, m, D)
    P = sim.syndrome_distribution(params, states)
    for i in range(states.shape[0]):
        sv = Statevector(build_syndrome_circuit(cfg, params, states[i])).data
        blocks = sv.reshape(2 ** L, 2 ** n)                   # row bit l = ancilla l
        worst_vec = max(worst_vec, float(np.abs(blocks - sec[:, i, :]).max()))
        worst_prob = max(worst_prob, float(
            np.abs((np.abs(blocks) ** 2).sum(axis=1) - P[i]).max()))
OUT["V3_circuit_block_vector_max_dev"] = worst_vec
OUT["V3_circuit_pattern_prob_max_dev"] = worst_prob
assert worst_vec < 1e-9 and worst_prob < 1e-12
log(f"V3 circuit ground truth: block-vector dev {worst_vec:.2e}, "
    f"pattern-prob dev {worst_prob:.2e}")

# ---------------------------------------------------------------------------
# V4: failure-sector = derivative identity (logit coordinates)
# ---------------------------------------------------------------------------
worst_id = 0.0
for n, L, seed in ((4, 2, 21), (6, 3, 22)):
    cfg, params, states = random_setup(n, L, seed, m=2)
    sim = BatchedSLCU(cfg)
    sec = sim.syndrome_sector_states(params, states)
    _, derivs = sim.forward_and_derivatives(params, states)
    for j in range(L):
        a = _softmax(params[4 * j: 4 * j + 2])
        pattern = 1 << j                                     # only layer j fails
        pred = -derivs[4 * j] / np.sqrt(a[0] * a[1])
        worst_id = max(worst_id, float(np.abs(sec[pattern] - pred).max()))
OUT["V4_sector_derivative_identity_max_dev"] = worst_id
assert worst_id < 1e-9
log(f"V4 sector=derivative identity (logit coords) dev {worst_id:.2e}")

# ---------------------------------------------------------------------------
# V5: dephased-twin theorem (density-matrix simulation)
# ---------------------------------------------------------------------------
def dephased_pattern_probs_numeric(cfg, params, psi):
    """Explicit instrument simulation with ancilla dephasing after SELECT."""
    n, L = cfg.n_qubits, cfg.n_layers
    branches = [(np.array(psi, complex), 1.0, ())]           # (state, prob, pattern)
    for l in range(L):
        logits = params[4 * l: 4 * l + 2]
        theta, phi = params[4 * l + 2], params[4 * l + 3]
        a = _softmax(np.asarray(logits, float))
        c2, s2 = a[0], a[1]                                  # |R_{00}|^2, |R_{10}|^2
        cost, mix = branch_matrices(cfg, theta, phi)
        new = []
        for st, pr, pat in branches:
            for u, w in ((cost, c2), (mix, s2)):             # dephased branch choice
                phi_b = u @ st
                # PREP^dag then measure: outcome 0 w.p. |R_{y0}|^2, 1 w.p. |R_{y1}|^2
                p0 = c2 if w == c2 else s2                    # |R_{y0}|^2 for y = branch
                new.append((phi_b, pr * w * p0, pat + (0,)))
                new.append((phi_b, pr * w * (1 - p0), pat + (1,)))
        branches = new
    D = 2 ** L
    out = np.zeros(D)
    for st, pr, pat in branches:
        idx = sum(b << i for i, b in enumerate(pat))
        out[idx] += pr * float(np.vdot(st, st).real)
    return out


cfg, params, states = random_setup(4, 2, 31, m=4)
ref = dephased_syndrome_distribution(params, cfg.n_layers, states.shape[0])
worst_dep, spread = 0.0, 0.0
for i in range(states.shape[0]):
    num = dephased_pattern_probs_numeric(cfg, params, states[i])
    worst_dep = max(worst_dep, float(np.abs(num - ref[i]).max()))
spread = float(np.abs(ref - ref[0]).max())                   # data-independence
OUT["V5_dephased_numeric_vs_formula_max_dev"] = worst_dep
OUT["V5_dephased_data_independence_spread"] = spread
assert worst_dep < 1e-12 and spread == 0.0
log(f"V5 dephased twin: numeric-vs-formula dev {worst_dep:.2e}; "
    f"data-independent (spread {spread}) -> syndrome signal is purely "
    f"interference-borne")

# ---------------------------------------------------------------------------
# V6: pipeline smoke test on real ULB rows with stored EXP-B parameters
# ---------------------------------------------------------------------------
pipe = prepare_hsbc(8, seed=0)
with open("runs/hsbc_challenge/fit_v0/exp_b_scorer.json") as fh:
    params_b = np.array(json.load(fh)["n8_L2"]["S2_trained_mixer_params"])
cfg8 = SLCUStackConfig(n_qubits=8, n_layers=2, hp=pipe["model"].hp)
sim8 = BatchedSLCU(cfg8)
U = pipe["U"]["te"][:2000]
t0 = time.time()
feats = sim8.syndrome_features(params_b, product_state_batch(U).astype(complex))
dt = time.time() - t0
P = np.exp(feats) - 1e-6
OUT["V6_rows"] = int(len(U))
OUT["V6_row_sum_max_dev"] = float(np.abs(P.sum(axis=1) - 1.0).max())
OUT["V6_us_per_tx"] = float(1e6 * dt / len(U))
OUT["V6_pattern_prob_ranges"] = {
    f"pattern_{p:0{cfg8.n_layers}b}": [float(P[:, p].min()), float(P[:, p].max())]
    for p in range(2 ** cfg8.n_layers)}
assert OUT["V6_row_sum_max_dev"] < 1e-9
log(f"V6 pipeline: {len(U)} rows, {OUT['V6_us_per_tx']:.1f} us/tx, "
    f"row-sum dev {OUT['V6_row_sum_max_dev']:.2e}")

os.makedirs("runs/hsbc_challenge/engine_a_v1", exist_ok=True)
OUT["wall_seconds"] = time.time() - T0
save_json("runs/hsbc_challenge/engine_a_v1/verify_syndrome_v1.json", OUT)
log("ALL SYNDROME VERIFICATIONS PASSED -> "
    "runs/hsbc_challenge/engine_a_v1/verify_syndrome_v1.json")
