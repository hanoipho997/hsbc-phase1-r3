"""Candidate E: target-native compile artifact for the exact Engine-A stacked
cross-view LCU circuit (12 system + 3 ancilla qubits, L=3, seed-900 params).

  E1  Structured construction (single-qubit rotations + Pauli-string
      rotations only; controlled branches via the identity
      exp(-i a Pi_v (x) P) = exp(-i a/2 P) exp(-/+ i a/2 Z_anc P)); verified
      against the dense CrossViewSLCU sector states on random product rows.
  E2  Walsh spectrum of the fitted per-view tables: weight >= 3 coefficients
      must vanish (2-body Ising), otherwise the count changes.
  E3  Transpile DISTRIBUTIONS over seeds 0..9 for four targets:
        ion_all_to_all   basis {rx, ry, rz, rxx}, no coupling map (IonQ-like)
        cz_all_to_all    basis {cz, rz, sx, x}, no coupling map (audit method)
        iqm_grid_4x5     basis {r, cz}, 4x5 square-lattice coupling map
                         (IQM Garnet stand-in: 20 qubits, 31 couplers)
        linear_chain_15  basis {cz, rz, sx, x}, 15-qubit line (worst case)
  E4  Like-for-like at the audit instance (ULB n=8, L=2, thr-0.01 Ising):
      old controlled-gate build vs structured build, same transpile method
      as exp_e_depth.py (cz basis, O3, seed 7), plus seed distribution.
  E5  Unary-iteration neutrality at binary width: repo estimates for K=2,4,8.

Artifacts -> runs/hsbc_challenge/novelty_v1/native_compile_v1.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsbc_common import (
    CrossViewConfig, CrossViewSLCU, broadcast_view_diagonal, prepare_hsbc,
    product_state_batch,
)
from novelty_common_v1 import (
    build_structured_crossview_circuit, crossview_walsh, save_json_no_overwrite,
    sha256_file,
)
from fqe_ising.unary_iteration import (
    multicontrolled_toffoli_estimate, scratch_size, unary_iteration_toffoli_estimate,
)

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

T0 = time.time()
OUTD = "runs/hsbc_challenge/novelty_v1"
os.makedirs(OUTD, exist_ok=True)
OUT_PATH = os.path.join(OUTD, "native_compile_v1.json")
if os.path.exists(OUT_PATH):
    raise FileExistsError(OUT_PATH)
OUT = {"schema": "hsbc-novelty-native-compile-v1", "sources": {}}
SEEDS = list(range(10))


def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def two_qubit_stats(tq):
    ops = tq.count_ops()
    n2 = int(sum(v for k, v in ops.items() if k in ("cx", "cz", "rxx", "rzz", "swap", "ecr", "iswap")))
    return {"two_qubit_gates": n2, "depth": int(tq.depth()),
            "two_qubit_depth": int(tq.depth(lambda inst: inst.operation.num_qubits == 2)),
            "ops": {k: int(v) for k, v in ops.items()}}


def summarize(rows):
    keys = ("two_qubit_gates", "depth", "two_qubit_depth")
    return {k: {"min": int(min(r[k] for r in rows)), "median": float(np.median([r[k] for r in rows])),
                "max": int(max(r[k] for r in rows))} for k in keys}


# ---------------------------------------------------------------------------
# E1/E2: Engine-A circuit
# ---------------------------------------------------------------------------
import joblib
BASE = "runs/hsbc_challenge/engine_a_v1"
views = joblib.load(os.path.join(BASE, "views_v1.joblib"))
arms = json.load(open(os.path.join(BASE, "arms_v1", "arms_summary_v1.json")))
OUT["sources"]["views_v1.joblib"] = sha256_file(os.path.join(BASE, "views_v1.joblib"))
OUT["sources"]["arms_summary_v1.json"] = sha256_file(os.path.join(BASE, "arms_v1", "arms_summary_v1.json"))
VQ = views["view_qubits"]
name2idx = {"A": 0, "B": 1, "C": 2}
cfg = CrossViewConfig(
    n_qubits=12, views=tuple(tuple(VQ[v]) for v in ("A", "B", "C")),
    layer_pairs=tuple((name2idx[a], name2idx[b]) for a, b in views["layer_pairs"]),
    hp_views=tuple(broadcast_view_diagonal(np.array(views["hp_tables"][v]), VQ[v], 12)
                   for v in ("A", "B", "C")))
sim = CrossViewSLCU(cfg)
walsh = crossview_walsh(cfg)
OUT["E2_walsh"] = {
    "max_abs_weight_ge3": float(max(abs(c) for v in walsh for S, c in walsh[v].items() if len(S) >= 3)),
    "nonzero_weight1_per_view": [int(sum(abs(c) > 1e-12 for S, c in walsh[v].items() if len(S) == 1)) for v in walsh],
    "nonzero_weight2_per_view": [int(sum(abs(c) > 1e-12 for S, c in walsh[v].items() if len(S) == 2)) for v in walsh],
}
log(f"E2 Walsh: max weight>=3 coefficient {OUT['E2_walsh']['max_abs_weight_ge3']:.2e}")
params = np.array(arms["Q"]["seeds"]["900"]["params"])

rng = np.random.default_rng(20260904)
U = rng.uniform(0, 1, size=(3, 12))
states = product_state_batch(U).astype(complex)
sec = sim.syndrome_sector_states(params, states)
worst = 0.0
worst_prob = 0.0
Pd = sim.syndrome_distribution(params, states)
for i in range(3):
    qc = build_structured_crossview_circuit(cfg, params, u_row=U[i], walsh=walsh, dephase_marker=False)
    sv = Statevector(qc).data.reshape(2 ** 3, 2 ** 12)
    # the constant Walsh terms leave one global phase per layer: align it
    ph = np.vdot(sec[:, i, :].ravel(), sv.ravel())
    sv_aligned = sv * np.conj(ph / abs(ph))
    worst = max(worst, float(np.abs(sv_aligned - sec[:, i, :]).max()))
    worst_prob = max(worst_prob, float(np.abs((np.abs(sv) ** 2).sum(axis=1) - Pd[i]).max()))
OUT["E1_structured_vs_dense_max_dev_phase_aligned"] = worst
OUT["E1_structured_vs_dense_pattern_prob_max_dev"] = worst_prob
assert worst < 1e-9 and worst_prob < 1e-12, (worst, worst_prob)
qc0 = build_structured_crossview_circuit(cfg, params, u_row=U[0], walsh=walsh, dephase_marker=False)
OUT["E1_structured_counts"] = two_qubit_stats(qc0)
OUT["E1_structured_counts"]["qubits"] = qc0.num_qubits
log(f"E1 structured circuit verified ({worst:.2e}); pre-transpile "
    f"{OUT['E1_structured_counts']['two_qubit_gates']} two-qubit gates, depth {qc0.depth()}")

# analytic native accounting (no transpiler)
per_branch = {"controlled_rx_ZX_rotations": 4, "controlled_field_ZZ_rotations": 4,
              "controlled_coupling_ZZZ_rotations": 6}
OUT["E1_analytic_per_layer"] = {
    "branches": 2,
    "weight2_pauli_rotations_per_branch": 8,
    "weight3_pauli_rotations_per_branch": 6,
    "cx_per_weight3_rotation_ladder": 4,
    "two_qubit_gates_per_layer_structured": 2 * (8 + 6 * 4),
    "two_qubit_gates_L3_structured": 3 * 2 * (8 + 6 * 4),
    "note": "a weight-3 Z_anc Z_q1 Z_q2 rotation is one native gate only on hardware with a multi-qubit ZZZ primitive; on MS/ZZ/CZ hardware it needs a 2-CX ladder each side"}

# ---------------------------------------------------------------------------
# E3: transpile distributions
# ---------------------------------------------------------------------------
def grid_coupling(rows, cols):
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append([q, q + 1])
            if r + 1 < rows:
                edges.append([q, q + cols])
    return CouplingMap(edges + [[b, a] for a, b in edges])


TARGETS = {
    "ion_all_to_all": dict(basis_gates=["rx", "ry", "rz", "rxx"], coupling_map=None),
    "cz_all_to_all": dict(basis_gates=["cz", "rz", "sx", "x"], coupling_map=None),
    "iqm_grid_4x5": dict(basis_gates=["r", "cz"], coupling_map=grid_coupling(4, 5)),
    "linear_chain_15": dict(basis_gates=["cz", "rz", "sx", "x"],
                            coupling_map=CouplingMap.from_line(15)),
}
OUT["E3"] = {}
qc_bare = build_structured_crossview_circuit(cfg, params, u_row=U[0], walsh=walsh, dephase_marker=False)
for tag, kw in TARGETS.items():
    rows = []
    t0 = time.time()
    for s in SEEDS:
        tq = transpile(qc_bare, basis_gates=kw["basis_gates"], coupling_map=kw["coupling_map"],
                       optimization_level=3, seed_transpiler=s)
        st = two_qubit_stats(tq)
        st["seed"] = s
        rows.append(st)
    OUT["E3"][tag] = {"basis_gates": kw["basis_gates"],
                      "coupling": None if kw["coupling_map"] is None else f"{tag}",
                      "per_seed": rows, "summary": summarize(rows),
                      "wall_seconds": time.time() - t0}
    log(f"E3 {tag}: 2q gates min/median/max "
        f"{OUT['E3'][tag]['summary']['two_qubit_gates']} ; depth {OUT['E3'][tag]['summary']['depth']}")

# ---------------------------------------------------------------------------
# E4: like-for-like at the audit instance (ULB n=8, L=2)
# ---------------------------------------------------------------------------
pipe = prepare_hsbc(8, seed=0)
model = pipe["model"]
h = -model.a
J = {(i, j): -model.b[i, j] for i in range(8) for j in range(i + 1, 8) if abs(model.b[i, j]) > 0.01}
scale = float(model.energies.std())
theta, phi, prep = 0.9, 0.4, 1.1


def old_build(n, L):
    sys_q = QuantumRegister(n, "s")
    anc = QuantumRegister(L, "a")
    circ = QuantumCircuit(sys_q, anc)
    circ.h(sys_q)
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


def zrot(circ, qubits, t):
    if len(qubits) == 1:
        circ.rz(2 * t, qubits[0])
    elif len(qubits) == 2:
        circ.rzz(2 * t, qubits[0], qubits[1])
    else:
        for a, b in zip(qubits[:-1], qubits[1:]):
            circ.cx(a, b)
        circ.rz(2 * t, qubits[-1])
        for a, b in reversed(list(zip(qubits[:-1], qubits[1:]))):
            circ.cx(a, b)


def new_build(n, L):
    """Same operator as old_build, structured: branch0 (anc=0) = Ising phase,
    branch1 (anc=1) = RX mixer."""
    circ = QuantumCircuit(n + L)
    circ.h(range(n))
    for l in range(L):
        aq = n + l
        circ.ry(prep, aq)
        # branch 0 controlled on |0>: exp(-i a Pi_0 P) = exp(-i a/2 P) exp(-i a/2 Z_anc P)
        for e in range(n):
            if abs(h[e]) > 1e-12:
                a = theta * h[e] / scale
                zrot(circ, [e], a / 2)
                zrot(circ, [aq, e], a / 2)
        for (e, f), j in J.items():
            a = theta * j / scale
            zrot(circ, [e, f], a / 2)
            zrot(circ, [aq, e, f], a / 2)
        # branch 1 controlled on |1>: crx(2 phi) = exp(-i phi Pi_1 X)
        for e in range(n):
            circ.rx(phi, e)
            circ.h(e)
            zrot(circ, [aq, e], -phi / 2)
            circ.h(e)
        circ.ry(-prep, aq)
    return circ


old8 = old_build(8, 2)
new8 = new_build(8, 2)
from qiskit.quantum_info import Operator
dev_ops = float(np.abs(Operator(old8).data - Operator(new8).data).max())
OUT["E4_ulb_n8_L2"] = {"couplings_thr0.01": len(J), "old_vs_new_operator_max_dev": dev_ops}
assert dev_ops < 1e-9, dev_ops
for tag, circ in (("old_controlled_gates", old8), ("structured", new8)):
    tq7 = transpile(circ, basis_gates=["cz", "rz", "sx", "x"], optimization_level=3, seed_transpiler=7)
    rows = []
    for s in SEEDS:
        tq = transpile(circ, basis_gates=["cz", "rz", "sx", "x"], optimization_level=3, seed_transpiler=s)
        st = two_qubit_stats(tq)
        st["seed"] = s
        rows.append(st)
    OUT["E4_ulb_n8_L2"][tag] = {"seed7_audit_method": two_qubit_stats(tq7),
                                "seed_distribution": summarize(rows), "per_seed": rows}
    log(f"E4 {tag}: seed-7 cz {OUT['E4_ulb_n8_L2'][tag]['seed7_audit_method']['two_qubit_gates']} "
        f"depth {OUT['E4_ulb_n8_L2'][tag]['seed7_audit_method']['depth']}; "
        f"seed distribution {OUT['E4_ulb_n8_L2'][tag]['seed_distribution']['two_qubit_gates']}")

# ---------------------------------------------------------------------------
# E5: unary iteration neutrality
# ---------------------------------------------------------------------------
OUT["E5_unary_iteration"] = {
    str(K): {"selection_qubits": int(np.ceil(np.log2(K))) if K > 1 else 0,
             "scratch_qubits": int(scratch_size(K)),
             "multicontrolled_toffoli_estimate": int(multicontrolled_toffoli_estimate(K)),
             "unary_iteration_toffoli_estimate": int(unary_iteration_toffoli_estimate(K))}
    for K in (2, 4, 8, 16)}
OUT["E5_note"] = ("With K=2 branches the selection register is one qubit, scratch is zero and "
                  "every branch is already singly controlled; unary iteration saves nothing at "
                  "binary width and only matters once K>=4 branches share a layer.")
OUT["wall_seconds"] = time.time() - T0
save_json_no_overwrite(OUT_PATH, OUT)
log(f"DONE -> {OUT_PATH}")
