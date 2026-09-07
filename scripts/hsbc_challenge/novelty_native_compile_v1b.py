"""Candidate E, part b: the SAME Engine-A operator built two ways —
(old) Qiskit controlled gates crx/crz and cx-crz-cx for couplings, as in
exp_e_depth.py; (structured) uncontrolled rotation + Z_anc-string rotation —
transpiled to the same four targets over seeds 0..9.  Also the ULB n=8/L=2
audit instance on the ion-like rxx basis.  Reports distributions, not the
best compile.

Artifacts -> runs/hsbc_challenge/novelty_v1/native_compile_v1b.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsbc_common import (
    CrossViewConfig, CrossViewSLCU, _softmax, broadcast_view_diagonal, prepare_hsbc,
    product_state_batch,
)
from novelty_common_v1 import (
    build_structured_crossview_circuit, crossview_walsh, save_json_no_overwrite, sha256_file,
)
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import CouplingMap

T0 = time.time()
OUTD = "runs/hsbc_challenge/novelty_v1"
OUT_PATH = os.path.join(OUTD, "native_compile_v1b.json")
if os.path.exists(OUT_PATH):
    raise FileExistsError(OUT_PATH)
OUT = {"schema": "hsbc-novelty-native-compile-v1b"}
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
    "linear_chain_15": dict(basis_gates=["cz", "rz", "sx", "x"], coupling_map=CouplingMap.from_line(15)),
}


def sweep(circ, targets):
    out = {}
    for tag, kw in targets.items():
        rows = []
        for s in SEEDS:
            tq = transpile(circ, basis_gates=kw["basis_gates"], coupling_map=kw["coupling_map"],
                           optimization_level=3, seed_transpiler=s)
            st = two_qubit_stats(tq)
            st["seed"] = s
            rows.append(st)
        out[tag] = {"summary": summarize(rows), "per_seed": rows}
    return out


# ---------------------------------------------------------------------------
# Engine-A circuit, old controlled-gate construction
# ---------------------------------------------------------------------------
import joblib
BASE = "runs/hsbc_challenge/engine_a_v1"
views = joblib.load(os.path.join(BASE, "views_v1.joblib"))
arms = json.load(open(os.path.join(BASE, "arms_v1", "arms_summary_v1.json")))
VQ = views["view_qubits"]
name2idx = {"A": 0, "B": 1, "C": 2}
cfg = CrossViewConfig(
    n_qubits=12, views=tuple(tuple(VQ[v]) for v in ("A", "B", "C")),
    layer_pairs=tuple((name2idx[a], name2idx[b]) for a, b in views["layer_pairs"]),
    hp_views=tuple(broadcast_view_diagonal(np.array(views["hp_tables"][v]), VQ[v], 12)
                   for v in ("A", "B", "C")))
sim = CrossViewSLCU(cfg)
walsh = crossview_walsh(cfg)
params = np.array(arms["Q"]["seeds"]["900"]["params"])
rng = np.random.default_rng(20260904)
U = rng.uniform(0, 1, size=(2, 12))


def old_style_engine_a(u_row):
    n, L = 12, 3
    qc = QuantumCircuit(n + L)
    for e in range(n):
        qc.ry(np.pi * float(u_row[e]), e)
    for l in range(L):
        p = params[6 * l: 6 * l + 6]
        a = _softmax(np.asarray(p[:2], float))
        prep = 2.0 * np.arctan2(np.sqrt(a[1]), np.sqrt(a[0]))
        anc = n + l
        qc.ry(prep, anc)
        consts = []
        for branch in range(2):
            view = cfg.layer_pairs[l][branch]
            theta, phi = p[2 + 2 * branch], p[3 + 2 * branch]
            qubits = list(cfg.views[view])
            if branch == 0:
                qc.x(anc)
            for q in qubits:                      # controlled RX(2 phi)
                qc.crx(2.0 * phi, anc, q)
            for S, c in walsh[view].items():      # controlled exp(-i theta c Z_S)
                if len(S) == 0:
                    consts.append(theta * c)
                    continue
                if len(S) == 1:
                    qc.crz(2.0 * theta * c, anc, qubits[S[0]])
                else:
                    e, f = qubits[S[0]], qubits[S[1]]
                    qc.cx(e, f)
                    qc.crz(2.0 * theta * c, anc, f)
                    qc.cx(e, f)
            if branch == 0:
                qc.x(anc)
        qc.p(-(consts[1] - consts[0]), anc)
        qc.ry(-prep, anc)
    return qc


sec = sim.syndrome_sector_states(params, product_state_batch(U).astype(complex))
Pd = sim.syndrome_distribution(params, product_state_batch(U).astype(complex))
worst = 0.0
for i in range(2):
    sv = Statevector(old_style_engine_a(U[i])).data.reshape(8, 4096)
    worst = max(worst, float(np.abs((np.abs(sv) ** 2).sum(axis=1) - Pd[i]).max()))
OUT["engine_a_old_style_pattern_prob_max_dev"] = worst
assert worst < 1e-12, worst
old_qc = old_style_engine_a(U[0])
new_qc = build_structured_crossview_circuit(cfg, params, u_row=U[0], walsh=walsh, dephase_marker=False)
OUT["engine_a"] = {"old_controlled_gates": {"pre_transpile": two_qubit_stats(old_qc), "targets": sweep(old_qc, TARGETS)},
                   "structured": {"pre_transpile": two_qubit_stats(new_qc), "targets": sweep(new_qc, TARGETS)}}
for build in ("old_controlled_gates", "structured"):
    for tag, v in OUT["engine_a"][build]["targets"].items():
        log(f"Engine-A {build:20s} {tag:16s} 2q {v['summary']['two_qubit_gates']} depth {v['summary']['depth']}")

# ---------------------------------------------------------------------------
# ULB n=8/L=2 audit instance on the ion-like basis
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
    circ = QuantumCircuit(n + L)
    circ.h(range(n))
    for l in range(L):
        aq = n + l
        circ.ry(prep, aq)
        for e in range(n):
            if abs(h[e]) > 1e-12:
                a = theta * h[e] / scale
                zrot(circ, [e], a / 2)
                zrot(circ, [aq, e], a / 2)
        for (e, f), j in J.items():
            a = theta * j / scale
            zrot(circ, [e, f], a / 2)
            zrot(circ, [aq, e, f], a / 2)
        for e in range(n):
            circ.rx(phi, e)
            circ.h(e)
            zrot(circ, [aq, e], -phi / 2)
            circ.h(e)
        circ.ry(-prep, aq)
    return circ


old8, new8 = old_build(8, 2), new_build(8, 2)
assert float(np.abs(Operator(old8).data - Operator(new8).data).max()) < 1e-9
T8 = {"ion_all_to_all": TARGETS["ion_all_to_all"], "cz_all_to_all": TARGETS["cz_all_to_all"],
      "linear_chain_10": dict(basis_gates=["cz", "rz", "sx", "x"], coupling_map=CouplingMap.from_line(10))}
OUT["ulb_n8_L2"] = {"couplings_thr0.01": len(J),
                    "old_controlled_gates": sweep(old8, T8), "structured": sweep(new8, T8)}
for build in ("old_controlled_gates", "structured"):
    for tag, v in OUT["ulb_n8_L2"][build].items():
        log(f"ULB n8 L2 {build:20s} {tag:16s} 2q {v['summary']['two_qubit_gates']} depth {v['summary']['depth']}")
OUT["note"] = ("In a CX/CZ basis an RZZ costs two CX, the same as a CRZ, so the structured form "
               "doubles the rotation count; it is cheaper only where an arbitrary-angle two-qubit "
               "rotation (rxx/MS/ZZ) is native.")
OUT["wall_seconds"] = time.time() - T0
save_json_no_overwrite(OUT_PATH, OUT)
log(f"DONE -> {OUT_PATH}")
