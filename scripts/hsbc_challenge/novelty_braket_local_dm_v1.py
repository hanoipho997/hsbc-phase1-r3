"""Candidate A, Braket-SDK stage: reduced instance (6 system + 3 ancilla
qubits, three 2-qubit views, L=3) of the stacked cross-view LCU built with the
Amazon Braket SDK and executed on the Braket LOCAL density-matrix simulator
("braket_dm") with the partial-dephasing intervention as an explicit
PhaseFlip(q/2) channel on each ancilla between SELECT and PREP^dagger, plus
an optional depolarizing gate-noise stand-in.

Runs in an isolated venv containing only numpy + amazon-braket-sdk; the
repository is not imported.  Input:
runs/hsbc_challenge/novelty_v1/braket_reduced_instance_v1.npz (written by
novelty_dephasing_witness_v1.py, contains the exact Gram-path P_q reference).

This is NOT a Braket managed-simulator (DM1) or QPU execution; swapping the
device object (`AwsDevice(arn)`) is the only change needed for that stage,
which requires AWS credentials and budget approval.

Artifacts -> runs/hsbc_challenge/novelty_v1/braket_local_dm_v1.json
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import time

import numpy as np
from braket.circuits import Circuit, Gate, Noise
from braket.devices import LocalSimulator

T0 = time.time()
INP = "runs/hsbc_challenge/novelty_v1/braket_reduced_instance_v1.npz"
OUT_PATH = "runs/hsbc_challenge/novelty_v1/braket_local_dm_v1.json"
if os.path.exists(OUT_PATH):
    raise FileExistsError(OUT_PATH)


def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


d = np.load(INP, allow_pickle=True)
q_grid = [float(q) for q in d["q_grid"]]
U = d["U"]
params = d["params"]
P_exact = d["P_exact"]                      # (len q, rows, 2^L)
views = [tuple(int(x) for x in v) for v in d["views"]]
pairs = [tuple(int(x) for x in p) for p in d["pairs"]]
walsh = [{ast.literal_eval(k): float(v) for k, v in json.loads(s).items()} for s in d["walsh_keys"]]
n = sum(len(v) for v in views)
L = len(pairs)
import braket
OUT = {"schema": "hsbc-novelty-braket-local-dm-v1", "input": INP, "input_sha256": sha256_file(INP),
       "braket_sdk_version": getattr(braket, "__version__", "unknown"),
       "device": "LocalSimulator('braket_dm')", "n_system": n, "n_layers": L, "q_grid": q_grid}


def softmax(v):
    e = np.exp(v - np.max(v))
    return e / e.sum()


def zstring(c, qubits, t):
    """exp(-i t Z...Z) on the listed qubits."""
    if abs(t) < 1e-15:
        return
    if len(qubits) == 1:
        c.rz(qubits[0], 2 * t)
    elif len(qubits) == 2:
        c.zz(qubits[0], qubits[1], 2 * t)
    else:
        for a, b in zip(qubits[:-1], qubits[1:]):
            c.cnot(a, b)
        c.rz(qubits[-1], 2 * t)
        for a, b in reversed(list(zip(qubits[:-1], qubits[1:]))):
            c.cnot(a, b)


def build(u_row, q, p1=0.0, p2=0.0, identical=False):
    c = Circuit()
    for e in range(n):
        c.ry(e, np.pi * float(u_row[e]))
    for l in range(L):
        p = params[6 * l: 6 * l + 6]
        a = softmax(p[:2])
        prep = 2.0 * np.arctan2(np.sqrt(a[1]), np.sqrt(a[0]))
        anc = n + l
        c.ry(anc, prep)
        consts = []
        for branch in range(2):
            view = pairs[l][0] if (branch == 0 or identical) else pairs[l][1]
            theta = p[2] if (branch == 0 or identical) else p[4]
            phi = p[3] if (branch == 0 or identical) else p[5]
            sgn = -1.0 if branch == 1 else 1.0
            qubits = list(views[view])
            for qb in qubits:                       # controlled RX(2 phi)
                c.rx(qb, phi)
                c.h(qb)
                c.zz(anc, qb, sgn * phi)            # exp(-i (sgn phi/2) Z_anc Z_q) -> zz(angle = sgn*phi)
                c.h(qb)
            for S, coef in walsh[view].items():
                if len(S) == 0:
                    consts.append(theta * coef)
                    continue
                alpha = theta * coef
                zstring(c, [qubits[pos] for pos in S], alpha / 2)
                zstring(c, [anc] + [qubits[pos] for pos in S], sgn * alpha / 2)
        c.phaseshift(anc, -(consts[1] - consts[0]))
        if q > 0:
            c.phase_flip(anc, q / 2)
        c.ry(anc, -prep)
    return c


def with_noise(c, p1, p2):
    if p1 > 0:
        c = c.copy()
        c.apply_gate_noise(Noise.Depolarizing(p1), target_gates=[Gate.Rx, Gate.Ry, Gate.H])
    if p2 > 0:
        c = c.copy()
        c.apply_gate_noise(Noise.TwoQubitDepolarizing(p2), target_gates=[Gate.ZZ, Gate.CNot])
    return c


def pattern_probs(vals, targets):
    """Braket Probability result over `targets` (first target = most
    significant bit) -> pattern vector with bit l = ancilla l."""
    vals = np.asarray(vals)
    A = len(targets)
    out = np.zeros(2 ** A)
    for idx in range(2 ** A):
        bits = [(idx >> (A - 1 - k)) & 1 for k in range(A)]      # big-endian per target order
        pat = sum(bits[k] << k for k in range(A))
        out[pat] = vals[idx]
    return out


def W(P):
    m = P.shape[0]
    D = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(m, 1)
    return float(D[iu].mean())


dev = LocalSimulator("braket_dm")
ancs = list(range(n, n + L))
res = {"ideal_exact_dm": {}, "noise_standins": {}}
worst = 0.0
Wq = []
for k, q in enumerate(q_grid):
    Pk = []
    for i in range(U.shape[0]):
        c = build(U[i], q)
        c.probability(target=ancs)
        r = dev.run(c, shots=0).result()
        Pk.append(pattern_probs(r.values[0], ancs))
    Pk = np.stack(Pk)
    worst = max(worst, float(np.abs(Pk - P_exact[k]).max()))
    Wq.append(W(Pk))
    log(f"q={q}: max |P_dm - P_exact| = {np.abs(Pk - P_exact[k]).max():.2e}; W = {Wq[-1]:.5g}")
res["ideal_exact_dm"] = {"max_dev_vs_gram_path": worst, "W": Wq,
                         "W_exact_reference": [W(P) for P in P_exact]}
res["circuit_summary"] = {"qubits": n + L, "instructions": len(build(U[0], 0.0).instructions),
                          "two_qubit_gates": int(sum(1 for ins in build(U[0], 0.0).instructions
                                                     if len(ins.target) == 2))}
# identical-branch reference law
ref_dev = 0.0
for k, q in enumerate(q_grid):
    c = build(U[0], q, identical=True)
    c.probability(target=ancs)
    P = pattern_probs(dev.run(c, shots=0).result().values[0], ancs)
    for l in range(L):
        a = softmax(params[6 * l: 6 * l + 2])
        pf = P[((np.arange(2 ** L) >> l) & 1) == 1].sum()
        ref_dev = max(ref_dev, abs(pf - 2 * a[0] * a[1] * q))
res["identical_branch_reference_law_max_dev"] = float(ref_dev)
log(f"identical-branch reference law max dev {ref_dev:.2e}")

for tag, (p1, p2) in {"low": (5e-4, 5e-3), "high": (2e-3, 2e-2)}.items():
    Wn = []
    for k, q in enumerate(q_grid):
        Pk = []
        for i in range(U.shape[0]):
            c = with_noise(build(U[i], q), p1, p2)
            c.probability(target=ancs)
            Pk.append(pattern_probs(dev.run(c, shots=0).result().values[0], ancs))
        Wn.append(W(np.stack(Pk)))
    res["noise_standins"][tag] = {"p1": p1, "p2": p2, "W_exact_under_noise": Wn}
    log(f"noise {tag}: W(q) = {['%.5g' % w for w in Wn]}")

# finite-shot sampling on the local DM simulator at S=2000 (q=0 and q=1)
S = 2000
samp = {}
for q in (0.0, 1.0):
    counts = []
    for i in range(U.shape[0]):
        c = build(U[i], q)
        r = dev.run(c, shots=S).result()
        meas = np.asarray(r.measurements)[:, ancs]          # (S, L) with column k = ancilla k
        pat = (meas * (1 << np.arange(L))[None, :]).sum(axis=1)
        counts.append(np.bincount(pat, minlength=2 ** L))
    counts = np.stack(counts).astype(float)
    Sv = counts.sum(axis=1)
    phat = counts / Sv[:, None]
    Q = (counts * (counts - 1)).sum(axis=1) / (Sv * (Sv - 1))
    D = ((phat[:, None, :] - phat[None, :, :]) ** 2).sum(-1)
    corr = (1 - Q) / Sv
    est = D - corr[:, None] - corr[None, :]
    iu = np.triu_indices(counts.shape[0], 1)
    samp[str(q)] = float(est[iu].mean())
res["sampled_S2000_W_hat"] = samp
OUT["results"] = res
OUT["wall_seconds"] = time.time() - T0
with open(OUT_PATH, "w") as fh:
    json.dump(OUT, fh, indent=2)
log(f"DONE -> {OUT_PATH}")
