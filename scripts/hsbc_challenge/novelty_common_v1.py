"""Shared helpers for the 2026-09-04 novelty-dossier experiments.

  * Gram-path expansion of the stacked-LCU syndrome law under per-ancilla
    partial dephasing D_q = (1-q) id + q Delta_Z inserted between SELECT and
    PREP^dagger (exact; polynomial in 1-q of degree <= 2L).
  * Unbiased finite-shot U-statistic for the between-row diversity W.
  * Walsh (Z-string) decomposition of per-view diagonal tables and a
    structured Qiskit circuit builder for the cross-view stack using only
    single-qubit rotations and Pauli-string rotations (no dense unitaries),
    with a labelled identity on every ancilla at the dephasing insertion
    point so Aer noise models can attach the intervention.

Conventions follow hsbc_common.py: little-endian bit order (qubit e <-> bit
e), softmax PREP weights, branch = RX first then diagonal phase, ancilla of
layer l is qubit n + l, pattern bit l = outcome of ancilla l.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from hsbc_common import (
    BatchedSLCU, CrossViewSLCU, _apply_rx_all, _softmax,
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json_no_overwrite(path: str, payload: dict) -> None:
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite {path}")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)


# ---------------------------------------------------------------------------
# Gram-path expansion
# ---------------------------------------------------------------------------

def stride_of(sim) -> int:
    return 6 if isinstance(sim, CrossViewSLCU) else 4


def branch_apply(sim, states, layer, branch, params):
    """Raw (weight-free) branch unitary of `layer` applied to `states`."""
    if isinstance(sim, CrossViewSLCU):
        p = params[6 * layer: 6 * layer + 6]
        return sim._branch_apply(states, layer, branch,
                                 p[2 + 2 * branch], p[3 + 2 * branch])
    p = params[4 * layer: 4 * layer + 4]
    if branch == 0:
        return states * np.exp(-1j * p[2] * sim.cfg.hp)[None, :]
    return _apply_rx_all(states, sim.cfg.n_qubits, p[3])


def path_states(sim, params, states, identical_branches=False):
    """All 2^L branch-path states, shape (2^L, m, D); path index bit l =
    branch chosen at layer l.  With identical_branches=True both branches of
    every layer are branch 0 (reference-circuit control)."""
    paths = [np.asarray(states, complex)]
    for l in range(sim.cfg.n_layers):
        b1 = 0 if identical_branches else 1
        paths = ([branch_apply(sim, s, l, 0, params) for s in paths]
                 + [branch_apply(sim, s, l, b1, params) for s in paths])
    return np.stack(paths)


def pattern_coefficients(params, n_layers, stride) -> np.ndarray:
    """c[a, y] = prod_l coef(a_l, y_l): coef(0,y)=a_y, coef(1,0)=-sqrt(a0 a1),
    coef(1,1)=+sqrt(a0 a1)."""
    A = 2 ** n_layers
    c = np.ones((A, A))
    idx = np.arange(A)
    for l in range(n_layers):
        a = _softmax(np.asarray(params[stride * l: stride * l + 2], float))
        r = np.sqrt(a[0] * a[1])
        coef = np.array([[a[0], a[1]], [-r, r]])
        abits = (idx >> l) & 1
        c *= coef[abits[:, None], abits[None, :]]
    return c


def hamming_matrix(n_layers) -> np.ndarray:
    y = np.arange(2 ** n_layers)
    d = np.zeros((len(y), len(y)), int)
    for l in range(n_layers):
        d += ((y[:, None] >> l) & 1) ^ ((y[None, :] >> l) & 1)
    return d


def path_gram(sim, params, states, identical_branches=False):
    P = path_states(sim, params, states, identical_branches)
    return np.einsum("ymd,zmd->myz", P.conj(), P)          # G[m, y, z] = <path_y|path_z>


def partial_dephased_distribution(sim, params, states, q_values,
                                  identical_branches=False, gram=None):
    """P_q(a|x) for every q: shape (len(q_values), m, 2^L)."""
    L = sim.cfg.n_layers
    G = path_gram(sim, params, states, identical_branches) if gram is None else gram
    c = pattern_coefficients(params, L, stride_of(sim))
    dH = hamming_matrix(L)
    out = np.empty((len(q_values), G.shape[0], 2 ** L))
    for k, q in enumerate(q_values):
        W = (1.0 - float(q)) ** dH
        out[k] = np.einsum("ay,az,yz,myz->ma", c, c, W, G).real
    return out


def phase_twirled_distribution(sim, params, states, rng, n_draws=256):
    """Monte-Carlo average of the coherent syndrome law over a uniform random
    RZ angle on every ancilla between SELECT and PREP^dagger.  The branch-1
    path amplitude of layer l picks up exp(i eta_l); the average of the
    cross terms vanishes, so the limit is the q=1 law."""
    L = sim.cfg.n_layers
    G = path_gram(sim, params, states)
    c = pattern_coefficients(params, L, stride_of(sim))
    y = np.arange(2 ** L)
    acc = np.zeros((G.shape[0], 2 ** L))
    for _ in range(n_draws):
        eta = rng.uniform(0, 2 * np.pi, size=L)
        ph = np.exp(1j * (((y[:, None] >> np.arange(L)[None, :]) & 1) * eta[None, :]).sum(1))
        cph = c * ph[None, :]                                  # c[a,y] e^{i eta.y}
        acc += np.einsum("ay,az,myz->ma", cph, cph.conj(), G).real
    return acc / n_draws


def between_row_diversity(P: np.ndarray) -> float:
    """W = 2/[m(m-1)] sum_{i<j} ||P_i - P_j||^2 for P of shape (m, A)."""
    m = P.shape[0]
    D = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(m, 1)
    return float(D[iu].mean())


def unbiased_between_row_diversity(counts: np.ndarray) -> float:
    """Unbiased U-statistic for W from integer shot counts (m, A)."""
    counts = np.asarray(counts, float)
    S = counts.sum(axis=1)
    phat = counts / S[:, None]
    Q = (counts * (counts - 1.0)).sum(axis=1) / (S * (S - 1.0))
    D = ((phat[:, None, :] - phat[None, :, :]) ** 2).sum(-1)
    corr = (1.0 - Q) / S
    est = D - corr[:, None] - corr[None, :]
    iu = np.triu_indices(counts.shape[0], 1)
    return float(est[iu].mean())


def sample_counts(P: np.ndarray, shots: int, rng) -> np.ndarray:
    """Multinomial shot counts per row for a probability matrix (m, A)."""
    P = np.clip(P, 0.0, None)
    P = P / P.sum(axis=1, keepdims=True)
    return np.stack([rng.multinomial(shots, row) for row in P])


# ---------------------------------------------------------------------------
# Walsh decomposition and structured circuit
# ---------------------------------------------------------------------------

def walsh_z_coefficients(table: np.ndarray, k: int) -> dict:
    """table[j] over j in 0..2^k-1 (bit p of j = bit of view qubit p) as
    E = sum_S c_S prod_{p in S} Z_p, with Z_p eigenvalue +1 on |0>.
    Returns {tuple(sorted S): c_S}."""
    table = np.asarray(table, float)
    idx = np.arange(2 ** k)
    z = 1.0 - 2.0 * ((idx[:, None] >> np.arange(k)[None, :]) & 1)
    out = {}
    for mask in range(2 ** k):
        S = tuple(p for p in range(k) if (mask >> p) & 1)
        chi = np.prod(z[:, list(S)], axis=1) if S else np.ones(2 ** k)
        out[S] = float((table * chi).sum() / 2 ** k)
    return out


def _z_string_rotation(qc, qubits, t):
    """exp(-i t Z_{q1} ... Z_{qk}) with a CX ladder onto the last qubit."""
    qubits = list(qubits)
    if abs(t) < 1e-15:
        return
    if len(qubits) == 1:
        qc.rz(2.0 * t, qubits[0])
        return
    if len(qubits) == 2:
        qc.rzz(2.0 * t, qubits[0], qubits[1])
        return
    for a, b in zip(qubits[:-1], qubits[1:]):
        qc.cx(a, b)
    qc.rz(2.0 * t, qubits[-1])
    for a, b in reversed(list(zip(qubits[:-1], qubits[1:]))):
        qc.cx(a, b)


def _controlled_z_string_rotation(qc, anc, ctrl_value, qubits, alpha):
    """exp(-i alpha Pi_v(anc) (x) Z_S) = exp(-i alpha/2 Z_S) exp(-/+ i alpha/2 Z_anc Z_S),
    minus sign for ctrl_value=0, plus for ctrl_value=1."""
    _z_string_rotation(qc, qubits, alpha / 2.0)
    sgn = -1.0 if ctrl_value == 1 else 1.0
    _z_string_rotation(qc, [anc] + list(qubits), sgn * alpha / 2.0)


def _controlled_rx(qc, anc, ctrl_value, q, phi):
    """Controlled exp(-i phi X_q): exp(-i phi/2 X_q) exp(-/+ i phi/2 Z_anc X_q)."""
    qc.rx(phi, q)
    sgn = -1.0 if ctrl_value == 1 else 1.0
    qc.h(q)
    qc.rzz(2.0 * sgn * phi / 2.0, anc, q)
    qc.h(q)


def build_structured_crossview_circuit(cfg, params, u_row=None, walsh=None,
                                       identical_branches=False,
                                       dephase_marker=True,
                                       measure_ancillas=False,
                                       init_state=None):
    """Structured Qiskit circuit for a CrossViewConfig stack.

    walsh: dict view -> {S: c_S} over the view's local bit positions (from
    walsh_z_coefficients of the 2^k view table).  u_row: quantile features
    (n,) for the product-state input via RY(pi u); or init_state (2^n) via
    initialize (referee use only).
    """
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    n, L = cfg.n_qubits, cfg.n_layers
    qr = QuantumRegister(n + L, "q")
    regs = [qr]
    cr = None
    if measure_ancillas:
        cr = ClassicalRegister(L, "c")
        regs.append(cr)
    qc = QuantumCircuit(*regs)
    if init_state is not None:
        # exact unitary whose first column is the state (Aer density-matrix
        # method rejects `initialize`; a dense unitary is supported)
        from qiskit.circuit.library import UnitaryGate
        psi = np.asarray(init_state, complex)
        psi = psi / np.linalg.norm(psi)
        D = len(psi)
        rng_u = np.random.default_rng(int(abs(psi[0].real) * 1e6) % 100000)
        A = rng_u.normal(size=(D, D)) + 1j * rng_u.normal(size=(D, D))
        A[:, 0] = psi
        Qm, R = np.linalg.qr(A)
        Qm[:, 0] *= np.exp(1j * np.angle(R[0, 0])) if abs(R[0, 0]) > 0 else 1.0
        # ensure first column equals psi exactly up to phase; fix phase
        ph = np.vdot(Qm[:, 0], psi)
        Qm[:, 0] *= ph / abs(ph)
        assert np.abs(Qm[:, 0] - psi).max() < 1e-12, "state embedding failed"
        qc.append(UnitaryGate(Qm), list(range(n)))
    else:
        for e in range(n):
            qc.ry(np.pi * float(u_row[e]), e)
    for l in range(L):
        p = params[6 * l: 6 * l + 6]
        a = _softmax(np.asarray(p[:2], float))
        prep = 2.0 * np.arctan2(np.sqrt(a[1]), np.sqrt(a[0]))
        anc = n + l
        qc.ry(prep, anc)
        vpair = cfg.layer_pairs[l]
        consts = []
        for branch in range(2):
            view = vpair[0] if (branch == 0 or identical_branches) else vpair[1]
            theta = p[2] if (branch == 0 or identical_branches) else p[4]
            phi = p[3] if (branch == 0 or identical_branches) else p[5]
            qubits = list(cfg.views[view])
            # mixer first
            for q in qubits:
                _controlled_rx(qc, anc, branch, q, phi)
            # then diagonal phase exp(-i theta sum_S c_S Z_S)
            for S, c in walsh[view].items():
                if len(S) == 0:
                    consts.append(theta * c)
                    continue
                _controlled_z_string_rotation(
                    qc, anc, branch, [qubits[pos] for pos in S], theta * c)
        # relative branch phase from the constant terms: diag(e^{-i c0}, e^{-i c1})
        qc.p(-(consts[1] - consts[0]), anc)
        if dephase_marker:
            qc.id(anc)
        qc.ry(-prep, anc)
    if measure_ancillas:
        for l in range(L):
            qc.measure(n + l, l)
    return qc


def crossview_walsh(cfg) -> dict:
    """Walsh coefficients of every view table from the full-space diagonal."""
    out = {}
    for v, qubits in enumerate(cfg.views):
        k = len(qubits)
        # recover the 2^k local table from the broadcast full-space diagonal
        idx = np.zeros(2 ** k, int)
        for j in range(2 ** k):
            full = 0
            for pos, q in enumerate(qubits):
                full |= ((j >> pos) & 1) << q
            idx[j] = full
        table = np.asarray(cfg.hp_views[v])[idx]
        out[v] = walsh_z_coefficients(table, k)
    return out


def counts_to_pattern_probs(counts: dict, n_layers: int, shots: int) -> np.ndarray:
    """Qiskit counts (keys little-endian strings, cbit 0 rightmost) ->
    pattern probability vector with pattern bit l = ancilla l."""
    out = np.zeros(2 ** n_layers)
    for key, c in counts.items():
        bits = key.replace(" ", "")[::-1]
        pat = sum(int(bits[l]) << l for l in range(n_layers))
        out[pat] += c
    return out / shots
