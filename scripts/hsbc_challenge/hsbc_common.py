"""Shared library for the HSBC fraud-detection fit assessment.

Implements the one-class spectral-filter scorer built from this repo's
stacked-LCU machinery, plus the data layer and classical references needed
for a matched E.ON-vs-HSBC comparison:

  * ULB European Cardholder data (OpenML id 1597), stratified splits,
    single-feature-AUC feature selection, per-feature quantile transform.
  * Quantile-angle product encoding  |phi_i(x)> = cos(pi u/2)|0> + sin(pi u/2)|1>.
  * Pseudo-likelihood Ising fit of legitimate behaviour:
      E(z) = -(sum_i a_i z_i + sum_{i<j} b_ij z_i z_j),  z in {+-1}^n,
    fitted per-bit with L2 logistic regression on legit-only training rows.
  * Diagonal filter scorers (classical closed forms, Prop 1) and the trained
    S-LCU scorer  score(x) = ||A(params)|phi(x)>||^2 = p_s(x).
  * A batched structured simulator for the E.ON ansatz family
    (per-layer LCU of [e^{-i theta H'}, RX(phi)^{otimes n}]) with analytic
    parameter derivatives, verified against DifferentiableStackedLCU.

Conventions follow scripts/eon_challenge/eon_slcu_feasibility.py:
little-endian bit order (qubit e <-> bit e of the basis index), softmax
PREP weights, and the truncated-Cauchy LCHS quadrature of eon_lchs_probe.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from fqe_ising.stacked_lcu import (
    DifferentiableStackedLCU,
    DifferentiableStackedLCULayer,
    DifferentiableUnitaryTerm,
)

# ---------------------------------------------------------------------------
# data layer
# ---------------------------------------------------------------------------

ULB_NPZ = "runs/hsbc_challenge/data/ulb_creditcard.npz"


def load_ulb(path: str = ULB_NPZ):
    d = np.load(path, allow_pickle=True)
    X = np.asarray(d["X"], float)
    y = np.asarray(d["y"], int)
    cols = [str(c) for c in d["cols"]]
    return X, y, cols


def stratified_split(y: np.ndarray, fracs=(0.6, 0.2, 0.2), seed: int = 0):
    """Deterministic stratified train/val/test index split."""
    rng = np.random.default_rng(seed)
    idx_tr, idx_va, idx_te = [], [], []
    for cls in np.unique(y):
        ids = np.flatnonzero(y == cls)
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(fracs[0] * n))
        n_va = int(round(fracs[1] * n))
        idx_tr.append(ids[:n_tr])
        idx_va.append(ids[n_tr : n_tr + n_va])
        idx_te.append(ids[n_tr + n_va :])
    return (np.sort(np.concatenate(idx_tr)),
            np.sort(np.concatenate(idx_va)),
            np.sort(np.concatenate(idx_te)))


def single_feature_auc(x: np.ndarray, y: np.ndarray) -> float:
    """AUC-ROC of a single raw feature (rank statistic, ties averaged)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(auc)


def select_features(X, y, n_feat: int) -> np.ndarray:
    """Top-n features by |single-feature AUC - 0.5| on the given rows."""
    scores = np.array(
        [abs(single_feature_auc(X[:, j], y) - 0.5) for j in range(X.shape[1])]
    )
    return np.argsort(-scores)[:n_feat]


class QuantileTransform:
    """Per-feature empirical CDF u in [0,1], fitted on training rows."""

    def __init__(self, X_train: np.ndarray):
        self.sorted_cols = [np.sort(X_train[:, j]) for j in range(X_train.shape[1])]

    def __call__(self, X: np.ndarray) -> np.ndarray:
        U = np.empty_like(X, dtype=float)
        for j, col in enumerate(self.sorted_cols):
            # midrank CDF: average of left and right insertion positions
            lo = np.searchsorted(col, X[:, j], side="left")
            hi = np.searchsorted(col, X[:, j], side="right")
            U[:, j] = 0.5 * (lo + hi) / len(col)
        return np.clip(U, 0.0, 1.0)


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def hard_bits(U: np.ndarray) -> np.ndarray:
    """Median split: bit 1 iff u > 1/2 (z = +1 <-> bit 0)."""
    return (U > 0.5).astype(int)


def angle_amplitudes(U: np.ndarray):
    """Per-qubit amplitudes (amp0, amp1) of the quantile-angle encoding."""
    alpha = np.pi * U
    return np.cos(alpha / 2.0), np.sin(alpha / 2.0)


def product_state_batch(U: np.ndarray) -> np.ndarray:
    """Return (m, 2^n) matrix of encoded product-state amplitudes.

    Index bit e (little-endian) is qubit e, matching the E.ON scripts.
    """
    a0, a1 = angle_amplitudes(U)
    m, n = U.shape
    amps = np.ones((m, 1))
    for e in range(n):
        amps = np.concatenate(
            [amps * a0[:, e : e + 1], amps * a1[:, e : e + 1]], axis=1
        )
    return amps


def product_probs_batch(U: np.ndarray) -> np.ndarray:
    """(m, 2^n) matrix of basis-state probabilities |<z|phi(x)>|^2."""
    amps = product_state_batch(U)
    return amps ** 2


# ---------------------------------------------------------------------------
# Ising pseudo-likelihood fit (legit-only)
# ---------------------------------------------------------------------------

@dataclass
class IsingModel:
    a: np.ndarray                # fields, shape (n,)
    b: np.ndarray                # couplings, symmetric, zero diagonal, (n, n)
    energies: np.ndarray = field(default=None)   # all 2^n raw energies
    hp: np.ndarray = field(default=None)         # normalized to [0, 1]

    @property
    def n(self) -> int:
        return len(self.a)

    def energy_bits(self, bits: np.ndarray) -> np.ndarray:
        """Raw energy of hard bit rows (m, n); z = 1 - 2*bit."""
        z = 1.0 - 2.0 * bits
        return -(z @ self.a + 0.5 * np.einsum("mi,ij,mj->m", z, self.b, z))

    def tabulate(self) -> None:
        n = self.n
        idx = np.arange(2 ** n)
        zbits = ((idx[:, None] >> np.arange(n)[None, :]) & 1)
        z = 1.0 - 2.0 * zbits
        self.energies = -(z @ self.a + 0.5 * np.einsum("mi,ij,mj->m", z, self.b, z))
        span = float(self.energies.max() - self.energies.min())
        self.hp = (self.energies - self.energies.min()) / span


def fit_ising_pseudolikelihood(bits: np.ndarray, l2_c: float = 1.0,
                               max_rows: int = 80_000, seed: int = 0) -> IsingModel:
    """Per-bit logistic regression fit of P(z_i=+1 | z_-i) = sigma(2(a_i + sum_j b_ij z_j))."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    if len(bits) > max_rows:
        bits = bits[rng.choice(len(bits), max_rows, replace=False)]
    z = 1.0 - 2.0 * bits.astype(float)
    n = z.shape[1]
    a = np.zeros(n)
    b = np.zeros((n, n))
    for i in range(n):
        target = (z[:, i] > 0).astype(int)
        others = np.delete(np.arange(n), i)
        clf = LogisticRegression(C=l2_c, max_iter=2000)
        clf.fit(z[:, others], target)
        a[i] = 0.5 * float(clf.intercept_[0])
        b[i, others] = 0.5 * clf.coef_[0]
    b = 0.5 * (b + b.T)
    np.fill_diagonal(b, 0.0)
    model = IsingModel(a=a, b=b)
    model.tabulate()
    return model


# ---------------------------------------------------------------------------
# diagonal filter scorers (classical closed forms; Prop 1)
# ---------------------------------------------------------------------------

def lchs_cauchy_filter(hp: np.ndarray, tau: float, nodes: int, cutoff: float):
    """Truncated-Cauchy trapezoid LCHS filter F approx e^{-tau hp} (eon_lchs_probe.py).

    Returns (F complex vector over the spectrum, 1-norm of the weights).
    """
    k = np.linspace(-cutoff, cutoff, nodes)
    w = (2 * cutoff / (nodes - 1)) / (np.pi * (1 + k ** 2))
    F = (w[:, None] * np.exp(1j * np.outer(k, tau * hp))).sum(axis=0)
    return F, float(np.abs(w).sum())


def diagonal_ps_scores(probs: np.ndarray, filter_diag: np.ndarray) -> np.ndarray:
    """p_s(x) = E_{z~q_x}[|f(E(z))|^2] for a diagonal filter (Prop 1 closed form)."""
    g = np.abs(filter_diag) ** 2
    return probs @ g


# ---------------------------------------------------------------------------
# batched structured S-LCU simulator (E.ON ansatz family)
# ---------------------------------------------------------------------------

@dataclass
class SLCUStackConfig:
    """L layers; each layer = softmax(logits) LCU of [e^{-i theta H'}, RX(phi)^n].

    Per-layer parameter block: [logit0, logit1, theta, phi]  (4 per layer),
    identical to the E.ON feasibility ansatz.
    """
    n_qubits: int
    n_layers: int
    hp: np.ndarray  # diagonal of H' over the 2^n basis (any real normalization)

    @property
    def n_params(self) -> int:
        return 4 * self.n_layers


def _softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


def _apply_rx_all(states: np.ndarray, n: int, phi: float,
                  deriv: bool = False) -> np.ndarray:
    """Apply RX(2*phi)^{otimes n} (matching the E.ON convention rx(phi) with
    entries cos(phi), -i sin(phi)) to a batch of states, shape (m, 2^n).

    With deriv=True returns d/dphi of the applied result.
    """
    c, s = np.cos(phi), np.sin(phi)
    out = states
    m = states.shape[0]
    for e in range(n):
        v = out.reshape(m, -1, 2, 2 ** e)
        v0 = v[:, :, 0, :]
        v1 = v[:, :, 1, :]
        new = np.empty_like(v)
        new[:, :, 0, :] = c * v0 - 1j * s * v1
        new[:, :, 1, :] = -1j * s * v0 + c * v1
        out = new.reshape(m, -1)
    if not deriv:
        return out
    # d/dphi of a tensor product: sum over qubits of the single-qubit derivative
    dtotal = np.zeros_like(out)
    for e in range(n):
        v = out.reshape(m, -1, 2, 2 ** e)  # differentiate qubit e *after* full apply:
        # (d RX/dphi) RX^{-1} = -i X  acting on qubit e
        v0 = v[:, :, 0, :]
        v1 = v[:, :, 1, :]
        dv = np.empty_like(v)
        dv[:, :, 0, :] = -1j * v1
        dv[:, :, 1, :] = -1j * v0
        dtotal += dv.reshape(m, -1)
    return dtotal


class BatchedSLCU:
    """Fast batched forward/derivative evaluation of the E.ON stack.

    States are (m, 2^n) rows. Forward output is A(params) applied to each row;
    derivatives are with respect to the 4L stack parameters. Verified against
    DifferentiableStackedLCU (see verify_against_repo).
    """

    def __init__(self, config: SLCUStackConfig):
        self.cfg = config

    def _layer_apply(self, states, logits, theta, phi, which=None):
        """Apply one layer; `which` in {None,'logit0','logit1','theta','phi',
        'fail'} selects the derivative or failure-sector operator instead of
        the plain (success) operator.

        Success block  K = a0*Cost + a1*Mix   (softmax weights a).
        Failure block  F = sqrt(a0*a1)*(Mix - Cost): the <1|W|0> block of the
        PREP-SELECT-PREP^dagger layer with PREP|0> = sqrt(a0)|0>+sqrt(a1)|1>.
        Note d K / d logit0 = a0*a1*(Cost - Mix) = -sqrt(a0*a1) * F, the
        logit-coordinate form of the failure-sector = derivative identity.
        """
        a = _softmax(np.asarray(logits, float))
        phase = np.exp(-1j * theta * self.cfg.hp)[None, :]
        cost = states * phase
        mix = _apply_rx_all(states, self.cfg.n_qubits, phi)
        if which is None:
            return a[0] * cost + a[1] * mix
        if which == "fail":
            return np.sqrt(a[0] * a[1]) * (mix - cost)
        if which in ("logit0", "logit1"):
            k = 0 if which == "logit0" else 1
            # d a / d logit_k = a * (delta - a[k])
            da = a * ((np.arange(2) == k).astype(float) - a[k])
            return da[0] * cost + da[1] * mix
        if which == "theta":
            return a[0] * (cost * (-1j * self.cfg.hp)[None, :])
        if which == "phi":
            return a[1] * _apply_rx_all(states, self.cfg.n_qubits, phi, deriv=True)
        raise ValueError(which)

    def forward(self, params: np.ndarray, states: np.ndarray) -> np.ndarray:
        out = states
        for l in range(self.cfg.n_layers):
            p = params[4 * l : 4 * l + 4]
            out = self._layer_apply(out, p[:2], p[2], p[3])
        return out

    # ---- ancilla-syndrome machinery -------------------------------------
    #
    # One circuit's joint Z-basis ancilla record carries the full pattern
    # distribution P(a|x) over a in {0,1}^L, where pattern a applies the
    # sector chain  M_L^{a_L} ... M_1^{a_1} |phi(x)>  with M^0 = K (success
    # block) and M^1 = F (failure block).  P(a|x) = ||psi_a||^2, and
    # completeness (K^dag K + F^dag F = I per layer) guarantees
    # sum_a P(a|x) = 1 for normalized inputs.  P(0...0|x) equals the p_s
    # returned by `forward`.  Pattern index convention: bit l (little-endian)
    # = outcome of layer-l's ancilla.

    def syndrome_sector_states(self, params: np.ndarray,
                               states: np.ndarray) -> np.ndarray:
        """All 2^L sector states, shape (2^L, m, D)."""
        sectors = [states]
        for l in range(self.cfg.n_layers):
            p = params[4 * l : 4 * l + 4]
            new = [None] * (2 * len(sectors))
            for i, s in enumerate(sectors):
                new[i] = self._layer_apply(s, p[:2], p[2], p[3])
                new[i + len(sectors)] = self._layer_apply(
                    s, p[:2], p[2], p[3], which="fail")
            sectors = new
        return np.stack(sectors)

    def syndrome_distribution(self, params: np.ndarray,
                              states: np.ndarray) -> np.ndarray:
        """P(a|x) for all patterns, shape (m, 2^L); row sums = 1."""
        sec = self.syndrome_sector_states(params, states)
        return np.einsum("pmd,pmd->mp", sec.conj(), sec).real

    def syndrome_features(self, params: np.ndarray, states: np.ndarray,
                          eps: float = 1e-6) -> np.ndarray:
        """Log-probability syndrome features log(P(a|x)+eps), shape (m, 2^L)."""
        return np.log(self.syndrome_distribution(params, states) + eps)


    def forward_and_derivatives(self, params: np.ndarray, states: np.ndarray):
        """Return (A states, dA states) with derivative axis first, shape (P, m, D)."""
        cfg = self.cfg
        m, D = states.shape
        # forward prefix states after each layer
        prefix = [states]
        for l in range(cfg.n_layers):
            p = params[4 * l : 4 * l + 4]
            prefix.append(self._layer_apply(prefix[-1], p[:2], p[2], p[3]))
        derivs = np.zeros((cfg.n_params, m, D), complex)
        for l in range(cfg.n_layers):
            p = params[4 * l : 4 * l + 4]
            for j, which in enumerate(("logit0", "logit1", "theta", "phi")):
                d = self._layer_apply(prefix[l], p[:2], p[2], p[3], which=which)
                for l2 in range(l + 1, cfg.n_layers):
                    p2 = params[4 * l2 : 4 * l2 + 4]
                    d = self._layer_apply(d, p2[:2], p2[2], p2[3])
                derivs[4 * l + j] = d
        return prefix[-1], derivs


# ---------------------------------------------------------------------------
# cross-view stacked LCU (Engine-A preregistration section 5; milestone M1)
# ---------------------------------------------------------------------------

def _apply_rx_subset(states: np.ndarray, qubits, phi: float,
                     deriv: bool = False) -> np.ndarray:
    """Apply RX(2*phi) (entries cos phi, -i sin phi) on the listed qubits of a
    batch of states, shape (m, 2^n).  With deriv=True return d/dphi of the
    applied result (sum of -iX insertions over the subset)."""
    c, s = np.cos(phi), np.sin(phi)
    out = states
    m = states.shape[0]
    for e in qubits:
        v = out.reshape(m, -1, 2, 2 ** e)
        v0, v1 = v[:, :, 0, :], v[:, :, 1, :]
        new = np.empty_like(v)
        new[:, :, 0, :] = c * v0 - 1j * s * v1
        new[:, :, 1, :] = -1j * s * v0 + c * v1
        out = new.reshape(m, -1)
    if not deriv:
        return out
    dtotal = np.zeros_like(out)
    for e in qubits:
        v = out.reshape(m, -1, 2, 2 ** e)
        v0, v1 = v[:, :, 0, :], v[:, :, 1, :]
        dv = np.empty_like(v)
        dv[:, :, 0, :] = -1j * v1
        dv[:, :, 1, :] = -1j * v0
        dtotal += dv.reshape(m, -1)
    return dtotal


def broadcast_view_diagonal(table: np.ndarray, qubits, n_qubits: int) -> np.ndarray:
    """Lift a per-view energy table over the view's bits (little-endian within
    the listed qubit order) to a full-space diagonal of length 2^n."""
    idx = np.arange(2 ** n_qubits)
    sub = np.zeros_like(idx)
    for pos, q in enumerate(qubits):
        sub |= (((idx >> q) & 1) << pos)
    return np.asarray(table, float)[sub]


@dataclass
class CrossViewConfig:
    """L cross-view layers; layer j acts on view pair (v_j, w_j) with branches
    U_{j0} = e^{-i theta_{j0} H_{v_j}} RX_{v_j}(phi_{j0}) and
    U_{j1} = e^{-i theta_{j1} H_{w_j}} RX_{w_j}(phi_{j1})
    (rightmost acts first: phase after mixer).  Per-layer parameter block:
    [logit0, logit1, theta0, phi0, theta1, phi1] (6 per layer), matching the
    repo DifferentiableStackedLCULayer ordering [logits..., term params...].

    `hp_views[v]` is the FULL-SPACE diagonal of the view Hamiltonian
    (use broadcast_view_diagonal; Engine-A convention: per-view [0,1] span
    normalization, recorded as prereg clarification A1).
    """
    n_qubits: int
    views: tuple
    layer_pairs: tuple
    hp_views: tuple

    @property
    def n_layers(self) -> int:
        return len(self.layer_pairs)

    @property
    def n_params(self) -> int:
        return 6 * len(self.layer_pairs)


class CrossViewSLCU:
    """Batched cross-view stacked LCU with the same API as BatchedSLCU
    (forward / forward_and_derivatives / syndrome_*), verified against a
    repo-dense twin and a Qiskit circuit in verify_crossview_v1.py."""

    def __init__(self, config: CrossViewConfig):
        self.cfg = config

    def _branch_apply(self, states, layer, branch, theta, phi, dtheta=False,
                      dphi=False):
        v, w = self.cfg.layer_pairs[layer]
        view = v if branch == 0 else w
        qubits = self.cfg.views[view]
        hp = self.cfg.hp_views[view]
        if dphi:
            mixed = _apply_rx_subset(states, qubits, phi, deriv=True)
        else:
            mixed = _apply_rx_subset(states, qubits, phi)
        out = mixed * np.exp(-1j * theta * hp)[None, :]
        if dtheta:
            out = out * (-1j * hp)[None, :]
        return out

    def _layer_apply(self, states, layer, params_l, which=None):
        a = _softmax(np.asarray(params_l[:2], float))
        t0, f0, t1, f1 = params_l[2], params_l[3], params_l[4], params_l[5]
        if which in (None, "fail", "logit0", "logit1"):
            b0 = self._branch_apply(states, layer, 0, t0, f0)
            b1 = self._branch_apply(states, layer, 1, t1, f1)
            if which is None:
                return a[0] * b0 + a[1] * b1
            if which == "fail":
                return np.sqrt(a[0] * a[1]) * (b1 - b0)
            k = 0 if which == "logit0" else 1
            da = a * ((np.arange(2) == k).astype(float) - a[k])
            return da[0] * b0 + da[1] * b1
        if which == "theta0":
            return a[0] * self._branch_apply(states, layer, 0, t0, f0, dtheta=True)
        if which == "phi0":
            return a[0] * self._branch_apply(states, layer, 0, t0, f0, dphi=True)
        if which == "theta1":
            return a[1] * self._branch_apply(states, layer, 1, t1, f1, dtheta=True)
        if which == "phi1":
            return a[1] * self._branch_apply(states, layer, 1, t1, f1, dphi=True)
        raise ValueError(which)

    def forward(self, params: np.ndarray, states: np.ndarray) -> np.ndarray:
        out = states
        for l in range(self.cfg.n_layers):
            out = self._layer_apply(out, l, params[6 * l: 6 * l + 6])
        return out

    def forward_and_derivatives(self, params: np.ndarray, states: np.ndarray):
        cfg = self.cfg
        m, D = states.shape
        prefix = [states]
        for l in range(cfg.n_layers):
            prefix.append(self._layer_apply(prefix[-1], l,
                                            params[6 * l: 6 * l + 6]))
        derivs = np.zeros((cfg.n_params, m, D), complex)
        names = ("logit0", "logit1", "theta0", "phi0", "theta1", "phi1")
        for l in range(cfg.n_layers):
            for j, which in enumerate(names):
                d = self._layer_apply(prefix[l], l, params[6 * l: 6 * l + 6],
                                      which=which)
                for l2 in range(l + 1, cfg.n_layers):
                    d = self._layer_apply(d, l2, params[6 * l2: 6 * l2 + 6])
                derivs[6 * l + j] = d
        return prefix[-1], derivs

    def syndrome_sector_states(self, params: np.ndarray,
                               states: np.ndarray) -> np.ndarray:
        sectors = [states]
        for l in range(self.cfg.n_layers):
            p = params[6 * l: 6 * l + 6]
            new = [None] * (2 * len(sectors))
            for i, s in enumerate(sectors):
                new[i] = self._layer_apply(s, l, p)
                new[i + len(sectors)] = self._layer_apply(s, l, p, which="fail")
            sectors = new
        return np.stack(sectors)

    def syndrome_distribution(self, params: np.ndarray,
                              states: np.ndarray) -> np.ndarray:
        sec = self.syndrome_sector_states(params, states)
        return np.einsum("pmd,pmd->mp", sec.conj(), sec).real

    def syndrome_features(self, params: np.ndarray, states: np.ndarray,
                          eps: float = 1e-6) -> np.ndarray:
        return np.log(self.syndrome_distribution(params, states) + eps)


def crossview_repo_stack(config: CrossViewConfig) -> DifferentiableStackedLCU:
    """Independent dense twin of CrossViewSLCU through the repository's
    differentiable classes (verification referee; small sizes only)."""
    from functools import reduce
    D = 2 ** config.n_qubits
    I2 = np.eye(2)
    X = np.array([[0.0, 1.0], [1.0, 0.0]])

    def rx_dense(qubits, phi):
        rx = np.array([[np.cos(phi), -1j * np.sin(phi)],
                       [-1j * np.sin(phi), np.cos(phi)]])
        factors = [rx if q in qubits else I2
                   for q in reversed(range(config.n_qubits))]
        return reduce(np.kron, factors).astype(complex)

    def xsum_dense(qubits):
        out = np.zeros((D, D), complex)
        for e in qubits:
            factors = [X if q == e else I2
                       for q in reversed(range(config.n_qubits))]
            out += reduce(np.kron, factors)
        return out

    def branch_term(view):
        qubits = config.views[view]
        hp = config.hp_views[view]
        xs = xsum_dense(qubits)

        def ev(params):
            theta, phi = float(params[0]), float(params[1])
            phase = np.diag(np.exp(-1j * theta * hp))
            mix = rx_dense(qubits, phi)
            u = phase @ mix
            du_theta = np.diag(-1j * hp) @ u
            du_phi = phase @ ((-1j * xs) @ mix)
            return u, np.stack([du_theta, du_phi])
        return DifferentiableUnitaryTerm(D, 2, ev)

    layers = [
        DifferentiableStackedLCULayer([branch_term(v), branch_term(w)])
        for (v, w) in config.layer_pairs
    ]
    return DifferentiableStackedLCU(layers)


def dephased_syndrome_distribution(params: np.ndarray, n_layers: int,
                                   n_samples: int,
                                   stride: int = 4) -> np.ndarray:
    """Pattern distribution of the fully within-layer-dephased twin.

    If the layer ancilla is dephased in the computational basis between
    SELECT and PREP^dagger, the outcome probability of layer l becomes
    P(fail) = sum_y |R_{y1}|^2 ||U_y psi||^2 = a1*a0 + a0*a1 = 2*a0*a1 ---
    independent of the data state (branch unitaries preserve norm).  The
    joint pattern distribution is therefore the data-independent product of
    Bernoulli(2*a_l0*a_l1) factors: the dephased syndrome carries exactly
    zero transaction information.  (This makes the coherent-vs-dephased
    comparison a mechanism statement, not a tunable ablation; the honest
    classical twin for *performance* comparisons is the exact 2^L-path
    overlap expansion, which reproduces the coherent features classically at
    small n.)
    """
    probs_fail = []
    for l in range(n_layers):
        a = _softmax(np.asarray(params[stride * l : stride * l + 2], float))
        probs_fail.append(2.0 * a[0] * a[1])
    D = 2 ** n_layers
    out = np.empty(D)
    for pattern in range(D):
        p = 1.0
        for l in range(n_layers):
            f = probs_fail[l]
            p *= f if (pattern >> l) & 1 else (1.0 - f)
        out[pattern] = p
    return np.tile(out, (n_samples, 1))


def repo_stack(config: SLCUStackConfig) -> DifferentiableStackedLCU:
    """The same ansatz built through the repo's differentiable dense classes."""
    D = 2 ** config.n_qubits
    n = config.n_qubits
    hp = config.hp

    def cost_term():
        def ev(params):
            theta = float(params[0])
            u = np.diag(np.exp(-1j * theta * hp))
            du = (-1j * np.diag(hp)) @ u
            return u, du[None, :, :]
        return DifferentiableUnitaryTerm(D, 1, ev)

    def mixer_term():
        from functools import reduce
        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        xsum = sum(
            reduce(np.kron, [X if q == e else np.eye(2) for q in reversed(range(n))])
            for e in range(n)
        ).astype(complex)

        def ev(params):
            phi = float(params[0])
            rx = np.array([[np.cos(phi), -1j * np.sin(phi)],
                           [-1j * np.sin(phi), np.cos(phi)]])
            from functools import reduce as red
            u = red(np.kron, [rx] * n).astype(complex)
            du = (-1j * xsum) @ u
            return u, du[None, :, :]
        return DifferentiableUnitaryTerm(D, 1, ev)

    layers = [
        DifferentiableStackedLCULayer([cost_term(), mixer_term()])
        for _ in range(config.n_layers)
    ]
    return DifferentiableStackedLCU(layers)


def verify_against_repo(config: SLCUStackConfig, seed: int = 0,
                        n_states: int = 3, atol: float = 1e-9) -> float:
    """Max abs deviation between BatchedSLCU and DifferentiableStackedLCU."""
    rng = np.random.default_rng(seed)
    params = rng.normal(scale=0.4, size=config.n_params)
    D = 2 ** config.n_qubits
    states = rng.normal(size=(n_states, D)) + 1j * rng.normal(size=(n_states, D))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    fast = BatchedSLCU(config)
    v_fast, d_fast = fast.forward_and_derivatives(params, states)
    stack = repo_stack(config)
    worst = 0.0
    for i in range(n_states):
        v_ref, d_ref = stack.unnormalized_state_and_derivatives(params, states[i])
        worst = max(worst, float(np.abs(v_fast[i] - v_ref).max()))
        worst = max(worst, float(np.abs(d_fast[:, i, :] - d_ref).max()))
    if worst > atol:
        raise AssertionError(f"BatchedSLCU mismatch vs repo: {worst:.3e}")
    return worst


# ---------------------------------------------------------------------------
# scorer training (pairwise separation loss on p_s)
# ---------------------------------------------------------------------------

def ps_scores(sim: BatchedSLCU, params: np.ndarray, states: np.ndarray) -> np.ndarray:
    v = sim.forward(params, states)
    return np.einsum("md,md->m", v.conj(), v).real


def pairwise_loss_and_grad(sim: BatchedSLCU, params: np.ndarray,
                           legit_states: np.ndarray, fraud_states: np.ndarray,
                           kappa: float = 12.0):
    """Pairwise logistic ranking loss  L = mean_{f,l} log(1 + exp(-kappa (s_l - s_f))).

    Legitimate rows should score HIGH (they pass the filter), fraud rows LOW.
    Returns (loss, grad, mean legit p_s, mean fraud p_s).
    """
    vl, dl = sim.forward_and_derivatives(params, legit_states)
    vf, df = sim.forward_and_derivatives(params, fraud_states)
    sl = np.einsum("md,md->m", vl.conj(), vl).real
    sf = np.einsum("md,md->m", vf.conj(), vf).real
    dsl = 2.0 * np.einsum("pmd,md->pm", dl.conj(), vl).real
    dsf = 2.0 * np.einsum("pmd,md->pm", df.conj(), vf).real
    diff = sl[None, :] - sf[:, None]              # (n_f, n_l)
    sig = 1.0 / (1.0 + np.exp(np.clip(kappa * diff, -60, 60)))
    loss = float(np.mean(np.log1p(np.exp(np.clip(-kappa * diff, -60, 60)))))
    n_f, n_l = diff.shape
    w_l = sig.sum(axis=0) / (n_f * n_l)           # weight per legit sample
    w_f = sig.sum(axis=1) / (n_f * n_l)           # weight per fraud sample
    grad = -kappa * (dsl @ w_l - dsf @ w_f)
    return loss, grad, float(sl.mean()), float(sf.mean())


def batch_fs_metric(sim: BatchedSLCU, params: np.ndarray,
                    states: np.ndarray) -> np.ndarray:
    """Batch-averaged Fubini-Study metric of the normalized per-sample outputs."""
    v, d = sim.forward_and_derivatives(params, states)
    norms2 = np.einsum("md,md->m", v.conj(), v).real
    psi = v / np.sqrt(norms2)[:, None]
    dpsi = d / np.sqrt(norms2)[None, :, None]
    overl = np.einsum("pmd,md->pm", dpsi.conj(), psi)
    g = np.einsum("pmd,qmd->pqm", dpsi.conj(), dpsi).real
    g -= np.einsum("pm,qm->pqm", overl, overl.conj()).real
    metric = g.mean(axis=2)
    return 0.5 * (metric + metric.T)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def ranking_metrics(y_true: np.ndarray, fraud_score: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score
    return {
        "auc_roc": float(roc_auc_score(y_true, fraud_score)),
        "auprc": float(average_precision_score(y_true, fraud_score)),
    }


def f1_at_best_threshold(y_val, s_val, y_test, s_test) -> dict:
    """Pick the F1-max threshold on validation, report test P/R/F1 there."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_val, s_val)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    k = int(np.nanargmax(f1[:-1]))
    t = float(thr[k])
    pred = (s_test >= t).astype(int)
    tp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {
        "threshold": t,
        "precision": float(p),
        "recall": float(r),
        "f1": float(2 * p * r / max(p + r, 1e-12)),
    }


def bootstrap_auprc_ci(y, s, n_boot: int = 1000, seed: int = 0):
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].sum() == 0:
            continue
        vals.append(average_precision_score(y[b], s[b]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_delta_auprc(y, s_a, s_b, n_boot: int = 1000, seed: int = 0):
    """Paired bootstrap CI for AUPRC(a) - AUPRC(b) on the same rows."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].sum() == 0:
            continue
        vals.append(average_precision_score(y[b], s_a[b])
                    - average_precision_score(y[b], s_b[b]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


def shot_noise_scores(ps: np.ndarray, shots: int, seed: int = 0) -> np.ndarray:
    """Finite-shot Bernoulli estimate of each p_s score."""
    rng = np.random.default_rng(seed)
    p = np.clip(ps, 0.0, 1.0)
    return rng.binomial(shots, p) / shots


def save_json(path: str, payload: dict) -> None:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)


def keyed_noise(ids, n_feat: int = 8, base_seed: int = 7) -> np.ndarray:
    """Deterministic per-row N(0,1) noise keyed by (base_seed, row id).

    Order- and batch-invariant: the same id always gets the same noise
    vector, regardless of which rows are generated together (Engine-A
    amendment A6 — the C-f noise-control features).
    """
    ids = np.asarray(ids, dtype=np.int64)
    out = np.empty((len(ids), n_feat))
    for i, t in enumerate(ids):
        out[i] = np.random.default_rng([base_seed, int(t)]).normal(size=n_feat)
    return out


# ---------------------------------------------------------------------------
# shared end-to-end pipeline prep (must mirror exp_b_scorer.py exactly)
# ---------------------------------------------------------------------------

def prepare_hsbc(n_feat: int, seed: int = 0, splits=None):
    """Rebuild the EXP-B data pipeline deterministically.

    Returns dict with splits, selected features, quantile transform, U/bits/y
    per split, and the legit-only pseudo-likelihood Ising model (tabulated).
    """
    X, y, cols = load_ulb()
    if splits is None:
        idx_tr, idx_va, idx_te = stratified_split(y, seed=seed)
    else:
        idx_tr, idx_va, idx_te = splits
    feat = select_features(X[idx_tr], y[idx_tr], n_feat)
    qt = QuantileTransform(X[np.ix_(idx_tr, feat)])
    U = {k: qt(X[np.ix_(i, feat)])
         for k, i in (("tr", idx_tr), ("va", idx_va), ("te", idx_te))}
    yy = {"tr": y[idx_tr], "va": y[idx_va], "te": y[idx_te]}
    bits_tr = hard_bits(U["tr"])
    model = fit_ising_pseudolikelihood(bits_tr[yy["tr"] == 0], seed=seed)
    return {
        "X": X, "y": y, "cols": cols,
        "idx": {"tr": idx_tr, "va": idx_va, "te": idx_te},
        "feat": feat, "feature_names": [cols[j] for j in feat],
        "qt": qt, "U": U, "y_split": yy, "model": model,
    }


# ---------------------------------------------------------------------------
# matched-size E.ON congestion QUBO (same family as eon_slcu_feasibility.py)
# ---------------------------------------------------------------------------

def make_eon_qubo(n_cand: int = 8, m_lines: int = 4, seed: int = 3):
    """Seeded congestion-relief QUBO in the E.ON toy family.

    Returns dict with raw energies over all 2^n plans, optimum, and the
    normalized diagonals used by the ansatz ([-1,1] "norm" and [0,1] "hp").
    """
    rng = np.random.default_rng(seed)
    R = np.zeros((m_lines, n_cand))
    for e in range(n_cand):
        lines = rng.choice(m_lines, size=rng.integers(1, 4), replace=False)
        R[lines, e] = rng.uniform(0.2, 0.9, size=len(lines))
    f0 = rng.uniform(0.8, 1.3, size=m_lines)
    w = np.ones(m_lines)
    cb = rng.uniform(0.25, 0.55, size=n_cand)
    lam = 1.0
    plans = np.array([[(i >> e) & 1 for e in range(n_cand)]
                      for i in range(2 ** n_cand)], float)
    over = f0[None, :] - plans @ R.T
    energies = (over ** 2) @ w + lam * (plans @ cb)
    idx_opt = int(np.argmin(energies))
    shift = float(energies.mean())
    scale = float(np.max(np.abs(energies - shift)))
    span = float(energies.max() - energies.min())
    return {
        "n": n_cand, "R": R, "f0": f0, "cb": cb,
        "energies": energies, "idx_opt": idx_opt,
        "e_opt": float(energies[idx_opt]),
        "gap": float(np.sort(energies)[1] - energies[idx_opt]),
        "norm": (energies - shift) / scale,          # spectrum in [-1, 1]
        "hp": (energies - energies.min()) / span,     # spectrum in [0, 1]
        "shift": shift, "scale": scale,
    }
