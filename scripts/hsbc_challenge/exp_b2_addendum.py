"""EXP-B2 addendum: (1) alias-corrected LCHS truncation study for the ranking
deliverable; (2) paired-bootstrap resolution of the mixer question (trained
L=1 mixer stack vs analytic S1 vs trained diagonal S2d on identical test rows).

The flat truncated-Cauchy quadrature F(E) = sum_j w_j e^{i k_j tau E} is
periodic in tau*E with alias period 2*pi/dk = pi*(M-1)/K.  The E.ON probe's
(M, K) pairs keep K/(M-1) ~ 0.8, i.e. period ~3.9 -- fine for *concentration*
(only the spectral floor matters) but rank-UNFAITHFUL for scoring whenever
tau * span >= period (high energies wrap around to high scores).  Ranking
needs period > tau*span with margin, plus small Cauchy tail 2/(pi*K).

Artifacts -> runs/hsbc_challenge/fit_v0/exp_b2_addendum.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hsbc_common import (
    BatchedSLCU, SLCUStackConfig, bootstrap_delta_auprc, lchs_cauchy_filter,
    pairwise_loss_and_grad, prepare_hsbc, product_probs_batch,
    product_state_batch, ranking_metrics, save_json,
)

SEED = 0
TAU = 4.0
CHUNK = 8192
T0 = time.time()
OUT = {"tau": TAU}


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


pipe = prepare_hsbc(8, seed=SEED)
model = pipe["model"]
U_tr, U_va, U_te = pipe["U"]["tr"], pipe["U"]["va"], pipe["U"]["te"]
y_tr, y_va, y_te = (pipe["y_split"][k] for k in ("tr", "va", "te"))

# ---------------------------------------------------------------------------
# 1. alias-aware LCHS truncation study
# ---------------------------------------------------------------------------
ite = np.exp(-TAU * model.hp)
s1_te = -probs_matmul(U_te, (ite ** 2)[:, None])[:, 0]
rows = {}
for M, K in ((8, 6.0), (16, 12.0), (64, 12.0), (128, 12.0), (256, 25.0)):
    F, w1 = lchs_cauchy_filter(model.hp, TAU, M, K)
    g = (np.abs(F) ** 2) / (w1 ** 2)
    s = -probs_matmul(U_te, g[:, None])[:, 0]
    period = np.pi * (M - 1) / K
    rows[f"M{M}_K{int(K)}"] = {
        "alias_period_over_tau_span": float(period / TAU),
        "cauchy_tail_bound": float(2 / (np.pi * K)),
        "filter_sup_err": float(np.max(np.abs(F - ite))),
        "weight_1norm": w1,
        **ranking_metrics(y_te, s),
    }
    log(f"LCHS M={M} K={K}: period/(tau*span)={period/TAU:.2f} "
        f"supErr={rows[f'M{M}_K{int(K)}']['filter_sup_err']:.3f} "
        f"AUPRC={rows[f'M{M}_K{int(K)}']['auprc']:.4f}")
OUT["lchs_truncation"] = rows
OUT["s1_reference"] = ranking_metrics(y_te, s1_te)

# ---------------------------------------------------------------------------
# 2. mixer question: trained L=1 stacks (EXP-E protocol) vs S1 vs S2d, paired
# ---------------------------------------------------------------------------
from sklearn.metrics import average_precision_score

fraud_states = product_state_batch(U_tr[y_tr == 1]).astype(complex)
legit_idx = np.flatnonzero(y_tr == 0)
rng_v = np.random.default_rng(SEED)
val_sel = np.sort(np.concatenate(
    [np.flatnonzero(y_va == 1),
     rng_v.choice(np.flatnonzero(y_va == 0), 8000, replace=False)]))
U_va_sel, y_va_sel = U_va[val_sel], y_va[val_sel]

cfg1 = SLCUStackConfig(n_qubits=8, n_layers=1, hp=model.hp)
sim1 = BatchedSLCU(cfg1)


def train_L1(seed, iters=300):
    rng = np.random.default_rng(seed)
    params = np.array([*rng.normal(scale=0.15, size=2),
                       rng.uniform(0.5, 2.5), rng.normal(scale=0.15)])
    m1 = np.zeros(4)
    m2 = np.zeros(4)
    best = (-1.0, params.copy())
    for it in range(iters):
        batch = rng.choice(legit_idx, 384, replace=False)
        legit_states = product_state_batch(U_tr[batch]).astype(complex)
        _, grad, _, _ = pairwise_loss_and_grad(sim1, params, legit_states,
                                               fraud_states)
        m1 = 0.9 * m1 + 0.1 * grad
        m2 = 0.999 * m2 + 0.001 * grad * grad
        params = params - 0.05 * (m1 / (1 - 0.9 ** (it + 1))) / (
            np.sqrt(m2 / (1 - 0.999 ** (it + 1))) + 1e-8)
        if (it + 1) % 20 == 0 or it == iters - 1:
            ap = average_precision_score(y_va_sel, -ps_chunked(sim1, params, U_va_sel))
            if ap > best[0]:
                best = (ap, params.copy())
    return best


best = (-1.0, None)
per_seed = []
for s in (300, 301, 302, 303, 304):
    ap, p = train_L1(s)
    per_seed.append(float(ap))
    if ap > best[0]:
        best = (ap, p)
s2m_te = -ps_chunked(sim1, best[1], U_te)
OUT["S2_L1_val_sub_auprc_per_seed"] = per_seed
OUT["S2_L1_test"] = ranking_metrics(y_te, s2m_te)
log(f"trained L=1 mixer: val-sub per seed {np.round(per_seed,4).tolist()}, "
    f"test AUPRC {OUT['S2_L1_test']['auprc']:.4f}")

expb = json.load(open("runs/hsbc_challenge/fit_v0/exp_b_scorer.json"))["n8_L2"]
cfg2 = SLCUStackConfig(n_qubits=8, n_layers=2, hp=model.hp)
sim2 = BatchedSLCU(cfg2)
s2d_te = -ps_chunked(sim2, np.array(expb["S2d_trained_diagonal_params"]), U_te)
s2b_te = -ps_chunked(sim2, np.array(expb["S2_trained_mixer_params"]), U_te)

deltas = {}
for name, (a, b) in {
    "S2_L1_mixer - S1": (s2m_te, s1_te),
    "S2_L1_mixer - S2d": (s2m_te, s2d_te),
    "S2_expB_L2_mixer - S1": (s2b_te, s1_te),
}.items():
    mean, lo, hi = bootstrap_delta_auprc(y_te, a, b, seed=SEED)
    deltas[name] = {"mean": mean, "ci95": [lo, hi]}
    log(f"paired dAUPRC {name}: {mean:+.4f} [{lo:+.4f},{hi:+.4f}]")
OUT["mixer_paired_deltas"] = deltas

OUT["wall_seconds"] = time.time() - T0
save_json("runs/hsbc_challenge/fit_v0/exp_b2_addendum.json", OUT)
log("EXP-B2 done -> runs/hsbc_challenge/fit_v0/exp_b2_addendum.json")
