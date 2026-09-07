"""EXP-C: numerical verification of Prop 1 (commuting stacks dequantize).

  C1  n=12 HSBC scorer, diagonal-only stack: the classical product-measure
      Monte-Carlo estimator of p_s(x) converges at the same 1/sqrt(N) rate as
      hardware shots (slope fit + side-by-side RMSE).
  C2  n=8 E.ON sampler, diagonal-only stack: classical rejection sampling
      reproduces the postselected conditional distribution, with acceptance
      rate >= the quantum joint-postselection rate (g_max <= 1).
  C3  mixers on: the product-measure shortcut is *biased* (error does not
      shrink with N) -- the non-commuting layer is where any quantum content
      lives.  (This does not prove classical hardness; small-n dense/TN
      simulation obviously remains possible.)

Artifacts -> runs/hsbc_challenge/fit_v0/exp_c_dequantization.json
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hsbc_common import (
    BatchedSLCU, SLCUStackConfig, angle_amplitudes, make_eon_qubo,
    prepare_hsbc, product_probs_batch, product_state_batch, save_json,
)

SEED = 0
T0 = time.time()
OUT = {}


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def diagonal_filter(params, hp, n_layers):
    """f(E) = prod_l (a_l0 e^{-i theta_l E} + a_l1) on the spectrum grid."""
    f = np.ones_like(hp, complex)
    for l in range(n_layers):
        logits = params[4 * l: 4 * l + 2]
        a = np.exp(logits - logits.max())
        a = a / a.sum()
        f = f * (a[0] * np.exp(-1j * params[4 * l + 2] * hp) + a[1])
    return f


# ---------------------------------------------------------------------------
# C1: score-estimator rate equivalence (n=12, diagonal-only)
# ---------------------------------------------------------------------------
pipe = prepare_hsbc(12, seed=SEED)
model = pipe["model"]
U_te, y_te = pipe["U"]["te"], pipe["y_split"]["te"]
rng = np.random.default_rng(SEED)
rows = np.sort(np.concatenate([
    rng.choice(np.flatnonzero(y_te == 0), 40, replace=False),
    rng.choice(np.flatnonzero(y_te == 1), 10, replace=False)]))
U_sel = U_te[rows]

params_diag = np.array([0.10, -0.10, 1.2, 0.0, 0.05, 0.20, 0.8, 0.0])
g = np.abs(diagonal_filter(params_diag, model.hp, 2)) ** 2
exact = product_probs_batch(U_sel) @ g

# cross-check against the circuit-level batched simulator
cfg = SLCUStackConfig(n_qubits=12, n_layers=2, hp=model.hp)
sim = BatchedSLCU(cfg)
v = sim.forward(params_diag, product_state_batch(U_sel).astype(complex))
ps_circuit = np.einsum("md,md->m", v.conj(), v).real
assert np.allclose(ps_circuit, exact, atol=1e-10)
log(f"C1: closed form == circuit p_s (max dev {np.abs(ps_circuit-exact).max():.2e})")

_, a1 = angle_amplitudes(U_sel)
p1 = a1 ** 2                       # per-qubit P(bit=1)
pow2 = 2 ** np.arange(12)
rates = {}
for N in (100, 1000, 10_000, 100_000):
    est = np.empty(len(rows))
    for i in range(len(rows)):
        bits = rng.random((N, 12)) < p1[i]
        est[i] = g[bits @ pow2].mean()
    rmse_mc = float(np.sqrt(np.mean((est - exact) ** 2)))
    rmse_shots = float(np.mean(np.sqrt(exact * (1 - exact) / N)))
    rates[str(N)] = {"rmse_classical_mc": rmse_mc,
                     "rmse_quantum_shots_analytic": rmse_shots}
    log(f"C1: N={N:>6d}  MC rmse {rmse_mc:.5f}  vs shot rmse {rmse_shots:.5f}")
Ns = np.array([100, 1000, 10_000, 100_000], float)
slope = float(np.polyfit(np.log(Ns),
                         np.log([rates[str(int(N))]["rmse_classical_mc"] for N in Ns]), 1)[0])
rates["log_log_slope"] = slope
OUT["C1_estimator_rate"] = rates
log(f"C1: MC error slope vs N = {slope:.3f} (theory -0.5)")

# ---------------------------------------------------------------------------
# C2: rejection sampling reproduces the E.ON conditional distribution (n=8)
# ---------------------------------------------------------------------------
inst = make_eon_qubo(n_cand=8, seed=3)
f = diagonal_filter(params_diag, inst["norm"], 2)
g = np.abs(f) ** 2
g_max = float(g.max())
p_s_quantum = float(g.mean())            # uniform |+>^n input
accept_classical = float(g.mean() / g_max)
cond_exact = g / g.sum()

rng2 = np.random.default_rng(SEED + 1)
target = 20_000
accepted = []
draws = 0
while len(accepted) < target:
    z = rng2.integers(0, 2 ** 8, size=50_000)
    keep = rng2.random(50_000) < g[z] / g_max
    accepted.extend(z[keep].tolist())
    draws += 50_000
accepted = np.array(accepted[:target])
emp = np.bincount(accepted, minlength=2 ** 8) / target
tv_emp = 0.5 * float(np.abs(emp - cond_exact).sum())
ref = rng2.multinomial(target, cond_exact) / target
tv_ref = 0.5 * float(np.abs(ref - cond_exact).sum())
OUT["C2_rejection_sampler"] = {
    "g_max": g_max, "g_max_le_1": bool(g_max <= 1.0 + 1e-12),
    "quantum_joint_postselect_rate": p_s_quantum,
    "classical_acceptance_rate": accept_classical,
    "classical_rate_advantage": accept_classical / p_s_quantum,
    "tv_rejection_vs_exact": tv_emp,
    "tv_multinomial_reference": tv_ref,
    "P_opt_given_success_exact": float(cond_exact[inst["idx_opt"]]),
    "P_opt_given_success_sampled": float(emp[inst["idx_opt"]]),
}
log(f"C2: accept_classical={accept_classical:.4f} >= p_s={p_s_quantum:.4f}; "
    f"TV sampled {tv_emp:.4f} vs multinomial ref {tv_ref:.4f}")

# ---------------------------------------------------------------------------
# C3: with mixers the product-measure shortcut is biased
# ---------------------------------------------------------------------------
params_mix = params_diag.copy()
params_mix[3] = 0.35
params_mix[7] = -0.30
v = sim.forward(params_mix, product_state_batch(U_sel).astype(complex))
ps_mix = np.einsum("md,md->m", v.conj(), v).real
g_diag = np.abs(diagonal_filter(params_mix, model.hp, 2)) ** 2
surrogate = product_probs_batch(U_sel) @ g_diag
bias = float(np.sqrt(np.mean((surrogate - ps_mix) ** 2)))
OUT["C3_mixer_bias"] = {
    "rmse_commuting_surrogate_vs_true_ps": bias,
    "mc_rmse_at_N100000_for_scale": rates["100000"]["rmse_classical_mc"],
    "mixer_angles": [0.35, -0.30],
}
log(f"C3: commuting-surrogate bias {bias:.5f} "
    f"(vs N=1e5 MC noise {rates['100000']['rmse_classical_mc']:.5f})")

OUT["wall_seconds"] = time.time() - T0
save_json("runs/hsbc_challenge/fit_v0/exp_c_dequantization.json", OUT)
log("EXP-C done -> runs/hsbc_challenge/fit_v0/exp_c_dequantization.json")
