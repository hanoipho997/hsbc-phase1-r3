"""Candidate B referee: is failure-sector "gradient recycling" a new
estimator, and what does it cost in shots?

Checks, on trained parameters (ULB L=1 seed-500 replication params, ULB
n=8/L=2 EXP-B params, Engine-A n=12/L=3 seed-900 params) and ULB
VALIDATION rows (test rows untouched):

  B1  Identity: d p_s / d beta_j = 2 <X_{a_j} Pi_{not j}> = 4 Re <psi_0 | f_j>,
      with beta_j the physical PREP angle (a0 = cos^2 beta), against central
      finite differences through the softmax-logit map.
  B2  Coordinate rewrite: the X-basis ancilla readout equals the two-point
      parameter-shift rule applied to the PREP^dagger (uncompute) angle
      alone, P0(beta, beta+pi/4) - P0(beta, beta-pi/4), and by PREP/PREP^dagger
      symmetry d p_s/d beta = 2 x that.  Verified for every layer.
  B3  Frequency content of p_s in beta_j: last layer has only frequency 4;
      inner layers have frequencies {2, 4}.  Both the single-frequency
      2-point rule (last layer) and the two-frequency 4-point rule
      (Wierichs-type, shifts pi/8 and 3pi/8) reproduce the exact derivative.
  B4  Shot cost at fixed gradient MSE, per layer, averaged over rows:
      E1  X-readout (one setting, same circuit):        var = 4 (p_s + p_fj - <XPi>^2)/N
      E1' uncompute-only shift run as two circuits:      var from two Bernoulli estimates
      E2  physical-beta shift rule (2- or 4-point):      var from shifted Bernoulli estimates
      E3  Hadamard test of Re<psi_0|f_j> (extra control): var = 4 (1 - <X>^2)/N  (analytic)
      Reported as shots N to reach MSE 1e-4 per component, plus settings and
      distinct circuits.  Monte-Carlo sanity check of E1/E2 at N=4000.

Artifacts -> runs/hsbc_challenge/novelty_v1/gradient_recycling_v1.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsbc_common import (
    BatchedSLCU, CrossViewConfig, CrossViewSLCU, SLCUStackConfig, _softmax,
    broadcast_view_diagonal, prepare_hsbc, product_state_batch,
)
from novelty_common_v1 import (
    path_states, pattern_coefficients, save_json_no_overwrite, sha256_file,
    stride_of,
)

T0 = time.time()
OUTD = "runs/hsbc_challenge/novelty_v1"
os.makedirs(OUTD, exist_ok=True)
OUT_PATH = os.path.join(OUTD, "gradient_recycling_v1.json")
OUT = {"schema": "hsbc-novelty-gradient-recycling-v1", "sources": {}}


def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def logits_from_beta(beta):
    """softmax(l0, l1) = (cos^2 beta, sin^2 beta) with l0 = 0 (any real beta)."""
    t2 = np.tan(beta) ** 2
    return np.array([0.0, float(np.log(np.clip(t2, 1e-300, 1e300)))])


def beta_from_logits(logits):
    a = _softmax(np.asarray(logits, float))
    return float(np.arctan2(np.sqrt(a[1]), np.sqrt(a[0])))


def set_layer_beta(params, l, stride, beta):
    p = params.copy()
    p[stride * l: stride * l + 2] = logits_from_beta(beta)
    return p


def ps_of(sim, params, states):
    v = sim.forward(params, states)
    return np.einsum("md,md->m", v.conj(), v).real


def sectors(sim, params, states):
    return sim.syndrome_sector_states(params, states)      # (2^L, m, D)


def uncompute_shifted_p0(sim, params, states, l, delta):
    """P(all ancillas 0) when only layer l's PREP^dagger angle is shifted by
    delta: amplitude = cos(b+d) sqrt(a0) path0 + sin(b+d) sqrt(a1) path1 in
    layer l, everything else unchanged.  Computed from the pattern-coefficient
    expansion with the layer-l success coefficients replaced."""
    L, stride = sim.cfg.n_layers, stride_of(sim)
    P = path_states(sim, params, states)                    # (Y, m, D)
    c = pattern_coefficients(params, L, stride)[0].copy()   # success row c[0, y]
    a = _softmax(params[stride * l: stride * l + 2])
    b = beta_from_logits(params[stride * l: stride * l + 2])
    y = np.arange(2 ** L)
    bit = (y >> l) & 1
    base = np.where(bit == 0, a[0], a[1])                   # original layer-l factor
    new = np.where(bit == 0, np.cos(b + delta) * np.sqrt(a[0]),
                   np.sin(b + delta) * np.sqrt(a[1]))
    c2 = c / base * new
    v = np.einsum("y,ymd->md", c2, P)
    return np.einsum("md,md->m", v.conj(), v).real


# ---------------------------------------------------------------------------
# instances
# ---------------------------------------------------------------------------
pipe = prepare_hsbc(8, seed=0)
U_va = pipe["U"]["va"][:512]
states8 = product_state_batch(U_va).astype(complex)
cfg8 = SLCUStackConfig(n_qubits=8, n_layers=1, hp=pipe["model"].hp)

c1 = json.load(open("runs/hsbc_challenge/audit_v1/priority_attacks_v1.json"))
OUT["sources"]["priority_attacks_v1.json"] = sha256_file(
    "runs/hsbc_challenge/audit_v1/priority_attacks_v1.json")
params_L1 = np.array(c1["C1_mixer_replication"]["per_seed"][0]["params"])
assert c1["C1_mixer_replication"]["per_seed"][0]["seed"] == 500

eb = json.load(open("runs/hsbc_challenge/fit_v0/exp_b_scorer.json"))
OUT["sources"]["exp_b_scorer.json"] = sha256_file("runs/hsbc_challenge/fit_v0/exp_b_scorer.json")
params_L2 = np.array(eb["n8_L2"]["S2_trained_mixer_params"])
cfg8L2 = SLCUStackConfig(n_qubits=8, n_layers=2, hp=pipe["model"].hp)

import joblib
views = joblib.load("runs/hsbc_challenge/engine_a_v1/views_v1.joblib")
arms = json.load(open("runs/hsbc_challenge/engine_a_v1/arms_v1/arms_summary_v1.json"))
OUT["sources"]["arms_summary_v1.json"] = sha256_file(
    "runs/hsbc_challenge/engine_a_v1/arms_v1/arms_summary_v1.json")
OUT["sources"]["views_v1.joblib"] = sha256_file("runs/hsbc_challenge/engine_a_v1/views_v1.joblib")
VQ = views["view_qubits"]
name2idx = {"A": 0, "B": 1, "C": 2}
cfg12 = CrossViewConfig(
    n_qubits=12, views=tuple(tuple(VQ[v]) for v in ("A", "B", "C")),
    layer_pairs=tuple((name2idx[a], name2idx[b]) for a, b in views["layer_pairs"]),
    hp_views=tuple(broadcast_view_diagonal(np.array(views["hp_tables"][v]), VQ[v], 12)
                   for v in ("A", "B", "C")))
params_12 = np.array(arms["Q"]["seeds"]["900"]["params"])
# Engine-A-style inputs: 12 quantile features; use 256 synthetic rows drawn
# from the view quantile transform of random uniforms is NOT the data; instead
# reuse ULB validation quantile features tiled to 12 columns (mechanism-only,
# any product state is a valid input to the identity checks).
rng = np.random.default_rng(20260904)
U12 = np.column_stack([U_va[:256], U_va[:256, :4]])
states12 = product_state_batch(U12).astype(complex)

INSTANCES = [
    ("ULB_n8_L1_seed500", BatchedSLCU(cfg8), params_L1, states8),
    ("ULB_n8_L2_expB", BatchedSLCU(cfg8L2), params_L2, states8),
    ("EngineA_n12_L3_seed900", CrossViewSLCU(cfg12), params_12, states12),
]

results = {}
for name, sim, params, states in INSTANCES:
    L, stride = sim.cfg.n_layers, stride_of(sim)
    m = states.shape[0]
    sec = sectors(sim, params, states)
    psi0 = sec[0]
    ps = np.einsum("md,md->m", psi0.conj(), psi0).real
    rec = {"n_layers": L, "rows": int(m), "mean_ps": float(ps.mean()),
           "layers": {}}
    for l in range(L):
        fj = sec[1 << l]
        pfj = np.einsum("md,md->m", fj.conj(), fj).real
        x_exp = 2.0 * np.einsum("md,md->m", psi0.conj(), fj).real        # <X_j Pi>
        grad_readout = 2.0 * x_exp                                       # claimed d p_s / d beta_j
        # B1: finite differences in the physical beta through the logit map
        b = beta_from_logits(params[stride * l: stride * l + 2])
        h = 1e-5
        fd = (ps_of(sim, set_layer_beta(params, l, stride, b + h), states)
              - ps_of(sim, set_layer_beta(params, l, stride, b - h), states)) / (2 * h)
        dev_B1 = float(np.abs(grad_readout - fd).max())
        # B2: uncompute-only two-point shift
        p_plus = uncompute_shifted_p0(sim, params, states, l, +np.pi / 4)
        p_minus = uncompute_shifted_p0(sim, params, states, l, -np.pi / 4)
        dev_B2_x = float(np.abs((p_plus - p_minus) - x_exp).max())
        dev_B2_grad = float(np.abs(2.0 * (p_plus - p_minus) - grad_readout).max())
        # sanity: unshifted uncompute reproduces p_s
        dev_B2_zero = float(np.abs(uncompute_shifted_p0(sim, params, states, l, 0.0) - ps).max())
        # B3: physical-beta shift rules
        s4 = np.pi / 8
        f_p = ps_of(sim, set_layer_beta(params, l, stride, b + s4), states)
        f_m = ps_of(sim, set_layer_beta(params, l, stride, b - s4), states)
        grad_single = 2.0 * (f_p - f_m)                                   # exact iff only freq 4
        dev_single = float(np.abs(grad_single - fd).max())
        # two-frequency rule, shifts s1=pi/8, s2=3pi/8: f(x+s)-f(x-s) = 2 sum_w d_w sin(w s)
        s1, s2 = np.pi / 8, 3 * np.pi / 8
        g1 = ps_of(sim, set_layer_beta(params, l, stride, b + s1), states) \
            - ps_of(sim, set_layer_beta(params, l, stride, b - s1), states)
        g2 = ps_of(sim, set_layer_beta(params, l, stride, b + s2), states) \
            - ps_of(sim, set_layer_beta(params, l, stride, b - s2), states)
        M = np.array([[2 * np.sin(2 * s1), 2 * np.sin(4 * s1)],
                      [2 * np.sin(2 * s2), 2 * np.sin(4 * s2)]])
        Minv = np.linalg.inv(M)
        d_coef = Minv @ np.stack([g1, g2])                                # (2, m): d_2, d_4
        grad_two = 2.0 * d_coef[0] + 4.0 * d_coef[1]
        dev_two = float(np.abs(grad_two - fd).max())
        # derivative coefficient vector for the 4-point rule in terms of the 4 evaluations
        # f' = w1 (f(+s1) - f(-s1)) + w2 (f(+s2) - f(-s2)),  [w1, w2] = [2, 4] @ Minv
        w = np.array([2.0, 4.0]) @ Minv
        freq2_amplitude = float(np.abs(d_coef[0]).max())
        # B4: exact variances per shot (times N), averaged over rows
        p_pi = ps + pfj                                                   # P(Pi = 1)
        var_E1 = 4.0 * (p_pi - x_exp ** 2)                                # X-readout, all N shots
        var_E1p = 4.0 * 2.0 * (p_plus * (1 - p_plus) + p_minus * (1 - p_minus))  # two circuits, N/2 each
        var_E2s = 4.0 * 2.0 * (f_p * (1 - f_p) + f_m * (1 - f_m))          # single-freq 2-point, N/2 each
        fs = [ps_of(sim, set_layer_beta(params, l, stride, b + s), states) for s in (s1, -s1, s2, -s2)]
        coefs = np.array([w[0], -w[0], w[1], -w[1]])
        var_E2t = 4.0 * sum(coefs[k] ** 2 * fs[k] * (1 - fs[k]) for k in range(4))  # 4-point, N/4 each
        x_ht = 2.0 * np.einsum("md,md->m", psi0.conj(), fj).real          # Hadamard-test observable
        var_E3 = 4.0 * (1.0 - x_ht ** 2)                                  # controlled circuit, all N shots
        mse_target = 1e-4
        lay = {
            "beta": b, "a0": float(np.cos(b) ** 2),
            "mean_p_single_failure": float(pfj.mean()),
            "mean_x_readout": float(x_exp.mean()),
            "B1_readout_vs_finite_difference_max_dev": dev_B1,
            "B2_uncompute_shift_equals_X_readout_max_dev": dev_B2_x,
            "B2_uncompute_shift_gradient_max_dev": dev_B2_grad,
            "B2_unshifted_reproduces_ps_max_dev": dev_B2_zero,
            "B3_single_frequency_rule_max_dev": dev_single,
            "B3_two_frequency_rule_max_dev": dev_two,
            "B3_frequency2_amplitude_max": freq2_amplitude,
            "B4_var_times_N_mean": {
                "E1_X_readout": float(var_E1.mean()),
                "E1p_uncompute_shift_two_circuits": float(var_E1p.mean()),
                "E2_single_frequency_shift": float(var_E2s.mean()),
                "E2_two_frequency_shift": float(var_E2t.mean()),
                "E3_hadamard_test": float(var_E3.mean()),
            },
            "B4_shots_to_mse_1e-4_mean_row": {
                "E1_X_readout": float(var_E1.mean() / mse_target),
                "E1p_uncompute_shift_two_circuits": float(var_E1p.mean() / mse_target),
                "E2_single_frequency_shift": float(var_E2s.mean() / mse_target),
                "E2_two_frequency_shift": float(var_E2t.mean() / mse_target),
                "E3_hadamard_test": float(var_E3.mean() / mse_target),
            },
            "B4_ratio_E1_over_E2_valid_rule": float(
                var_E1.mean() / (var_E2s.mean() if dev_single < 1e-8 else var_E2t.mean())),
            "valid_physical_shift_rule": "single_frequency_2point" if dev_single < 1e-8 else "two_frequency_4point",
        }
        # Monte-Carlo sanity check at N = 4000 on the first 64 rows
        N = 4000
        mc_rng = np.random.default_rng(1000 + l)
        idx = slice(0, min(64, m))
        pp, pm_, pplus, pminus = p_pi[idx], x_exp[idx], p_plus[idx], p_minus[idx]
        # E1: outcomes +1 w.p. (p_pi + x)/2, -1 w.p. (p_pi - x)/2, 0 otherwise
        prob_plus = (pp + pm_) / 2
        prob_minus = (pp - pm_) / 2
        est = []
        for i in range(len(pp)):
            pr = np.clip(np.array([prob_plus[i], prob_minus[i], 1 - prob_plus[i] - prob_minus[i]]), 0.0, None)
            pr = pr / pr.sum()                       # P(Pi=1)=1 exactly at L=1 gives -1e-16 residuals
            draws = mc_rng.choice([1.0, -1.0, 0.0], size=N, p=pr)
            est.append(2.0 * draws.mean())
        mse_E1 = float(np.mean((np.array(est) - grad_readout[idx]) ** 2))
        if dev_single < 1e-8:
            e2 = 2.0 * (mc_rng.binomial(N // 2, f_p[idx]) / (N // 2)
                        - mc_rng.binomial(N // 2, f_m[idx]) / (N // 2))
        else:
            e2 = sum(coefs[k] * (mc_rng.binomial(N // 4, fs[k][idx]) / (N // 4)) for k in range(4))
        mse_E2 = float(np.mean((e2 - fd[idx]) ** 2))
        lay["B4_monte_carlo_N4000"] = {
            "E1_mse": mse_E1, "E1_predicted_mse": float(var_E1[idx].mean() / N),
            "E2_mse": mse_E2,
            "E2_predicted_mse": float((var_E2s if dev_single < 1e-8 else var_E2t)[idx].mean() / N),
        }
        rec["layers"][str(l)] = lay
        log(f"{name} layer {l}: B1 {dev_B1:.2e} B2 {dev_B2_x:.2e} B3 single {dev_single:.2e} "
            f"two-freq {dev_two:.2e}; var*N E1 {var_E1.mean():.3f} E2 "
            f"{(var_E2s if dev_single < 1e-8 else var_E2t).mean():.3f} "
            f"(ratio {lay['B4_ratio_E1_over_E2_valid_rule']:.2f})")
    results[name] = rec

OUT["results"] = results
OUT["settings_accounting"] = {
    "E1_X_readout": "1 extra measurement setting per layer on the unshifted circuit; both shifted uncompute angles are read from the two X outcomes of the same shots",
    "E1p": "2 distinct circuits per layer (uncompute angle +-pi/4), Z-basis readout",
    "E2_single_frequency": "2 distinct circuits per layer (physical PREP and PREP^dagger both shifted by +-pi/8), Z-basis readout",
    "E2_two_frequency": "4 distinct circuits per layer (shifts +-pi/8, +-3pi/8), Z-basis readout",
    "E3_hadamard_test": "1 extra control qubit and controlled branch unitaries; strictly more circuit than E1 for the same observable",
}
OUT["wall_seconds"] = time.time() - T0
save_json_no_overwrite(OUT_PATH, OUT)
log(f"DONE -> {OUT_PATH}")
