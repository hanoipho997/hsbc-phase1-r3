"""Candidate A runner: partial-dephasing interference witness, simulator stage.

Protocol: docs/hsbc_challenge_dephasing_witness_preregistration_v1.md
(SHA-256 printed at start).  Stage 1 verifies the Gram-path simulator on
random states with no transaction data; stage 2 computes the preregistered
data-bearing curves on 32 IEEE-CIS VALIDATION rows.  The sealed test parquet
is never opened.

Usage:
    python novelty_dephasing_witness_v1.py [--noise-shots 2000] [--skip-noise]

Artifacts -> runs/hsbc_challenge/novelty_v1/dephasing_witness_v1.json
             runs/hsbc_challenge/novelty_v1/dephasing_witness_curves_v1.npz
             runs/hsbc_challenge/novelty_v1/braket_reduced_instance_v1.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsbc_common import (
    CrossViewConfig, CrossViewSLCU, _softmax, broadcast_view_diagonal,
    dephased_syndrome_distribution, product_state_batch,
)
from novelty_common_v1 import (
    between_row_diversity, build_structured_crossview_circuit, counts_to_pattern_probs,
    crossview_walsh, partial_dephased_distribution, path_gram,
    pattern_coefficients, phase_twirled_distribution, sample_counts,
    save_json_no_overwrite, sha256_file, unbiased_between_row_diversity,
)

ap = argparse.ArgumentParser()
ap.add_argument("--noise-shots", type=int, default=2000)
ap.add_argument("--skip-noise", action="store_true")
ap.add_argument("--noise-seed-only", type=int, default=900)
ARGS = ap.parse_args()

T0 = time.time()
PROTOCOL = "docs/hsbc_challenge_dephasing_witness_preregistration_v1.md"
OUTD = "runs/hsbc_challenge/novelty_v1"
os.makedirs(OUTD, exist_ok=True)
OUT_JSON = os.path.join(OUTD, "dephasing_witness_v1.json")
OUT_NPZ = os.path.join(OUTD, "dephasing_witness_curves_v1.npz")
OUT_BRAKET = os.path.join(OUTD, "braket_reduced_instance_v1.npz")
for p in (OUT_JSON, OUT_NPZ, OUT_BRAKET):
    if os.path.exists(p):
        raise FileExistsError(f"refusing to overwrite {p}")
Q_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
Q_FINE = [round(x, 3) for x in np.linspace(0, 1, 21)]
OUT = {"schema": "hsbc-novelty-dephasing-witness-v1",
       "protocol": {"path": PROTOCOL, "sha256": sha256_file(PROTOCOL)},
       "q_grid": Q_GRID, "sources": {}, "stage1": {}, "stage2": {}}


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


log(f"PROTOCOL {PROTOCOL} SHA-256 = {OUT['protocol']['sha256']}")

# ---------------------------------------------------------------------------
# Stage 1: simulator gates on random states (no transaction data)
# ---------------------------------------------------------------------------
from qiskit.quantum_info import Kraus
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error


def random_config(n, views, pairs, seed):
    r = np.random.default_rng(seed)
    hp_views = tuple(broadcast_view_diagonal(r.uniform(0, 1, size=2 ** len(v)), v, n)
                     for v in views)
    cfg = CrossViewConfig(n_qubits=n, views=tuple(tuple(v) for v in views),
                          layer_pairs=tuple(pairs), hp_views=hp_views)
    params = r.normal(scale=0.5, size=cfg.n_params)
    states = r.normal(size=(3, 2 ** n)) + 1j * r.normal(size=(3, 2 ** n))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    return cfg, params, states


def kraus_dephase(q):
    return Kraus([np.sqrt(1 - q / 2) * np.eye(2), np.sqrt(q / 2) * np.diag([1.0, -1.0])])


def dm_referee_probs(cfg, params, state, q_list_per_layer, walsh, identical=False):
    """Aer density-matrix run of the structured circuit with explicit Kraus
    dephasing (possibly a sequence of strengths) at the insertion point."""
    from qiskit import QuantumCircuit
    n, L = cfg.n_qubits, cfg.n_layers
    base = build_structured_crossview_circuit(cfg, params, walsh=walsh, init_state=state,
                                              identical_branches=identical,
                                              dephase_marker=True)
    qc = QuantumCircuit(n + L)
    for inst in base.data:
        if inst.operation.name == "id":
            anc = base.find_bit(inst.qubits[0]).index
            for q in q_list_per_layer[anc - n]:
                if q > 0:
                    qc.append(kraus_dephase(q).to_instruction(), [anc])
            continue
        qc.append(inst.operation, [base.find_bit(b).index for b in inst.qubits])
    qc.save_probabilities(qubits=list(range(n, n + L)))
    sim = AerSimulator(method="density_matrix")
    res = sim.run(qc).result()
    return np.asarray(res.data(0)["probabilities"])


s1 = {}
worst = {"q0_vs_coherent": 0.0, "q1_vs_dephased_formula": 0.0, "dm_referee": 0.0,
         "composition": 0.0, "identical_branch_law": 0.0, "identical_branch_W": 0.0,
         "two_point_twirl": 0.0, "polynomial_fit_residual": 0.0}
for n, views, pairs, seed in ((4, [(0, 1), (2, 3)], [(0, 1), (1, 0)], 11),
                              (6, [(0, 1), (2, 3), (4, 5)], [(0, 1), (1, 2), (0, 2)], 12)):
    cfg, params, states = random_config(n, views, pairs, seed)
    sim = CrossViewSLCU(cfg)
    L = cfg.n_layers
    walsh = crossview_walsh(cfg)
    Pq = partial_dephased_distribution(sim, params, states, Q_GRID)
    worst["q0_vs_coherent"] = max(worst["q0_vs_coherent"], float(
        np.abs(Pq[0] - sim.syndrome_distribution(params, states)).max()))
    worst["q1_vs_dephased_formula"] = max(worst["q1_vs_dephased_formula"], float(
        np.abs(Pq[-1] - dephased_syndrome_distribution(params, L, states.shape[0], stride=6)).max()))
    # density-matrix Kraus referee at every q, all layers same q
    for k, q in enumerate(Q_GRID):
        for i in range(states.shape[0]):
            ref = dm_referee_probs(cfg, params, states[i], [[q]] * L, walsh)
            worst["dm_referee"] = max(worst["dm_referee"], float(np.abs(ref - Pq[k, i]).max()))
    # composition: q1 then q2 on every ancilla == q_eff = 1-(1-q1)(1-q2)
    q1, q2 = 0.3, 0.5
    qeff = 1 - (1 - q1) * (1 - q2)
    Pe = partial_dephased_distribution(sim, params, states, [qeff])[0]
    for i in range(states.shape[0]):
        ref = dm_referee_probs(cfg, params, states[i], [[q1, q2]] * L, walsh)
        worst["composition"] = max(worst["composition"], float(np.abs(ref - Pe[i]).max()))
    # identical-branch reference: P(fail_l) = 2 a0 a1 q, W = 0
    for k, q in enumerate(Q_GRID):
        Pr = partial_dephased_distribution(sim, params, states, [q], identical_branches=True)[0]
        for l in range(L):
            a = _softmax(params[6 * l: 6 * l + 2])
            pf = Pr[:, ((np.arange(2 ** L) >> l) & 1) == 1].sum(axis=1)
            worst["identical_branch_law"] = max(worst["identical_branch_law"],
                                                float(np.abs(pf - 2 * a[0] * a[1] * q).max()))
        worst["identical_branch_W"] = max(worst["identical_branch_W"], between_row_diversity(Pr))
        # circuit-level check of the reference at this q on one state
        ref = dm_referee_probs(cfg, params, states[0], [[q]] * L, walsh, identical=True)
        worst["identical_branch_law"] = max(worst["identical_branch_law"], float(np.abs(ref - Pr[0]).max()))
    # exact two-point phase twirl {0, pi} on every ancilla equals the q=1 law
    G = path_gram(sim, params, states)
    c = pattern_coefficients(params, L, 6)
    y = np.arange(2 ** L)
    acc = np.zeros_like(Pq[0])
    for mask in range(2 ** L):
        eta = np.array([np.pi * ((mask >> l) & 1) for l in range(L)])
        ph = np.exp(1j * (((y[:, None] >> np.arange(L)[None, :]) & 1) * eta[None, :]).sum(1))
        cph = c * ph[None, :]
        acc += np.einsum("ay,az,myz->ma", cph, cph.conj(), G).real
    acc /= 2 ** L
    worst["two_point_twirl"] = max(worst["two_point_twirl"], float(np.abs(acc - Pq[-1]).max()))
    # uniform twirl Monte Carlo (descriptive)
    tw = phase_twirled_distribution(sim, params, states, np.random.default_rng(5), n_draws=400)
    s1[f"n{n}_L{L}_uniform_twirl_mc_dev_400draws"] = float(np.abs(tw - Pq[-1]).max())
    # exact polynomial structure: W(q) is a polynomial of degree <= 2L in (1-q)
    Wf = np.array([between_row_diversity(partial_dephased_distribution(sim, params, states, [q])[0])
                   for q in Q_FINE])
    coef = np.polyfit(1 - np.array(Q_FINE), Wf, 2 * L)
    worst["polynomial_fit_residual"] = max(worst["polynomial_fit_residual"],
                                           float(np.abs(np.polyval(coef, 1 - np.array(Q_FINE)) - Wf).max()))
    # sampled stochastic-Z realization through Aer statevector + noise model on 'id'
    m = states.shape[0]
    S = 4000
    for k, q in enumerate(Q_GRID):
        nm = NoiseModel(basis_gates=["ry", "rx", "rz", "h", "p", "cx", "rzz", "id", "initialize"])
        if q > 0:
            for l in range(L):
                nm.add_quantum_error(pauli_error([("Z", q / 2), ("I", 1 - q / 2)]), "id", [n + l])
        aer = AerSimulator(method="statevector", noise_model=nm, seed_simulator=100 + k)
        counts = []
        for i in range(m):
            qc = build_structured_crossview_circuit(cfg, params, walsh=walsh, init_state=states[i],
                                                    measure_ancillas=True)
            res = aer.run(qc, shots=S).result().get_counts()
            counts.append(np.rint(counts_to_pattern_probs(res, L, S) * S).astype(int))
        counts = np.stack(counts)
        s1[f"n{n}_L{L}_aer_sampled_q{q}_max_abs_dev_probs_S{S}"] = float(
            np.abs(counts / S - Pq[k]).max())
        s1[f"n{n}_L{L}_aer_sampled_q{q}_W_hat_vs_exact"] = [
            unbiased_between_row_diversity(counts), between_row_diversity(Pq[k])]
    log(f"stage 1 n={n} L={L}: worst so far {json.dumps({k: f'{v:.2e}' for k, v in worst.items()})}")

OUT["stage1"] = {"worst_deviations": worst, "descriptive": s1}
GATE1 = (worst["q0_vs_coherent"] <= 1e-12 and worst["q1_vs_dephased_formula"] <= 1e-12
         and worst["dm_referee"] <= 1e-10 and worst["composition"] <= 1e-10
         and worst["identical_branch_law"] <= 1e-10 and worst["two_point_twirl"] <= 1e-12)
OUT["stage1"]["simulator_gate_pass"] = bool(GATE1)
log(f"STAGE 1 simulator gate: {'PASS' if GATE1 else 'FAIL'}")
if not GATE1:
    OUT["wall_seconds"] = time.time() - T0
    save_json_no_overwrite(OUT_JSON, OUT)
    sys.exit("simulator gate failed; stopping before data-bearing stage")

# ---------------------------------------------------------------------------
# Stage 2: preregistered rows and curves (IEEE-CIS validation only)
# ---------------------------------------------------------------------------
import joblib
import pandas as pd

BASE = "runs/hsbc_challenge/engine_a_v1"
TRAINVAL = "runs/hsbc_challenge/data/ieee/trainval.parquet"
assert "SEALED" not in TRAINVAL
views = joblib.load(os.path.join(BASE, "views_v1.joblib"))
routing = json.load(open(os.path.join(BASE, "routing_freeze_v1.json")))
arms = json.load(open(os.path.join(BASE, "arms_v1", "arms_summary_v1.json")))
for f in ("views_v1.joblib", "routing_freeze_v1.json", "arms_v1/arms_summary_v1.json"):
    OUT["sources"][f] = sha256_file(os.path.join(BASE, f))
t_lo = routing["routing_thresholds"]["t_lo_val_q988"]
t_hi = routing["routing_thresholds"]["t_hi_val_q998"]

cols = list(dict.fromkeys(["TransactionID", "TransactionDT", "isFraud", "split"] + views["features"]))
df = pd.read_parquet(TRAINVAL, columns=cols)
va = df[df["split"] == "val"].reset_index(drop=True)
del df
pval = 0.5 * (np.load(os.path.join(BASE, "backbone_vm", "best_xgb_val_pred.npy")).astype(np.float64)
              + np.load(os.path.join(BASE, "backbone_vm", "best_lgb_val_pred.npy")).astype(np.float64))
assert len(pval) == len(va), (len(pval), len(va))
order = np.argsort(va["TransactionDT"].to_numpy(), kind="mergesort")
region = np.where(pval > t_hi, 2, np.where(pval > t_lo, 1, 0))          # 0 below, 1 band, 2 above
rng = np.random.default_rng(20260904)
sel = []
quart = np.array_split(order, 4)
for qi, chunk in enumerate(quart):
    for reg, k in ((0, 2), (1, 4), (2, 2)):
        cand = chunk[region[chunk] == reg]
        pick = rng.choice(cand, size=k, replace=False)
        sel.extend(int(i) for i in pick)
sel = np.array(sel)
rows = va.iloc[sel].copy()
rows["hour"] = ((rows["TransactionDT"] // 3600) % 24).astype(float)


def views_u(fr):
    out = []
    for c in views["features"]:
        s = fr[c]
        if c in views["freq_maps"]:
            s = s.astype(str).map({k: float(v) for k, v in views["freq_maps"][c].items()}) \
                if s.dtype == object else s.astype(float)
        x = np.nan_to_num(s.to_numpy(dtype=float), nan=-1.0)
        col = views["sorted_cols"][c]
        lo = np.searchsorted(col, x, side="left")
        hi = np.searchsorted(col, x, side="right")
        out.append(np.clip(0.5 * (lo + hi) / len(col), 0.0, 1.0))
    return np.column_stack(out)


U = views_u(rows)
states = product_state_batch(U).astype(complex)
labels = rows["isFraud"].to_numpy()
OUT["stage2"]["rows"] = {
    "n_rows": int(len(sel)), "selection_seed": 20260904,
    "transaction_ids": [int(t) for t in rows["TransactionID"].to_numpy()],
    "regions": [int(r) for r in region[sel]],
    "time_quartile": [int(i // 8) for i in range(len(sel))],
    "backbone_scores": [float(p) for p in pval[sel]],
    "fraud_labels_disclosed_post_hoc": [int(l) for l in labels],
    "fraud_count": int(labels.sum()),
}
log(f"stage 2 rows selected: {len(sel)} (frauds {int(labels.sum())}, disclosed post hoc)")

VQ = views["view_qubits"]
name2idx = {"A": 0, "B": 1, "C": 2}
cfg = CrossViewConfig(
    n_qubits=12, views=tuple(tuple(VQ[v]) for v in ("A", "B", "C")),
    layer_pairs=tuple((name2idx[a], name2idx[b]) for a, b in views["layer_pairs"]),
    hp_views=tuple(broadcast_view_diagonal(np.array(views["hp_tables"][v]), VQ[v], 12)
                   for v in ("A", "B", "C")))
sim = CrossViewSLCU(cfg)
L = cfg.n_layers
walsh = crossview_walsh(cfg)
OUT["stage2"]["walsh_max_weight_ge3_coefficient"] = float(max(
    abs(c) for v in walsh for S, c in walsh[v].items() if len(S) >= 3))

seeds = sorted(arms["Q"]["seeds"].keys())
curves = {}
npz = {"q_grid": np.array(Q_GRID), "q_fine": np.array(Q_FINE), "U": U, "labels": labels}
S_PRIMARY, S_SECONDARY = 2000, 4000
BOOT = 2000
for sd in seeds:
    params = np.array(arms["Q"]["seeds"][sd]["params"])
    G = path_gram(sim, params, states)
    Pq = partial_dephased_distribution(sim, params, states, Q_GRID, gram=G)
    Pfine = partial_dephased_distribution(sim, params, states, Q_FINE, gram=G)
    Wq = [between_row_diversity(P) for P in Pq]
    Wfine = [between_row_diversity(P) for P in Pfine]
    Pref = partial_dephased_distribution(sim, params, states, Q_GRID, identical_branches=True)
    Wref = [between_row_diversity(P) for P in Pref]
    # sanity: coherent formula matches the production syndrome features
    dev_prod = float(np.abs(Pq[0] - sim.syndrome_distribution(params, states)).max())
    # finite-shot sampled ideal intervention (distributionally identical to the
    # ideal Aer stochastic-Z circuit; equivalence shown in stage 1)
    rec = {"W_exact": Wq, "W_fine": Wfine, "W_reference_identical_branches": Wref,
           "coherent_vs_production_syndrome_max_dev": dev_prod,
           "area_under_W": float(np.trapezoid(Wfine, Q_FINE)),
           "sampled": {}}
    for S in (S_PRIMARY, S_SECONDARY):
        srng = np.random.default_rng(777 + S)
        counts = [sample_counts(P, S, srng) for P in Pq]
        What = [unbiased_between_row_diversity(cn) for cn in counts]
        # row bootstrap of the contrast W_hat(0)-W_hat(1) (and W_hat(0)-W_hat(0.5))
        brng = np.random.default_rng(20260904)
        d01, d05 = [], []
        m = len(sel)
        for _ in range(BOOT):
            idx = brng.integers(0, m, m)
            idx = np.unique(idx) if len(np.unique(idx)) >= 2 else idx
            d01.append(unbiased_between_row_diversity(counts[0][idx])
                       - unbiased_between_row_diversity(counts[-1][idx]))
            d05.append(unbiased_between_row_diversity(counts[0][idx])
                       - unbiased_between_row_diversity(counts[2][idx]))
        # shot-replicate spread (20 independent shot draws)
        reps = []
        for r in range(20):
            rr = np.random.default_rng(5000 + 100 * S + r)
            reps.append([unbiased_between_row_diversity(sample_counts(P, S, rr)) for P in Pq])
        reps = np.array(reps)
        rec["sampled"][str(S)] = {
            "W_hat": What,
            "contrast_W0_minus_W1": {"point": What[0] - What[-1],
                                     "row_bootstrap_97.5pct": [float(np.percentile(d01, 1.25)),
                                                               float(np.percentile(d01, 98.75))]},
            "contrast_W0_minus_W05": {"point": What[0] - What[2],
                                      "row_bootstrap_97.5pct": [float(np.percentile(d05, 1.25)),
                                                                float(np.percentile(d05, 98.75))]},
            "shot_replicates_mean": reps.mean(axis=0).tolist(),
            "shot_replicates_std": reps.std(axis=0, ddof=1).tolist(),
        }
    curves[sd] = rec
    npz[f"P_q_seed{sd}"] = Pq
    npz[f"P_fine_seed{sd}"] = Pfine
    log(f"seed {sd}: W exact {['%.4g' % w for w in Wq]}; W_hat(S=2000) "
        f"{['%.4g' % w for w in rec['sampled']['2000']['W_hat']]}; "
        f"contrast 97.5% {rec['sampled']['2000']['contrast_W0_minus_W1']['row_bootstrap_97.5pct']}")
OUT["stage2"]["seeds"] = curves
# row-bootstrap unique-index note: with replacement resampling duplicates rows;
# the U-statistic on duplicated rows is biased toward zero, so duplicates are
# collapsed (unique) before recomputing; recorded as an analysis detail.
OUT["stage2"]["bootstrap_note"] = "row bootstrap collapses duplicated rows before recomputing the U-statistic (duplicates would bias W toward zero)"

# primary-1 gate: all five seeds' 97.5% intervals strictly above zero at S=2000
gate_p1 = all(curves[sd]["sampled"]["2000"]["contrast_W0_minus_W1"]["row_bootstrap_97.5pct"][0] > 0
              for sd in seeds)
OUT["stage2"]["primary1_resolvability_gate_pass_S2000"] = bool(gate_p1)
log(f"PRIMARY 1 resolvability gate (S=2000, all seeds): {'PASS' if gate_p1 else 'FAIL'}")

# ---------------------------------------------------------------------------
# Stage 2b: device-noise stand-in on the structured 15-qubit circuit (Aer)
# ---------------------------------------------------------------------------
NOISE_SETTINGS = {"low": (5e-4, 5e-3, 5e-3), "high": (2e-3, 2e-2, 2e-2)}
OUT["stage2"]["noise_standin"] = {"settings": NOISE_SETTINGS,
                                  "seed_used": str(ARGS.noise_seed_only),
                                  "shots": ARGS.noise_shots,
                                  "note": "1q depolarizing on ry/rx/h only (rz/p treated as virtual), 2q depolarizing on cx/rzz, symmetric readout error on ancillas; assumption, not a Braket device model"}
if not ARGS.skip_noise:
    sd = str(ARGS.noise_seed_only)
    params = np.array(arms["Q"]["seeds"][sd]["params"])
    S = ARGS.noise_shots
    circuits = {}
    for i in range(len(sel)):
        qc = build_structured_crossview_circuit(cfg, params, u_row=U[i], walsh=walsh,
                                                measure_ancillas=True)
        circuits[i] = qc
    ops = circuits[0].count_ops()
    OUT["stage2"]["noise_standin"]["circuit_ops"] = {k: int(v) for k, v in ops.items()}
    OUT["stage2"]["noise_standin"]["two_qubit_gates"] = int(ops.get("cx", 0) + ops.get("rzz", 0))
    # ideal-circuit sanity at q=0 through Aer (no noise): matches exact P_0
    aer0 = AerSimulator(method="statevector", seed_simulator=1)
    t0 = time.time()
    res0 = aer0.run(circuits[0], shots=S).result().get_counts()
    OUT["stage2"]["noise_standin"]["ideal_row0_q0_max_dev"] = float(
        np.abs(counts_to_pattern_probs(res0, L, S) - npz[f"P_q_seed{sd}"][0, 0]).max())
    log(f"Aer ideal sanity row 0: dev {OUT['stage2']['noise_standin']['ideal_row0_q0_max_dev']:.3e} "
        f"({time.time()-t0:.1f}s for {S} shots)")
    for tag, (p1, p2, r) in NOISE_SETTINGS.items():
        What = []
        counts_all = []
        t0 = time.time()
        for k, q in enumerate(Q_GRID):
            nm = NoiseModel(basis_gates=["ry", "rx", "rz", "h", "p", "cx", "rzz", "id"])
            nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["ry", "rx", "h"])
            nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx", "rzz"])
            for l in range(L):
                if q > 0:
                    nm.add_quantum_error(pauli_error([("Z", q / 2), ("I", 1 - q / 2)]), "id", [12 + l])
                nm.add_readout_error(ReadoutError([[1 - r, r], [r, 1 - r]]), [12 + l])
            aer = AerSimulator(method="statevector", noise_model=nm, seed_simulator=300 + k)
            cnts = []
            for i in range(len(sel)):
                res = aer.run(circuits[i], shots=S).result().get_counts()
                cnts.append(np.rint(counts_to_pattern_probs(res, L, S) * S).astype(int))
            cnts = np.stack(cnts)
            counts_all.append(cnts)
            What.append(unbiased_between_row_diversity(cnts))
            log(f"  noise {tag} q={q}: W_hat {What[-1]:.5g} ({time.time()-t0:.0f}s)")
        brng = np.random.default_rng(20260904)
        d01 = []
        m = len(sel)
        for _ in range(BOOT):
            idx = np.unique(brng.integers(0, m, m))
            d01.append(unbiased_between_row_diversity(counts_all[0][idx])
                       - unbiased_between_row_diversity(counts_all[-1][idx]))
        OUT["stage2"]["noise_standin"][tag] = {
            "W_hat": What,
            "contrast_W0_minus_W1": {"point": What[0] - What[-1],
                                     "row_bootstrap_97.5pct": [float(np.percentile(d01, 1.25)),
                                                               float(np.percentile(d01, 98.75))]},
            "wall_seconds": time.time() - t0,
        }
        npz[f"noise_{tag}_counts"] = np.stack(counts_all)
    gate_p2 = OUT["stage2"]["noise_standin"]["low"]["contrast_W0_minus_W1"]["row_bootstrap_97.5pct"][0] > 0
    OUT["stage2"]["primary2_low_noise_gate_pass"] = bool(gate_p2)
    log(f"PRIMARY 2 low-noise gate: {'PASS' if gate_p2 else 'FAIL'}")

# ---------------------------------------------------------------------------
# reduced instance for the Braket local density-matrix run (n=6, L=3)
# ---------------------------------------------------------------------------
rcfg, rparams, _ = random_config(6, [(0, 1), (2, 3), (4, 5)], [(0, 1), (1, 2), (0, 2)], 20260904)
rsim = CrossViewSLCU(rcfg)
Ured = np.column_stack([U[:8, 0:2], U[:8, 4:6], U[:8, 8:10]])
rstates = product_state_batch(Ured).astype(complex)
rP = partial_dephased_distribution(rsim, rparams, rstates, Q_GRID)
rwalsh = crossview_walsh(rcfg)
np.savez(OUT_BRAKET, q_grid=np.array(Q_GRID), U=Ured, params=rparams,
         hp_view0=np.asarray(rcfg.hp_views[0])[[0, 1, 2, 3]],
         P_exact=rP,
         walsh_keys=np.array([json.dumps({str(k): v for k, v in rwalsh[vw].items()}) for vw in range(3)]),
         views=np.array(rcfg.views), pairs=np.array(rcfg.layer_pairs))
OUT["braket_reduced_instance"] = {"path": OUT_BRAKET, "n_qubits": 6, "layers": 3, "rows": 8,
                                  "W_exact": [between_row_diversity(P) for P in rP]}

np.savez(OUT_NPZ, **npz)
OUT["artifacts"] = {"curves_npz": OUT_NPZ, "curves_npz_sha256": sha256_file(OUT_NPZ),
                    "braket_npz_sha256": sha256_file(OUT_BRAKET)}
OUT["wall_seconds"] = time.time() - T0
save_json_no_overwrite(OUT_JSON, OUT)
log(f"DONE -> {OUT_JSON}")
