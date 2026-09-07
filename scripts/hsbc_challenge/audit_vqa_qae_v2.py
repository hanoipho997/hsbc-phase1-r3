"""Prospectively frozen VQC and matched trash-QAE audit for HSBC v2.

Scope
-----
This script implements *only* sections 3.2 and 3.4(1) of
``docs/hsbc_challenge_quantum_rivals_audit_v2.md``.  It deliberately does not
implement the quantum-kernel or hybrid-QNN/autoencoder rows.

The default invocation performs deterministic synthetic simulator checks and
exits without loading HSBC/ULB data.  A real audit run requires the explicit
``--run`` flag.  Existing result artifacts are protected unless
``--overwrite`` is also supplied.

Prospectively fixed circuit conventions
----------------------------------------
VQC (entangling): on eight little-endian qubits, apply

    RY(pi*u[0:8]) -> CZ ring -> RY(theta[0:8])
    -> RY(pi*u[0:8]) -> sequential directed CNOT ring
    1->2->...->7->0, then 0->1 -> Pr(qubit 0 = 1).

The CZ ring uses the edge list (0,1),...,(6,7),(7,0); the CNOT cycle uses the
exact directed order (1,2),...,(6,7),(7,0),(0,1).  The mandatory
no-entanglement control omits both entangling rings and is otherwise identical,
including initialization, legitimate batches, SPSA perturbations, validation,
and score readout.  Weighted BCE gives total weight 1/2 to each class.

Trash-QAE applies RY(pi*u), then RY(theta[0:8]), then the same sequential
directed CNOT ring 0->1->...->7->0.  Its loss and anomaly score are the mean
occupations of trash qubits (4,5,6,7).  All QAE fitting and checkpoint
selection use legitimate rows only.

Prospectively fixed SPSA and checkpoint conventions
---------------------------------------------------
All circuit rows use exactly 300 maximum SPSA updates, no restarts, no angle
wrapping, and no gradient clipping.  For zero-based update k,

    a_k = 0.08 * ((1 + 30) / (k + 1 + 30))**0.602
    c_k = 0.10 / (k + 1)**0.101.

Each perturbation is an eight-entry iid Rademacher vector from the frozen seed;
theta is initialized iid Normal(0, 0.1) from that same generator.  The update
uses two objective calls,

    ghat = (loss(theta+c_k*delta)-loss(theta-c_k*delta))/(2*c_k) * delta
    theta <- theta-a_k*ghat.

VQC seeds are 820..824.  Every update contains all 295 training frauds plus
384 legitimate rows sampled without replacement.  Full, unresampled
validation AUPRC is checked after updates 20,40,...; an improvement is strictly
greater than the incumbent (absolute tie tolerance 1e-15), so the earliest
checkpoint wins ties.  Five consecutive non-improving checks stop training.

QAE seeds are 840..844.  Every update contains 384 legitimate rows sampled
without replacement.  Legitimate-only full-validation mean trash occupation
is checked after updates 20,40,...,300.  All 15 checkpoints are evaluated and
the earliest strict minimum (absolute tie tolerance 1e-15) is retained; the
frozen QAE protocol did not specify a patience stop.

Compute-count definition
------------------------
One ``sample_circuit_forward_equivalent`` is one exact model-probability
evaluation for one transaction at one parameter vector.  Thus an SPSA update
costs twice its batch size, a validation pass costs its number of scored rows,
and the final full-test pass costs 56,962.  Chunking is an implementation
detail and does not alter the count.  Optimizer arithmetic, metric calculation,
and the classical logistic control cost zero circuit forwards.  Training,
validation, and final test inference all count against the frozen six-million
per-seed cap.  A prospective call that would exceed the cap is not made and the
seed is retained as ``BUDGET_CENSORED``.  Non-finite loss, score, parameter, or
gradient makes the frozen seed ``DIVERGED``; failed seeds are never replaced.

Primary aggregation and outputs
-------------------------------
Each primary circuit score is the arithmetic mean of all five frozen full-test
score arrays.  If any seed fails, the ensemble is invalid and is represented by
NaNs rather than silently taking a successful-seed mean.  The real run writes
``runs/hsbc_challenge/audit_v2/vqa_qae_v2.json`` and ``vqa_qae_v2.npz`` by
default.  It also reports full-test AUPRC, AUC-ROC, recall above the legitimate
validation-score 0.999 ``method="higher"`` quantile, realized validation/test
false-positive rates, and 2,000 paired Poisson(1) row bootstraps with 98.75%
simultaneous intervals for the in-scope comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from hsbc_common import ULB_NPZ, prepare_hsbc, product_state_batch, ranking_metrics  # noqa: E402


PROTOCOL_PATH = REPO_ROOT / "docs/hsbc_challenge_quantum_rivals_audit_v2.md"
AUDIT_V2_ROOT = REPO_ROOT / "runs/hsbc_challenge/audit_v2"
DEFAULT_OUTPUT = REPO_ROOT / "runs/hsbc_challenge/audit_v2/vqa_qae_v2.json"
DEFAULT_SCORES = REPO_ROOT / "runs/hsbc_challenge/audit_v2/vqa_qae_v2.npz"
DEFAULT_REFERENCES = (
    REPO_ROOT / "runs/hsbc_challenge/audit_v1/occ_tournament_scores_v1.npz"
)
ULB_DATA_PATH = REPO_ROOT / ULB_NPZ

N_QUBITS = 8
STATE_DIM = 2**N_QUBITS
VQC_SEEDS = tuple(range(820, 825))
QAE_SEEDS = tuple(range(840, 845))
MAX_UPDATES = 300
BATCH_LEGIT = 384
CHECKPOINT_EVERY = 20
VQC_PATIENCE_CHECKS = 5
FORWARD_CAP = 6_000_000
CHUNK = 4096
TIE_ATOL = 1e-15
PROB_EPS = 1e-6
FPR_TARGET = 1e-3
N_BOOT = 2_000
BOOTSTRAP_SEED = 20260831
SIMULTANEOUS_CI = 0.9875

SPSA = {
    "a_initial": 0.08,
    "c_initial": 0.10,
    "stability": 30.0,
    "alpha": 0.602,
    "gamma": 0.101,
    "perturbation": "iid Rademacher {-1,+1}, eight entries",
    "initialization": "iid Normal(0,0.1), eight entries",
    "angle_wrapping": False,
    "gradient_clipping": False,
    "restarts": 0,
}

EXPECTED = {
    "rows": {"tr": 170_884, "va": 56_961, "te": 56_962},
    "frauds": {"tr": 295, "va": 98, "te": 99},
}
EXPECTED_FEATURE_NAMES = ("V14", "V4", "V12", "V11", "V10", "V3", "V16", "V2")
EXPECTED_SOURCE_SHA256 = {
    "ULB source": "40ad6e73b1ef5c2b42265b7a02e89caf9738f09520770451a6c8d4491f4d79d6",
    "v1 references": "be2e24943012c329ebc2203594598c24198329e8f433923961fc5b86e17c1b8c",
}

VQC_TOPOLOGY = {
    "qubits": 8,
    "statevector_dimension": 256,
    "basis_order": "little-endian; integer bit q is qubit q",
    "ordered_operations": [
        "RY(pi*u_q) for q=0..7",
        "CZ(q,(q+1) mod 8) for q=0..7",
        "RY(theta_q) for q=0..7",
        "RY(pi*u_q) for q=0..7",
        "sequential CNOT cycle 1->2->...->7->0, then 0->1",
        "measure Pr(qubit 0 = 1)",
    ],
    "trainable_angles": 8,
    "data_appearances": 2,
    "one_qubit_rotations_per_forward": 24,
    "cz_per_forward": 8,
    "cnot_per_forward": 8,
    "readout": "single-qubit probability Pr(q0=1)",
    "exact_inference_readouts_per_transaction": 1,
    "simulation": "exact dense statevector; no finite-shot or hardware claim",
    "no_entanglement_control": "omit the CZ ring and directed CNOT ring; all else identical",
}

QAE_TOPOLOGY = {
    "qubits": 8,
    "statevector_dimension": 256,
    "basis_order": "little-endian; integer bit q is qubit q",
    "ordered_operations": [
        "RY(pi*u_q) for q=0..7",
        "RY(theta_q) for q=0..7",
        "sequential CNOT q->(q+1) mod 8 for q=0..7",
        "measure mean Pr(q=1) over q=4,5,6,7",
    ],
    "trainable_angles": 8,
    "data_appearances": 1,
    "one_qubit_rotations_per_forward": 16,
    "cnot_per_forward": 8,
    "trash_qubits": [4, 5, 6, 7],
    "readout": "mean trash occupation; also anomaly score",
    "exact_inference_readouts_per_transaction": 4,
    "simulation": "exact dense statevector; no finite-shot or hardware claim",
}

COMPUTE_DEFINITION = {
    "unit": "sample_circuit_forward_equivalent",
    "definition": (
        "one exact model-probability evaluation for one transaction at one "
        "parameter vector"
    ),
    "spsa_update": "2 * batch_rows (plus and minus objective evaluations)",
    "validation": "number of validation rows scored at the checkpoint",
    "test": "number of full-test rows scored once at selected checkpoint",
    "chunking": "does not change count",
    "not_counted": ["optimizer arithmetic", "metric calculation", "logistic control"],
    "per_seed_cap": FORWARD_CAP,
    "cap_policy": (
        "reserve before evaluation; if the next call exceeds cap, do not make it "
        "and retain seed as BUDGET_CENSORED"
    ),
}


class BudgetExceeded(RuntimeError):
    """Raised before a circuit evaluation that would exceed the frozen cap."""


class Diverged(RuntimeError):
    """Raised when a frozen seed develops a non-finite value."""


@dataclass
class ForwardCounter:
    cap: int = FORWARD_CAP
    total: int = 0
    by_stage: dict[str, int] = field(default_factory=dict)

    def reserve(self, stage: str, rows: int) -> None:
        rows = int(rows)
        if rows < 0:
            raise ValueError("forward count cannot be negative")
        if self.total + rows > self.cap:
            raise BudgetExceeded(
                f"next {stage} call ({rows}) would exceed {self.cap}; current={self.total}"
            )
        self.total += rows
        self.by_stage[stage] = self.by_stage.get(stage, 0) + rows

    def metadata(self) -> dict:
        return {
            "unit": COMPUTE_DEFINITION["unit"],
            "total": int(self.total),
            "cap": int(self.cap),
            "by_stage": {k: int(v) for k, v in sorted(self.by_stage.items())},
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(repr(array.shape).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def spsa_scales(update_index_zero_based: int) -> tuple[float, float]:
    k = float(update_index_zero_based)
    a_k = SPSA["a_initial"] * (
        (1.0 + SPSA["stability"]) / (k + 1.0 + SPSA["stability"])
    ) ** SPSA["alpha"]
    c_k = SPSA["c_initial"] / (k + 1.0) ** SPSA["gamma"]
    return float(a_k), float(c_k)


def _apply_ry(states: np.ndarray, n: int, qubit: int, angle) -> np.ndarray:
    """Apply RY in place-by-copy to a batch using little-endian qubit order."""

    m = len(states)
    angle = np.asarray(angle)
    if angle.ndim == 0:
        cosine, sine = np.cos(angle / 2.0), np.sin(angle / 2.0)
    else:
        if angle.shape != (m,):
            raise ValueError(f"batched angle must have shape {(m,)}, got {angle.shape}")
        cosine = np.cos(angle / 2.0)[:, None, None]
        sine = np.sin(angle / 2.0)[:, None, None]
    reshaped = states.reshape(m, -1, 2, 2**qubit)
    zero = reshaped[:, :, 0, :].copy()
    one = reshaped[:, :, 1, :].copy()
    out = np.empty_like(reshaped)
    out[:, :, 0, :] = cosine * zero - sine * one
    out[:, :, 1, :] = sine * zero + cosine * one
    return out.reshape(m, -1)


def _ring_cz_phase(n: int) -> np.ndarray:
    indices = np.arange(2**n)
    phase = np.ones(2**n, dtype=float)
    for control in range(n):
        target = (control + 1) % n
        both_one = (((indices >> control) & 1) == 1) & (
            ((indices >> target) & 1) == 1
        )
        phase[both_one] *= -1.0
    return phase


def _apply_cnot(states: np.ndarray, control: int, target: int) -> np.ndarray:
    """Apply a CNOT permutation to a batch of little-endian statevectors."""

    if control == target:
        raise ValueError("CNOT control and target must differ")
    dimension = states.shape[1]
    indices = np.arange(dimension)
    permutation = indices.copy()
    control_on = ((indices >> control) & 1).astype(bool)
    permutation[control_on] ^= 1 << target
    # CNOT is self-inverse, so output[j] = input[permutation[j]].
    return states[:, permutation]


def _apply_directed_cnot_cycle(
    states: np.ndarray, n: int, start_control: int
) -> np.ndarray:
    """Apply one directed nearest-neighbour cycle from ``start_control``."""

    if not 0 <= start_control < n:
        raise ValueError("CNOT-cycle start control must name a qubit")
    out = states
    for offset in range(n):
        control = (start_control + offset) % n
        out = _apply_cnot(out, control, (control + 1) % n)
    return out


def _occupation(states: np.ndarray, qubit: int) -> np.ndarray:
    probabilities = np.square(np.abs(states))
    indices = np.arange(states.shape[1])
    return probabilities[:, ((indices >> qubit) & 1).astype(bool)].sum(axis=1)


def vqc_probabilities(U: np.ndarray, theta: np.ndarray, entangling: bool) -> np.ndarray:
    """Exact-statevector VQC fraud probabilities under the frozen topology."""

    U = np.asarray(U, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if U.ndim != 2 or U.shape[1] != N_QUBITS:
        raise ValueError(f"U must have shape (rows,{N_QUBITS})")
    if theta.shape != (N_QUBITS,):
        raise ValueError(f"theta must have shape ({N_QUBITS},)")
    state = product_state_batch(U).astype(float, copy=False)
    if entangling:
        state = state * _ring_cz_phase(N_QUBITS)[None, :]
    for qubit in range(N_QUBITS):
        state = _apply_ry(state, N_QUBITS, qubit, theta[qubit])
    for qubit in range(N_QUBITS):
        state = _apply_ry(state, N_QUBITS, qubit, np.pi * U[:, qubit])
    if entangling:
        state = _apply_directed_cnot_cycle(state, N_QUBITS, start_control=1)
    score = _occupation(state, qubit=0)
    if not np.all(np.isfinite(score)):
        raise Diverged("non-finite VQC probability")
    return score


def qae_trash_scores(U: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Exact mean trash occupation under the frozen eight-angle QAE."""

    U = np.asarray(U, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if U.ndim != 2 or U.shape[1] != N_QUBITS:
        raise ValueError(f"U must have shape (rows,{N_QUBITS})")
    if theta.shape != (N_QUBITS,):
        raise ValueError(f"theta must have shape ({N_QUBITS},)")
    state = product_state_batch(U).astype(float, copy=False)
    for qubit in range(N_QUBITS):
        state = _apply_ry(state, N_QUBITS, qubit, theta[qubit])
    state = _apply_directed_cnot_cycle(state, N_QUBITS, start_control=0)
    occupations = np.column_stack(
        [_occupation(state, qubit) for qubit in QAE_TOPOLOGY["trash_qubits"]]
    )
    score = occupations.mean(axis=1)
    if not np.all(np.isfinite(score)):
        raise Diverged("non-finite QAE trash score")
    return score


def _chunked_scores(
    U: np.ndarray,
    theta: np.ndarray,
    model: Callable[[np.ndarray, np.ndarray], np.ndarray],
    counter: ForwardCounter,
    stage: str,
) -> np.ndarray:
    # Preflight the complete logical pass without charging work. This preserves
    # the fail-before-evaluation cap rule while the per-chunk reservations below
    # make failure-path accounting equal the calls actually attempted.
    if counter.total + len(U) > counter.cap:
        raise BudgetExceeded(
            f"next {stage} pass ({len(U)}) would exceed {counter.cap}; "
            f"current={counter.total}"
        )
    pieces = []
    for start in range(0, len(U), CHUNK):
        chunk = U[start : start + CHUNK]
        counter.reserve(stage, len(chunk))
        pieces.append(model(chunk, theta))
    score = np.concatenate(pieces) if pieces else np.empty(0, dtype=float)
    if score.shape != (len(U),) or not np.all(np.isfinite(score)):
        raise Diverged(f"invalid score vector in {stage}")
    return score


def weighted_bce(y: np.ndarray, probability: np.ndarray) -> float:
    """BCE with exactly one-half total weight on each class."""

    y = np.asarray(y, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), PROB_EPS, 1.0 - PROB_EPS)
    fraud = y == 1
    legit = y == 0
    if not np.any(fraud) or not np.any(legit):
        raise ValueError("weighted BCE requires both classes")
    loss = -0.5 * np.mean(np.log(probability[fraud]))
    loss -= 0.5 * np.mean(np.log1p(-probability[legit]))
    if not np.isfinite(loss):
        raise Diverged("non-finite weighted BCE")
    return float(loss)


def full_metrics(
    y_test: np.ndarray,
    test_score: np.ndarray,
    validation_legitimate_score: np.ndarray,
    fpr: float = FPR_TARGET,
) -> dict:
    """Test rankings and an operating point fixed on legitimate validation only."""

    y_test = np.asarray(y_test, dtype=int)
    test_score = np.asarray(test_score, dtype=float)
    validation_legitimate_score = np.asarray(
        validation_legitimate_score, dtype=float
    )
    invalid = (
        test_score.shape != y_test.shape
        or validation_legitimate_score.ndim != 1
        or len(validation_legitimate_score) == 0
        or not np.all(np.isfinite(test_score))
        or not np.all(np.isfinite(validation_legitimate_score))
    )
    if invalid:
        return {
            "status": "INVALID_NONFINITE_SCORE",
            "auprc": None,
            "auc_roc": None,
            "operating_point_rule": (
                "score > legitimate-validation quantile(0.999, method='higher')"
            ),
            "operating_threshold": None,
            "realized_validation_legitimate_fpr": None,
            "realized_test_fpr": None,
            "test_recall_at_validation_target_fpr_1e-3": None,
        }
    legitimate_test = y_test == 0
    fraud_test = y_test == 1
    if not np.any(legitimate_test) or not np.any(fraud_test):
        raise ValueError("test metrics require both legitimate and fraud rows")
    threshold = float(
        np.quantile(
            validation_legitimate_score,
            1.0 - fpr,
            method="higher",
        )
    )
    metrics = ranking_metrics(y_test, test_score)
    return {
        "status": "SUCCESS",
        "auprc": metrics["auprc"],
        "auc_roc": metrics["auc_roc"],
        "operating_point_rule": (
            "score > legitimate-validation quantile(0.999, method='higher')"
        ),
        "operating_threshold": threshold,
        "validation_legitimate_rows": int(len(validation_legitimate_score)),
        "realized_validation_legitimate_fpr": float(
            np.mean(validation_legitimate_score > threshold)
        ),
        "realized_test_fpr": float(np.mean(test_score[legitimate_test] > threshold)),
        "test_recall_at_validation_target_fpr_1e-3": float(
            np.mean(test_score[fraud_test] > threshold)
        ),
    }


def _vqc_seed(
    seed: int,
    U_tr: np.ndarray,
    y_tr: np.ndarray,
    U_va: np.ndarray,
    y_va: np.ndarray,
    U_te: np.ndarray,
    y_te: np.ndarray,
    entangling: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train one frozen VQC seed, retaining failures rather than replacing them."""

    started = time.perf_counter()
    counter = ForwardCounter()
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 0.1, size=N_QUBITS)
    initial_theta = theta.copy()
    fraud_indices = np.flatnonzero(y_tr == 1)
    legit_indices = np.flatnonzero(y_tr == 0)
    legit_validation = np.flatnonzero(y_va == 0)
    best_theta = None
    best_validation_legitimate_score = np.full(len(legit_validation), np.nan)
    best_validation = -np.inf
    best_update = None
    checks_without_improvement = 0
    history: list[dict] = []
    updates_completed = 0
    status = "SUCCESS"
    failure = None
    model = lambda rows, params: vqc_probabilities(rows, params, entangling=entangling)

    try:
        for update in range(1, MAX_UPDATES + 1):
            chosen_legit = rng.choice(legit_indices, size=BATCH_LEGIT, replace=False)
            batch_indices = np.concatenate([fraud_indices, chosen_legit])
            U_batch = U_tr[batch_indices]
            y_batch = y_tr[batch_indices]
            delta = rng.choice(np.array([-1.0, 1.0]), size=N_QUBITS)
            a_k, c_k = spsa_scales(update - 1)
            counter.reserve("spsa_train_plus", len(U_batch))
            loss_plus = weighted_bce(y_batch, model(U_batch, theta + c_k * delta))
            counter.reserve("spsa_train_minus", len(U_batch))
            loss_minus = weighted_bce(y_batch, model(U_batch, theta - c_k * delta))
            gradient = ((loss_plus - loss_minus) / (2.0 * c_k)) * delta
            if not np.all(np.isfinite(gradient)):
                raise Diverged("non-finite SPSA gradient")
            theta = theta - a_k * gradient
            if not np.all(np.isfinite(theta)):
                raise Diverged("non-finite VQC theta")
            updates_completed = update

            if update % CHECKPOINT_EVERY == 0:
                validation_score = _chunked_scores(
                    U_va, theta, model, counter, "validation_full"
                )
                from sklearn.metrics import average_precision_score

                validation_auprc = float(average_precision_score(y_va, validation_score))
                if not np.isfinite(validation_auprc):
                    raise Diverged("non-finite validation AUPRC")
                improved = validation_auprc > best_validation + TIE_ATOL
                if improved:
                    best_validation = validation_auprc
                    best_theta = theta.copy()
                    best_validation_legitimate_score = validation_score[
                        legit_validation
                    ].copy()
                    best_update = update
                    checks_without_improvement = 0
                else:
                    checks_without_improvement += 1
                history.append(
                    {
                        "update": update,
                        "validation_auprc": validation_auprc,
                        "strict_improvement": bool(improved),
                        "checks_without_improvement": checks_without_improvement,
                        "a_k": a_k,
                        "c_k": c_k,
                    }
                )
                if checks_without_improvement >= VQC_PATIENCE_CHECKS:
                    break

        if best_theta is None:
            raise Diverged("no finite VQC checkpoint")
        test_score = _chunked_scores(U_te, best_theta, model, counter, "test_full")
        metrics = full_metrics(
            y_te, test_score, best_validation_legitimate_score
        )
    except BudgetExceeded as exc:
        status, failure = "BUDGET_CENSORED", str(exc)
        test_score = np.full(len(U_te), np.nan)
        metrics = full_metrics(
            y_te, test_score, best_validation_legitimate_score
        )
    except (Diverged, FloatingPointError) as exc:
        status, failure = "DIVERGED", str(exc)
        test_score = np.full(len(U_te), np.nan)
        metrics = full_metrics(
            y_te, test_score, best_validation_legitimate_score
        )

    metadata = {
        "seed": seed,
        "status": status,
        "failure": failure,
        "variant": "entangling" if entangling else "no_entanglement_control",
        "updates_completed": updates_completed,
        "maximum_updates": MAX_UPDATES,
        "train_rows_per_update": int(len(fraud_indices) + BATCH_LEGIT),
        "training_frauds_per_update": int(len(fraud_indices)),
        "training_legitimate_per_update": BATCH_LEGIT,
        "checkpoint_history": history,
        "checkpoint_rule": {
            "metric": "full-validation AUPRC on all unresampled rows",
            "cadence_updates": CHECKPOINT_EVERY,
            "patience_checks": VQC_PATIENCE_CHECKS,
            "tie_rule": f"earliest wins; strict improvement > incumbent + {TIE_ATOL}",
        },
        "selected_update": best_update,
        "selected_validation_auprc": (
            float(best_validation) if np.isfinite(best_validation) else None
        ),
        "selected_validation_legitimate_rows_for_operating_point": int(
            len(legit_validation)
        ),
        "initial_theta": initial_theta.tolist(),
        "selected_theta": best_theta.tolist() if best_theta is not None else None,
        "spsa": SPSA,
        "metrics": metrics,
        "resource_count": counter.metadata(),
        "wall_time_scope": "all training, validation, and test work under exact statevector simulation",
        "wall_time_seconds": float(time.perf_counter() - started),
    }
    return test_score, best_validation_legitimate_score, metadata


def _qae_seed(
    seed: int,
    U_tr: np.ndarray,
    y_tr: np.ndarray,
    U_va: np.ndarray,
    y_va: np.ndarray,
    U_te: np.ndarray,
    y_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train one frozen legitimate-only trash-QAE seed."""

    started = time.perf_counter()
    counter = ForwardCounter()
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 0.1, size=N_QUBITS)
    initial_theta = theta.copy()
    legit_train = np.flatnonzero(y_tr == 0)
    legit_validation = np.flatnonzero(y_va == 0)
    best_theta = None
    best_validation_legitimate_score = np.full(len(legit_validation), np.nan)
    best_validation = np.inf
    best_update = None
    history: list[dict] = []
    updates_completed = 0
    status = "SUCCESS"
    failure = None

    try:
        for update in range(1, MAX_UPDATES + 1):
            batch_indices = rng.choice(legit_train, size=BATCH_LEGIT, replace=False)
            U_batch = U_tr[batch_indices]
            delta = rng.choice(np.array([-1.0, 1.0]), size=N_QUBITS)
            a_k, c_k = spsa_scales(update - 1)
            counter.reserve("spsa_train_plus", len(U_batch))
            loss_plus = float(np.mean(qae_trash_scores(U_batch, theta + c_k * delta)))
            counter.reserve("spsa_train_minus", len(U_batch))
            loss_minus = float(np.mean(qae_trash_scores(U_batch, theta - c_k * delta)))
            if not np.isfinite(loss_plus) or not np.isfinite(loss_minus):
                raise Diverged("non-finite QAE SPSA loss")
            gradient = ((loss_plus - loss_minus) / (2.0 * c_k)) * delta
            if not np.all(np.isfinite(gradient)):
                raise Diverged("non-finite QAE SPSA gradient")
            theta = theta - a_k * gradient
            if not np.all(np.isfinite(theta)):
                raise Diverged("non-finite QAE theta")
            updates_completed = update

            if update % CHECKPOINT_EVERY == 0:
                validation_score = _chunked_scores(
                    U_va[legit_validation],
                    theta,
                    qae_trash_scores,
                    counter,
                    "validation_legitimate_only",
                )
                validation_loss = float(np.mean(validation_score))
                if not np.isfinite(validation_loss):
                    raise Diverged("non-finite QAE validation loss")
                improved = validation_loss < best_validation - TIE_ATOL
                if improved:
                    best_validation = validation_loss
                    best_theta = theta.copy()
                    best_validation_legitimate_score = validation_score.copy()
                    best_update = update
                history.append(
                    {
                        "update": update,
                        "validation_legitimate_mean_trash_occupation": validation_loss,
                        "strict_improvement": bool(improved),
                        "a_k": a_k,
                        "c_k": c_k,
                    }
                )

        if best_theta is None:
            raise Diverged("no finite QAE checkpoint")
        test_score = _chunked_scores(
            U_te, best_theta, qae_trash_scores, counter, "test_full"
        )
        metrics = full_metrics(
            y_te, test_score, best_validation_legitimate_score
        )
    except BudgetExceeded as exc:
        status, failure = "BUDGET_CENSORED", str(exc)
        test_score = np.full(len(U_te), np.nan)
        metrics = full_metrics(
            y_te, test_score, best_validation_legitimate_score
        )
    except (Diverged, FloatingPointError) as exc:
        status, failure = "DIVERGED", str(exc)
        test_score = np.full(len(U_te), np.nan)
        metrics = full_metrics(
            y_te, test_score, best_validation_legitimate_score
        )

    metadata = {
        "seed": seed,
        "status": status,
        "failure": failure,
        "variant": "matched_eight_angle_trash_qae",
        "updates_completed": updates_completed,
        "maximum_updates": MAX_UPDATES,
        "training_rows_per_update": BATCH_LEGIT,
        "training_information": "legitimate rows only",
        "validation_rows": int(len(legit_validation)),
        "validation_information": "legitimate rows only; fraud labels excluded",
        "checkpoint_history": history,
        "checkpoint_rule": {
            "metric": "mean trash occupation on every legitimate validation row",
            "cadence_updates": CHECKPOINT_EVERY,
            "patience": None,
            "candidate_updates": list(range(CHECKPOINT_EVERY, MAX_UPDATES + 1, CHECKPOINT_EVERY)),
            "tie_rule": f"earliest wins; strict improvement < incumbent - {TIE_ATOL}",
        },
        "selected_update": best_update,
        "selected_validation_legitimate_mean_trash_occupation": (
            float(best_validation) if np.isfinite(best_validation) else None
        ),
        "initial_theta": initial_theta.tolist(),
        "selected_theta": best_theta.tolist() if best_theta is not None else None,
        "spsa": SPSA,
        "metrics": metrics,
        "resource_count": counter.metadata(),
        "wall_time_scope": "all training, validation, and test work under exact statevector simulation",
        "wall_time_seconds": float(time.perf_counter() - started),
    }
    return test_score, best_validation_legitimate_score, metadata


def _ensemble(seed_scores: list[np.ndarray], seed_metadata: list[dict]) -> tuple[np.ndarray, str]:
    if len(seed_scores) != 5 or any(row["status"] != "SUCCESS" for row in seed_metadata):
        return np.full_like(seed_scores[0], np.nan, dtype=float), "INVALID_SEED_FAILURE"
    return np.mean(np.stack(seed_scores, axis=0), axis=0), "SUCCESS"


def _seed_family_completion(seed_metadata: list[dict]) -> dict:
    """Summarize all frozen rows without converting zero survivors to COMPLETE."""

    requested = len(seed_metadata)
    successful = sum(row["status"] == "SUCCESS" for row in seed_metadata)
    if requested and successful == requested:
        status = "COMPLETE"
    elif successful:
        status = "PARTIAL_FAILED_SEEDS"
    else:
        status = "FAILED_NO_SURVIVOR"
    return {
        "status": status,
        "requested_seed_count": requested,
        "successful_seed_count": successful,
        "failed_seed_count": requested - successful,
        "failed_seed_rows_retained": True,
        "resource_counters_retained_for_failed_seeds": True,
    }


def _fit_logistic_control(
    U_tr: np.ndarray,
    y_tr: np.ndarray,
    U_va: np.ndarray,
    y_va: np.ndarray,
    U_te: np.ndarray,
    y_te: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fixed, untuned balanced logistic control on the same eight inputs."""

    from sklearn.linear_model import LogisticRegression

    started = time.perf_counter()
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=3000,
        tol=1e-8,
        random_state=20260831,
    )
    model.fit(U_tr, y_tr)
    validation_score = model.predict_proba(U_va)[:, 1]
    test_score = model.predict_proba(U_te)[:, 1]
    metadata = {
        "status": "SUCCESS",
        "information_bracket": "supervised",
        "inputs": "the same eight transformed U features",
        "selection": "none; C and solver fixed prospectively",
        "class_weight": "balanced",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 3000,
        "tol": 1e-8,
        "n_iter": int(np.max(model.n_iter_)),
        "converged_before_cap": bool(np.max(model.n_iter_) < 3000),
        "parameters_including_intercept": int(model.coef_.size + model.intercept_.size),
        "validation_ranking_metrics_report_only_not_selected": ranking_metrics(
            y_va, validation_score
        ),
        "metrics": full_metrics(
            y_te,
            test_score,
            validation_score[y_va == 0],
        ),
        "circuit_forward_equivalents": 0,
        "wall_time_seconds": float(time.perf_counter() - started),
    }
    return test_score, metadata


def _weighted_ap_sorted(
    y_sorted: np.ndarray, weights_sorted: np.ndarray, group_ends: np.ndarray
) -> np.ndarray:
    positive_weight = weights_sorted * y_sorted[None, :]
    cumulative_positive = np.cumsum(positive_weight, axis=1, dtype=float)
    cumulative_all = np.cumsum(weights_sorted, axis=1, dtype=float)
    total_positive = cumulative_positive[:, -1]
    positive_at_end = cumulative_positive[:, group_ends]
    all_at_end = cumulative_all[:, group_ends]
    precision = np.divide(
        positive_at_end,
        all_at_end,
        out=np.zeros_like(positive_at_end),
        where=all_at_end > 0,
    )
    group_positive = np.diff(
        np.column_stack([np.zeros(len(weights_sorted)), positive_at_end]), axis=1
    )
    return np.divide(
        np.sum(precision * group_positive, axis=1),
        total_positive,
        out=np.full(len(weights_sorted), np.nan),
        where=total_positive > 0,
    )


def paired_poisson_summary(
    y: np.ndarray,
    score_map: dict[str, np.ndarray],
    comparison_pairs: list[tuple[str, str, str]],
) -> dict:
    """One common set of Poisson row weights for all 98.75% comparisons."""

    from sklearn.metrics import average_precision_score

    y = np.asarray(y, dtype=int)
    finite_score_map = {
        name: np.asarray(score, dtype=float)
        for name, score in score_map.items()
        if np.all(np.isfinite(score))
    }
    omitted = sorted(set(score_map) - set(finite_score_map))
    orders = {
        name: np.argsort(-np.asarray(score), kind="mergesort")
        for name, score in finite_score_map.items()
    }
    group_ends = {}
    for name, order in orders.items():
        sorted_score = finite_score_map[name][order]
        group_ends[name] = np.flatnonzero(
            np.r_[sorted_score[1:] != sorted_score[:-1], True]
        )
    point = {
        name: float(average_precision_score(y, score))
        for name, score in finite_score_map.items()
    }
    draws: dict[str, list[float]] = {name: [] for name in finite_score_map}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    made = 0
    while draws and made < N_BOOT:
        take = min(16, N_BOOT - made)
        weights = rng.poisson(1.0, size=(take, len(y))).astype(np.float32)
        valid = (weights[:, y == 1].sum(axis=1) > 0) & (
            weights[:, y == 0].sum(axis=1) > 0
        )
        if not np.any(valid):
            continue
        weights = weights[valid]
        for name, order in orders.items():
            values = _weighted_ap_sorted(y[order], weights[:, order], group_ends[name])
            draws[name].extend(values[np.isfinite(values)].tolist())
        made = min(len(values) for values in draws.values())
    arrays = {name: np.asarray(values[:N_BOOT]) for name, values in draws.items()}
    tail = (1.0 - SIMULTANEOUS_CI) / 2.0
    quantiles = [tail, 1.0 - tail]
    comparisons = {}
    for label, first, second in comparison_pairs:
        if first not in arrays or second not in arrays:
            missing = [name for name in (first, second) if name not in arrays]
            comparisons[label] = {
                "status": "NOT_AVAILABLE",
                "first": first,
                "second": second,
                "missing_nonfinite_methods": missing,
                "reason": "one or both named score arrays are non-finite",
            }
            continue
        delta = arrays[first] - arrays[second]
        interval = [float(value) for value in np.quantile(delta, quantiles)]
        point_delta = float(point[first] - point[second])
        if interval[0] <= 0.0 <= interval[1] or abs(point_delta) <= 0.01:
            decision = "PRACTICAL_TIE"
        elif interval[0] > 0.0:
            decision = "FIRST_WINS"
        elif interval[1] < 0.0:
            decision = "SECOND_WINS"
        else:
            decision = "PRACTICAL_TIE"
        comparisons[label] = {
            "status": "AVAILABLE",
            "first": first,
            "second": second,
            "point_auprc_first_minus_second": point_delta,
            "simultaneous_ci_98_75_percent": interval,
            "decision": decision,
        }
    available_pairs = sum(
        row["status"] == "AVAILABLE" for row in comparisons.values()
    )
    if available_pairs == len(comparison_pairs):
        status = "SUCCESS"
    elif available_pairs:
        status = "PARTIAL_WITH_UNAVAILABLE_PAIRS"
    else:
        status = "NO_AVAILABLE_COMPARISONS"
    return {
        "status": status,
        "bootstrap": "paired Poisson(1) weights shared across methods and rows",
        "seed": BOOTSTRAP_SEED,
        "n_boot": N_BOOT if arrays else 0,
        "requested_n_boot": N_BOOT,
        "simultaneous_interval_level": SIMULTANEOUS_CI,
        "method_point_auprc": point,
        "method_simultaneous_ci_98_75_percent": {
            name: [float(value) for value in np.quantile(values, quantiles)]
            for name, values in arrays.items()
        },
        "omitted_nonfinite_methods": omitted,
        "available_comparison_count": available_pairs,
        "requested_comparison_count": len(comparison_pairs),
        "comparisons": comparisons,
    }


def _validate_data(pipe: dict) -> None:
    observed_feature_names = tuple(str(value) for value in pipe["feature_names"])
    if observed_feature_names != EXPECTED_FEATURE_NAMES:
        raise RuntimeError(
            "frozen feature identity mismatch: "
            f"observed={observed_feature_names}, expected={EXPECTED_FEATURE_NAMES}"
        )
    feature_indices = np.asarray(pipe["feat"], dtype=int)
    if feature_indices.shape != (N_QUBITS,) or len(np.unique(feature_indices)) != N_QUBITS:
        raise RuntimeError(f"invalid frozen feature indices: {feature_indices.tolist()}")
    names_from_source_columns = tuple(
        str(pipe["cols"][index]) for index in feature_indices
    )
    if names_from_source_columns != EXPECTED_FEATURE_NAMES:
        raise RuntimeError(
            "feature indices do not resolve to the frozen source columns: "
            f"{names_from_source_columns}"
        )

    all_labels = np.asarray(pipe["y"], dtype=int)
    split_indices = []
    for split in ("tr", "va", "te"):
        U = pipe["U"][split]
        y = np.asarray(pipe["y_split"][split], dtype=int)
        indices = np.asarray(pipe["idx"][split], dtype=int)
        if U.shape != (EXPECTED["rows"][split], N_QUBITS):
            raise RuntimeError(f"unexpected {split} U shape: {U.shape}")
        if len(y) != EXPECTED["rows"][split]:
            raise RuntimeError(f"unexpected {split} y length: {len(y)}")
        if indices.shape != (EXPECTED["rows"][split],):
            raise RuntimeError(f"unexpected {split} row-index shape: {indices.shape}")
        if not np.array_equal(y, all_labels[indices]):
            raise RuntimeError(f"{split} row-label identity mismatch against ULB source")
        if int(np.sum(y)) != EXPECTED["frauds"][split]:
            raise RuntimeError(f"unexpected {split} fraud count: {int(np.sum(y))}")
        if not np.all(np.isin(y, (0, 1))):
            raise RuntimeError(f"non-binary labels in {split}")
        if not np.all(np.isfinite(U)) or np.min(U) < 0.0 or np.max(U) > 1.0:
            raise RuntimeError(f"invalid transformed values in {split}")
        split_indices.append(indices)
    joined_indices = np.concatenate(split_indices)
    if len(np.unique(joined_indices)) != len(joined_indices):
        raise RuntimeError("frozen train/validation/test row indices overlap")
    if len(joined_indices) != len(all_labels) or not np.array_equal(
        np.sort(joined_indices), np.arange(len(all_labels))
    ):
        raise RuntimeError("frozen train/validation/test indices do not partition ULB rows")


def synthetic_self_check() -> dict:
    """Deterministic, data-free checks; never calls ``prepare_hsbc``."""

    rng = np.random.default_rng(1701)
    U = rng.uniform(0.0, 1.0, size=(7, N_QUBITS))
    theta = rng.normal(0.0, 0.2, size=N_QUBITS)

    encoded = product_state_batch(U)
    norms = np.sum(encoded**2, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-12):
        raise AssertionError("angle encoding lost normalization")

    entangling_score = vqc_probabilities(U, theta, entangling=True)
    no_entanglement_score = vqc_probabilities(U, theta, entangling=False)
    qae_score = qae_trash_scores(U, theta)
    for name, score in (
        ("vqc_entangling", entangling_score),
        ("vqc_no_entanglement", no_entanglement_score),
        ("qae", qae_score),
    ):
        if score.shape != (len(U),) or np.min(score) < -1e-12 or np.max(score) > 1 + 1e-12:
            raise AssertionError(f"{name} returned invalid probabilities")
    if np.allclose(entangling_score, no_entanglement_score, atol=1e-12):
        raise AssertionError("entangling and no-entanglement VQC controls are identical")
    expected_no_ent_q0 = np.sin(np.pi * U[:, 0] + theta[0] / 2.0) ** 2
    if not np.allclose(no_entanglement_score, expected_no_ent_q0, atol=1e-12):
        raise AssertionError("no-entanglement VQC disagrees with its analytic q0 result")

    sensitivity_step = 0.731
    angle_sensitivity = {}
    for name, scorer in (
        ("vqc_entangling", lambda params: vqc_probabilities(U, params, entangling=True)),
        ("qae", lambda params: qae_trash_scores(U, params)),
    ):
        baseline = scorer(theta)
        changes = []
        for index in range(N_QUBITS):
            perturbed = theta.copy()
            perturbed[index] += sensitivity_step
            changes.append(float(np.max(np.abs(scorer(perturbed) - baseline))))
        if any(change <= 1e-9 for change in changes):
            raise AssertionError(
                f"{name} is not sensitive to every trainable angle: {changes}"
            )
        angle_sensitivity[name] = {
            "perturbation": sensitivity_step,
            "max_absolute_score_change_by_angle": changes,
            "minimum_change": float(min(changes)),
            "required_strictly_greater_than": 1e-9,
        }

    qae_zero = qae_trash_scores(np.zeros((1, N_QUBITS)), np.zeros(N_QUBITS))
    qae_basis_q6 = np.zeros((1, N_QUBITS))
    qae_basis_q6[0, 6] = 1.0
    qae_basis_score = qae_trash_scores(qae_basis_q6, np.zeros(N_QUBITS))
    if not np.allclose(qae_zero, 0.0, atol=1e-12):
        raise AssertionError("QAE does not preserve the all-zero state")
    if not np.allclose(qae_basis_score, 0.5, atol=1e-12):
        raise AssertionError("QAE directed-ring/trash readout failed a basis-state oracle")

    basis = np.eye(4)
    mapped = _apply_cnot(basis, control=0, target=1)
    expected_permutation = np.array([0, 3, 2, 1])
    if not np.array_equal(mapped, basis[:, expected_permutation]):
        raise AssertionError("CNOT little-endian permutation is wrong")
    if not np.array_equal(_apply_cnot(mapped, 0, 1), basis):
        raise AssertionError("CNOT is not self-inverse")

    operating_oracle = full_metrics(
        np.array([0, 0, 1, 1]),
        np.array([1.0, 3.0, 2.0, 4.0]),
        np.array([0.0, 1.0, 2.0, 3.0]),
        fpr=0.5,
    )
    expected_operating_values = {
        "operating_threshold": 2.0,
        "realized_validation_legitimate_fpr": 0.25,
        "realized_test_fpr": 0.5,
        "test_recall_at_validation_target_fpr_1e-3": 0.5,
    }
    for key, expected_value in expected_operating_values.items():
        if not np.isclose(operating_oracle[key], expected_value, atol=0.0):
            raise AssertionError(
                f"validation-fixed operating-point oracle failed for {key}"
            )

    no_survivor = _seed_family_completion(
        [{"status": "DIVERGED"}, {"status": "BUDGET_CENSORED"}]
    )
    if no_survivor["status"] != "FAILED_NO_SURVIVOR":
        raise AssertionError("zero-survivor family was incorrectly marked complete")

    scales = np.asarray([spsa_scales(k) for k in range(MAX_UPDATES)])
    if not (np.all(np.diff(scales[:, 0]) < 0) and np.all(np.diff(scales[:, 1]) < 0)):
        raise AssertionError("SPSA schedules must decrease strictly")

    counter = ForwardCounter(cap=10)
    counter.reserve("first", 4)
    try:
        counter.reserve("overflow", 7)
    except BudgetExceeded:
        pass
    else:
        raise AssertionError("forward cap did not fail closed")
    if counter.total != 4:
        raise AssertionError("rejected reservation changed the forward count")

    return {
        "status": "PASS",
        "uses_hsbc_data": False,
        "checks": [
            "angle-encoding normalization",
            "VQC/QAE output range and shape",
            "entangling-control nonidentity on fixed synthetic rows",
            "no-entanglement VQC analytic q0 oracle",
            "nonzero score sensitivity to every VQC and QAE trainable angle",
            "QAE zero-state and basis-state trash-score oracles",
            "little-endian CNOT truth table and involution",
            "legitimate-validation higher-quantile threshold and strict comparison",
            "zero-survivor family cannot report COMPLETE",
            "strictly decreasing frozen SPSA schedules",
            "fail-closed circuit-forward cap",
        ],
        "angle_sensitivity": angle_sensitivity,
        "synthetic_rows": len(U),
    }


def _artifact_manifest(arrays: dict[str, np.ndarray]) -> dict:
    return {
        name: {
            "shape": list(np.asarray(array).shape),
            "dtype": str(np.asarray(array).dtype),
            "sha256": sha256_array(np.asarray(array)),
        }
        for name, array in sorted(arrays.items())
    }


def _same_existing_file(first: Path, second: Path) -> bool:
    return first.exists() and second.exists() and os.path.samefile(first, second)


def _validate_output_targets(
    output: Path,
    scores_output: Path,
    references_path: Path,
) -> tuple[Path, Path]:
    """Enforce clarification-3 target containment and immutable-input separation."""

    output = output.resolve(strict=False)
    scores_output = scores_output.resolve(strict=False)
    audit_root = AUDIT_V2_ROOT.resolve(strict=False)
    targets = (("JSON", output, ".json"), ("NPZ", scores_output, ".npz"))
    for label, target, suffix in targets:
        if not target.is_relative_to(audit_root) or target == audit_root:
            raise ValueError(f"{label} target must be a file under {audit_root}: {target}")
        if target.suffix.lower() != suffix:
            raise ValueError(f"{label} target must end in {suffix}: {target}")
        if target.exists() and not target.is_file():
            raise ValueError(f"{label} target exists but is not a regular file: {target}")
    if output == scores_output or _same_existing_file(output, scores_output):
        raise ValueError("JSON and NPZ output targets must be distinct files")

    immutable_sources = {
        "script": Path(__file__).resolve(),
        "protocol": PROTOCOL_PATH.resolve(),
        "ULB source": ULB_DATA_PATH.resolve(),
        "v1 references": references_path.resolve(),
    }
    for target_label, target, _ in targets:
        for source_label, source in immutable_sources.items():
            if target == source or _same_existing_file(target, source):
                raise ValueError(
                    f"{target_label} target aliases immutable {source_label}: {target}"
                )
    return output, scores_output


def _output_snapshot(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    if not path.is_file():
        raise ValueError(f"output target is not a regular file: {path}")
    stat = path.stat()
    return {
        "exists": True,
        "sha256": sha256_file(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def _make_stage_path(final_path: Path, suffix: str) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{final_path.name}.stage-",
        suffix=suffix,
        dir=final_path.parent,
    )
    os.close(descriptor)
    return Path(name)


def _verify_staged_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with np.load(path, allow_pickle=False) as staged:
        if set(staged.files) != set(arrays):
            raise RuntimeError("staged NPZ keys do not match the in-memory score bundle")
        for name, expected in arrays.items():
            observed = np.asarray(staged[name])
            if sha256_array(observed) != sha256_array(np.asarray(expected)):
                raise RuntimeError(f"staged NPZ array mismatch: {name}")


def _publish_staged_pair(
    output: Path,
    scores_output: Path,
    arrays: dict[str, np.ndarray],
    payload: dict,
    initial_snapshots: dict[str, dict],
    immutable_hashes: dict[str, tuple[Path, str]],
) -> None:
    """Stage and validate both artifacts before coordinated final publication."""

    staged_scores = _make_stage_path(scores_output, ".npz")
    staged_json = _make_stage_path(output, ".json")
    backups: dict[str, Path] = {}
    lock_path = AUDIT_V2_ROOT / ".vqa_qae_v2.publish.lock"
    lock_descriptor = None
    lock_acquired = False
    published = False
    try:
        np.savez_compressed(staged_scores, **arrays)
        _verify_staged_npz(staged_scores, arrays)
        payload["scores_sha256"] = sha256_file(staged_scores)
        payload["publication"] = {
            "policy": "both artifacts staged and verified before coordinated final publication",
            "root": str(AUDIT_V2_ROOT.resolve()),
            "json_target": str(output),
            "npz_target": str(scores_output),
        }
        staged_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        decoded = json.loads(staged_json.read_text())
        if decoded.get("scores_sha256") != payload["scores_sha256"]:
            raise RuntimeError("staged JSON does not retain the staged NPZ hash")
        for label, (source_path, expected_hash) in immutable_hashes.items():
            if sha256_file(source_path) != expected_hash:
                raise RuntimeError(
                    f"{label} changed while outputs were staged; refusing mixed-provenance publication"
                )

        AUDIT_V2_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            lock_descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise RuntimeError(f"another vqa/qae publication holds {lock_path}") from exc
        lock_acquired = True
        os.write(lock_descriptor, f"pid={os.getpid()}\n".encode())
        os.close(lock_descriptor)
        lock_descriptor = None

        current_snapshots = {
            "json": _output_snapshot(output),
            "npz": _output_snapshot(scores_output),
        }
        if current_snapshots != initial_snapshots:
            raise RuntimeError(
                "output targets changed during execution; refusing to overwrite concurrent artifacts"
            )

        for label, final_path in (("json", output), ("npz", scores_output)):
            if initial_snapshots[label]["exists"]:
                backup = _make_stage_path(final_path, ".backup")
                shutil.copy2(final_path, backup)
                backups[label] = backup

        try:
            staged_scores.replace(scores_output)
            staged_json.replace(output)
            published = True
        except Exception:
            for label, final_path in (("json", output), ("npz", scores_output)):
                backup = backups.get(label)
                if backup is not None and backup.exists():
                    backup.replace(final_path)
                elif not initial_snapshots[label]["exists"] and final_path.exists():
                    final_path.unlink()
            raise
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if lock_acquired and lock_path.exists():
            lock_path.unlink()
        for path in (staged_scores, staged_json, *backups.values()):
            if path.exists():
                path.unlink()
    if not published:
        raise RuntimeError("staged output pair was not published")


def run_hsbc(
    output: Path,
    scores_output: Path,
    references_path: Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Execute the explicitly authorized, prospectively frozen HSBC run."""

    if Path.cwd().resolve() != REPO_ROOT:
        raise RuntimeError(
            f"run from repository root {REPO_ROOT}; hsbc_common uses a frozen relative data path"
        )
    references_path = references_path.resolve()
    output, scores_output = _validate_output_targets(
        output, scores_output, references_path
    )
    initial_output_snapshots = {
        "json": _output_snapshot(output),
        "npz": _output_snapshot(scores_output),
    }
    if not overwrite and any(
        snapshot["exists"] for snapshot in initial_output_snapshots.values()
    ):
        raise FileExistsError(
            "v2 output already exists; refuse silent replacement (pass --overwrite explicitly)"
        )
    if not ULB_DATA_PATH.exists():
        raise FileNotFoundError(f"frozen ULB artifact missing: {ULB_DATA_PATH}")
    if not references_path.exists():
        raise FileNotFoundError(f"frozen v1 reference artifact missing: {references_path}")
    overall_started = time.perf_counter()
    run_started_utc = datetime.now(timezone.utc).isoformat()
    script_path = Path(__file__).resolve()
    script_sha256_at_start = sha256_file(script_path)
    protocol_sha256_at_start = sha256_file(PROTOCOL_PATH)
    ulb_sha256_at_start = sha256_file(ULB_DATA_PATH)
    references_sha256_at_start = sha256_file(references_path)
    observed_frozen_source_hashes = {
        "ULB source": ulb_sha256_at_start,
        "v1 references": references_sha256_at_start,
    }
    if observed_frozen_source_hashes != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            "frozen source hash mismatch before model scoring: "
            f"observed={observed_frozen_source_hashes}, "
            f"expected={EXPECTED_SOURCE_SHA256}"
        )
    self_check = synthetic_self_check()
    pipe = prepare_hsbc(N_QUBITS, seed=0)
    _validate_data(pipe)
    U_tr, U_va, U_te = (pipe["U"][split] for split in ("tr", "va", "te"))
    y_tr, y_va, y_te = (
        np.asarray(pipe["y_split"][split], dtype=int) for split in ("tr", "va", "te")
    )

    references = np.load(references_path, allow_pickle=False)
    required = ("n8_y", "n8_S1_analytic", "n8_S2_L1_seed500")
    missing = [name for name in required if name not in references.files]
    if missing:
        raise RuntimeError(f"missing frozen v1 reference arrays: {missing}")
    reference_y = np.asarray(references["n8_y"], dtype=int)
    if reference_y.shape != y_te.shape or not np.array_equal(y_te, reference_y):
        raise RuntimeError("v2 test labels do not match frozen v1 reference rows")
    for name in required[1:]:
        values = np.asarray(references[name], dtype=float)
        if values.shape != y_te.shape or not np.all(np.isfinite(values)):
            raise RuntimeError(f"invalid frozen v1 reference score array: {name}")
    pre_score_validation = {
        "status": "PASS",
        "performed_before_model_or_test_scoring": True,
        "expected_split_rows_and_frauds": EXPECTED,
        "expected_feature_names": list(EXPECTED_FEATURE_NAMES),
        "observed_feature_names": list(pipe["feature_names"]),
        "source_hashes_expected": EXPECTED_SOURCE_SHA256,
        "source_hashes_observed": observed_frozen_source_hashes,
        "source_hash_match": True,
        "split_row_label_identity_against_ulb_source": True,
        "test_row_label_identity_against_v1_references": True,
    }

    arrays: dict[str, np.ndarray] = {"n8_y": y_te.copy()}
    vqc_ent_scores, vqc_ent_meta = [], []
    vqc_noent_scores, vqc_noent_meta = [], []
    vqc_ent_validation_legitimate = []
    vqc_noent_validation_legitimate = []
    for seed in VQC_SEEDS:
        score, validation_legitimate_score, metadata = _vqc_seed(
            seed, U_tr, y_tr, U_va, y_va, U_te, y_te, entangling=True
        )
        arrays[f"vqc_entangling_seed{seed}"] = score
        vqc_ent_scores.append(score)
        vqc_ent_validation_legitimate.append(validation_legitimate_score)
        vqc_ent_meta.append(metadata)

        score, validation_legitimate_score, metadata = _vqc_seed(
            seed, U_tr, y_tr, U_va, y_va, U_te, y_te, entangling=False
        )
        arrays[f"vqc_no_entanglement_seed{seed}"] = score
        vqc_noent_scores.append(score)
        vqc_noent_validation_legitimate.append(validation_legitimate_score)
        vqc_noent_meta.append(metadata)

    vqc_ensemble, vqc_status = _ensemble(vqc_ent_scores, vqc_ent_meta)
    noent_ensemble, noent_status = _ensemble(vqc_noent_scores, vqc_noent_meta)
    vqc_validation_ensemble, _ = _ensemble(
        vqc_ent_validation_legitimate, vqc_ent_meta
    )
    noent_validation_ensemble, _ = _ensemble(
        vqc_noent_validation_legitimate, vqc_noent_meta
    )
    arrays["vqc_entangling_ensemble_mean5"] = vqc_ensemble
    arrays["vqc_no_entanglement_ensemble_mean5"] = noent_ensemble

    logistic_score, logistic_meta = _fit_logistic_control(
        U_tr, y_tr, U_va, y_va, U_te, y_te
    )
    arrays["balanced_logistic_control"] = logistic_score

    qae_scores, qae_meta = [], []
    qae_validation_legitimate = []
    for seed in QAE_SEEDS:
        score, validation_legitimate_score, metadata = _qae_seed(
            seed, U_tr, y_tr, U_va, y_va, U_te, y_te
        )
        arrays[f"qae_trash_seed{seed}"] = score
        qae_scores.append(score)
        qae_validation_legitimate.append(validation_legitimate_score)
        qae_meta.append(metadata)
    qae_ensemble, qae_status = _ensemble(qae_scores, qae_meta)
    qae_validation_ensemble, _ = _ensemble(
        qae_validation_legitimate, qae_meta
    )
    arrays["qae_trash_ensemble_mean5"] = qae_ensemble

    arrays["reference_S1_analytic"] = np.asarray(references["n8_S1_analytic"])
    arrays["reference_S2_L1_seed500"] = np.asarray(references["n8_S2_L1_seed500"])
    if "n8_XGB_subset" in references.files:
        arrays["reference_XGB_subset"] = np.asarray(references["n8_XGB_subset"])

    score_map = {
        "S2_L1_seed500": arrays["reference_S2_L1_seed500"],
        "VQC_entangling_mean5": vqc_ensemble,
        "VQC_no_entanglement_mean5": noent_ensemble,
        "balanced_logistic": logistic_score,
        "S1_analytic": arrays["reference_S1_analytic"],
        "trash_QAE_mean5": qae_ensemble,
    }
    comparisons = paired_poisson_summary(
        y_te,
        score_map,
        [
            ("S2_stack_minus_VQC", "S2_L1_seed500", "VQC_entangling_mean5"),
            ("VQC_minus_logistic", "VQC_entangling_mean5", "balanced_logistic"),
            (
                "VQC_minus_no_entanglement",
                "VQC_entangling_mean5",
                "VQC_no_entanglement_mean5",
            ),
            ("S1_stack_minus_QAE", "S1_analytic", "trash_QAE_mean5"),
        ],
    )
    family_completion = {
        "vqc_entangling": _seed_family_completion(vqc_ent_meta),
        "vqc_no_entanglement": _seed_family_completion(vqc_noent_meta),
        "trash_qae": _seed_family_completion(qae_meta),
    }
    family_statuses = {
        row["status"] for row in family_completion.values()
    }
    if "FAILED_NO_SURVIVOR" in family_statuses:
        overall_status = "FAILED_NO_SURVIVOR"
    elif family_statuses != {"COMPLETE"}:
        overall_status = "PARTIAL_FAILED_SEEDS"
    elif comparisons["status"] != "SUCCESS":
        overall_status = "PARTIAL_UNAVAILABLE_COMPARISONS"
    else:
        overall_status = "COMPLETE"

    if sha256_file(script_path) != script_sha256_at_start:
        raise RuntimeError("script changed during execution; refuse to write mixed-provenance output")
    if sha256_file(PROTOCOL_PATH) != protocol_sha256_at_start:
        raise RuntimeError("protocol changed during execution; refuse to write mixed-provenance output")
    if sha256_file(references_path) != references_sha256_at_start:
        raise RuntimeError("v1 references changed during execution; refuse mixed-provenance output")
    if sha256_file(ULB_DATA_PATH) != ulb_sha256_at_start:
        raise RuntimeError("ULB source changed during execution; refuse mixed-provenance output")

    result = {
        "schema": "hsbc-quantum-rivals-vqa-qae-v2/1.0.0",
        "status": overall_status,
        "run_started_utc": run_started_utc,
        "run_completed_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(script_path),
        "script_sha256": script_sha256_at_start,
        "protocol": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha256_at_start,
        "scope": ["section 3.2 re-uploading VQC", "section 3.4(1) matched trash-QAE"],
        "explicitly_out_of_scope": [
            "section 3.1 quantum kernels",
            "section 3.3 hybrid QNN",
            "sections 3.4(2)-(3) hybrid autoencoder and mandatory twins",
        ],
        "prospective_implementation_freeze": {
            "vqc_topology": VQC_TOPOLOGY,
            "qae_topology": QAE_TOPOLOGY,
            "spsa": SPSA,
            "compute_count": COMPUTE_DEFINITION,
            "maximum_updates": MAX_UPDATES,
            "checkpoint_every_updates": CHECKPOINT_EVERY,
            "vqc_patience_checks": VQC_PATIENCE_CHECKS,
            "tie_absolute_tolerance": TIE_ATOL,
            "primary_ensemble": "arithmetic mean of all five frozen full-test score arrays",
            "failed_ensemble_policy": "invalid NaN ensemble; never mean only successful seeds",
            "simulation_disclosure": (
                "exact dense statevector is a simulation ceiling, not a shot-matched or "
                "hardware result"
            ),
        },
        "synthetic_self_check": self_check,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
        "data": {
            "dataset": "ULB credit-card fraud",
            "source_path": str(ULB_DATA_PATH),
            "source_sha256": ulb_sha256_at_start,
            "prepare_call": "prepare_hsbc(8, seed=0)",
            "rows": EXPECTED["rows"],
            "frauds": EXPECTED["frauds"],
            "feature_indices": [int(value) for value in pipe["feat"]],
            "feature_names": list(pipe["feature_names"]),
            "representation_disclosure": (
                "label-screened eight-feature representation; QAE is legitimate-only only "
                "conditional on that supervised screen"
            ),
            "split_array_sha256": {
                "train_labels": sha256_array(y_tr),
                "validation_labels": sha256_array(y_va),
                "test_labels": sha256_array(y_te),
                "train_transformed_features": sha256_array(U_tr),
                "validation_transformed_features": sha256_array(U_va),
                "test_transformed_features": sha256_array(U_te),
            },
            "pre_score_validation": pre_score_validation,
        },
        "references": {
            "path": str(references_path),
            "sha256": references_sha256_at_start,
            "arrays": [name for name in required if name != "n8_y"],
        },
        "vqc": {
            "information_bracket": "supervised",
            "topology": VQC_TOPOLOGY,
            "seeds": list(VQC_SEEDS),
            "entangling_family_completion": family_completion["vqc_entangling"],
            "entangling_per_seed": vqc_ent_meta,
            "entangling_ensemble": {
                "status": vqc_status,
                "aggregation": "arithmetic mean of all five frozen seed score arrays",
                "metrics": full_metrics(
                    y_te, vqc_ensemble, vqc_validation_ensemble
                ),
            },
            "no_entanglement_family_completion": family_completion[
                "vqc_no_entanglement"
            ],
            "no_entanglement_per_seed": vqc_noent_meta,
            "no_entanglement_ensemble": {
                "status": noent_status,
                "aggregation": "arithmetic mean of all five frozen seed score arrays",
                "metrics": full_metrics(
                    y_te, noent_ensemble, noent_validation_ensemble
                ),
            },
            "balanced_logistic_control": logistic_meta,
        },
        "qae": {
            "information_bracket": "conditional one-class on label-screened representation",
            "topology": QAE_TOPOLOGY,
            "seeds": list(QAE_SEEDS),
            "family_completion": family_completion["trash_qae"],
            "per_seed": qae_meta,
            "ensemble": {
                "status": qae_status,
                "aggregation": "arithmetic mean of all five frozen seed score arrays",
                "metrics": full_metrics(
                    y_te, qae_ensemble, qae_validation_ensemble
                ),
            },
            "classical_autoencoder_twin": "NOT_IN_SCOPE: frozen section 3.4(3), not this script",
        },
        "paired_bootstrap": comparisons,
        "decision_scope": (
            "Only these preregistered VQC and matched trash-QAE instances; no family-wide "
            "or quantum-advantage inference is licensed"
        ),
        "scores_output": str(scores_output),
        "score_array_manifest": _artifact_manifest(arrays),
        "wall_time_seconds": float(time.perf_counter() - overall_started),
    }

    _publish_staged_pair(
        output,
        scores_output,
        arrays,
        result,
        initial_output_snapshots,
        {
            "script": (script_path, script_sha256_at_start),
            "protocol": (PROTOCOL_PATH, protocol_sha256_at_start),
            "v1 references": (references_path, references_sha256_at_start),
            "ULB source": (ULB_DATA_PATH, ulb_sha256_at_start),
        },
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="explicitly authorize loading ULB/HSBC data and writing v2 outcomes",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scores-output", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of existing v2 JSON/NPZ artifacts (requires --run)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    self_check = synthetic_self_check()
    if not args.run:
        print(json.dumps(self_check, indent=2, sort_keys=True))
        print("HSBC/ULB data NOT loaded; pass --run for the frozen audit execution.")
        return 0

    result = run_hsbc(
        args.output,
        args.scores_output,
        args.references,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "scores_output": str(args.scores_output.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
