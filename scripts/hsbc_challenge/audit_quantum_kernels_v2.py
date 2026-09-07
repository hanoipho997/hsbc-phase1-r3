"""Preregistered HSBC v2 quantum-kernel completion audit (section 3.1 only).

The governing protocol is ``docs/hsbc_challenge_quantum_rivals_audit_v2.md``.
This file materializes section 3.1 before any v2 HSBC outcome is inspected.
Dataset execution is deliberately fail-closed: invoking the script without
``--run`` performs synthetic and direct-Qiskit checks only and writes no result
artifact.

Two implementation details were not algebraically fixed by the prose protocol
and are therefore frozen here, in code and in every result JSON:

* Ring-IQP map.  Put ``x_j = 2 u_j - 1``.  Each of ``depth`` re-upload blocks is
  ``H`` on every qubit, then ``RZ(lambda*pi*x_j)`` on qubit ``j``, then
  ``RZZ(lambda*pi*x_j*x_{j+1})`` on every nearest-neighbour ring edge.  Qubit
  indices use Qiskit's little-endian statevector convention.
* Classical RBF bandwidth twin.  For each product-kernel bandwidth
  ``lambda`` use ``gamma=(lambda*pi/2)^2``.  This is the local Gaussian
  approximation to the product fidelity kernel.  The frozen lambda and C grids
  are otherwise unchanged.
* Conditional one-class kernels from pre-outcome clarification 2 use a linear
  OC-SVM on standardized Nyström features.  This keeps the Nyström map as the
  sole kernel approximation instead of silently applying a second RBF kernel.

Primary outputs, written only by an explicit ``--run``:

  runs/hsbc_challenge/audit_v2/quantum_kernels_v2.json
  runs/hsbc_challenge/audit_v2/quantum_kernels_v2.npz

The run retains every validation candidate, failed/nonconvergent status, exact
resource accounting, and full-test score arrays.  It never balances or
subsamples validation/test rows.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hsbc_common import prepare_hsbc  # noqa: E402


SCHEMA = "hsbc-quantum-kernels-v2"
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
PROTOCOL = REPO_ROOT / "docs/hsbc_challenge_quantum_rivals_audit_v2.md"
V1_SCORES = REPO_ROOT / "runs/hsbc_challenge/audit_v1/occ_tournament_scores_v1.npz"
SOURCE_DATA = REPO_ROOT / "runs/hsbc_challenge/data/ulb_creditcard.npz"
AUDIT_V2_DIRECTORY = REPO_ROOT / "runs/hsbc_challenge/audit_v2"
DEFAULT_JSON = AUDIT_V2_DIRECTORY / "quantum_kernels_v2.json"
DEFAULT_NPZ = AUDIT_V2_DIRECTORY / "quantum_kernels_v2.npz"

SPLIT_SEED = 0
AUDIT_SEED = 20260831
BOOTSTRAP_SEED = 20260831
N_BOOTSTRAP = 2_000
TARGET_FPR = 1e-3
N_QUBITS = 8
STATEVECTOR_DIMENSION = 2**N_QUBITS
TRAIN_LEGIT_CAP = 2_000
LANDMARKS_PER_CLASS = 32
N_LANDMARKS = 2 * LANDMARKS_PER_CLASS
KERNEL_CHUNK = 2_048
STATE_CHUNK = 512

PRODUCT_LAMBDAS = (0.125, 0.25, 0.5, 1.0, 2.0)
IQP_DEPTHS = (1, 2)
IQP_LAMBDAS = (0.5, 1.0, 2.0)
C_GRID = (0.1, 1.0, 10.0)
RBF_TWIN_GAMMAS = tuple((lam * np.pi / 2.0) ** 2 for lam in PRODUCT_LAMBDAS)
OC_MAP_NU = 0.001727
OC_NU_GRID = (0.0005, 0.001, 0.001727, 0.003, 0.005)
OC_TARGET_LEGITIMATE_VALIDATION_FPR = 0.001
OC_CLASSICAL_LAMBDAS = (0.5, 1.0, 2.0)
OC_CLASSICAL_GAMMAS = tuple(
    (lam * np.pi / 2.0) ** 2 for lam in OC_CLASSICAL_LAMBDAS
)

SVC_TOL = 1e-3
SVC_MAX_ITER = 100_000
LOGISTIC_TOL = 1e-6
LOGISTIC_MAX_ITER = 5_000
NYSTROM_RCOND = 1e-10
FAMILYWISE_ALPHA = 0.05
N_COPRIMARY_FAMILIES = 4
SIMULTANEOUS_ALPHA = FAMILYWISE_ALPHA / N_COPRIMARY_FAMILIES

EXPECTED_INPUT_SHA256 = {
    "protocol": "de10a4640e310986ea94ed52a35a7edba2bb32fceac7e84116dc2c0bc5ebc115",
    "ULB_source": "40ad6e73b1ef5c2b42265b7a02e89caf9738f09520770451a6c8d4491f4d79d6",
    "v1_reference_scores": "be2e24943012c329ebc2203594598c24198329e8f433923961fc5b86e17c1b8c",
}
EXPECTED_FEATURE_NAMES = ("V14", "V4", "V12", "V11", "V10", "V3", "V16", "V2")
EXPECTED_SPLIT_COUNTS = {
    "tr": {"rows": 170_884, "frauds": 295},
    "va": {"rows": 56_961, "frauds": 98},
    "te": {"rows": 56_962, "frauds": 99},
}

IQP_FORMULA = {
    "centered_feature": "x_j = 2*u_j - 1",
    "block_order": "H_all -> RZ_all -> RZZ_nearest_neighbor_ring",
    "rz_angle": "lambda*pi*x_j",
    "rzz_angle": "lambda*pi*x_j*x_(j+1)",
    "depth": list(IQP_DEPTHS),
    "lambda_grid": list(IQP_LAMBDAS),
    "endianness": "Qiskit little-endian; qubit q is bit 2**q",
}


def log(message: str) -> None:
    print(message, flush=True)


class MethodAuditFailure(RuntimeError):
    """Carry partial candidate/resource evidence through the top-level guard."""

    def __init__(self, record: dict):
        self.record = record
        super().__init__(str(record.get("error", "method audit failed")))


def failed_method_record(
    method: str,
    error: str,
    resources: dict,
    **partial: object,
) -> dict:
    return {
        "method": method,
        "status": "FAILED",
        "error": error,
        **partial,
        "selected": {"status": "NOT_RUN"},
        "test": {"status": "NOT_RUN"},
        "resources": copy.deepcopy(resources),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_input_hashes() -> dict[str, str]:
    return {
        "protocol": sha256(PROTOCOL),
        "ULB_source": sha256(SOURCE_DATA),
        "v1_reference_scores": sha256(V1_SCORES),
        "script": sha256(SCRIPT_PATH),
    }


def prepare_and_verify_frozen_inputs(
    *,
    expected_script_hash: str,
    expected_test_labels: np.ndarray | None = None,
) -> tuple[dict, dict[str, np.ndarray], dict]:
    """Hash-lock inputs, rebuild the split, and verify v1 row-label identity."""

    if Path.cwd().resolve() != REPO_ROOT:
        raise RuntimeError(
            f"dataset run must start from frozen repository root {REPO_ROOT}; "
            f"got {Path.cwd().resolve()}"
        )
    hashes = frozen_input_hashes()
    for name, expected in EXPECTED_INPUT_SHA256.items():
        if hashes[name] != expected:
            raise RuntimeError(
                f"frozen input hash mismatch for {name}: expected {expected}, got {hashes[name]}"
            )
    if hashes["script"] != expected_script_hash:
        raise RuntimeError("kernel script changed after the run's source freeze")

    pipe = prepare_hsbc(N_QUBITS, seed=SPLIT_SEED)
    feature_names = tuple(pipe["feature_names"])
    if feature_names != EXPECTED_FEATURE_NAMES:
        raise RuntimeError(
            f"frozen feature mismatch: expected {EXPECTED_FEATURE_NAMES}, got {feature_names}"
        )
    observed_counts = {
        key: {
            "rows": int(len(pipe["idx"][key])),
            "frauds": int(np.sum(pipe["y_split"][key])),
        }
        for key in ("tr", "va", "te")
    }
    if observed_counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(
            f"frozen split-count mismatch: expected {EXPECTED_SPLIT_COUNTS}, got {observed_counts}"
        )
    all_labels = np.asarray(pipe["y"], int)
    split_index_hashes = {}
    split_label_hashes = {}
    split_indices = []
    for key in ("tr", "va", "te"):
        indices = np.asarray(pipe["idx"][key], int)
        split_labels = np.asarray(pipe["y_split"][key], int)
        if indices.shape != (EXPECTED_SPLIT_COUNTS[key]["rows"],):
            raise RuntimeError(f"frozen {key} index shape mismatch: {indices.shape}")
        if np.any(indices < 0) or np.any(indices >= len(all_labels)):
            raise RuntimeError(f"frozen {key} contains an out-of-range source-row index")
        if not np.array_equal(split_labels, all_labels[indices]):
            raise RuntimeError(f"frozen {key} row-label identity mismatch against ULB source")
        split_indices.append(indices)
        split_index_hashes[key] = hashlib.sha256(indices.tobytes()).hexdigest()
        split_label_hashes[key] = hashlib.sha256(split_labels.tobytes()).hexdigest()
    joined_indices = np.concatenate(split_indices)
    if len(np.unique(joined_indices)) != len(joined_indices):
        raise RuntimeError("frozen train/validation/test source-row indices overlap")
    if len(joined_indices) != len(all_labels) or not np.array_equal(
        np.sort(joined_indices), np.arange(len(all_labels))
    ):
        raise RuntimeError("frozen train/validation/test indices do not partition ULB rows")
    y_test = np.asarray(pipe["y_split"]["te"], int)
    with np.load(V1_SCORES, allow_pickle=False) as v1:
        v1_y = np.asarray(v1["n8_y"], int)
        if not np.array_equal(v1_y, y_test):
            raise RuntimeError("v1 reference row labels do not match the rebuilt seed-0 test split")
        references = {
            "S1_analytic": np.asarray(v1["n8_S1_analytic"], float),
            "S2_L1_seed500": np.asarray(v1["n8_S2_L1_seed500"], float),
            "XGB_subset": np.asarray(v1["n8_XGB_subset"], float),
        }
    for name, score in references.items():
        if score.shape != y_test.shape or not np.all(np.isfinite(score)):
            raise RuntimeError(
                f"v1 reference {name} is nonfinite or not aligned to frozen test rows"
            )
    if expected_test_labels is not None and not np.array_equal(
        y_test, np.asarray(expected_test_labels, int)
    ):
        raise RuntimeError("rebuilt test labels changed since the initial run preflight")
    snapshot = {
        "hashes": hashes,
        "expected_hashes": dict(EXPECTED_INPUT_SHA256),
        "features": list(feature_names),
        "split_counts": observed_counts,
        "split_row_label_identity": {key: True for key in ("tr", "va", "te")},
        "split_indices_nonoverlapping_and_partition_source": True,
        "split_index_sha256": split_index_hashes,
        "split_label_sha256": split_label_hashes,
        "v1_row_label_identity": True,
        "test_label_sha256": hashlib.sha256(y_test.tobytes()).hexdigest(),
    }
    return pipe, references, snapshot


def package_versions() -> dict:
    import scipy
    import sklearn

    try:
        import qiskit

        qiskit_version = qiskit.__version__
    except Exception as exc:  # pragma: no cover - recorded on a failed environment
        qiskit_version = f"UNAVAILABLE: {type(exc).__name__}: {exc}"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "qiskit": qiskit_version,
    }


def fit_legitimate_validation_threshold(
    legitimate_validation_score: np.ndarray,
    target_fpr: float = TARGET_FPR,
) -> dict:
    """Freeze a strict anomaly threshold from legitimate validation rows only."""

    score = np.asarray(legitimate_validation_score, float).reshape(-1)
    if len(score) == 0 or not np.all(np.isfinite(score)):
        raise ValueError("legitimate validation threshold requires finite nonempty scores")
    threshold = float(np.quantile(score, 1.0 - target_fpr, method="higher"))
    return {
        "threshold": threshold,
        "source": "legitimate validation scores only",
        "source_rows": int(len(score)),
        "target_fpr": float(target_fpr),
        "quantile": float(1.0 - target_fpr),
        "quantile_method": "higher",
        "decision_rule": "anomaly iff score > threshold",
        "achieved_legitimate_validation_fpr": float(np.mean(score > threshold)),
        "test_labels_or_scores_used_for_threshold": False,
    }


def ranking_metrics(
    y: np.ndarray,
    score: np.ndarray,
    legitimate_validation_threshold: dict,
) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y, int)
    score = np.asarray(score, float)
    if len(score) != len(y) or not np.all(np.isfinite(score)):
        raise ValueError("ranking score is nonfinite or has the wrong row count")
    threshold = float(legitimate_validation_threshold["threshold"])
    prediction = score > threshold
    return {
        "auprc": float(average_precision_score(y, score)),
        "auc_roc": float(roc_auc_score(y, score)),
        "recall_at_fpr_1e-3_validation_legit_threshold": float(
            np.mean(prediction[y == 1])
        ),
        "observed_test_fpr_at_validation_legit_threshold": float(
            np.mean(prediction[y == 0])
        ),
        "secondary_threshold": legitimate_validation_threshold,
    }


def validation_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y, int)
    score = np.asarray(score, float)
    if len(score) != len(y) or not np.all(np.isfinite(score)):
        raise ValueError("validation score is nonfinite or has the wrong row count")
    return {
        "auprc": float(average_precision_score(y, score)),
        "auc_roc": float(roc_auc_score(y, score)),
    }


def _weighted_ap_sorted(
    y_sorted: np.ndarray,
    weights_sorted: np.ndarray,
    group_ends: np.ndarray,
) -> np.ndarray:
    pos_w = weights_sorted * y_sorted[None, :]
    cum_pos = np.cumsum(pos_w, axis=1, dtype=float)
    cum_all = np.cumsum(weights_sorted, axis=1, dtype=float)
    total_pos = cum_pos[:, -1]
    cum_pos_group = cum_pos[:, group_ends]
    cum_all_group = cum_all[:, group_ends]
    precision = np.divide(
        cum_pos_group,
        cum_all_group,
        out=np.zeros_like(cum_pos_group),
        where=cum_all_group > 0,
    )
    group_pos = np.diff(
        np.column_stack([np.zeros(len(weights_sorted)), cum_pos_group]), axis=1
    )
    return np.divide(
        np.sum(precision * group_pos, axis=1),
        total_pos,
        out=np.full(len(weights_sorted), np.nan),
        where=total_pos > 0,
    )


def paired_ap_bootstrap(
    y: np.ndarray,
    score_map: dict[str, np.ndarray],
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Paired Poisson(1) AP bootstrap using identical weights for every score."""

    from sklearn.metrics import average_precision_score

    y = np.asarray(y, int)
    scores = {name: np.asarray(score, float) for name, score in score_map.items()}
    if any(len(score) != len(y) for score in scores.values()):
        raise ValueError("paired bootstrap requires identical test rows")
    orders = {name: np.argsort(-score, kind="mergesort") for name, score in scores.items()}
    group_ends = {
        name: np.flatnonzero(np.r_[score[order][1:] != score[order][:-1], True])
        for name, (score, order) in {
            key: (scores[key], orders[key]) for key in scores
        }.items()
    }
    draws = {name: [] for name in scores}
    rng = np.random.default_rng(seed)
    made = 0
    while made < n_boot:
        take = min(24, n_boot - made)
        weights = rng.poisson(1.0, size=(take, len(y))).astype(np.float32)
        valid = (weights[:, y == 1].sum(axis=1) > 0) & (
            weights[:, y == 0].sum(axis=1) > 0
        )
        if not np.any(valid):
            continue
        weights = weights[valid]
        for name, order in orders.items():
            ap = _weighted_ap_sorted(y[order], weights[:, order], group_ends[name])
            draws[name].extend(ap[np.isfinite(ap)].tolist())
        made = min(len(values) for values in draws.values())
    arrays = {name: np.asarray(values[:n_boot]) for name, values in draws.items()}
    points = {name: float(average_precision_score(y, score)) for name, score in scores.items()}
    summary = {
        "bootstrap": "paired Poisson(1) test-row weights",
        "seed": seed,
        "n_boot": n_boot,
        "ap_point": points,
        "ap_ci95": {
            name: [float(v) for v in np.quantile(values, [0.025, 0.975])]
            for name, values in arrays.items()
        },
    }
    return summary, arrays


def paired_comparison(
    method: str,
    reference: str,
    points: dict[str, float],
    draws: dict[str, np.ndarray],
) -> dict:
    delta = draws[method] - draws[reference]
    q_sim = [SIMULTANEOUS_ALPHA / 2.0, 1.0 - SIMULTANEOUS_ALPHA / 2.0]
    return {
        "orientation": f"AUPRC({method}) - AUPRC({reference})",
        "point": float(points[method] - points[reference]),
        "ci95": [float(v) for v in np.quantile(delta, [0.025, 0.975])],
        "ci98_75_simultaneous": [float(v) for v in np.quantile(delta, q_sim)],
        "simultaneous_alpha": SIMULTANEOUS_ALPHA,
    }


def product_fidelity_kernel(
    A: np.ndarray,
    B: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    out = np.ones((len(A), len(B)), dtype=np.float64)
    coefficient = 0.5 * bandwidth * np.pi
    for j in range(A.shape[1]):
        out *= np.cos(coefficient * (A[:, j, None] - B[None, :, j])) ** 2
    return out


def rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    a2 = np.einsum("ij,ij->i", A, A)[:, None]
    b2 = np.einsum("ij,ij->i", B, B)[None, :]
    distance2 = np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)
    return np.exp(-float(gamma) * distance2)


def _apply_h_all(states: np.ndarray, n_qubits: int) -> np.ndarray:
    out = np.asarray(states, complex)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for qubit in range(n_qubits):
        view = out.reshape(len(out), -1, 2, 2**qubit)
        zero = view[:, :, 0, :].copy()
        one = view[:, :, 1, :].copy()
        view[:, :, 0, :] = (zero + one) * inv_sqrt2
        view[:, :, 1, :] = (zero - one) * inv_sqrt2
    return out


def _apply_rz_rows(states: np.ndarray, qubit: int, angle: np.ndarray) -> None:
    view = states.reshape(len(states), -1, 2, 2**qubit)
    phase = np.exp(0.5j * np.asarray(angle, float))[:, None, None]
    view[:, :, 0, :] *= phase.conj()
    view[:, :, 1, :] *= phase


def _apply_rzz_rows(
    states: np.ndarray,
    qubit_a: int,
    qubit_b: int,
    angle: np.ndarray,
) -> None:
    indices = np.arange(states.shape[1])
    parity = ((indices >> qubit_a) ^ (indices >> qubit_b)) & 1
    sign = 1.0 - 2.0 * parity
    states *= np.exp(-0.5j * np.asarray(angle, float)[:, None] * sign[None, :])


def ring_iqp_states(U: np.ndarray, depth: int, bandwidth: float) -> np.ndarray:
    """Exact ring-IQP states using the formula frozen in this script header."""

    U = np.asarray(U, float)
    if U.ndim != 2 or U.shape[1] != N_QUBITS:
        raise ValueError(f"ring IQP expects rows of {N_QUBITS} transformed features")
    if depth not in IQP_DEPTHS:
        raise ValueError(f"depth must be one of {IQP_DEPTHS}")
    state = np.zeros((len(U), STATEVECTOR_DIMENSION), dtype=np.complex128)
    state[:, 0] = 1.0
    centered = 2.0 * U - 1.0
    for _ in range(depth):
        state = _apply_h_all(state, N_QUBITS)
        for qubit in range(N_QUBITS):
            _apply_rz_rows(state, qubit, bandwidth * np.pi * centered[:, qubit])
        for qubit in range(N_QUBITS):
            neighbour = (qubit + 1) % N_QUBITS
            angle = bandwidth * np.pi * centered[:, qubit] * centered[:, neighbour]
            _apply_rzz_rows(state, qubit, neighbour, angle)
    return state


def state_fidelity_kernel(A_states: np.ndarray, B_states: np.ndarray) -> np.ndarray:
    overlaps = np.asarray(A_states, complex).conj() @ np.asarray(B_states, complex).T
    return np.abs(overlaps) ** 2


def pauli_observables(n_qubits: int = N_QUBITS) -> list[tuple[tuple[int, str], ...]]:
    axes = ("X", "Y", "Z")
    observables: list[tuple[tuple[int, str], ...]] = []
    for qubit in range(n_qubits):
        for axis in axes:
            observables.append(((qubit, axis),))
    for qubit in range(n_qubits):
        neighbour = (qubit + 1) % n_qubits
        for axis_a in axes:
            for axis_b in axes:
                observables.append(((qubit, axis_a), (neighbour, axis_b)))
    if len(observables) != 96:
        raise AssertionError(f"expected 96 projected observables, got {len(observables)}")
    return observables


def _pauli_action(
    observable: tuple[tuple[int, str], ...],
    dimension: int = STATEVECTOR_DIMENSION,
) -> tuple[np.ndarray, np.ndarray]:
    source_indices = np.arange(dimension)
    flip_mask = 0
    for qubit, axis in observable:
        if axis in ("X", "Y"):
            flip_mask ^= 1 << qubit
    source_indices = source_indices ^ flip_mask
    phase = np.ones(dimension, dtype=complex)
    for qubit, axis in observable:
        bit = (source_indices >> qubit) & 1
        if axis == "Z":
            phase *= 1.0 - 2.0 * bit
        elif axis == "Y":
            phase *= 1j * (1.0 - 2.0 * bit)
        elif axis != "X":
            raise ValueError(f"unknown Pauli axis {axis!r}")
    return source_indices, phase


PAULI_ACTIONS = tuple(_pauli_action(observable) for observable in pauli_observables())


def projected_pauli_features_from_states(states: np.ndarray) -> tuple[np.ndarray, float]:
    states = np.asarray(states, complex)
    features = np.empty((len(states), len(PAULI_ACTIONS)), dtype=float)
    max_imaginary = 0.0
    for column, (source, phase) in enumerate(PAULI_ACTIONS):
        value = np.einsum(
            "bi,bi->b", states.conj(), states[:, source] * phase[None, :]
        )
        max_imaginary = max(max_imaginary, float(np.max(np.abs(value.imag))))
        features[:, column] = value.real
    return features, max_imaginary


def projected_iqp_features(
    U: np.ndarray,
    depth: int,
    bandwidth: float,
    chunk: int = STATE_CHUNK,
) -> tuple[np.ndarray, dict]:
    blocks = []
    max_norm_deviation = 0.0
    max_imaginary = 0.0
    for start in range(0, len(U), chunk):
        states = ring_iqp_states(U[start : start + chunk], depth, bandwidth)
        norms = np.einsum("bi,bi->b", states.conj(), states).real
        max_norm_deviation = max(max_norm_deviation, float(np.max(np.abs(norms - 1.0))))
        features, imaginary = projected_pauli_features_from_states(states)
        max_imaginary = max(max_imaginary, imaginary)
        blocks.append(features)
    return np.concatenate(blocks), {
        "max_state_norm_deviation": max_norm_deviation,
        "max_pauli_expectation_imaginary": max_imaginary,
    }


def nystrom_inverse_sqrt(K_landmarks: np.ndarray) -> tuple[np.ndarray, dict]:
    sym = 0.5 * (np.asarray(K_landmarks, float) + np.asarray(K_landmarks, float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    largest = max(float(np.max(eigenvalues)), 1.0)
    cutoff = NYSTROM_RCOND * largest
    retained = eigenvalues > cutoff
    inv_sqrt = (eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])) @ (
        eigenvectors[:, retained].T
    )
    return inv_sqrt, {
        "minimum_eigenvalue_before_repair": float(np.min(eigenvalues)),
        "maximum_eigenvalue": float(np.max(eigenvalues)),
        "pseudoinverse_cutoff": cutoff,
        "retained_rank": int(np.sum(retained)),
        "discarded_eigenvalues": int(np.sum(~retained)),
        "psd_repair": "symmetrize then Moore-Penrose inverse square root; discard eigenvalues <= rcond*max",
    }


def nystrom_features(
    K_to_landmarks: np.ndarray,
    inverse_sqrt: np.ndarray,
) -> np.ndarray:
    return np.asarray(K_to_landmarks, float) @ np.asarray(inverse_sqrt, float)


def _fit_precomputed_svc(K_train: np.ndarray, y_train: np.ndarray, C: float):
    from sklearn.svm import SVC

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SVC(
            C=float(C),
            kernel="precomputed",
            class_weight="balanced",
            tol=SVC_TOL,
            max_iter=SVC_MAX_ITER,
        ).fit(K_train, y_train)
    return model, [str(item.message) for item in caught]


def _svc_metadata(model, warnings_seen: list[str]) -> dict:
    return {
        "support_vectors": int(len(model.support_)),
        "support_vectors_by_class": [int(v) for v in model.n_support_],
        "fit_status": int(model.fit_status_),
        "iterations": [int(v) for v in np.atleast_1d(model.n_iter_)],
        "warnings": warnings_seen,
        "converged": bool(model.fit_status_ == 0),
    }


def _fit_logistic(features: np.ndarray, y: np.ndarray, C: float):
    from sklearn.linear_model import LogisticRegression

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = LogisticRegression(
            l1_ratio=0.0,
            C=float(C),
            class_weight="balanced",
            solver="lbfgs",
            tol=LOGISTIC_TOL,
            max_iter=LOGISTIC_MAX_ITER,
            random_state=AUDIT_SEED,
        ).fit(features, y)
    return model, [str(item.message) for item in caught]


def _logistic_metadata(model, warnings_seen: list[str]) -> dict:
    return {
        "trainable_head_parameters": int(model.coef_.size + model.intercept_.size),
        "iterations": [int(v) for v in np.atleast_1d(model.n_iter_)],
        "warnings": warnings_seen,
        "converged": bool(np.max(model.n_iter_) < LOGISTIC_MAX_ITER),
    }


def _fit_one_class_svm(features: np.ndarray, nu: float):
    from sklearn.svm import OneClassSVM

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = OneClassSVM(
            kernel="linear",
            nu=float(nu),
            tol=SVC_TOL,
            max_iter=SVC_MAX_ITER,
        ).fit(features)
    return model, [str(item.message) for item in caught]


def _one_class_metadata(model, warnings_seen: list[str]) -> dict:
    fit_status = int(getattr(model, "fit_status_", 0))
    iterations = [int(v) for v in np.atleast_1d(getattr(model, "n_iter_", []))]
    return {
        "support_vectors": int(len(model.support_)),
        "fit_status": fit_status,
        "iterations": iterations,
        "warnings": warnings_seen,
        "converged": bool(fit_status == 0),
    }


def legitimate_validation_selection_metrics(
    legitimate_margin: np.ndarray,
) -> dict:
    margin = np.asarray(legitimate_margin, float)
    if not np.all(np.isfinite(margin)):
        raise ValueError("legitimate validation margin is nonfinite")
    fpr = float(np.mean(margin < 0.0))
    return {
        "legitimate_rows": int(len(margin)),
        "legitimate_validation_fpr": fpr,
        "absolute_fpr_target_error": abs(fpr - OC_TARGET_LEGITIMATE_VALIDATION_FPR),
        "mean_legitimate_validation_margin": float(np.mean(margin)),
        "target_fpr": OC_TARGET_LEGITIMATE_VALIDATION_FPR,
        "fraud_rows_used": 0,
    }


def _choose_one_class_candidate(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row.get("status") == "OK"]
    if not eligible:
        raise RuntimeError("no converged one-class validation candidate")
    return min(
        eligible,
        key=lambda row: (
            row["legitimate_validation"]["absolute_fpr_target_error"],
            -row["legitimate_validation"]["mean_legitimate_validation_margin"],
            row["candidate_index"],
        ),
    )


def _stream_precomputed_scores(
    rows: np.ndarray,
    train_rows: np.ndarray,
    models: dict[float, object],
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    chunk: int = KERNEL_CHUNK,
) -> dict[float, np.ndarray]:
    blocks = {C: [] for C in models}
    for start in range(0, len(rows), chunk):
        K = kernel(rows[start : start + chunk], train_rows)
        for C, model in models.items():
            blocks[C].append(model.decision_function(K))
    return {C: np.concatenate(parts) for C, parts in blocks.items()}


def _choose_candidate(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row.get("status") == "OK"]
    if not eligible:
        raise RuntimeError("no successful validation candidate")
    # Python max is stable, so exact ties retain the earliest frozen-grid row.
    return max(eligible, key=lambda row: row["validation"]["auprc"])


def select_capped_training_and_landmarks(pipe: dict) -> dict:
    """Freeze the common capped train rows and one shared class-stratified landmark set."""

    y_train = np.asarray(pipe["y_split"]["tr"], int)
    legitimate = np.flatnonzero(y_train == 0)
    fraud = np.flatnonzero(y_train == 1)
    rng_cap = np.random.default_rng(AUDIT_SEED)
    chosen_legitimate = rng_cap.choice(
        legitimate, min(TRAIN_LEGIT_CAP, len(legitimate)), replace=False
    )
    capped = np.sort(np.concatenate([chosen_legitimate, fraud]))
    capped_y = y_train[capped]

    # "Without labels within class": labels fix the 32/32 allocation, while the
    # draw inside each class is uniform and uses no score or feature information.
    rng_landmarks = np.random.default_rng(AUDIT_SEED)
    capped_legitimate = np.flatnonzero(capped_y == 0)
    capped_fraud = np.flatnonzero(capped_y == 1)
    if len(capped_legitimate) < LANDMARKS_PER_CLASS or len(capped_fraud) < LANDMARKS_PER_CLASS:
        raise RuntimeError("not enough capped rows for the frozen 32/32 landmark allocation")
    landmark_positions = np.concatenate(
        [
            rng_landmarks.choice(capped_legitimate, LANDMARKS_PER_CLASS, replace=False),
            rng_landmarks.choice(capped_fraud, LANDMARKS_PER_CLASS, replace=False),
        ]
    )
    one_class_legitimate = np.sort(chosen_legitimate)
    rng_one_class_landmarks = np.random.default_rng(AUDIT_SEED)
    one_class_landmark_positions = rng_one_class_landmarks.choice(
        len(one_class_legitimate), N_LANDMARKS, replace=False
    )
    return {
        "capped_positions_in_train_split": capped,
        "capped_original_row_indices": np.asarray(pipe["idx"]["tr"])[capped],
        "landmark_positions_in_cap": landmark_positions,
        "landmark_positions_in_train_split": capped[landmark_positions],
        "landmark_original_row_indices": np.asarray(pipe["idx"]["tr"])[
            capped[landmark_positions]
        ],
        "one_class_legitimate_positions_in_train_split": one_class_legitimate,
        "one_class_legitimate_original_row_indices": np.asarray(pipe["idx"]["tr"])[
            one_class_legitimate
        ],
        "one_class_landmark_positions_in_legitimate_cap": one_class_landmark_positions,
        "one_class_landmark_positions_in_train_split": one_class_legitimate[
            one_class_landmark_positions
        ],
        "one_class_landmark_original_row_indices": np.asarray(pipe["idx"]["tr"])[
            one_class_legitimate[one_class_landmark_positions]
        ],
        "capped_class_counts": {
            "legitimate": int(np.sum(capped_y == 0)),
            "fraud": int(np.sum(capped_y == 1)),
        },
        "landmark_class_counts": {
            "legitimate": int(np.sum(capped_y[landmark_positions] == 0)),
            "fraud": int(np.sum(capped_y[landmark_positions] == 1)),
        },
        "one_class_train_rows": int(len(one_class_legitimate)),
        "one_class_landmark_rows": int(len(one_class_landmark_positions)),
    }


def tune_capped_precomputed_svc(
    method: str,
    train_rows: np.ndarray,
    y_train: np.ndarray,
    validation_rows: np.ndarray,
    y_validation: np.ndarray,
    test_rows: np.ndarray,
    y_test: np.ndarray,
    map_specs: list[dict],
    kernel_for_spec: Callable[[np.ndarray, np.ndarray, dict], np.ndarray],
) -> tuple[dict, np.ndarray]:
    """Full-validation map/C grid for the product and classical RBF SVCs."""

    candidates: list[dict] = []
    best: dict | None = None
    n_train = len(train_rows)
    resources = {
        "training_gram_entries_actual": 0,
        "training_gram_unique_symmetric_entries": 0,
        "validation_kernel_entries_actual": 0,
        "validation_decision_support_evaluations": 0,
        "test_kernel_entries_actual": 0,
        "candidate_maps_started": 0,
        "candidate_maps_completed": 0,
        "candidate_heads_started": 0,
        "candidate_heads_completed": 0,
    }

    for map_index, spec in enumerate(map_specs):
        resources["candidate_maps_started"] += 1
        try:
            K_train = kernel_for_spec(train_rows, train_rows, spec)
            resources["training_gram_entries_actual"] += int(n_train * n_train)
            resources["training_gram_unique_symmetric_entries"] += int(
                n_train * (n_train + 1) // 2
            )
            models: dict[float, object] = {}
            model_meta: dict[float, dict] = {}
            for C in C_GRID:
                resources["candidate_heads_started"] += 1
                try:
                    model, warning_rows = _fit_precomputed_svc(K_train, y_train, C)
                    models[C] = model
                    model_meta[C] = _svc_metadata(model, warning_rows)
                    resources["candidate_heads_completed"] += 1
                except Exception as exc:
                    candidates.append(
                        {
                            "candidate_index": len(candidates),
                            "map_index": map_index,
                            "map": spec,
                            "C": C,
                            "status": "FAILED",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            if not models:
                continue
            kernel = lambda A, B, frozen=spec: kernel_for_spec(A, B, frozen)
            validation_scores = _stream_precomputed_scores(
                validation_rows, train_rows, models, kernel
            )
            resources["candidate_maps_completed"] += 1
            resources["validation_kernel_entries_actual"] += int(
                len(validation_rows) * n_train
            )
            for C in C_GRID:
                if C not in models:
                    continue
                meta = model_meta[C]
                status = "OK" if meta["converged"] else "NONCONVERGED"
                row = {
                    "candidate_index": len(candidates),
                    "map_index": map_index,
                    "map": spec,
                    "C": C,
                    "status": status,
                    "validation": validation_metrics(y_validation, validation_scores[C]),
                    **meta,
                }
                candidates.append(row)
                resources["validation_decision_support_evaluations"] += int(
                    len(validation_rows) * meta["support_vectors"]
                )
                if status == "OK" and (
                    best is None
                    or row["validation"]["auprc"] > best["row"]["validation"]["auprc"]
                ):
                    best = {
                        "row": row,
                        "model": models[C],
                        "spec": spec,
                        "validation_score": validation_scores[C],
                    }
        except Exception as exc:
            candidates.append(
                {
                    "candidate_index": len(candidates),
                    "map_index": map_index,
                    "map": spec,
                    "C": None,
                    "status": "FAILED_MAP",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if best is None:
        raise MethodAuditFailure(
            failed_method_record(
                method,
                f"{method}: no converged validation candidate",
                resources,
                validation_candidates=candidates,
            )
        )
    winning_kernel = lambda A, B: kernel_for_spec(A, B, best["spec"])
    test_score = _stream_precomputed_scores(
        test_rows,
        train_rows,
        {float(best["row"]["C"]): best["model"]},
        winning_kernel,
    )[float(best["row"]["C"])]
    resources["test_kernel_entries_actual"] = int(len(test_rows) * n_train)
    resources["implied_kernel_evaluations_per_transaction"] = int(
        best["row"]["support_vectors"]
    )
    threshold = fit_legitimate_validation_threshold(
        best["validation_score"][np.asarray(y_validation, int) == 0]
    )
    return {
        "method": method,
        "status": "OK",
        "selection_metric": "full-validation AUPRC",
        "tie_break": "earliest row in frozen map-major then C-major grid",
        "validation_candidates": candidates,
        "selected": {
            "map": best["spec"],
            "C": best["row"]["C"],
            "candidate_index": best["row"]["candidate_index"],
            "validation": best["row"]["validation"],
            "support_vectors": best["row"]["support_vectors"],
        },
        "test": ranking_metrics(y_test, test_score, threshold),
        "resources": resources,
        "svc": {
            "class_weight": "balanced",
            "tolerance": SVC_TOL,
            "max_iter": SVC_MAX_ITER,
            "train_rows": int(n_train),
        },
    }, test_score


def run_product_fidelity_svc(
    U_train: np.ndarray,
    y_train: np.ndarray,
    U_validation: np.ndarray,
    y_validation: np.ndarray,
    U_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict, np.ndarray]:
    specs = [{"lambda": bandwidth} for bandwidth in PRODUCT_LAMBDAS]
    result, score = tune_capped_precomputed_svc(
        "bandwidth_tuned_product_fidelity_QSVC",
        U_train,
        y_train,
        U_validation,
        y_validation,
        U_test,
        y_test,
        specs,
        lambda A, B, spec: product_fidelity_kernel(A, B, spec["lambda"]),
    )
    result["implementation_definition"] = {
        "kernel": "product_j cos^2(lambda*pi*(u_j-v_j)/2)",
        "lambda_grid": list(PRODUCT_LAMBDAS),
        "evaluation": "exact analytic product-state fidelity; no finite-shot estimate",
        "PSD_repair": "none",
        "qubits": N_QUBITS,
        "statevector_dimension_if_materialized": STATEVECTOR_DIMENSION,
    }
    result["resources"].update(
        {
            "feature_map_trainable_parameters": 0,
            "svc_dual_coefficients_plus_intercept_proxy": int(
                result["selected"]["support_vectors"] + 1
            ),
            "simulator_statevector_materializations": 0,
            "overlap_evaluations_per_transaction": int(
                result["selected"]["support_vectors"]
            ),
            "hardware_implied_overlap_circuit_executions_per_transaction_per_shot": int(
                result["selected"]["support_vectors"]
            ),
            "hardware_implied_data_state_preparations_per_transaction_per_shot": int(
                result["selected"]["support_vectors"]
            ),
            "hardware_implied_support_state_preparations_per_transaction_per_shot": int(
                result["selected"]["support_vectors"]
            ),
            "shot_accounting": (
                "counts are per shot of every overlap circuit; no finite hardware "
                "shot count was frozen or executed"
            ),
            "evaluation_backend": "exact analytic fidelity kernel",
            "finite_shots_used": 0,
        }
    )
    return result, score


def run_classical_rbf_svc(
    U_train: np.ndarray,
    y_train: np.ndarray,
    U_validation: np.ndarray,
    y_validation: np.ndarray,
    U_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict, np.ndarray]:
    specs = [
        {"lambda_twin": bandwidth, "gamma": gamma}
        for bandwidth, gamma in zip(PRODUCT_LAMBDAS, RBF_TWIN_GAMMAS)
    ]
    result, score = tune_capped_precomputed_svc(
        "classical_RBF_SVC_twin",
        U_train,
        y_train,
        U_validation,
        y_validation,
        U_test,
        y_test,
        specs,
        lambda A, B, spec: rbf_kernel(A, B, spec["gamma"]),
    )
    result["implementation_definition"] = {
        "gamma_formula": "gamma=(lambda*pi/2)^2",
        "lambda_grid": list(PRODUCT_LAMBDAS),
        "gamma_grid": list(RBF_TWIN_GAMMAS),
        "reason": "local Gaussian twin of the product fidelity bandwidth grid",
        "PSD_repair": "none",
    }
    result["resources"].update(
        {
            "trainable_quantum_parameters": 0,
            "svc_dual_coefficients_plus_intercept_proxy": int(
                result["selected"]["support_vectors"] + 1
            ),
            "finite_shots_used": 0,
            "quantum_hardware_circuits": 0,
            "evaluation_backend": "classical analytic RBF kernel",
        }
    )
    return result, score


def _iqp_landmark_features(
    U: np.ndarray,
    depth: int,
    bandwidth: float,
    landmark_states: np.ndarray,
    inverse_sqrt: np.ndarray,
    chunk: int = STATE_CHUNK,
) -> tuple[np.ndarray, dict]:
    blocks = []
    max_norm_deviation = 0.0
    for start in range(0, len(U), chunk):
        states = ring_iqp_states(U[start : start + chunk], depth, bandwidth)
        norms = np.einsum("bi,bi->b", states.conj(), states).real
        max_norm_deviation = max(max_norm_deviation, float(np.max(np.abs(norms - 1.0))))
        blocks.append(nystrom_features(state_fidelity_kernel(states, landmark_states), inverse_sqrt))
    return np.concatenate(blocks), {"max_state_norm_deviation": max_norm_deviation}


def run_iqp_fidelity_nystrom(
    U_train: np.ndarray,
    y_train: np.ndarray,
    U_validation: np.ndarray,
    y_validation: np.ndarray,
    U_test: np.ndarray,
    y_test: np.ndarray,
    landmark_positions: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Six-map IQP selection at C=1, then frozen C tuning for the winning map."""

    from sklearn.preprocessing import StandardScaler

    map_candidates: list[dict] = []
    best: dict | None = None
    resources = {
        "statevector_dimension": STATEVECTOR_DIMENSION,
        "landmarks": N_LANDMARKS,
        "training_landmark_entries": 0,
        "landmark_gram_entries": 0,
        "validation_landmark_entries": 0,
        "test_landmark_entries": 0,
        "simulator_statevector_materializations_map_selection": 0,
        "simulator_statevector_materializations_test": 0,
        "candidate_maps_started": 0,
        "candidate_maps_completed": 0,
        "candidate_heads_started": 0,
        "candidate_heads_completed": 0,
    }
    for depth in IQP_DEPTHS:
        for bandwidth in IQP_LAMBDAS:
            spec = {"depth": depth, "lambda": bandwidth}
            resources["candidate_maps_started"] += 1
            try:
                train_states = ring_iqp_states(U_train, depth, bandwidth)
                resources["simulator_statevector_materializations_map_selection"] += int(
                    len(U_train)
                )
                train_norms = np.einsum("bi,bi->b", train_states.conj(), train_states).real
                landmark_states = train_states[landmark_positions]
                K_landmarks = state_fidelity_kernel(landmark_states, landmark_states)
                resources["landmark_gram_entries"] += int(N_LANDMARKS**2)
                inverse_sqrt, repair = nystrom_inverse_sqrt(K_landmarks)
                train_landmark_kernel = state_fidelity_kernel(
                    train_states, landmark_states
                )
                resources["training_landmark_entries"] += int(
                    len(U_train) * N_LANDMARKS
                )
                train_features = nystrom_features(train_landmark_kernel, inverse_sqrt)
                validation_features, validation_state_meta = _iqp_landmark_features(
                    U_validation,
                    depth,
                    bandwidth,
                    landmark_states,
                    inverse_sqrt,
                )
                resources["simulator_statevector_materializations_map_selection"] += int(
                    len(U_validation)
                )
                resources["validation_landmark_entries"] += int(
                    len(U_validation) * N_LANDMARKS
                )
                scaler = StandardScaler().fit(train_features)
                train_scaled = scaler.transform(train_features)
                validation_scaled = scaler.transform(validation_features)
                model, warning_rows = _fit_logistic(train_scaled, y_train, C=1.0)
                score = model.decision_function(validation_scaled)
                model_meta = _logistic_metadata(model, warning_rows)
                status = "OK" if model_meta["converged"] else "NONCONVERGED"
                row = {
                    "candidate_index": len(map_candidates),
                    "map": spec,
                    "head_C": 1.0,
                    "status": status,
                    "validation": validation_metrics(y_validation, score),
                    "nystrom_psd": repair,
                    "max_train_state_norm_deviation": float(
                        np.max(np.abs(train_norms - 1.0))
                    ),
                    **validation_state_meta,
                    **model_meta,
                }
                map_candidates.append(row)
                resources["candidate_maps_completed"] += 1
                if status == "OK" and (
                    best is None
                    or row["validation"]["auprc"] > best["row"]["validation"]["auprc"]
                ):
                    best = {
                        "row": row,
                        "spec": spec,
                        "landmark_states": landmark_states,
                        "inverse_sqrt": inverse_sqrt,
                        "scaler": scaler,
                        "train_scaled": train_scaled,
                        "validation_scaled": validation_scaled,
                        "C1_model": model,
                        "C1_warnings": warning_rows,
                    }
            except Exception as exc:
                map_candidates.append(
                    {
                        "candidate_index": len(map_candidates),
                        "map": spec,
                        "head_C": 1.0,
                        "status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    if best is None:
        raise MethodAuditFailure(
            failed_method_record(
                "ring_IQP_fidelity_Nystrom_logistic",
                "ring-IQP Nyström: no converged map candidate",
                resources,
                feature_map_selection={
                    "status": "FAILED",
                    "candidates": map_candidates,
                    "selected": {"status": "NOT_RUN"},
                },
                head_selection={"status": "NOT_RUN", "candidates": []},
            )
        )

    head_candidates: list[dict] = []
    head_models: dict[float, object] = {}
    head_validation_scores: dict[float, np.ndarray] = {}
    for C in C_GRID:
        resources["candidate_heads_started"] += 1
        try:
            if C == 1.0:
                model = best["C1_model"]
                warning_rows = best["C1_warnings"]
            else:
                model, warning_rows = _fit_logistic(best["train_scaled"], y_train, C)
            score = model.decision_function(best["validation_scaled"])
            model_meta = _logistic_metadata(model, warning_rows)
            status = "OK" if model_meta["converged"] else "NONCONVERGED"
            row = {
                "candidate_index": len(head_candidates),
                "C": C,
                "status": status,
                "validation": validation_metrics(y_validation, score),
                **model_meta,
            }
            head_candidates.append(row)
            head_models[C] = model
            head_validation_scores[C] = np.asarray(score, float)
            resources["candidate_heads_completed"] += 1
        except Exception as exc:
            head_candidates.append(
                {
                    "candidate_index": len(head_candidates),
                    "C": C,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    try:
        selected_head = _choose_candidate(head_candidates)
    except RuntimeError as exc:
        raise MethodAuditFailure(
            failed_method_record(
                "ring_IQP_fidelity_Nystrom_logistic",
                f"head selection failed: {exc}",
                resources,
                feature_map_selection={
                    "status": "OK",
                    "candidates": map_candidates,
                    "selected": best["row"],
                },
                head_selection={
                    "status": "FAILED",
                    "candidates": head_candidates,
                    "selected": {"status": "NOT_RUN"},
                },
            )
        ) from exc
    selected_model = head_models[float(selected_head["C"])]
    threshold = fit_legitimate_validation_threshold(
        head_validation_scores[float(selected_head["C"])][
            np.asarray(y_validation, int) == 0
        ]
    )
    test_blocks = []
    max_test_norm_deviation = 0.0
    depth = int(best["spec"]["depth"])
    bandwidth = float(best["spec"]["lambda"])
    for start in range(0, len(U_test), STATE_CHUNK):
        states = ring_iqp_states(U_test[start : start + STATE_CHUNK], depth, bandwidth)
        norms = np.einsum("bi,bi->b", states.conj(), states).real
        max_test_norm_deviation = max(
            max_test_norm_deviation, float(np.max(np.abs(norms - 1.0)))
        )
        features = nystrom_features(
            state_fidelity_kernel(states, best["landmark_states"]), best["inverse_sqrt"]
        )
        test_blocks.append(selected_model.decision_function(best["scaler"].transform(features)))
    test_score = np.concatenate(test_blocks)
    resources["test_landmark_entries"] = int(len(U_test) * N_LANDMARKS)
    resources["simulator_statevector_materializations_test"] = int(len(U_test))
    resources["overlap_evaluations_per_transaction"] = N_LANDMARKS
    resources["hardware_implied_overlap_circuit_executions_per_transaction_per_shot"] = N_LANDMARKS
    resources["hardware_implied_data_state_preparations_per_transaction_per_shot"] = N_LANDMARKS
    resources["hardware_implied_landmark_state_preparations_per_transaction_per_shot"] = N_LANDMARKS
    resources["shot_accounting"] = (
        "64 independent fidelity-overlap circuits per transaction per shot; each "
        "circuit prepares one data and one landmark state; no finite hardware shot "
        "count was frozen or executed"
    )
    resources["feature_map_trainable_parameters"] = 0
    resources["trainable_logistic_head_parameters"] = int(
        selected_head["trainable_head_parameters"]
    )
    resources["exact_statevector_scores"] = True
    resources["evaluation_backend"] = "exact dense statevector simulator"
    resources["finite_shots_used"] = 0

    return {
        "method": "ring_IQP_fidelity_Nystrom_logistic",
        "status": "OK",
        "feature_map_selection": {
            "metric": "full-validation AUPRC at head C=1",
            "tie_break": "earliest depth-major then lambda-major frozen-grid row",
            "candidates": map_candidates,
            "selected": best["row"],
        },
        "head_selection": {
            "metric": "full-validation AUPRC after feature-map selection",
            "tie_break": "earliest C in frozen grid",
            "candidates": head_candidates,
            "selected": selected_head,
        },
        "test": ranking_metrics(y_test, test_score, threshold),
        "test_max_state_norm_deviation": max_test_norm_deviation,
        "resources": resources,
        "implementation_definition": {
            **IQP_FORMULA,
            "Nystrom_normalization": (
                "K_XL times the Moore-Penrose inverse square root of symmetrized K_LL"
            ),
            "standardization": "fit on capped training Nystrom features only",
        },
    }, test_score


def _classical_nystrom_map(
    rows: np.ndarray,
    landmark_rows: np.ndarray,
    gamma: float,
    inverse_sqrt: np.ndarray,
) -> np.ndarray:
    return nystrom_features(rbf_kernel(rows, landmark_rows, gamma), inverse_sqrt)


def run_classical_rbf_nystrom(
    U_train: np.ndarray,
    y_train: np.ndarray,
    U_validation: np.ndarray,
    y_validation: np.ndarray,
    U_test: np.ndarray,
    y_test: np.ndarray,
    landmark_positions: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """RBF Nyström/logistic twin using the identical landmarks and two-stage rule."""

    from sklearn.preprocessing import StandardScaler

    landmark_rows = U_train[landmark_positions]
    map_candidates: list[dict] = []
    best: dict | None = None
    resources = {
        "landmarks": N_LANDMARKS,
        "training_landmark_entries": 0,
        "landmark_gram_entries": 0,
        "validation_landmark_entries": 0,
        "test_landmark_entries": 0,
        "candidate_maps_started": 0,
        "candidate_maps_completed": 0,
        "candidate_heads_started": 0,
        "candidate_heads_completed": 0,
    }
    for bandwidth, gamma in zip(PRODUCT_LAMBDAS, RBF_TWIN_GAMMAS):
        spec = {"lambda_twin": bandwidth, "gamma": gamma}
        resources["candidate_maps_started"] += 1
        try:
            K_landmarks = rbf_kernel(landmark_rows, landmark_rows, gamma)
            resources["landmark_gram_entries"] += int(N_LANDMARKS**2)
            inverse_sqrt, repair = nystrom_inverse_sqrt(K_landmarks)
            train_features = _classical_nystrom_map(
                U_train, landmark_rows, gamma, inverse_sqrt
            )
            resources["training_landmark_entries"] += int(
                len(U_train) * N_LANDMARKS
            )
            validation_features = _classical_nystrom_map(
                U_validation, landmark_rows, gamma, inverse_sqrt
            )
            resources["validation_landmark_entries"] += int(
                len(U_validation) * N_LANDMARKS
            )
            scaler = StandardScaler().fit(train_features)
            train_scaled = scaler.transform(train_features)
            validation_scaled = scaler.transform(validation_features)
            model, warning_rows = _fit_logistic(train_scaled, y_train, C=1.0)
            score = model.decision_function(validation_scaled)
            model_meta = _logistic_metadata(model, warning_rows)
            status = "OK" if model_meta["converged"] else "NONCONVERGED"
            row = {
                "candidate_index": len(map_candidates),
                "map": spec,
                "head_C": 1.0,
                "status": status,
                "validation": validation_metrics(y_validation, score),
                "nystrom_psd": repair,
                **model_meta,
            }
            map_candidates.append(row)
            resources["candidate_maps_completed"] += 1
            if status == "OK" and (
                best is None
                or row["validation"]["auprc"] > best["row"]["validation"]["auprc"]
            ):
                best = {
                    "row": row,
                    "spec": spec,
                    "landmark_rows": landmark_rows,
                    "inverse_sqrt": inverse_sqrt,
                    "scaler": scaler,
                    "train_scaled": train_scaled,
                    "validation_scaled": validation_scaled,
                    "C1_model": model,
                    "C1_warnings": warning_rows,
                }
        except Exception as exc:
            map_candidates.append(
                {
                    "candidate_index": len(map_candidates),
                    "map": spec,
                    "head_C": 1.0,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if best is None:
        raise MethodAuditFailure(
            failed_method_record(
                "classical_RBF_Nystrom_logistic_twin",
                "classical RBF Nyström: no converged map candidate",
                resources,
                feature_map_selection={
                    "status": "FAILED",
                    "candidates": map_candidates,
                    "selected": {"status": "NOT_RUN"},
                },
                head_selection={"status": "NOT_RUN", "candidates": []},
            )
        )

    head_candidates: list[dict] = []
    head_models: dict[float, object] = {}
    head_validation_scores: dict[float, np.ndarray] = {}
    for C in C_GRID:
        resources["candidate_heads_started"] += 1
        try:
            if C == 1.0:
                model = best["C1_model"]
                warning_rows = best["C1_warnings"]
            else:
                model, warning_rows = _fit_logistic(best["train_scaled"], y_train, C)
            score = model.decision_function(best["validation_scaled"])
            model_meta = _logistic_metadata(model, warning_rows)
            status = "OK" if model_meta["converged"] else "NONCONVERGED"
            row = {
                "candidate_index": len(head_candidates),
                "C": C,
                "status": status,
                "validation": validation_metrics(y_validation, score),
                **model_meta,
            }
            head_candidates.append(row)
            head_models[C] = model
            head_validation_scores[C] = np.asarray(score, float)
            resources["candidate_heads_completed"] += 1
        except Exception as exc:
            head_candidates.append(
                {
                    "candidate_index": len(head_candidates),
                    "C": C,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    try:
        selected_head = _choose_candidate(head_candidates)
    except RuntimeError as exc:
        raise MethodAuditFailure(
            failed_method_record(
                "classical_RBF_Nystrom_logistic_twin",
                f"head selection failed: {exc}",
                resources,
                feature_map_selection={
                    "status": "OK",
                    "candidates": map_candidates,
                    "selected": best["row"],
                },
                head_selection={
                    "status": "FAILED",
                    "candidates": head_candidates,
                    "selected": {"status": "NOT_RUN"},
                },
            )
        ) from exc
    selected_model = head_models[float(selected_head["C"])]
    threshold = fit_legitimate_validation_threshold(
        head_validation_scores[float(selected_head["C"])][
            np.asarray(y_validation, int) == 0
        ]
    )
    gamma = float(best["spec"]["gamma"])
    test_features = _classical_nystrom_map(
        U_test, best["landmark_rows"], gamma, best["inverse_sqrt"]
    )
    test_score = selected_model.decision_function(best["scaler"].transform(test_features))
    resources["test_landmark_entries"] = int(len(U_test) * N_LANDMARKS)
    resources["implied_kernel_evaluations_per_transaction"] = N_LANDMARKS
    resources["quantum_hardware_circuits"] = 0
    resources["evaluation_backend"] = "classical analytic RBF Nyström map"
    resources["trainable_logistic_head_parameters"] = int(
        selected_head["trainable_head_parameters"]
    )
    return {
        "method": "classical_RBF_Nystrom_logistic_twin",
        "status": "OK",
        "feature_map_selection": {
            "metric": "full-validation AUPRC at head C=1",
            "tie_break": "earliest lambda/gamma row in frozen grid",
            "candidates": map_candidates,
            "selected": best["row"],
        },
        "head_selection": {
            "metric": "full-validation AUPRC after map selection",
            "tie_break": "earliest C in frozen grid",
            "candidates": head_candidates,
            "selected": selected_head,
        },
        "test": ranking_metrics(y_test, test_score, threshold),
        "resources": resources,
        "implementation_definition": {
            "gamma_formula": "gamma=(lambda*pi/2)^2",
            "lambda_grid": list(PRODUCT_LAMBDAS),
            "gamma_grid": list(RBF_TWIN_GAMMAS),
            "landmarks": "identical 32-legitimate/32-fraud train-only rows as IQP Nyström",
            "Nystrom_normalization": (
                "K_XL times the Moore-Penrose inverse square root of symmetrized K_LL"
            ),
            "standardization": "fit on capped training Nystrom features only",
        },
    }, test_score


def run_iqp_fidelity_nystrom_ocsvm(
    U_legitimate_train: np.ndarray,
    U_legitimate_validation: np.ndarray,
    U_test: np.ndarray,
    y_test: np.ndarray,
    landmark_positions: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Conditional one-class ring-IQP Nyström OC-SVM from clarification 2."""

    from sklearn.preprocessing import StandardScaler

    map_candidates: list[dict] = []
    map_bundles: dict[int, dict] = {}
    resources = {
        "statevector_dimension": STATEVECTOR_DIMENSION,
        "legitimate_training_rows": int(len(U_legitimate_train)),
        "legitimate_validation_rows": int(len(U_legitimate_validation)),
        "fraud_training_rows": 0,
        "fraud_validation_rows": 0,
        "legitimate_only_landmarks": N_LANDMARKS,
        "training_full_gram_entries": 0,
        "training_landmark_entries": 0,
        "landmark_gram_entries": 0,
        "legitimate_validation_landmark_entries": 0,
        "test_landmark_entries": 0,
        "simulator_statevector_materializations_map_selection": 0,
        "simulator_statevector_materializations_test": 0,
        "nu_selection_additional_landmark_entries": 0,
        "candidate_maps_started": 0,
        "candidate_maps_completed": 0,
        "candidate_nus_started": 0,
        "candidate_nus_completed": 0,
    }
    for depth in IQP_DEPTHS:
        for bandwidth in IQP_LAMBDAS:
            spec = {"depth": depth, "lambda": bandwidth}
            candidate_index = len(map_candidates)
            resources["candidate_maps_started"] += 1
            try:
                train_states = ring_iqp_states(U_legitimate_train, depth, bandwidth)
                resources["simulator_statevector_materializations_map_selection"] += int(
                    len(U_legitimate_train)
                )
                train_norms = np.einsum("bi,bi->b", train_states.conj(), train_states).real
                landmark_states = train_states[landmark_positions]
                K_landmarks = state_fidelity_kernel(landmark_states, landmark_states)
                resources["landmark_gram_entries"] += int(N_LANDMARKS**2)
                inverse_sqrt, repair = nystrom_inverse_sqrt(K_landmarks)
                train_landmark_kernel = state_fidelity_kernel(
                    train_states, landmark_states
                )
                resources["training_landmark_entries"] += int(
                    len(U_legitimate_train) * N_LANDMARKS
                )
                train_features = nystrom_features(train_landmark_kernel, inverse_sqrt)
                validation_features, validation_state_meta = _iqp_landmark_features(
                    U_legitimate_validation,
                    depth,
                    bandwidth,
                    landmark_states,
                    inverse_sqrt,
                )
                resources["simulator_statevector_materializations_map_selection"] += int(
                    len(U_legitimate_validation)
                )
                resources["legitimate_validation_landmark_entries"] += int(
                    len(U_legitimate_validation) * N_LANDMARKS
                )
                scaler = StandardScaler().fit(train_features)
                train_scaled = scaler.transform(train_features)
                validation_scaled = scaler.transform(validation_features)
                model, warning_rows = _fit_one_class_svm(train_scaled, OC_MAP_NU)
                margin = np.asarray(model.decision_function(validation_scaled)).reshape(-1)
                model_meta = _one_class_metadata(model, warning_rows)
                status = "OK" if model_meta["converged"] else "NONCONVERGED"
                row = {
                    "candidate_index": candidate_index,
                    "map": spec,
                    "nu": OC_MAP_NU,
                    "status": status,
                    "legitimate_validation": legitimate_validation_selection_metrics(margin),
                    "nystrom_psd": repair,
                    "max_train_state_norm_deviation": float(
                        np.max(np.abs(train_norms - 1.0))
                    ),
                    **validation_state_meta,
                    **model_meta,
                }
                map_candidates.append(row)
                map_bundles[candidate_index] = {
                    "row": row,
                    "spec": spec,
                    "landmark_states": landmark_states,
                    "inverse_sqrt": inverse_sqrt,
                    "scaler": scaler,
                    "train_scaled": train_scaled,
                    "validation_scaled": validation_scaled,
                }
                resources["candidate_maps_completed"] += 1
            except Exception as exc:
                map_candidates.append(
                    {
                        "candidate_index": candidate_index,
                        "map": spec,
                        "nu": OC_MAP_NU,
                        "status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    try:
        selected_map = _choose_one_class_candidate(map_candidates)
    except RuntimeError as exc:
        raise MethodAuditFailure(
            failed_method_record(
                "ring_IQP_fidelity_Nystrom_OCSVM",
                f"map selection failed: {exc}",
                resources,
                map_selection={
                    "status": "FAILED",
                    "candidates": map_candidates,
                    "selected": {"status": "NOT_RUN"},
                },
                nu_selection={"status": "NOT_RUN", "candidates": []},
            )
        ) from exc
    best = map_bundles[int(selected_map["candidate_index"])]

    nu_candidates: list[dict] = []
    nu_models: dict[float, object] = {}
    for nu in OC_NU_GRID:
        candidate_index = len(nu_candidates)
        resources["candidate_nus_started"] += 1
        try:
            model, warning_rows = _fit_one_class_svm(best["train_scaled"], nu)
            margin = np.asarray(model.decision_function(best["validation_scaled"])).reshape(-1)
            model_meta = _one_class_metadata(model, warning_rows)
            status = "OK" if model_meta["converged"] else "NONCONVERGED"
            row = {
                "candidate_index": candidate_index,
                "nu": nu,
                "status": status,
                "legitimate_validation": legitimate_validation_selection_metrics(margin),
                **model_meta,
            }
            nu_candidates.append(row)
            nu_models[nu] = model
            resources["candidate_nus_completed"] += 1
        except Exception as exc:
            nu_candidates.append(
                {
                    "candidate_index": candidate_index,
                    "nu": nu,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    try:
        selected_nu = _choose_one_class_candidate(nu_candidates)
    except RuntimeError as exc:
        raise MethodAuditFailure(
            failed_method_record(
                "ring_IQP_fidelity_Nystrom_OCSVM",
                f"nu selection failed: {exc}",
                resources,
                map_selection={
                    "status": "OK",
                    "candidates": map_candidates,
                    "selected": selected_map,
                },
                nu_selection={
                    "status": "FAILED",
                    "candidates": nu_candidates,
                    "selected": {"status": "NOT_RUN"},
                },
            )
        ) from exc
    selected_model = nu_models[float(selected_nu["nu"])]
    selected_validation_score = -np.asarray(
        selected_model.decision_function(best["validation_scaled"]), float
    ).reshape(-1)
    threshold = fit_legitimate_validation_threshold(selected_validation_score)
    depth = int(best["spec"]["depth"])
    bandwidth = float(best["spec"]["lambda"])
    test_blocks = []
    max_test_norm_deviation = 0.0
    for start in range(0, len(U_test), STATE_CHUNK):
        states = ring_iqp_states(U_test[start : start + STATE_CHUNK], depth, bandwidth)
        norms = np.einsum("bi,bi->b", states.conj(), states).real
        max_test_norm_deviation = max(
            max_test_norm_deviation, float(np.max(np.abs(norms - 1.0)))
        )
        features = nystrom_features(
            state_fidelity_kernel(states, best["landmark_states"]), best["inverse_sqrt"]
        )
        margin = selected_model.decision_function(best["scaler"].transform(features))
        test_blocks.append(-np.asarray(margin).reshape(-1))
    test_score = np.concatenate(test_blocks)
    resources["test_landmark_entries"] = int(len(U_test) * N_LANDMARKS)
    resources["simulator_statevector_materializations_test"] = int(len(U_test))
    resources["overlap_evaluations_per_transaction"] = N_LANDMARKS
    resources["hardware_implied_overlap_circuit_executions_per_transaction_per_shot"] = N_LANDMARKS
    resources["hardware_implied_data_state_preparations_per_transaction_per_shot"] = N_LANDMARKS
    resources["hardware_implied_landmark_state_preparations_per_transaction_per_shot"] = N_LANDMARKS
    resources["shot_accounting"] = (
        "64 independent fidelity-overlap circuits per transaction per shot; each "
        "circuit prepares one data and one landmark state; no finite hardware shot "
        "count was frozen or executed"
    )
    resources["feature_map_trainable_parameters"] = 0
    resources["selected_OCSVM_support_vectors"] = int(selected_nu["support_vectors"])
    resources["exact_statevector_scores"] = True
    resources["evaluation_backend"] = "exact dense statevector simulator"
    resources["finite_shots_used"] = 0
    return {
        "method": "ring_IQP_fidelity_Nystrom_OCSVM",
        "status": "OK",
        "information_bracket": "conditional one-class after shared supervised feature screen",
        "anomaly_score": "negative OC-SVM decision function",
        "map_selection": {
            "fixed_nu": OC_MAP_NU,
            "rule": (
                "minimize abs(legitimate_validation_FPR-0.001), then maximize mean "
                "legitimate validation margin, then earliest frozen grid row"
            ),
            "candidates": map_candidates,
            "selected": selected_map,
        },
        "nu_selection": {
            "grid": list(OC_NU_GRID),
            "rule": (
                "minimize abs(legitimate_validation_FPR-0.001), then maximize mean "
                "legitimate validation margin, then earliest frozen nu row"
            ),
            "candidates": nu_candidates,
            "selected": selected_nu,
        },
        "test": ranking_metrics(y_test, test_score, threshold),
        "test_max_state_norm_deviation": max_test_norm_deviation,
        "resources": resources,
        "implementation_definition": {
            **IQP_FORMULA,
            "Nystrom_normalization": (
                "K_XL times the Moore-Penrose inverse square root of symmetrized K_LL"
            ),
            "standardization": "fit on 2,000 legitimate training Nystrom rows only",
            "OCSVM_kernel": "linear on Nystrom features; no second kernel",
            "OCSVM_tolerance": SVC_TOL,
            "OCSVM_max_iter": SVC_MAX_ITER,
            "validation_fraud_labels_used": False,
        },
    }, test_score


def run_classical_rbf_nystrom_ocsvm(
    U_legitimate_train: np.ndarray,
    U_legitimate_validation: np.ndarray,
    U_test: np.ndarray,
    y_test: np.ndarray,
    landmark_positions: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Conditional one-class RBF Nyström OC-SVM twin from clarification 2."""

    from sklearn.preprocessing import StandardScaler

    landmark_rows = U_legitimate_train[landmark_positions]
    map_candidates: list[dict] = []
    map_bundles: dict[int, dict] = {}
    resources = {
        "legitimate_training_rows": int(len(U_legitimate_train)),
        "legitimate_validation_rows": int(len(U_legitimate_validation)),
        "fraud_training_rows": 0,
        "fraud_validation_rows": 0,
        "legitimate_only_landmarks": N_LANDMARKS,
        "training_full_gram_entries": 0,
        "training_landmark_entries": 0,
        "landmark_gram_entries": 0,
        "legitimate_validation_landmark_entries": 0,
        "test_landmark_entries": 0,
        "nu_selection_additional_landmark_entries": 0,
        "candidate_maps_started": 0,
        "candidate_maps_completed": 0,
        "candidate_nus_started": 0,
        "candidate_nus_completed": 0,
    }
    for bandwidth, gamma in zip(OC_CLASSICAL_LAMBDAS, OC_CLASSICAL_GAMMAS):
        spec = {"lambda_twin": bandwidth, "gamma": gamma}
        candidate_index = len(map_candidates)
        resources["candidate_maps_started"] += 1
        try:
            K_landmarks = rbf_kernel(landmark_rows, landmark_rows, gamma)
            resources["landmark_gram_entries"] += int(N_LANDMARKS**2)
            inverse_sqrt, repair = nystrom_inverse_sqrt(K_landmarks)
            train_features = _classical_nystrom_map(
                U_legitimate_train, landmark_rows, gamma, inverse_sqrt
            )
            resources["training_landmark_entries"] += int(
                len(U_legitimate_train) * N_LANDMARKS
            )
            validation_features = _classical_nystrom_map(
                U_legitimate_validation, landmark_rows, gamma, inverse_sqrt
            )
            resources["legitimate_validation_landmark_entries"] += int(
                len(U_legitimate_validation) * N_LANDMARKS
            )
            scaler = StandardScaler().fit(train_features)
            train_scaled = scaler.transform(train_features)
            validation_scaled = scaler.transform(validation_features)
            model, warning_rows = _fit_one_class_svm(train_scaled, OC_MAP_NU)
            margin = np.asarray(model.decision_function(validation_scaled)).reshape(-1)
            model_meta = _one_class_metadata(model, warning_rows)
            status = "OK" if model_meta["converged"] else "NONCONVERGED"
            row = {
                "candidate_index": candidate_index,
                "map": spec,
                "nu": OC_MAP_NU,
                "status": status,
                "legitimate_validation": legitimate_validation_selection_metrics(margin),
                "nystrom_psd": repair,
                **model_meta,
            }
            map_candidates.append(row)
            map_bundles[candidate_index] = {
                "row": row,
                "spec": spec,
                "inverse_sqrt": inverse_sqrt,
                "scaler": scaler,
                "train_scaled": train_scaled,
                "validation_scaled": validation_scaled,
            }
            resources["candidate_maps_completed"] += 1
        except Exception as exc:
            map_candidates.append(
                {
                    "candidate_index": candidate_index,
                    "map": spec,
                    "nu": OC_MAP_NU,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    try:
        selected_map = _choose_one_class_candidate(map_candidates)
    except RuntimeError as exc:
        raise MethodAuditFailure(
            failed_method_record(
                "classical_RBF_Nystrom_OCSVM_twin",
                f"map selection failed: {exc}",
                resources,
                map_selection={
                    "status": "FAILED",
                    "candidates": map_candidates,
                    "selected": {"status": "NOT_RUN"},
                },
                nu_selection={"status": "NOT_RUN", "candidates": []},
            )
        ) from exc
    best = map_bundles[int(selected_map["candidate_index"])]

    nu_candidates: list[dict] = []
    nu_models: dict[float, object] = {}
    for nu in OC_NU_GRID:
        candidate_index = len(nu_candidates)
        resources["candidate_nus_started"] += 1
        try:
            model, warning_rows = _fit_one_class_svm(best["train_scaled"], nu)
            margin = np.asarray(model.decision_function(best["validation_scaled"])).reshape(-1)
            model_meta = _one_class_metadata(model, warning_rows)
            status = "OK" if model_meta["converged"] else "NONCONVERGED"
            row = {
                "candidate_index": candidate_index,
                "nu": nu,
                "status": status,
                "legitimate_validation": legitimate_validation_selection_metrics(margin),
                **model_meta,
            }
            nu_candidates.append(row)
            nu_models[nu] = model
            resources["candidate_nus_completed"] += 1
        except Exception as exc:
            nu_candidates.append(
                {
                    "candidate_index": candidate_index,
                    "nu": nu,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    try:
        selected_nu = _choose_one_class_candidate(nu_candidates)
    except RuntimeError as exc:
        raise MethodAuditFailure(
            failed_method_record(
                "classical_RBF_Nystrom_OCSVM_twin",
                f"nu selection failed: {exc}",
                resources,
                map_selection={
                    "status": "OK",
                    "candidates": map_candidates,
                    "selected": selected_map,
                },
                nu_selection={
                    "status": "FAILED",
                    "candidates": nu_candidates,
                    "selected": {"status": "NOT_RUN"},
                },
            )
        ) from exc
    selected_model = nu_models[float(selected_nu["nu"])]
    selected_validation_score = -np.asarray(
        selected_model.decision_function(best["validation_scaled"]), float
    ).reshape(-1)
    threshold = fit_legitimate_validation_threshold(selected_validation_score)
    gamma = float(best["spec"]["gamma"])
    test_features = _classical_nystrom_map(
        U_test, landmark_rows, gamma, best["inverse_sqrt"]
    )
    margin = selected_model.decision_function(best["scaler"].transform(test_features))
    test_score = -np.asarray(margin).reshape(-1)
    resources["test_landmark_entries"] = int(len(U_test) * N_LANDMARKS)
    resources["implied_kernel_evaluations_per_transaction"] = N_LANDMARKS
    resources["selected_OCSVM_support_vectors"] = int(selected_nu["support_vectors"])
    resources["finite_shots_used"] = 0
    resources["quantum_hardware_circuits"] = 0
    resources["evaluation_backend"] = "classical analytic RBF Nyström map"
    return {
        "method": "classical_RBF_Nystrom_OCSVM_twin",
        "status": "OK",
        "information_bracket": "conditional one-class after shared supervised feature screen",
        "anomaly_score": "negative OC-SVM decision function",
        "map_selection": {
            "fixed_nu": OC_MAP_NU,
            "rule": (
                "minimize abs(legitimate_validation_FPR-0.001), then maximize mean "
                "legitimate validation margin, then earliest frozen grid row"
            ),
            "candidates": map_candidates,
            "selected": selected_map,
        },
        "nu_selection": {
            "grid": list(OC_NU_GRID),
            "rule": (
                "minimize abs(legitimate_validation_FPR-0.001), then maximize mean "
                "legitimate validation margin, then earliest frozen nu row"
            ),
            "candidates": nu_candidates,
            "selected": selected_nu,
        },
        "test": ranking_metrics(y_test, test_score, threshold),
        "resources": resources,
        "implementation_definition": {
            "gamma_formula": "gamma=(lambda*pi/2)^2",
            "lambda_grid": list(OC_CLASSICAL_LAMBDAS),
            "gamma_grid": list(OC_CLASSICAL_GAMMAS),
            "landmarks": "same 64 legitimate-only train rows as ring-IQP OC-SVM",
            "Nystrom_normalization": (
                "K_XL times the Moore-Penrose inverse square root of symmetrized K_LL"
            ),
            "standardization": "fit on 2,000 legitimate training Nystrom rows only",
            "OCSVM_kernel": "linear on Nystrom features; no second kernel",
            "OCSVM_tolerance": SVC_TOL,
            "OCSVM_max_iter": SVC_MAX_ITER,
            "validation_fraud_labels_used": False,
        },
    }, test_score


def sklearn_gamma_scale(features: np.ndarray) -> float:
    variance = float(np.var(np.asarray(features, float)))
    return 1.0 / (features.shape[1] * variance) if variance > 0.0 else 1.0


def run_projected_iqp_svc(
    U_train: np.ndarray,
    y_train: np.ndarray,
    U_validation: np.ndarray,
    y_validation: np.ndarray,
    U_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Projected 96-observable IQP map selection, then C tuning for its RBF SVC."""

    from sklearn.preprocessing import StandardScaler

    map_candidates: list[dict] = []
    best: dict | None = None
    n_train = len(U_train)
    resources = {
        "statevector_dimension": STATEVECTOR_DIMENSION,
        "projected_expectations": 96,
        "simulator_statevector_materializations_map_selection": 0,
        "expectation_values_map_selection": 0,
        "training_gram_entries_map_selection": 0,
        "validation_kernel_entries_map_selection": 0,
        "validation_kernel_entries_head_selection": 0,
        "test_kernel_entries": 0,
        "simulator_statevector_materializations_test": 0,
        "expectation_values_test": 0,
        "candidate_maps_started": 0,
        "candidate_maps_completed": 0,
        "candidate_heads_started": 0,
        "candidate_heads_completed": 0,
    }
    for depth in IQP_DEPTHS:
        for bandwidth in IQP_LAMBDAS:
            spec = {"depth": depth, "lambda": bandwidth}
            resources["candidate_maps_started"] += 1
            try:
                train_features, train_projection_meta = projected_iqp_features(
                    U_train, depth, bandwidth
                )
                resources["simulator_statevector_materializations_map_selection"] += int(
                    len(U_train)
                )
                resources["expectation_values_map_selection"] += int(
                    96 * len(U_train)
                )
                validation_features, validation_projection_meta = projected_iqp_features(
                    U_validation, depth, bandwidth
                )
                resources["simulator_statevector_materializations_map_selection"] += int(
                    len(U_validation)
                )
                resources["expectation_values_map_selection"] += int(
                    96 * len(U_validation)
                )
                scaler = StandardScaler().fit(train_features)
                train_scaled = scaler.transform(train_features)
                validation_scaled = scaler.transform(validation_features)
                gamma = sklearn_gamma_scale(train_scaled)
                K_train = rbf_kernel(train_scaled, train_scaled, gamma)
                resources["training_gram_entries_map_selection"] += int(
                    n_train * n_train
                )
                model, warning_rows = _fit_precomputed_svc(K_train, y_train, C=1.0)
                validation_score = _stream_precomputed_scores(
                    validation_scaled,
                    train_scaled,
                    {1.0: model},
                    lambda A, B, frozen_gamma=gamma: rbf_kernel(A, B, frozen_gamma),
                )[1.0]
                resources["validation_kernel_entries_map_selection"] += int(
                    len(U_validation) * n_train
                )
                model_meta = _svc_metadata(model, warning_rows)
                status = "OK" if model_meta["converged"] else "NONCONVERGED"
                row = {
                    "candidate_index": len(map_candidates),
                    "map": spec,
                    "C": 1.0,
                    "gamma": gamma,
                    "status": status,
                    "validation": validation_metrics(y_validation, validation_score),
                    "train_projection_checks": train_projection_meta,
                    "validation_projection_checks": validation_projection_meta,
                    **model_meta,
                }
                map_candidates.append(row)
                resources["candidate_maps_completed"] += 1
                if status == "OK" and (
                    best is None
                    or row["validation"]["auprc"] > best["row"]["validation"]["auprc"]
                ):
                    best = {
                        "row": row,
                        "spec": spec,
                        "scaler": scaler,
                        "train_scaled": train_scaled,
                        "validation_scaled": validation_scaled,
                        "gamma": gamma,
                        "K_train": K_train,
                        "C1_model": model,
                        "C1_warnings": warning_rows,
                    }
            except Exception as exc:
                map_candidates.append(
                    {
                        "candidate_index": len(map_candidates),
                        "map": spec,
                        "C": 1.0,
                        "status": "FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    if best is None:
        raise MethodAuditFailure(
            failed_method_record(
                "projected_ring_IQP_96feature_RBF_SVC",
                "projected ring-IQP: no converged map candidate",
                resources,
                feature_map_selection={
                    "status": "FAILED",
                    "candidates": map_candidates,
                    "selected": {"status": "NOT_RUN"},
                },
                head_selection={"status": "NOT_RUN", "candidates": []},
            )
        )

    head_models: dict[float, object] = {}
    head_meta: dict[float, dict] = {}
    head_failures: dict[float, str] = {}
    for C in C_GRID:
        resources["candidate_heads_started"] += 1
        try:
            if C == 1.0:
                model = best["C1_model"]
                warning_rows = best["C1_warnings"]
            else:
                model, warning_rows = _fit_precomputed_svc(best["K_train"], y_train, C)
            head_models[C] = model
            head_meta[C] = _svc_metadata(model, warning_rows)
            resources["candidate_heads_completed"] += 1
        except Exception as exc:
            head_failures[C] = f"{type(exc).__name__}: {exc}"
    validation_scores = _stream_precomputed_scores(
        best["validation_scaled"],
        best["train_scaled"],
        head_models,
        lambda A, B: rbf_kernel(A, B, best["gamma"]),
    )
    resources["validation_kernel_entries_head_selection"] = int(
        len(U_validation) * n_train
    )
    head_candidates: list[dict] = []
    for C in C_GRID:
        if C in head_failures:
            head_candidates.append(
                {
                    "candidate_index": len(head_candidates),
                    "C": C,
                    "status": "FAILED",
                    "error": head_failures[C],
                }
            )
            continue
        meta = head_meta[C]
        status = "OK" if meta["converged"] else "NONCONVERGED"
        head_candidates.append(
            {
                "candidate_index": len(head_candidates),
                "C": C,
                "status": status,
                "validation": validation_metrics(y_validation, validation_scores[C]),
                **meta,
            }
        )
    try:
        selected_head = _choose_candidate(head_candidates)
    except RuntimeError as exc:
        raise MethodAuditFailure(
            failed_method_record(
                "projected_ring_IQP_96feature_RBF_SVC",
                f"head selection failed: {exc}",
                resources,
                feature_map_selection={
                    "status": "OK",
                    "candidates": map_candidates,
                    "selected": best["row"],
                },
                head_selection={
                    "status": "FAILED",
                    "candidates": head_candidates,
                    "selected": {"status": "NOT_RUN"},
                },
            )
        ) from exc
    selected_model = head_models[float(selected_head["C"])]
    threshold = fit_legitimate_validation_threshold(
        validation_scores[float(selected_head["C"])][
            np.asarray(y_validation, int) == 0
        ]
    )
    depth = int(best["spec"]["depth"])
    bandwidth = float(best["spec"]["lambda"])
    test_features, test_projection_meta = projected_iqp_features(U_test, depth, bandwidth)
    test_scaled = best["scaler"].transform(test_features)
    test_score = _stream_precomputed_scores(
        test_scaled,
        best["train_scaled"],
        {float(selected_head["C"]): selected_model},
        lambda A, B: rbf_kernel(A, B, best["gamma"]),
    )[float(selected_head["C"])]
    resources["test_kernel_entries"] = int(len(U_test) * n_train)
    resources["simulator_statevector_materializations_test"] = int(len(U_test))
    resources["expectation_values_test"] = int(96 * len(U_test))
    resources["observable_expectation_evaluations_per_transaction"] = 96
    resources["hardware_implied_observable_circuit_executions_per_transaction_per_shot"] = 96
    resources["hardware_implied_data_state_preparations_per_transaction_per_shot"] = 96
    resources["shot_accounting"] = (
        "naive ungrouped ceiling of 96 observable circuits and 96 data-state "
        "preparations per transaction per shot; no commuting-group compilation or "
        "finite hardware shot count was frozen or executed"
    )
    resources["implied_support_vector_evaluations_per_transaction"] = int(
        selected_head["support_vectors"]
    )
    resources["svc_dual_coefficients_plus_intercept_proxy"] = int(
        selected_head["support_vectors"] + 1
    )
    resources["feature_map_trainable_parameters"] = 0
    resources["exact_statevector_scores"] = True
    resources["evaluation_backend"] = "exact dense statevector simulator"
    resources["finite_shots_used"] = 0
    return {
        "method": "projected_ring_IQP_96feature_RBF_SVC",
        "status": "OK",
        "feature_map_selection": {
            "metric": "full-validation AUPRC at C=1 and gamma='scale'",
            "tie_break": "earliest depth-major then lambda-major frozen-grid row",
            "candidates": map_candidates,
            "selected": best["row"],
        },
        "head_selection": {
            "metric": "full-validation AUPRC after feature-map selection",
            "tie_break": "earliest C in frozen grid",
            "candidates": head_candidates,
            "selected": selected_head,
        },
        "test": ranking_metrics(y_test, test_score, threshold),
        "test_projection_checks": test_projection_meta,
        "resources": resources,
        "implementation_definition": {
            **IQP_FORMULA,
            "observables": "8*(X,Y,Z) + 8 ring edges*(X,Y,Z)x(X,Y,Z) = 96",
            "standardization": "fit on capped training projected features only",
            "rbf_gamma": "scikit-learn gamma='scale' evaluated after standardization",
            "RBF_PSD_repair": "none",
        },
    }, test_score


def qiskit_ring_iqp_state(u: np.ndarray, depth: int, bandwidth: float) -> np.ndarray:
    """Independent direct-Qiskit construction used only as a pre-outcome referee."""

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    u = np.asarray(u, float)
    if u.shape != (N_QUBITS,):
        raise ValueError(f"Qiskit referee expects one {N_QUBITS}-feature row")
    centered = 2.0 * u - 1.0
    circuit = QuantumCircuit(N_QUBITS)
    for _ in range(depth):
        for qubit in range(N_QUBITS):
            circuit.h(qubit)
        for qubit in range(N_QUBITS):
            circuit.rz(bandwidth * np.pi * centered[qubit], qubit)
        for qubit in range(N_QUBITS):
            neighbour = (qubit + 1) % N_QUBITS
            circuit.rzz(
                bandwidth * np.pi * centered[qubit] * centered[neighbour],
                qubit,
                neighbour,
            )
    return np.asarray(Statevector.from_instruction(circuit).data, complex)


def qiskit_pauli_expectations(state: np.ndarray) -> np.ndarray:
    from qiskit.quantum_info import Pauli, Statevector

    qiskit_state = Statevector(np.asarray(state, complex))
    values = []
    for observable in pauli_observables():
        label = ["I"] * N_QUBITS
        for qubit, axis in observable:
            label[N_QUBITS - 1 - qubit] = axis
        values.append(qiskit_state.expectation_value(Pauli("".join(label))))
    return np.asarray(values)


def synthetic_end_to_end_checks() -> dict:
    """Exercise every frozen model path on generated rows, never HSBC rows."""

    rng = np.random.default_rng(AUDIT_SEED + 99)
    train_legitimate = rng.beta(2.0, 4.0, size=(40, N_QUBITS))
    train_fraud = rng.beta(4.0, 2.0, size=(40, N_QUBITS))
    validation_legitimate = rng.beta(2.0, 4.0, size=(20, N_QUBITS))
    validation_fraud = rng.beta(4.0, 2.0, size=(20, N_QUBITS))
    test_legitimate = rng.beta(2.0, 4.0, size=(20, N_QUBITS))
    test_fraud = rng.beta(4.0, 2.0, size=(20, N_QUBITS))
    U_train = np.vstack([train_legitimate, train_fraud])
    U_validation = np.vstack([validation_legitimate, validation_fraud])
    U_test = np.vstack([test_legitimate, test_fraud])
    y_train = np.r_[np.zeros(40, int), np.ones(40, int)]
    y_validation = np.r_[np.zeros(20, int), np.ones(20, int)]
    y_test = np.r_[np.zeros(20, int), np.ones(20, int)]
    landmark_positions = np.r_[np.arange(32), np.arange(40, 72)]
    U_one_class_train = rng.beta(2.0, 4.0, size=(80, N_QUBITS))
    U_one_class_validation = rng.beta(2.0, 4.0, size=(40, N_QUBITS))
    one_class_landmark_positions = np.arange(N_LANDMARKS)

    calls = {
        "product_fidelity_QSVC": lambda: run_product_fidelity_svc(
            U_train, y_train, U_validation, y_validation, U_test, y_test
        ),
        "classical_RBF_SVC_twin": lambda: run_classical_rbf_svc(
            U_train, y_train, U_validation, y_validation, U_test, y_test
        ),
        "ring_IQP_fidelity_Nystrom": lambda: run_iqp_fidelity_nystrom(
            U_train,
            y_train,
            U_validation,
            y_validation,
            U_test,
            y_test,
            landmark_positions,
        ),
        "classical_RBF_Nystrom_twin": lambda: run_classical_rbf_nystrom(
            U_train,
            y_train,
            U_validation,
            y_validation,
            U_test,
            y_test,
            landmark_positions,
        ),
        "projected_ring_IQP_SVC": lambda: run_projected_iqp_svc(
            U_train, y_train, U_validation, y_validation, U_test, y_test
        ),
        "ring_IQP_fidelity_Nystrom_OCSVM": lambda: run_iqp_fidelity_nystrom_ocsvm(
            U_one_class_train,
            U_one_class_validation,
            U_test,
            y_test,
            one_class_landmark_positions,
        ),
        "classical_RBF_Nystrom_OCSVM_twin": lambda: run_classical_rbf_nystrom_ocsvm(
            U_one_class_train,
            U_one_class_validation,
            U_test,
            y_test,
            one_class_landmark_positions,
        ),
    }
    summary = {}
    for name, call in calls.items():
        record, score = call()
        if record.get("status") != "OK":
            raise AssertionError(f"synthetic end-to-end {name} status is not OK")
        if np.asarray(score).shape != (len(U_test),) or not np.all(np.isfinite(score)):
            raise AssertionError(f"synthetic end-to-end {name} emitted invalid scores")
        summary[name] = {
            "status": "PASS",
            "test_rows": int(len(score)),
            "selected_test_metrics_finite": True,
        }
    return summary


def run_self_checks() -> dict:
    """Synthetic-only norm, Hermiticity/PSD, and direct-Qiskit checks."""

    rng = np.random.default_rng(AUDIT_SEED)
    U = rng.uniform(0.03, 0.97, size=(5, N_QUBITS))
    max_norm_deviation = 0.0
    max_kernel_hermiticity = 0.0
    minimum_kernel_eigenvalue = np.inf
    max_qiskit_infidelity = 0.0
    max_qiskit_pauli_deviation = 0.0
    max_pauli_imaginary = 0.0
    max_pauli_bound_excess = 0.0

    checked_maps = []
    for depth in IQP_DEPTHS:
        for bandwidth in IQP_LAMBDAS:
            states = ring_iqp_states(U, depth, bandwidth)
            norms = np.einsum("bi,bi->b", states.conj(), states).real
            max_norm_deviation = max(
                max_norm_deviation, float(np.max(np.abs(norms - 1.0)))
            )
            kernel = state_fidelity_kernel(states, states)
            max_kernel_hermiticity = max(
                max_kernel_hermiticity, float(np.max(np.abs(kernel - kernel.T.conj())))
            )
            minimum_kernel_eigenvalue = min(
                minimum_kernel_eigenvalue,
                float(np.min(np.linalg.eigvalsh(0.5 * (kernel + kernel.T)))),
            )
            features, imaginary = projected_pauli_features_from_states(states)
            max_pauli_imaginary = max(max_pauli_imaginary, imaginary)
            max_pauli_bound_excess = max(
                max_pauli_bound_excess, float(max(np.max(np.abs(features)) - 1.0, 0.0))
            )
            qiskit_state = qiskit_ring_iqp_state(U[0], depth, bandwidth)
            fidelity = float(abs(np.vdot(qiskit_state, states[0])) ** 2)
            max_qiskit_infidelity = max(max_qiskit_infidelity, abs(1.0 - fidelity))
            qiskit_features = qiskit_pauli_expectations(qiskit_state)
            max_qiskit_pauli_deviation = max(
                max_qiskit_pauli_deviation,
                float(np.max(np.abs(qiskit_features.real - features[0]))),
            )
            max_pauli_imaginary = max(
                max_pauli_imaginary, float(np.max(np.abs(qiskit_features.imag)))
            )
            checked_maps.append({"depth": depth, "lambda": bandwidth})

    product = product_fidelity_kernel(U, U, PRODUCT_LAMBDAS[-1])
    product_hermiticity = float(np.max(np.abs(product - product.T)))
    product_diagonal_deviation = float(np.max(np.abs(np.diag(product) - 1.0)))
    product_minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(product)))

    # Exercise the Nyström algebra and both classical estimator types without
    # touching any repository dataset.
    synthetic_y = np.asarray([0, 0, 0, 1, 1], int)
    synthetic_landmarks = ring_iqp_states(U[:4], 1, 1.0)
    K_landmarks = state_fidelity_kernel(synthetic_landmarks, synthetic_landmarks)
    inverse_sqrt, repair = nystrom_inverse_sqrt(K_landmarks)
    synthetic_features = nystrom_features(
        state_fidelity_kernel(ring_iqp_states(U, 1, 1.0), synthetic_landmarks),
        inverse_sqrt,
    )
    logistic, logistic_warnings = _fit_logistic(synthetic_features, synthetic_y, C=1.0)
    if not np.all(np.isfinite(logistic.decision_function(synthetic_features))):
        raise AssertionError("synthetic Nyström/logistic score is nonfinite")
    K_product = product_fidelity_kernel(U, U, 0.5)
    svc, svc_warnings = _fit_precomputed_svc(K_product, synthetic_y, C=1.0)
    if not np.all(np.isfinite(svc.decision_function(K_product))):
        raise AssertionError("synthetic product-kernel SVC score is nonfinite")

    tolerances = {
        "norm": 1e-11,
        "kernel_hermiticity": 1e-12,
        "kernel_psd": -1e-10,
        "qiskit_infidelity": 1e-11,
        "qiskit_pauli": 1e-11,
        "pauli_imaginary": 1e-11,
        "pauli_bound_excess": 1e-11,
    }
    checks = {
        "state_norm": max_norm_deviation <= tolerances["norm"],
        "kernel_hermiticity": max_kernel_hermiticity <= tolerances["kernel_hermiticity"],
        "kernel_psd": minimum_kernel_eigenvalue >= tolerances["kernel_psd"],
        "direct_qiskit_state": max_qiskit_infidelity <= tolerances["qiskit_infidelity"],
        "direct_qiskit_pauli": max_qiskit_pauli_deviation <= tolerances["qiskit_pauli"],
        "pauli_hermiticity": max_pauli_imaginary <= tolerances["pauli_imaginary"],
        "pauli_bounds": max_pauli_bound_excess <= tolerances["pauli_bound_excess"],
        "product_kernel_hermiticity": product_hermiticity <= tolerances[
            "kernel_hermiticity"
        ],
        "product_kernel_diagonal": product_diagonal_deviation <= tolerances["norm"],
        "product_kernel_psd": product_minimum_eigenvalue >= tolerances["kernel_psd"],
    }
    synthetic_oc_rows = [
        {
            "candidate_index": 0,
            "status": "OK",
            "legitimate_validation": {
                "absolute_fpr_target_error": 5e-4,
                "mean_legitimate_validation_margin": 10.0,
            },
        },
        {
            "candidate_index": 1,
            "status": "OK",
            "legitimate_validation": {
                "absolute_fpr_target_error": 0.0,
                "mean_legitimate_validation_margin": 1.0,
            },
        },
        {
            "candidate_index": 2,
            "status": "OK",
            "legitimate_validation": {
                "absolute_fpr_target_error": 0.0,
                "mean_legitimate_validation_margin": 2.0,
            },
        },
        {
            "candidate_index": 3,
            "status": "OK",
            "legitimate_validation": {
                "absolute_fpr_target_error": 0.0,
                "mean_legitimate_validation_margin": 2.0,
            },
        },
    ]
    checks["one_class_lexicographic_selection"] = bool(
        _choose_one_class_candidate(synthetic_oc_rows)["candidate_index"] == 2
    )
    synthetic_legitimate_scores = np.linspace(-1.0, 1.0, 1_001)
    threshold_a = fit_legitimate_validation_threshold(synthetic_legitimate_scores)
    threshold_b = fit_legitimate_validation_threshold(synthetic_legitimate_scores.copy())
    checks["legitimate_only_threshold_deterministic"] = bool(
        threshold_a == threshold_b
        and not threshold_a["test_labels_or_scores_used_for_threshold"]
        and threshold_a["achieved_legitimate_validation_fpr"] <= TARGET_FPR
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"quantum-kernel self-check failure: {failed}")
    end_to_end = synthetic_end_to_end_checks()
    output_safety = synthetic_output_safety_checks()
    return {
        "status": "PASS",
        "scope": "synthetic rows only; no HSBC data loaded",
        "checked_maps": checked_maps,
        "checks": checks,
        "tolerances": tolerances,
        "max_state_norm_deviation": max_norm_deviation,
        "max_kernel_hermiticity": max_kernel_hermiticity,
        "minimum_kernel_eigenvalue": minimum_kernel_eigenvalue,
        "max_direct_qiskit_infidelity": max_qiskit_infidelity,
        "max_direct_qiskit_pauli_deviation": max_qiskit_pauli_deviation,
        "max_pauli_expectation_imaginary": max_pauli_imaginary,
        "max_pauli_bound_excess": max_pauli_bound_excess,
        "product_kernel": {
            "max_hermiticity_deviation": product_hermiticity,
            "max_diagonal_deviation": product_diagonal_deviation,
            "minimum_eigenvalue": product_minimum_eigenvalue,
        },
        "nystrom_psd_check": repair,
        "synthetic_end_to_end": end_to_end,
        "synthetic_output_safety": output_safety,
        "synthetic_logistic_warnings": logistic_warnings,
        "synthetic_svc_warnings": svc_warnings,
        "versions": package_versions(),
    }


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def protected_output_inputs() -> tuple[Path, ...]:
    return (SCRIPT_PATH, PROTOCOL.resolve(), SOURCE_DATA.resolve(), V1_SCORES.resolve())


def validate_output_targets(
    output: Path,
    scores_output: Path,
    *,
    audit_directory: Path = AUDIT_V2_DIRECTORY,
    protected_paths: tuple[Path, ...] | None = None,
) -> dict[str, Path]:
    """Resolve and reject unsafe, aliased, or out-of-scope output targets."""

    audit_lexical = Path(os.path.abspath(audit_directory))
    if audit_lexical.exists() and audit_lexical.is_symlink():
        raise ValueError(f"audit output directory must not be a symlink: {audit_lexical}")
    audit_resolved = audit_lexical.resolve(strict=False)
    protected = tuple(Path(path).resolve(strict=False) for path in (
        protected_paths if protected_paths is not None else protected_output_inputs()
    ))

    def resolve_one(path: Path, suffix: str, label: str) -> Path:
        lexical = Path(os.path.abspath(Path(path)))
        if lexical.suffix != suffix:
            raise ValueError(f"{label} must have suffix {suffix!r}: {lexical}")
        if lexical.exists() and lexical.is_symlink():
            raise ValueError(f"{label} must not be a symlink: {lexical}")
        if lexical.exists() and not lexical.is_file():
            raise ValueError(f"{label} must be a regular file target: {lexical}")
        resolved = lexical.resolve(strict=False)
        # The frozen outputs are direct children of the one audit directory.
        # This is intentionally stronger than merely sharing a string prefix.
        if resolved.parent != audit_resolved:
            raise ValueError(
                f"{label} must resolve directly under {audit_resolved}, got {resolved}"
            )
        for protected_path in protected:
            if resolved == protected_path:
                raise ValueError(f"{label} aliases protected input {protected_path}")
            if resolved.exists() and protected_path.exists():
                try:
                    if os.path.samefile(resolved, protected_path):
                        raise ValueError(f"{label} hard-links protected input {protected_path}")
                except FileNotFoundError:
                    pass
        return resolved

    output_resolved = resolve_one(Path(output), ".json", "JSON output")
    scores_resolved = resolve_one(Path(scores_output), ".npz", "NPZ output")
    if output_resolved == scores_resolved:
        raise ValueError("JSON and NPZ outputs must be distinct resolved files")
    if output_resolved.exists() and scores_resolved.exists():
        try:
            if os.path.samefile(output_resolved, scores_resolved):
                raise ValueError("JSON and NPZ outputs must not be hard-link aliases")
        except FileNotFoundError:
            pass
    return {
        "audit_directory": audit_resolved,
        "json": output_resolved,
        "npz": scores_resolved,
    }


def preflight_output_targets(
    output: Path,
    scores_output: Path,
    overwrite: bool,
    *,
    audit_directory: Path = AUDIT_V2_DIRECTORY,
    protected_paths: tuple[Path, ...] | None = None,
) -> dict[str, Path]:
    targets = validate_output_targets(
        output,
        scores_output,
        audit_directory=audit_directory,
        protected_paths=protected_paths,
    )
    existing = [str(targets[key]) for key in ("json", "npz") if targets[key].exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite frozen v2 artifacts without --overwrite: "
            + ", ".join(existing)
        )
    return targets


def _fsync_file(path: Path) -> None:
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_output_pair(
    json_path: Path,
    npz_path: Path,
    expected_scores: dict[str, np.ndarray],
    declared_final_npz: Path,
) -> dict:
    """Mechanically verify a staged or published JSON/NPZ pair."""

    json_path = Path(json_path)
    npz_path = Path(npz_path)
    if not json_path.is_file() or not npz_path.is_file():
        raise ValueError("both JSON and NPZ files must exist for pair verification")
    expected_keys = set(expected_scores)
    with np.load(npz_path, allow_pickle=False) as loaded:
        if set(loaded.files) != expected_keys:
            raise ValueError(
                f"NPZ keys differ: expected {sorted(expected_keys)}, got {sorted(loaded.files)}"
            )
        for key, expected in expected_scores.items():
            actual = np.asarray(loaded[key])
            expected_array = np.asarray(expected)
            if actual.dtype.hasobject:
                raise ValueError(f"NPZ key {key!r} has forbidden object dtype")
            if actual.shape != expected_array.shape or actual.dtype != expected_array.dtype:
                raise ValueError(f"NPZ key {key!r} shape/dtype mismatch")
            if not np.array_equal(actual, expected_array, equal_nan=True):
                raise ValueError(f"NPZ key {key!r} differs from staged source array")
            if np.issubdtype(actual.dtype, np.number) and not np.all(np.isfinite(actual)):
                raise ValueError(f"NPZ key {key!r} contains nonfinite values")
    if "y_test" not in expected_scores:
        raise ValueError("NPZ must contain y_test")
    y_test = np.asarray(expected_scores["y_test"])
    if y_test.ndim != 1 or not np.all(np.isin(y_test, [0, 1])):
        raise ValueError("y_test must be a one-dimensional binary array")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"JSON schema must be {SCHEMA!r}")
    if payload.get("status") not in {"COMPLETE", "FAILED"}:
        raise ValueError("JSON status must be COMPLETE or FAILED")
    artifacts = payload.get("artifacts", {})
    declared = Path(artifacts.get("scores", "")).resolve(strict=False)
    if declared != Path(declared_final_npz).resolve(strict=False):
        raise ValueError("JSON scores path does not name the final resolved NPZ target")
    npz_hash = sha256(npz_path)
    if artifacts.get("scores_sha256") != npz_hash:
        raise ValueError("JSON scores_sha256 does not match the verified NPZ")
    return {
        "json_bytes": int(json_path.stat().st_size),
        "npz_bytes": int(npz_path.stat().st_size),
        "npz_sha256": npz_hash,
        "score_keys": sorted(expected_keys),
        "test_rows": int(len(y_test)),
    }


def publish_output_pair(
    output: Path,
    scores_output: Path,
    result: dict,
    score_bundle: dict[str, np.ndarray],
    *,
    overwrite: bool,
    audit_directory: Path = AUDIT_V2_DIRECTORY,
    protected_paths: tuple[Path, ...] | None = None,
    pre_publish_guard: Callable[[], object] | None = None,
    failure_injector: Callable[[str], None] | None = None,
) -> dict:
    """Stage, verify, and rollback-safely publish the JSON/NPZ output pair."""

    targets = preflight_output_targets(
        output,
        scores_output,
        overwrite,
        audit_directory=audit_directory,
        protected_paths=protected_paths,
    )
    audit_root = targets["audit_directory"]
    audit_root.mkdir(parents=True, exist_ok=True)
    # Re-resolve after directory creation to close a symlink substitution gap.
    targets = preflight_output_targets(
        targets["json"],
        targets["npz"],
        overwrite,
        audit_directory=audit_root,
        protected_paths=protected_paths,
    )
    stage_directory = Path(
        tempfile.mkdtemp(prefix=".quantum_kernels_v2.stage.", dir=audit_root)
    )
    backup_directory: Path | None = None
    try:
        staged_npz = stage_directory / "quantum_kernels_v2.npz"
        staged_json = stage_directory / "quantum_kernels_v2.json"
        np.savez_compressed(staged_npz, **score_bundle)
        _fsync_file(staged_npz)
        staged_result = copy.deepcopy(result)
        staged_result.setdefault("artifacts", {})["scores"] = str(targets["npz"])
        staged_result["artifacts"]["scores_sha256"] = sha256(staged_npz)
        staged_result["publication"] = {
            "mode": "staged-verified coordinated replace with rollback",
            "json_commit_marker": str(targets["json"]),
            "resolved_json_target": str(targets["json"]),
            "resolved_npz_target": str(targets["npz"]),
            "staging_directory_parent": str(audit_root),
        }
        encoded = json.dumps(
            staged_result,
            indent=2,
            sort_keys=True,
            default=json_default,
            allow_nan=False,
        ) + "\n"
        staged_json.write_text(encoded, encoding="utf-8")
        _fsync_file(staged_json)
        _fsync_directory(stage_directory)
        staged_verification = verify_output_pair(
            staged_json, staged_npz, score_bundle, targets["npz"]
        )

        # The caller may re-hash/revalidate immutable scientific inputs here.
        # It runs after both staged files verify and immediately before backups
        # and final replacements.
        if pre_publish_guard is not None:
            pre_publish_guard()

        # Recheck aliases/existence immediately before publication.
        targets = preflight_output_targets(
            targets["json"],
            targets["npz"],
            overwrite,
            audit_directory=audit_root,
            protected_paths=protected_paths,
        )
        backup_directory = Path(
            tempfile.mkdtemp(prefix=".quantum_kernels_v2.backup.", dir=audit_root)
        )
        original_exists = {
            "npz": targets["npz"].exists(),
            "json": targets["json"].exists(),
        }
        backup_paths = {
            "npz": backup_directory / "original.npz",
            "json": backup_directory / "original.json",
        }
        original_hashes = {}
        for key in ("npz", "json"):
            if original_exists[key]:
                shutil.copy2(targets[key], backup_paths[key])
                _fsync_file(backup_paths[key])
                original_hashes[key] = sha256(backup_paths[key])
        _fsync_directory(backup_directory)

        replaced: list[str] = []
        try:
            os.replace(staged_npz, targets["npz"])
            replaced.append("npz")
            _fsync_directory(audit_root)
            if failure_injector is not None:
                failure_injector("after_npz_replace")
            os.replace(staged_json, targets["json"])
            replaced.append("json")
            _fsync_directory(audit_root)
            if failure_injector is not None:
                failure_injector("after_json_replace")
            published_verification = verify_output_pair(
                targets["json"], targets["npz"], score_bundle, targets["npz"]
            )
        except Exception as publish_error:
            rollback_errors = []
            for key in reversed(replaced):
                try:
                    if original_exists[key]:
                        os.replace(backup_paths[key], targets[key])
                    elif targets[key].exists() or targets[key].is_symlink():
                        targets[key].unlink()
                except Exception as rollback_error:  # pragma: no cover - catastrophic FS failure
                    rollback_errors.append(f"{key}: {type(rollback_error).__name__}: {rollback_error}")
            _fsync_directory(audit_root)
            for key, expected_hash in original_hashes.items():
                try:
                    if not targets[key].is_file() or sha256(targets[key]) != expected_hash:
                        rollback_errors.append(f"{key}: restored content hash mismatch")
                except Exception as rollback_error:  # pragma: no cover
                    rollback_errors.append(f"{key}: rollback verification failed: {rollback_error}")
            if rollback_errors:
                raise RuntimeError(
                    "output publication failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from publish_error
            raise

        return {
            "targets": {key: str(targets[key]) for key in ("json", "npz")},
            "staged_verification": staged_verification,
            "published_verification": published_verification,
            "overwrote_existing": original_exists,
        }
    finally:
        if backup_directory is not None:
            shutil.rmtree(backup_directory, ignore_errors=True)
        shutil.rmtree(stage_directory, ignore_errors=True)


def synthetic_output_safety_checks() -> dict:
    """Data-free path-alias, staging, verification, and rollback tests."""

    checks = {
        "outside_target_rejected": False,
        "wrong_suffix_rejected": False,
        "symlink_alias_rejected": False,
        "protected_hardlink_rejected": False,
        "pair_hardlink_alias_rejected": False,
        "staged_pair_published_and_verified": False,
        "overwrite_failure_restores_existing_pair": False,
        "fresh_failure_leaves_no_partial_pair": False,
        "nonfinite_stage_rejected_without_publish": False,
    }
    with tempfile.TemporaryDirectory(prefix="hsbc-kernel-output-safety-") as temporary:
        temporary_root = Path(temporary)
        audit_root = temporary_root / "runs/hsbc_challenge/audit_v2"
        audit_root.mkdir(parents=True)
        protected_root = temporary_root / "protected"
        protected_root.mkdir()
        protected_files = []
        for name in ("script.py", "protocol.md", "ulb.npz", "v1.npz"):
            path = protected_root / name
            path.write_bytes(("protected-" + name).encode())
            protected_files.append(path)
        protected = tuple(protected_files)

        try:
            validate_output_targets(
                temporary_root / "outside.json",
                audit_root / "scores.npz",
                audit_directory=audit_root,
                protected_paths=protected,
            )
        except ValueError:
            checks["outside_target_rejected"] = True
        try:
            validate_output_targets(
                audit_root / "wrong.txt",
                audit_root / "scores.npz",
                audit_directory=audit_root,
                protected_paths=protected,
            )
        except ValueError:
            checks["wrong_suffix_rejected"] = True

        symlink_alias = audit_root / "symlink_alias.json"
        symlink_alias.symlink_to(protected_files[1])
        try:
            validate_output_targets(
                symlink_alias,
                audit_root / "scores.npz",
                audit_directory=audit_root,
                protected_paths=protected,
            )
        except ValueError:
            checks["symlink_alias_rejected"] = True
        symlink_alias.unlink()

        hard_alias = audit_root / "hard_alias.json"
        os.link(protected_files[1], hard_alias)
        try:
            validate_output_targets(
                hard_alias,
                audit_root / "scores.npz",
                audit_directory=audit_root,
                protected_paths=protected,
            )
        except ValueError:
            checks["protected_hardlink_rejected"] = True
        hard_alias.unlink()

        pair_json = audit_root / "pair_alias.json"
        pair_npz = audit_root / "pair_alias.npz"
        pair_json.write_bytes(b"same inode")
        os.link(pair_json, pair_npz)
        try:
            validate_output_targets(
                pair_json,
                pair_npz,
                audit_directory=audit_root,
                protected_paths=protected,
            )
        except ValueError:
            checks["pair_hardlink_alias_rejected"] = True
        pair_json.unlink()
        pair_npz.unlink()

        final_json = audit_root / "synthetic.json"
        final_npz = audit_root / "synthetic.npz"
        scores = {
            "y_test": np.asarray([0, 1, 0, 1], dtype=np.int64),
            "synthetic_method": np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float64),
        }
        result = {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "artifacts": {},
            "scope": "synthetic output safety only",
        }
        publication = publish_output_pair(
            final_json,
            final_npz,
            result,
            scores,
            overwrite=False,
            audit_directory=audit_root,
            protected_paths=protected,
        )
        checks["staged_pair_published_and_verified"] = bool(
            publication["published_verification"]["test_rows"] == 4
        )
        old_json = final_json.read_bytes()
        old_npz = final_npz.read_bytes()
        replacement_scores = {
            "y_test": scores["y_test"].copy(),
            "synthetic_method": np.asarray([0.3, 0.7, 0.4, 0.6], dtype=np.float64),
        }

        def fail_after_npz(step: str) -> None:
            if step == "after_npz_replace":
                raise RuntimeError("synthetic injected publication failure")

        try:
            publish_output_pair(
                final_json,
                final_npz,
                result,
                replacement_scores,
                overwrite=True,
                audit_directory=audit_root,
                protected_paths=protected,
                failure_injector=fail_after_npz,
            )
        except RuntimeError as exc:
            if "synthetic injected" not in str(exc):
                raise
        checks["overwrite_failure_restores_existing_pair"] = bool(
            final_json.read_bytes() == old_json and final_npz.read_bytes() == old_npz
        )

        fresh_json = audit_root / "fresh.json"
        fresh_npz = audit_root / "fresh.npz"
        try:
            publish_output_pair(
                fresh_json,
                fresh_npz,
                result,
                scores,
                overwrite=False,
                audit_directory=audit_root,
                protected_paths=protected,
                failure_injector=fail_after_npz,
            )
        except RuntimeError as exc:
            if "synthetic injected" not in str(exc):
                raise
        checks["fresh_failure_leaves_no_partial_pair"] = bool(
            not fresh_json.exists() and not fresh_npz.exists()
        )

        bad_json = audit_root / "bad.json"
        bad_npz = audit_root / "bad.npz"
        bad_scores = {
            "y_test": scores["y_test"].copy(),
            "synthetic_method": np.asarray([0.1, np.nan, 0.2, 0.8], dtype=np.float64),
        }
        try:
            publish_output_pair(
                bad_json,
                bad_npz,
                result,
                bad_scores,
                overwrite=False,
                audit_directory=audit_root,
                protected_paths=protected,
            )
        except ValueError:
            checks["nonfinite_stage_rejected_without_publish"] = bool(
                not bad_json.exists() and not bad_npz.exists()
            )

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"synthetic output-safety checks failed: {failed}")
    return {"status": "PASS", "checks": checks, "scope": "temporary synthetic files only"}


def _method_guard(name: str, function: Callable[[], tuple[dict, np.ndarray]]):
    started = time.time()
    try:
        record, score = function()
        record["wall_seconds"] = time.time() - started
        log(f"{name}: completed validation selection and sealed full-test scoring")
        return record, score
    except MethodAuditFailure as exc:
        record = copy.deepcopy(exc.record)
        record["wall_seconds"] = time.time() - started
        record.setdefault("test", {"status": "NOT_RUN"})
        record.setdefault("resources", {})
        log(f"{name}: FAILED with retained candidates/resources: {exc}")
        return record, None
    except Exception as exc:
        log(f"{name}: FAILED: {type(exc).__name__}: {exc}")
        return {
            "method": name,
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "selected": {"status": "NOT_RUN"},
            "test": {"status": "NOT_RUN"},
            "resources": {},
            "wall_seconds": time.time() - started,
        }, None


def run_dataset_audit(output: Path, scores_output: Path, overwrite: bool = False) -> None:
    """Execute the frozen HSBC audit.  Call only through explicit ``--run``."""

    from datetime import datetime

    started = time.time()
    targets = preflight_output_targets(output, scores_output, overwrite)
    initial_script_hash = sha256(SCRIPT_PATH)
    self_checks = run_self_checks()
    if self_checks["status"] != "PASS":
        raise RuntimeError("synthetic/Qiskit checks did not pass")

    # This source/split/v1 identity gate precedes every HSBC model fit and every
    # HSBC test score.  Any mismatch aborts without publishing an artifact.
    pipe, references, initial_input_verification = prepare_and_verify_frozen_inputs(
        expected_script_hash=initial_script_hash
    )
    y_train = np.asarray(pipe["y_split"]["tr"], int)
    y_validation = np.asarray(pipe["y_split"]["va"], int)
    y_test = np.asarray(pipe["y_split"]["te"], int)
    U_validation = np.asarray(pipe["U"]["va"], float)
    U_test = np.asarray(pipe["U"]["te"], float)
    selection = select_capped_training_and_landmarks(pipe)
    capped = selection["capped_positions_in_train_split"]
    landmark_positions = selection["landmark_positions_in_cap"]
    U_all_train = np.asarray(pipe["U"]["tr"], float)
    U_train = U_all_train[capped]
    y_train_capped = y_train[capped]
    one_class_train_positions = selection["one_class_legitimate_positions_in_train_split"]
    one_class_landmark_positions = selection[
        "one_class_landmark_positions_in_legitimate_cap"
    ]
    U_one_class_train = U_all_train[one_class_train_positions]
    # This is the only use of validation labels for the OC rows: construct the
    # explicitly permitted legitimate-validation bracket.  Fraud validation
    # rows and their scores never enter map or nu selection.
    U_one_class_validation = U_validation[y_validation == 0]

    methods: dict[str, dict] = {}
    score_bundle: dict[str, np.ndarray] = {"y_test": y_test}

    methods["product_fidelity_QSVC"], score = _method_guard(
        "product_fidelity_QSVC",
        lambda: run_product_fidelity_svc(
            U_train,
            y_train_capped,
            U_validation,
            y_validation,
            U_test,
            y_test,
        ),
    )
    if score is not None:
        score_bundle["product_fidelity_QSVC"] = score

    methods["classical_RBF_SVC_twin"], score = _method_guard(
        "classical_RBF_SVC_twin",
        lambda: run_classical_rbf_svc(
            U_train,
            y_train_capped,
            U_validation,
            y_validation,
            U_test,
            y_test,
        ),
    )
    if score is not None:
        score_bundle["classical_RBF_SVC_twin"] = score

    methods["ring_IQP_fidelity_Nystrom"], score = _method_guard(
        "ring_IQP_fidelity_Nystrom",
        lambda: run_iqp_fidelity_nystrom(
            U_train,
            y_train_capped,
            U_validation,
            y_validation,
            U_test,
            y_test,
            landmark_positions,
        ),
    )
    if score is not None:
        score_bundle["ring_IQP_fidelity_Nystrom"] = score

    methods["classical_RBF_Nystrom_twin"], score = _method_guard(
        "classical_RBF_Nystrom_twin",
        lambda: run_classical_rbf_nystrom(
            U_train,
            y_train_capped,
            U_validation,
            y_validation,
            U_test,
            y_test,
            landmark_positions,
        ),
    )
    if score is not None:
        score_bundle["classical_RBF_Nystrom_twin"] = score

    methods["projected_ring_IQP_SVC"], score = _method_guard(
        "projected_ring_IQP_SVC",
        lambda: run_projected_iqp_svc(
            U_train,
            y_train_capped,
            U_validation,
            y_validation,
            U_test,
            y_test,
        ),
    )
    if score is not None:
        score_bundle["projected_ring_IQP_SVC"] = score

    methods["ring_IQP_fidelity_Nystrom_OCSVM"], score = _method_guard(
        "ring_IQP_fidelity_Nystrom_OCSVM",
        lambda: run_iqp_fidelity_nystrom_ocsvm(
            U_one_class_train,
            U_one_class_validation,
            U_test,
            y_test,
            one_class_landmark_positions,
        ),
    )
    if score is not None:
        score_bundle["ring_IQP_fidelity_Nystrom_OCSVM"] = score

    methods["classical_RBF_Nystrom_OCSVM_twin"], score = _method_guard(
        "classical_RBF_Nystrom_OCSVM_twin",
        lambda: run_classical_rbf_nystrom_ocsvm(
            U_one_class_train,
            U_one_class_validation,
            U_test,
            y_test,
            one_class_landmark_positions,
        ),
    )
    if score is not None:
        score_bundle["classical_RBF_Nystrom_OCSVM_twin"] = score

    score_bundle.update(references)

    scored_methods = {
        name: score
        for name, score in score_bundle.items()
        if name != "y_test" and len(score) == len(y_test)
    }
    bootstrap_summary, bootstrap_draws = paired_ap_bootstrap(y_test, scored_methods)
    for name, record in methods.items():
        if name in bootstrap_summary["ap_ci95"]:
            record["test"]["auprc_ci95"] = bootstrap_summary["ap_ci95"][name]

    comparisons = {}
    comparison_references = {
        "product_fidelity_QSVC": (
            "S2_L1_seed500",
            "S1_analytic",
            "XGB_subset",
            "classical_RBF_SVC_twin",
        ),
        "ring_IQP_fidelity_Nystrom": (
            "S2_L1_seed500",
            "S1_analytic",
            "XGB_subset",
            "classical_RBF_Nystrom_twin",
        ),
        "projected_ring_IQP_SVC": (
            "S2_L1_seed500",
            "S1_analytic",
            "XGB_subset",
            "classical_RBF_SVC_twin",
        ),
        "ring_IQP_fidelity_Nystrom_OCSVM": (
            "S1_analytic",
            "classical_RBF_Nystrom_OCSVM_twin",
        ),
        "classical_RBF_Nystrom_OCSVM_twin": ("S1_analytic",),
    }
    for quantum_method, reference_names in comparison_references.items():
        rows = {}
        for reference in reference_names:
            pair_name = f"vs_{reference}"
            missing = [
                name
                for name in (quantum_method, reference)
                if name not in scored_methods
            ]
            if missing:
                rows[pair_name] = {
                    "status": "NOT_AVAILABLE",
                    "missing_methods": missing,
                }
            else:
                rows[pair_name] = {
                    "status": "AVAILABLE",
                    **paired_comparison(
                        quantum_method,
                        reference,
                        bootstrap_summary["ap_point"],
                        bootstrap_draws,
                    ),
                }
        available = sum(row["status"] == "AVAILABLE" for row in rows.values())
        family_status = (
            "AVAILABLE"
            if available == len(rows)
            else "PARTIAL"
            if available
            else "NOT_AVAILABLE"
        )
        comparisons[quantum_method] = {"status": family_status, **rows}

    # Rebuild and revalidate once after all scoring, before result staging.  A
    # second identical check is passed into the publisher and runs after both
    # staged files verify, immediately before any final-target replacement.
    _, _, pre_staging_input_verification = prepare_and_verify_frozen_inputs(
        expected_script_hash=initial_script_hash,
        expected_test_labels=y_test,
    )
    failed_methods = sorted(
        name for name, record in methods.items() if record.get("status") != "OK"
    )
    result = {
        "schema": SCHEMA,
        "status": "FAILED" if failed_methods else "COMPLETE",
        "failed_methods": failed_methods,
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": initial_input_verification["hashes"]["protocol"],
        "script": str(SCRIPT_PATH),
        "script_sha256": initial_script_hash,
        "input_verification_before_scoring": initial_input_verification,
        "input_verification_before_staging": pre_staging_input_verification,
        "implementation_freeze": {
            "iqp_formula": IQP_FORMULA,
            "product_lambda_grid": list(PRODUCT_LAMBDAS),
            "iqp_depth_grid": list(IQP_DEPTHS),
            "iqp_lambda_grid": list(IQP_LAMBDAS),
            "C_grid": list(C_GRID),
            "classical_RBF_gamma_formula": "gamma=(lambda*pi/2)^2",
            "classical_RBF_gamma_grid": list(RBF_TWIN_GAMMAS),
            "conditional_one_class": {
                "map_nu": OC_MAP_NU,
                "nu_grid": list(OC_NU_GRID),
                "target_legitimate_validation_FPR": OC_TARGET_LEGITIMATE_VALIDATION_FPR,
                "classical_lambda_grid": list(OC_CLASSICAL_LAMBDAS),
                "classical_gamma_grid": list(OC_CLASSICAL_GAMMAS),
                "OCSVM_kernel": "linear on standardized Nystrom features",
                "training_rows": "2,000 legitimate only",
                "landmarks": "64 legitimate training rows shared by quantum/classical twins",
                "validation_selection": "legitimate rows only; no validation fraud score used",
            },
            "split_seed": SPLIT_SEED,
            "audit_and_landmark_seed": AUDIT_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "SVC": {
                "class_weight": "balanced",
                "tolerance": SVC_TOL,
                "max_iter": SVC_MAX_ITER,
            },
            "logistic": {
                "penalty": "L2",
                "class_weight": "balanced",
                "tolerance": LOGISTIC_TOL,
                "max_iter": LOGISTIC_MAX_ITER,
            },
            "Nystrom_rcond": NYSTROM_RCOND,
            "secondary_recall_threshold": {
                "target_FPR": TARGET_FPR,
                "fit_rows": "legitimate validation rows only",
                "quantile": 1.0 - TARGET_FPR,
                "quantile_method": "higher",
                "decision_rule": "anomaly iff score > threshold",
                "test_labels_or_scores_used_for_threshold": False,
            },
            "test_execution_guard": "explicit --run required",
        },
        "versions": package_versions(),
        "self_checks_before_data": self_checks,
        "shared_data": {
            "features": pipe["feature_names"],
            "split_rows": {
                key: int(len(pipe["idx"][key])) for key in ("tr", "va", "te")
            },
            "split_frauds": {
                key: int(np.sum(pipe["y_split"][key])) for key in ("tr", "va", "te")
            },
            "validation_resampled": False,
            "test_resampled": False,
            "conditional_label_screen_disclosure": (
                "all methods inherit prepare_hsbc's label-screened eight-feature representation"
            ),
        },
        "training_cap_and_landmarks": {
            "seed": AUDIT_SEED,
            "selection": "2,000 uniform legitimate rows plus all training frauds",
            "landmarks": (
                "32 uniform capped legitimate + 32 uniform capped fraud; no feature/score "
                "selection within class; identical rows for quantum/classical Nystrom"
            ),
            "one_class_selection": "the same 2,000 uniform legitimate training rows only",
            "one_class_landmarks": (
                "64 uniform rows from the legitimate cap; identical rows for the "
                "ring-IQP and classical RBF OC-SVM twins"
            ),
            **selection,
        },
        "methods": methods,
        "bootstrap": bootstrap_summary,
        "paired_comparisons": comparisons,
        "decision_interval": {
            "four_coprimary_families": N_COPRIMARY_FAMILIES,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "per_family_alpha": SIMULTANEOUS_ALPHA,
            "reported_simultaneous_interval": "98.75% percentile paired bootstrap",
        },
        "artifacts": {
            "scores": str(targets["npz"]),
            "scores_sha256": "FILLED_FROM_VERIFIED_STAGED_NPZ",
            "v1_reference_scores": str(V1_SCORES),
            "v1_reference_scores_sha256": initial_input_verification["hashes"][
                "v1_reference_scores"
            ],
            "source_data": str(SOURCE_DATA),
            "source_data_sha256": initial_input_verification["hashes"]["ULB_source"],
        },
        "wall_seconds": time.time() - started,
    }

    def final_input_guard() -> None:
        prepare_and_verify_frozen_inputs(
            expected_script_hash=initial_script_hash,
            expected_test_labels=y_test,
        )

    publication = publish_output_pair(
        targets["json"],
        targets["npz"],
        result,
        score_bundle,
        overwrite=overwrite,
        pre_publish_guard=final_input_guard,
    )
    log(f"published verified pair: {publication['targets']['json']}")
    log(f"published verified pair: {publication['targets']['npz']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="explicitly authorize the frozen HSBC data run; absent means self-check only",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--scores-output", type=Path, default=DEFAULT_NPZ)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing v2 outputs (never implied by --run)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run:
        checks = run_self_checks()
        print(json.dumps(checks, indent=2, sort_keys=True, default=json_default))
        print("SELF_CHECK_ONLY: no HSBC data loaded and no v2 artifact written")
        return
    run_dataset_audit(args.output, args.scores_output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
