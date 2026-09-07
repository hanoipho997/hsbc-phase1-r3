"""Frozen HSBC v2 hybrid-QML rivals: sections 3.3 and 3.4(2--3) only.

This file materializes the preregistered protocol in
``docs/hsbc_challenge_quantum_rivals_audit_v2.md`` without running it by
default.  A data-bearing execution requires the explicit ``--run`` flag.
Without that flag the script runs only deterministic synthetic norm,
parameter-shift, and end-to-end gradient checks and exits without creating an
artifact or loading the ULB/HSBC data.

Implemented, frozen conventions
-------------------------------

Hybrid QNN (Deloitte/AWS-shaped)
  * Dense 8->32 ReLU, dropout 0.3, Dense 32->9 ReLU, dropout 0.3.
  * The public three-qubit head is reproduced verbatim: RY(x0..x2),
    Rot(w0*x3,w1*x4,w2*x5) on q1,
    Rot(w3*x6,w4*x7,w5*x8) on q2, CNOT(1,2), RY(w6) on q2,
    CNOT(0,2), CNOT(1,2), and Z expectations on q0 and q2.
  * PennyLane's Rot(phi,theta,omega) convention is implemented as the gate
    sequence RZ(phi), RY(theta), RZ(omega).
  * There are 9 circuit inputs plus 7 trainable weights (16 differentiable
    variables), represented by 10 elementary rotation angles.  Exact
    parameter shifts are taken once per elementary angle and chained to all
    16 variables.  No angle is omitted.
  * The two expectations are logits.  Each update assigns total weight 1/2 to
    the 384 legitimate rows and total weight 1/2 to all 295 fraud rows.
  * The public front-end regularizers are retained: L2=0.01 on the first dense
    kernel and L1=L2=0.001 on the second.  Biases are unregularized.
  * Adam uses beta1=0.9, beta2=0.999, eps=1e-8; dense learning rate 1e-3 and
    quantum learning rate 1e-2.  Dropout is inverted dropout and is disabled
    for validation/test scoring.

Hybrid latent autoencoder
  * Tanh encoder 8->4, RY(pi*h) angle encoding, sequential CNOT ring
    0->1->2->3->0, one trainable RY(theta_q) per qubit, local-Z latent
    expectations, and a linear 4->8 decoder.
  * Direct-width-4 and width-8 tanh classical bottlenecks use linear decoders.
  * Reconstruction loss is mean squared error over rows and eight features;
    no weight regularization.  Adam constants match the QNN, with learning
    rate 1e-3 for classical tensors and 1e-2 for the four quantum angles.
  * Checkpoint selection uses legitimate-validation reconstruction loss only.
    IsolationForest is fitted only after checkpoint selection, on legitimate
    training latents, with 300 trees, max_samples=min(8192,n_train_legit),
    contamination='auto', and the model seed as random_state.

Common training and accounting
  * QNN seeds 830..834; HAE seeds 850..854; 300 updates; validation every 20
    updates; patience five checks; strict improvement, so the earliest exact
    tie wins.  Frozen failed seeds are retained and never replaced.
  * Sample-circuit forward equivalents count one exact statevector evaluation
    of one sample.  A quantum-gradient batch counts the unshifted forward plus
    two forwards per shifted elementary angle.  Validation, latent extraction,
    and test inference are counted.  The per-seed ceiling is 6,000,000.
  * Qubits are little-endian: basis-index bit q is qubit q.  Ring CNOTs are
    applied sequentially in the stated order.

The two output artifacts are written only by ``--run``:
``runs/hsbc_challenge/audit_v2/hybrid_qml_v2.json`` and
``runs/hsbc_challenge/audit_v2/hybrid_qml_v2.npz``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hsbc_common import ULB_NPZ, prepare_hsbc, ranking_metrics  # noqa: E402


SCRIPT_FILE = Path(__file__).resolve()
REPO_ROOT = SCRIPT_FILE.parents[2]
PROTOCOL_FILE = REPO_ROOT / "docs/hsbc_challenge_quantum_rivals_audit_v2.md"
PROTOCOL = f"{PROTOCOL_FILE} sections 3.3 and 3.4(2-3)"
DEFAULT_JSON = "runs/hsbc_challenge/audit_v2/hybrid_qml_v2.json"
DEFAULT_NPZ = "runs/hsbc_challenge/audit_v2/hybrid_qml_v2.npz"
DEFAULT_REFERENCES = "runs/hsbc_challenge/audit_v1/occ_tournament_scores_v1.npz"
EXPECTED_FEATURES = ("V14", "V4", "V12", "V11", "V10", "V3", "V16", "V2")

QNN_SEEDS = tuple(range(830, 835))
HAE_SEEDS = tuple(range(850, 855))
UPDATES = 300
BATCH_LEGIT = 384
CHECK_EVERY = 20
PATIENCE_CHECKS = 5
MAX_CIRCUIT_FORWARDS = 6_000_000
CHUNK = 8192
N_BOOT = 2000
BOOT_SEED = 20260831
SIMULTANEOUS_LEVEL = 0.9875

DROP_RATE = 0.3
QNN_DENSE_LR = 1e-3
QNN_QUANTUM_LR = 1e-2
HAE_CLASSICAL_LR = 1e-3
HAE_QUANTUM_LR = 1e-2
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
QNN_L2_FIRST = 1e-2
QNN_L1_SECOND = 1e-3
QNN_L2_SECOND = 1e-3

QNN_GATE_ANGLES = 10
QNN_DIFFERENTIABLE_VARIABLES = 16
HAE_GATE_ANGLES = 8

T0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_locked_paths(
    output: Path, scores_output: Path, references_path: Path
) -> tuple[Path, Path, Path, Path]:
    """Fail closed before touching HSBC data or an outcome target."""

    if Path.cwd().resolve() != REPO_ROOT:
        raise RuntimeError(f"run from frozen repository root {REPO_ROOT}")
    output = output.resolve(strict=False)
    scores_output = scores_output.resolve(strict=False)
    references_path = references_path.resolve(strict=True)
    data_path = (REPO_ROOT / ULB_NPZ).resolve(strict=True)
    allowed_parent = (REPO_ROOT / "runs/hsbc_challenge/audit_v2").resolve()
    expected_reference = (REPO_ROOT / DEFAULT_REFERENCES).resolve(strict=True)
    if output.parent != allowed_parent or scores_output.parent != allowed_parent:
        raise ValueError(f"outputs must be direct children of {allowed_parent}")
    if output.suffix != ".json" or scores_output.suffix != ".npz":
        raise ValueError("outcome targets must have distinct .json and .npz suffixes")
    if output == scores_output:
        raise ValueError("JSON and NPZ output targets must be distinct")
    if references_path != expected_reference:
        raise ValueError(f"references must be the frozen artifact {expected_reference}")
    forbidden = {
        SCRIPT_FILE,
        PROTOCOL_FILE.resolve(strict=True),
        data_path,
        references_path,
    }
    if output in forbidden or scores_output in forbidden:
        raise ValueError("outcome target aliases a frozen input/source")
    if output.exists() or scores_output.exists():
        raise FileExistsError("refusing to overwrite an existing v2 outcome artifact")
    return output, scores_output, references_path, data_path


def _stage_and_publish_pair(
    output: Path,
    scores_output: Path,
    score_bundle: dict[str, np.ndarray],
    payload: dict,
) -> None:
    """Validate both staged files, then publish NPZ first and JSON last."""

    output.parent.mkdir(parents=True, exist_ok=True)
    nonce = f"{os.getpid()}-{time.time_ns()}"
    staged_scores = scores_output.with_name(f".{scores_output.stem}.{nonce}.tmp.npz")
    staged_json = output.with_name(f".{output.stem}.{nonce}.tmp.json")
    published_scores = False
    try:
        np.savez_compressed(staged_scores, **score_bundle)
        with np.load(staged_scores, allow_pickle=False) as check:
            if set(check.files) != set(score_bundle):
                raise RuntimeError("staged NPZ key-set mismatch")
            for name, expected in score_bundle.items():
                observed = np.asarray(check[name])
                if observed.shape != np.asarray(expected).shape or not np.array_equal(
                    observed, np.asarray(expected), equal_nan=True
                ):
                    raise RuntimeError(f"staged NPZ array mismatch: {name}")
        payload["scores_sha256"] = sha256(staged_scores)
        staged_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        checked_payload = json.loads(staged_json.read_text())
        if checked_payload.get("scores_sha256") != sha256(staged_scores):
            raise RuntimeError("staged JSON/NPZ hash link mismatch")
        os.replace(staged_scores, scores_output)
        published_scores = True
        os.replace(staged_json, output)
    except Exception:
        for staged in (staged_scores, staged_json):
            if staged.exists():
                staged.unlink()
        if published_scores and scores_output.exists() and not output.exists():
            scores_output.unlink()
        raise


def _json_params(params: dict[str, np.ndarray]) -> dict[str, list]:
    return {name: np.asarray(value).tolist() for name, value in params.items()}


def _copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value).copy() for name, value in params.items()}


def _all_finite(params: dict[str, np.ndarray]) -> bool:
    return all(np.all(np.isfinite(value)) for value in params.values())


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=1, keepdims=True)


def _metrics(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    y_validation: np.ndarray,
    validation_scores: np.ndarray,
    fpr: float = 1e-3,
) -> dict:
    """Full-test rankings plus a legitimate-validation-fixed operating point."""

    test_values = np.asarray(test_scores, float)
    validation_values = np.asarray(validation_scores, float)
    validation_labels = np.asarray(y_validation, int)
    test_labels = np.asarray(y_test, int)
    if (
        len(validation_values) != len(validation_labels)
        or len(test_values) != len(test_labels)
        or not np.all(np.isfinite(validation_values))
        or not np.all(np.isfinite(test_values))
    ):
        return {"status": "NOT_AVAILABLE"}
    legitimate_validation = validation_values[validation_labels == 0]
    threshold = float(
        np.quantile(legitimate_validation, 1.0 - fpr, method="higher")
    )
    return {
        **ranking_metrics(test_labels, test_values),
        "operating_point_rule": (
            "score > legitimate-validation quantile(0.999, method='higher')"
        ),
        "operating_threshold": threshold,
        "realized_validation_legitimate_fpr": float(
            np.mean(legitimate_validation > threshold)
        ),
        "realized_test_fpr": float(np.mean(test_values[test_labels == 0] > threshold)),
        "test_recall_at_validation_target_fpr_1e-3": float(
            np.mean(test_values[test_labels == 1] > threshold)
        ),
    }


# ---------------------------------------------------------------------------
# Exact little-endian statevector primitives
# ---------------------------------------------------------------------------


def _zero_state(batch: int, n_qubits: int) -> np.ndarray:
    state = np.zeros((batch, 2**n_qubits), dtype=complex)
    state[:, 0] = 1.0
    return state


def _apply_ry(state: np.ndarray, n_qubits: int, qubit: int, angle) -> np.ndarray:
    angle = np.asarray(angle, float)
    if angle.ndim == 0:
        cosine = np.cos(angle / 2.0)
        sine = np.sin(angle / 2.0)
    else:
        cosine = np.cos(angle / 2.0)[:, None, None]
        sine = np.sin(angle / 2.0)[:, None, None]
    view = state.reshape(len(state), -1, 2, 2**qubit)
    value0 = view[:, :, 0, :].copy()
    value1 = view[:, :, 1, :].copy()
    out = np.empty_like(view)
    out[:, :, 0, :] = cosine * value0 - sine * value1
    out[:, :, 1, :] = sine * value0 + cosine * value1
    return out.reshape(len(state), 2**n_qubits)


def _apply_rz(state: np.ndarray, n_qubits: int, qubit: int, angle) -> np.ndarray:
    angle = np.asarray(angle, float)
    if angle.ndim == 0:
        phase0 = np.exp(-0.5j * angle)
        phase1 = np.exp(0.5j * angle)
    else:
        phase0 = np.exp(-0.5j * angle)[:, None, None]
        phase1 = np.exp(0.5j * angle)[:, None, None]
    view = state.reshape(len(state), -1, 2, 2**qubit).copy()
    view[:, :, 0, :] *= phase0
    view[:, :, 1, :] *= phase1
    return view.reshape(len(state), 2**n_qubits)


def _apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    if control == target:
        raise ValueError("CNOT control and target must differ")
    indices = np.arange(state.shape[1])
    mapped = indices ^ (((indices >> control) & 1) << target)
    out = np.empty_like(state)
    out[:, mapped] = state
    return out


def _z_expectations(state: np.ndarray, qubits: tuple[int, ...]) -> np.ndarray:
    probabilities = np.abs(state) ** 2
    indices = np.arange(state.shape[1])
    signs = np.column_stack([1.0 - 2.0 * ((indices >> q) & 1) for q in qubits])
    return probabilities @ signs


def _max_norm_residual(state: np.ndarray) -> float:
    return float(np.max(np.abs(np.sum(np.abs(state) ** 2, axis=1) - 1.0)))


# ---------------------------------------------------------------------------
# Verbatim Deloitte/AWS three-qubit quantum head
# ---------------------------------------------------------------------------


def qnn_gate_angles(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    inputs = np.asarray(inputs, float)
    weights = np.asarray(weights, float)
    if inputs.ndim != 2 or inputs.shape[1] != 9:
        raise ValueError(f"QNN head requires shape (batch,9), got {inputs.shape}")
    if weights.shape != (7,):
        raise ValueError(f"QNN head requires seven weights, got {weights.shape}")
    return np.column_stack(
        [
            inputs[:, 0],
            inputs[:, 1],
            inputs[:, 2],
            inputs[:, 3] * weights[0],
            inputs[:, 4] * weights[1],
            inputs[:, 5] * weights[2],
            inputs[:, 6] * weights[3],
            inputs[:, 7] * weights[4],
            inputs[:, 8] * weights[5],
            np.full(len(inputs), weights[6]),
        ]
    )


def qnn_state_from_angles(angles: np.ndarray) -> np.ndarray:
    angles = np.asarray(angles, float)
    if angles.ndim != 2 or angles.shape[1] != QNN_GATE_ANGLES:
        raise ValueError(f"expected (batch,{QNN_GATE_ANGLES}) angles, got {angles.shape}")
    state = _zero_state(len(angles), 3)
    for q in range(3):
        state = _apply_ry(state, 3, q, angles[:, q])
    # PennyLane Rot(phi,theta,omega) = RZ(phi), RY(theta), RZ(omega).
    state = _apply_rz(state, 3, 1, angles[:, 3])
    state = _apply_ry(state, 3, 1, angles[:, 4])
    state = _apply_rz(state, 3, 1, angles[:, 5])
    state = _apply_rz(state, 3, 2, angles[:, 6])
    state = _apply_ry(state, 3, 2, angles[:, 7])
    state = _apply_rz(state, 3, 2, angles[:, 8])
    state = _apply_cnot(state, 1, 2)
    state = _apply_ry(state, 3, 2, angles[:, 9])
    state = _apply_cnot(state, 0, 2)
    state = _apply_cnot(state, 1, 2)
    return state


def qnn_head_from_angles(angles: np.ndarray) -> np.ndarray:
    return _z_expectations(qnn_state_from_angles(angles), (0, 2))


def qnn_head(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return qnn_head_from_angles(qnn_gate_angles(inputs, weights))


def qnn_head_jacobians(
    inputs: np.ndarray,
    weights: np.ndarray,
    *,
    train_quantum: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """Return logits, dlogits/dinputs, dlogits/dweights, calls per sample.

    Shifts are applied to the ten elementary gate angles.  Product-angle chain
    rules then produce derivatives with respect to all nine inputs and seven
    weights.  A frozen head omits the final, weight-only shift.
    """

    inputs = np.asarray(inputs, float)
    weights = np.asarray(weights, float)
    angles = qnn_gate_angles(inputs, weights)
    logits = qnn_head_from_angles(angles)
    shifted_angles = QNN_GATE_ANGLES if train_quantum else QNN_GATE_ANGLES - 1
    gate_jac = np.zeros((len(inputs), 2, QNN_GATE_ANGLES), dtype=float)
    for index in range(shifted_angles):
        plus = angles.copy()
        minus = angles.copy()
        plus[:, index] += np.pi / 2.0
        minus[:, index] -= np.pi / 2.0
        gate_jac[:, :, index] = 0.5 * (
            qnn_head_from_angles(plus) - qnn_head_from_angles(minus)
        )

    input_jac = np.zeros((len(inputs), 2, 9), dtype=float)
    input_jac[:, :, :3] = gate_jac[:, :, :3]
    for index in range(3, 9):
        input_jac[:, :, index] = gate_jac[:, :, index] * weights[index - 3]

    weight_jac = None
    if train_quantum:
        weight_jac = np.zeros((len(inputs), 2, 7), dtype=float)
        for index in range(6):
            weight_jac[:, :, index] = (
                gate_jac[:, :, index + 3] * inputs[:, index + 3, None]
            )
        weight_jac[:, :, 6] = gate_jac[:, :, 9]
    return logits, input_jac, weight_jac, 1 + 2 * shifted_angles


# ---------------------------------------------------------------------------
# Four-qubit hybrid-autoencoder bottleneck
# ---------------------------------------------------------------------------


def hae_gate_angles(encoded: np.ndarray, theta: np.ndarray) -> np.ndarray:
    encoded = np.asarray(encoded, float)
    theta = np.asarray(theta, float)
    if encoded.ndim != 2 or encoded.shape[1] != 4:
        raise ValueError(f"HAE bottleneck requires shape (batch,4), got {encoded.shape}")
    if theta.shape != (4,):
        raise ValueError(f"HAE bottleneck requires four angles, got {theta.shape}")
    return np.column_stack([np.pi * encoded, np.broadcast_to(theta, (len(encoded), 4))])


def hae_state_from_angles(angles: np.ndarray) -> np.ndarray:
    angles = np.asarray(angles, float)
    if angles.ndim != 2 or angles.shape[1] != HAE_GATE_ANGLES:
        raise ValueError(f"expected (batch,{HAE_GATE_ANGLES}) angles, got {angles.shape}")
    state = _zero_state(len(angles), 4)
    for q in range(4):
        state = _apply_ry(state, 4, q, angles[:, q])
    for control, target in ((0, 1), (1, 2), (2, 3), (3, 0)):
        state = _apply_cnot(state, control, target)
    for q in range(4):
        state = _apply_ry(state, 4, q, angles[:, 4 + q])
    return state


def hae_bottleneck_from_angles(angles: np.ndarray) -> np.ndarray:
    return _z_expectations(hae_state_from_angles(angles), (0, 1, 2, 3))


def hae_bottleneck(encoded: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return hae_bottleneck_from_angles(hae_gate_angles(encoded, theta))


def hae_bottleneck_jacobians(
    encoded: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    angles = hae_gate_angles(encoded, theta)
    latent = hae_bottleneck_from_angles(angles)
    gate_jac = np.zeros((len(encoded), 4, HAE_GATE_ANGLES), dtype=float)
    for index in range(HAE_GATE_ANGLES):
        plus = angles.copy()
        minus = angles.copy()
        plus[:, index] += np.pi / 2.0
        minus[:, index] -= np.pi / 2.0
        gate_jac[:, :, index] = 0.5 * (
            hae_bottleneck_from_angles(plus) - hae_bottleneck_from_angles(minus)
        )
    encoded_jac = np.pi * gate_jac[:, :, :4]
    theta_jac = gate_jac[:, :, 4:]
    return latent, encoded_jac, theta_jac, 1 + 2 * HAE_GATE_ANGLES


# ---------------------------------------------------------------------------
# Initializers, losses, gradients, and Adam
# ---------------------------------------------------------------------------


def _he_normal(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    return rng.normal(scale=np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))


def _xavier_normal(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    return rng.normal(scale=np.sqrt(2.0 / (fan_in + fan_out)), size=(fan_in, fan_out))


def initialize_qnn(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3301]))
    return {
        "W1": _he_normal(rng, 8, 32),
        "b1": np.zeros(32),
        "W2": _he_normal(rng, 32, 9),
        "b2": np.zeros(9),
        "q": rng.normal(scale=0.1, size=7),
        "Wh": _he_normal(rng, 9, 2),
        "bh": np.zeros(2),
    }


def initialize_hae(seed: int) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3402]))
    four = {
        "We": _xavier_normal(rng, 8, 4),
        "be": np.zeros(4),
        "Wd": _xavier_normal(rng, 4, 8),
        "bd": np.zeros(8),
        "q": rng.normal(scale=0.1, size=4),
    }
    eight = {
        "We": _xavier_normal(rng, 8, 8),
        "be": np.zeros(8),
        "Wd": _xavier_normal(rng, 8, 8),
        "bd": np.zeros(8),
    }
    return {
        "hybrid_quantum": _copy_params(four),
        "direct4_classical": {k: v.copy() for k, v in four.items() if k != "q"},
        "width8_classical": _copy_params(eight),
    }


def _weighted_cross_entropy(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, int)
    probabilities = _softmax(logits)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("weighted cross-entropy batch must contain both classes")
    sample_weight = np.where(labels == 0, 0.5 / counts[0], 0.5 / counts[1])
    loss = -float(
        np.sum(sample_weight * np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-15, 1.0)))
    )
    grad = probabilities.copy()
    grad[np.arange(len(labels)), labels] -= 1.0
    grad *= sample_weight[:, None]
    return loss, grad, probabilities


def _dense_front(
    params: dict[str, np.ndarray],
    values: np.ndarray,
    mask1: np.ndarray | None,
    mask2: np.ndarray | None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    pre1 = values @ params["W1"] + params["b1"]
    hidden1 = np.maximum(pre1, 0.0)
    dropped1 = hidden1 if mask1 is None else hidden1 * mask1
    pre2 = dropped1 @ params["W2"] + params["b2"]
    hidden2 = np.maximum(pre2, 0.0)
    dropped2 = hidden2 if mask2 is None else hidden2 * mask2
    return dropped2, (pre1, hidden1, dropped1, pre2, hidden2)


def qnn_loss_and_gradients(
    params: dict[str, np.ndarray],
    values: np.ndarray,
    labels: np.ndarray,
    mode: str,
    mask1: np.ndarray | None,
    mask2: np.ndarray | None,
) -> tuple[float, dict[str, np.ndarray], int]:
    features, cache = _dense_front(params, values, mask1, mask2)
    if mode == "dense_head":
        logits = features @ params["Wh"] + params["bh"]
        input_jac = None
        weight_jac = None
        calls_per_sample = 0
    elif mode in ("trainable_quantum", "frozen_quantum"):
        logits, input_jac, weight_jac, calls_per_sample = qnn_head_jacobians(
            features,
            params["q"],
            train_quantum=(mode == "trainable_quantum"),
        )
    else:
        raise ValueError(f"unknown QNN mode {mode}")

    loss, dlogits, _ = _weighted_cross_entropy(logits, labels)
    gradients: dict[str, np.ndarray] = {}
    if mode == "dense_head":
        gradients["Wh"] = features.T @ dlogits
        gradients["bh"] = np.sum(dlogits, axis=0)
        dfeatures = dlogits @ params["Wh"].T
    else:
        assert input_jac is not None
        dfeatures = np.einsum("bo,boj->bj", dlogits, input_jac)
        if mode == "trainable_quantum":
            assert weight_jac is not None
            gradients["q"] = np.einsum("bo,bok->k", dlogits, weight_jac)

    pre1, _, dropped1, pre2, _ = cache
    if mask2 is not None:
        dfeatures = dfeatures * mask2
    dpre2 = dfeatures * (pre2 > 0.0)
    gradients["W2"] = dropped1.T @ dpre2
    gradients["b2"] = np.sum(dpre2, axis=0)
    ddropped1 = dpre2 @ params["W2"].T
    if mask1 is not None:
        ddropped1 = ddropped1 * mask1
    dpre1 = ddropped1 * (pre1 > 0.0)
    gradients["W1"] = values.T @ dpre1
    gradients["b1"] = np.sum(dpre1, axis=0)

    loss += QNN_L2_FIRST * float(np.sum(params["W1"] ** 2))
    loss += QNN_L1_SECOND * float(np.sum(np.abs(params["W2"])))
    loss += QNN_L2_SECOND * float(np.sum(params["W2"] ** 2))
    gradients["W1"] += 2.0 * QNN_L2_FIRST * params["W1"]
    gradients["W2"] += QNN_L1_SECOND * np.sign(params["W2"])
    gradients["W2"] += 2.0 * QNN_L2_SECOND * params["W2"]
    return loss, gradients, calls_per_sample


def qnn_scores(
    params: dict[str, np.ndarray],
    values: np.ndarray,
    mode: str,
    chunk: int = CHUNK,
) -> np.ndarray:
    rows = []
    for start in range(0, len(values), chunk):
        features, _ = _dense_front(params, values[start : start + chunk], None, None)
        if mode == "dense_head":
            logits = features @ params["Wh"] + params["bh"]
        else:
            logits = qnn_head(features, params["q"])
        rows.append(_softmax(logits)[:, 1])
    return np.concatenate(rows) if rows else np.empty(0)


def hae_loss_and_gradients(
    params: dict[str, np.ndarray],
    values: np.ndarray,
    mode: str,
) -> tuple[float, dict[str, np.ndarray], int]:
    pre = values @ params["We"] + params["be"]
    encoded = np.tanh(pre)
    if mode == "hybrid_quantum":
        latent, encoded_jac, theta_jac, calls_per_sample = hae_bottleneck_jacobians(
            encoded, params["q"]
        )
    elif mode in ("direct4_classical", "width8_classical"):
        latent = encoded
        encoded_jac = None
        theta_jac = None
        calls_per_sample = 0
    else:
        raise ValueError(f"unknown HAE mode {mode}")
    reconstruction = latent @ params["Wd"] + params["bd"]
    residual = reconstruction - values
    loss = float(np.mean(residual**2))
    dreconstruction = 2.0 * residual / residual.size
    gradients = {
        "Wd": latent.T @ dreconstruction,
        "bd": np.sum(dreconstruction, axis=0),
    }
    dlatent = dreconstruction @ params["Wd"].T
    if mode == "hybrid_quantum":
        assert encoded_jac is not None and theta_jac is not None
        dencoded = np.einsum("bo,boj->bj", dlatent, encoded_jac)
        gradients["q"] = np.einsum("bo,bok->k", dlatent, theta_jac)
    else:
        dencoded = dlatent
    dpre = dencoded * (1.0 - encoded**2)
    gradients["We"] = values.T @ dpre
    gradients["be"] = np.sum(dpre, axis=0)
    return loss, gradients, calls_per_sample


def hae_latents(
    params: dict[str, np.ndarray],
    values: np.ndarray,
    mode: str,
    chunk: int = CHUNK,
) -> np.ndarray:
    rows = []
    for start in range(0, len(values), chunk):
        part = values[start : start + chunk]
        encoded = np.tanh(part @ params["We"] + params["be"])
        if mode == "hybrid_quantum":
            encoded = hae_bottleneck(encoded, params["q"])
        rows.append(encoded)
    return np.vstack(rows) if rows else np.empty((0, params["be"].size))


def hae_reconstruction_loss(
    params: dict[str, np.ndarray],
    values: np.ndarray,
    mode: str,
    chunk: int = CHUNK,
) -> float:
    squared_sum = 0.0
    count = 0
    for start in range(0, len(values), chunk):
        part = values[start : start + chunk]
        latent = hae_latents(params, part, mode, chunk=len(part) or 1)
        reconstruction = latent @ params["Wd"] + params["bd"]
        squared_sum += float(np.sum((reconstruction - part) ** 2))
        count += int(part.size)
    return squared_sum / count


@dataclass
class AdamState:
    first: dict[str, np.ndarray]
    second: dict[str, np.ndarray]


def _adam_state(params: dict[str, np.ndarray], trainable: tuple[str, ...]) -> AdamState:
    return AdamState(
        first={name: np.zeros_like(params[name]) for name in trainable},
        second={name: np.zeros_like(params[name]) for name in trainable},
    )


def _adam_update(
    params: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    state: AdamState,
    update: int,
    learning_rates: dict[str, float],
) -> None:
    for name, first in state.first.items():
        gradient = gradients[name]
        second = state.second[name]
        first *= ADAM_BETA1
        first += (1.0 - ADAM_BETA1) * gradient
        second *= ADAM_BETA2
        second += (1.0 - ADAM_BETA2) * gradient * gradient
        first_hat = first / (1.0 - ADAM_BETA1**update)
        second_hat = second / (1.0 - ADAM_BETA2**update)
        params[name] -= learning_rates[name] * first_hat / (np.sqrt(second_hat) + ADAM_EPS)


@dataclass
class CircuitBudget:
    limit: int = MAX_CIRCUIT_FORWARDS
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return int(sum(self.counts.values()))

    def reserve(self, category: str, amount: int) -> bool:
        amount = int(amount)
        if self.total + amount > self.limit:
            return False
        self.counts[category] = self.counts.get(category, 0) + amount
        return True

    def metadata(self) -> dict:
        return {
            "unit": "sample-circuit forward equivalents",
            "limit": self.limit,
            "total": self.total,
            "counts": dict(sorted(self.counts.items())),
        }


def _dropout_masks(seed: int, update: int, batch: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, update, 3303]))
    keep = 1.0 - DROP_RATE
    mask1 = (rng.random((batch, 32)) < keep).astype(float) / keep
    mask2 = (rng.random((batch, 9)) < keep).astype(float) / keep
    return mask1, mask2


def _parameter_counts_qnn(mode: str) -> dict:
    front = 8 * 32 + 32 + 32 * 9 + 9
    if mode == "dense_head":
        return {"trainable": front + 9 * 2 + 2, "total": front + 9 * 2 + 2}
    if mode == "frozen_quantum":
        return {"trainable": front, "total": front + 7, "frozen": 7}
    return {"trainable": front + 7, "total": front + 7}


def _parameter_counts_hae(mode: str) -> dict:
    if mode == "width8_classical":
        total = 8 * 8 + 8 + 8 * 8 + 8
    else:
        total = 8 * 4 + 4 + 4 * 8 + 8 + (4 if mode == "hybrid_quantum" else 0)
    return {"trainable": total, "total": total}


def _qnn_measurement_ledger(mode: str, validation_rows: int, test_rows: int) -> dict:
    if mode == "dense_head":
        return {
            "quantum_measurements": "N/A",
            "reason": "classical dense-head twin",
        }
    return {
        "threshold_validation_rows": int(validation_rows),
        "test_rows": int(test_rows),
        "local_Z_observables_per_transaction": 2,
        "commuting_measurement_settings_per_transaction": 1,
        "test_expectation_values": int(2 * test_rows),
        "threshold_validation_expectation_values": int(2 * validation_rows),
        "threshold_validation_measurement_settings": int(validation_rows),
        "test_measurement_settings": int(test_rows),
        "shots": "NOT_RUN",
        "shot_matched": False,
        "note": "exact statevector expectations; finite-shot repetitions are not inferred",
    }


def _hae_measurement_ledger(
    mode: str,
    train_legitimate_rows: int,
    validation_rows: int,
    test_rows: int,
) -> dict:
    if mode != "hybrid_quantum":
        return {
            "quantum_measurements": "N/A",
            "reason": "classical latent-bottleneck twin",
        }
    return {
        "isolation_forest_fit_rows": int(train_legitimate_rows),
        "threshold_validation_rows": int(validation_rows),
        "test_rows": int(test_rows),
        "local_Z_observables_per_transaction": 4,
        "commuting_measurement_settings_per_transaction": 1,
        "isolation_forest_fit_expectation_values": int(4 * train_legitimate_rows),
        "test_expectation_values": int(4 * test_rows),
        "threshold_validation_expectation_values": int(4 * validation_rows),
        "isolation_forest_fit_measurement_settings": int(train_legitimate_rows),
        "test_measurement_settings": int(test_rows),
        "threshold_validation_measurement_settings": int(validation_rows),
        "shots": "NOT_RUN",
        "shot_matched": False,
        "note": "exact statevector expectations; finite-shot repetitions are not inferred",
    }


# ---------------------------------------------------------------------------
# Frozen training/checkpoint loops
# ---------------------------------------------------------------------------


def _qnn_mode_params(seed: int, mode: str) -> dict[str, np.ndarray]:
    initialized = initialize_qnn(seed)
    common = {name: initialized[name].copy() for name in ("W1", "b1", "W2", "b2")}
    if mode in ("trainable_quantum", "frozen_quantum"):
        common["q"] = initialized["q"].copy()
    elif mode == "dense_head":
        common["Wh"] = initialized["Wh"].copy()
        common["bh"] = initialized["bh"].copy()
    else:
        raise ValueError(mode)
    return common


def _qnn_trainable_names(mode: str) -> tuple[str, ...]:
    if mode == "trainable_quantum":
        return ("W1", "b1", "W2", "b2", "q")
    if mode == "frozen_quantum":
        return ("W1", "b1", "W2", "b2")
    if mode == "dense_head":
        return ("W1", "b1", "W2", "b2", "Wh", "bh")
    raise ValueError(mode)


def train_qnn_seed(
    U_train: np.ndarray,
    y_train: np.ndarray,
    U_validation: np.ndarray,
    y_validation: np.ndarray,
    U_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train one frozen QNN seed and return validation/full-test score arrays."""

    started = time.time()
    params = _qnn_mode_params(seed, mode)
    trainable = _qnn_trainable_names(mode)
    state = _adam_state(params, trainable)
    learning_rates = {
        name: (QNN_QUANTUM_LR if name == "q" else QNN_DENSE_LR) for name in trainable
    }
    budget = CircuitBudget()
    fraud = np.flatnonzero(y_train == 1)
    legitimate = np.flatnonzero(y_train == 0)
    if len(fraud) != 295:
        raise RuntimeError(f"frozen QNN protocol expects 295 train frauds, found {len(fraud)}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3304]))

    best_params: dict[str, np.ndarray] | None = None
    best_step: int | None = None
    best_validation_auprc = -np.inf
    best_validation_score: np.ndarray | None = None
    checks_without_improvement = 0
    history = []
    status = "COMPLETED"
    failure_reason = None
    final_update = 0

    try:
        for update in range(1, UPDATES + 1):
            chosen_legitimate = rng.choice(legitimate, BATCH_LEGIT, replace=False)
            batch = np.concatenate([chosen_legitimate, fraud])
            batch = batch[rng.permutation(len(batch))]
            mask1, mask2 = _dropout_masks(seed, update, len(batch))
            calls_per_sample = 0
            if mode == "trainable_quantum":
                calls_per_sample = 1 + 2 * QNN_GATE_ANGLES
            elif mode == "frozen_quantum":
                calls_per_sample = 1 + 2 * (QNN_GATE_ANGLES - 1)
            if not budget.reserve("training_parameter_shift", len(batch) * calls_per_sample):
                status = "BUDGET_CENSORED"
                failure_reason = f"circuit cap reached before update {update}"
                break
            loss, gradients, observed_calls = qnn_loss_and_gradients(
                params,
                U_train[batch],
                y_train[batch],
                mode,
                mask1,
                mask2,
            )
            if observed_calls != calls_per_sample:
                raise RuntimeError(
                    f"QNN circuit accounting mismatch: reserved {calls_per_sample}, "
                    f"observed {observed_calls}"
                )
            if not np.isfinite(loss) or not all(np.all(np.isfinite(g)) for g in gradients.values()):
                status = "FAILED"
                failure_reason = f"non-finite loss/gradient at update {update}"
                break
            _adam_update(params, gradients, state, update, learning_rates)
            final_update = update
            if not _all_finite(params):
                status = "FAILED"
                failure_reason = f"non-finite parameter at update {update}"
                break

            if update % CHECK_EVERY == 0:
                validation_calls = len(U_validation) if mode != "dense_head" else 0
                if not budget.reserve("validation", validation_calls):
                    status = "BUDGET_CENSORED"
                    failure_reason = f"circuit cap reached before validation at update {update}"
                    break
                validation_score = qnn_scores(params, U_validation, mode)
                from sklearn.metrics import average_precision_score

                validation_auprc = float(average_precision_score(y_validation, validation_score))
                history.append(
                    {
                        "update": update,
                        "train_objective": float(loss),
                        "validation_auprc": validation_auprc,
                        "circuit_forwards_after_check": budget.total,
                    }
                )
                if validation_auprc > best_validation_auprc:
                    best_validation_auprc = validation_auprc
                    best_step = update
                    best_params = _copy_params(params)
                    best_validation_score = np.asarray(validation_score, float).copy()
                    checks_without_improvement = 0
                else:
                    checks_without_improvement += 1
                    if checks_without_improvement >= PATIENCE_CHECKS:
                        status = "EARLY_STOPPED"
                        break
    except Exception as exc:  # retain frozen failed seeds instead of replacing them
        status = "FAILED"
        failure_reason = f"{type(exc).__name__}: {exc}"

    if best_params is None:
        return np.full(len(U_test), np.nan), np.full(len(U_validation), np.nan), {
            "seed": seed,
            "status": status if status != "COMPLETED" else "FAILED",
            "failure_reason": failure_reason or "no finite validation checkpoint",
            "final_update": final_update,
            "best_checkpoint_update": None,
            "validation_history": history,
            "parameter_counts": _parameter_counts_qnn(mode),
            "circuit_budget": budget.metadata(),
            "wall_seconds": time.time() - started,
        }

    params = best_params
    if status == "FAILED":
        return np.full(len(U_test), np.nan), np.full(len(U_validation), np.nan), {
            "seed": seed,
            "status": "FAILED",
            "failure_reason": failure_reason,
            "final_update": final_update,
            "best_checkpoint_update": best_step,
            "best_validation_auprc": best_validation_auprc,
            "validation_history": history,
            "parameter_counts": _parameter_counts_qnn(mode),
            "checkpoint_parameters_before_failure": _json_params(params),
            "circuit_budget": budget.metadata(),
            "wall_seconds": time.time() - started,
        }
    test_calls = len(U_test) if mode != "dense_head" else 0
    if not budget.reserve("test_inference", test_calls):
        return np.full(len(U_test), np.nan), np.full(len(U_validation), np.nan), {
            "seed": seed,
            "status": "BUDGET_CENSORED",
            "failure_reason": "circuit cap reached before full-test inference",
            "final_update": final_update,
            "best_checkpoint_update": best_step,
            "best_validation_auprc": best_validation_auprc,
            "validation_history": history,
            "parameter_counts": _parameter_counts_qnn(mode),
            "checkpoint_parameters": _json_params(params),
            "circuit_budget": budget.metadata(),
            "wall_seconds": time.time() - started,
        }
    if best_validation_score is None:
        raise RuntimeError("best QNN checkpoint is missing its validation score array")
    test_score = qnn_scores(params, U_test, mode)
    return test_score, best_validation_score, {
        "seed": seed,
        "status": status,
        "failure_reason": failure_reason,
        "final_update": final_update,
        "best_checkpoint_update": best_step,
        "best_validation_auprc": best_validation_auprc,
        "validation_history": history,
        "parameter_counts": _parameter_counts_qnn(mode),
        "checkpoint_parameters": _json_params(params),
        "test_metrics": _metrics(
            y_test, test_score, y_validation, best_validation_score
        ),
        "circuit_budget": budget.metadata(),
        "wall_seconds": time.time() - started,
    }


def _hae_mode_params(seed: int, mode: str) -> dict[str, np.ndarray]:
    return initialize_hae(seed)[mode]


def _hae_trainable_names(mode: str) -> tuple[str, ...]:
    if mode == "hybrid_quantum":
        return ("We", "be", "Wd", "bd", "q")
    return ("We", "be", "Wd", "bd")


def _count_tree_nodes(model) -> int:
    return int(sum(tree.tree_.node_count for tree in np.ravel(model.estimators_)))


def train_hae_seed(
    U_train: np.ndarray,
    y_train: np.ndarray,
    U_validation: np.ndarray,
    y_validation: np.ndarray,
    U_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train one legitimate-only HAE/twin seed and fit its latent IF scorer."""

    started = time.time()
    params = _hae_mode_params(seed, mode)
    trainable = _hae_trainable_names(mode)
    state = _adam_state(params, trainable)
    learning_rates = {
        name: (HAE_QUANTUM_LR if name == "q" else HAE_CLASSICAL_LR)
        for name in trainable
    }
    budget = CircuitBudget()
    legitimate_train = np.flatnonzero(y_train == 0)
    legitimate_validation = np.flatnonzero(y_validation == 0)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3404]))

    best_params: dict[str, np.ndarray] | None = None
    best_step: int | None = None
    best_validation_loss = np.inf
    checks_without_improvement = 0
    history = []
    status = "COMPLETED"
    failure_reason = None
    final_update = 0

    try:
        for update in range(1, UPDATES + 1):
            batch = rng.choice(legitimate_train, BATCH_LEGIT, replace=False)
            calls_per_sample = 1 + 2 * HAE_GATE_ANGLES if mode == "hybrid_quantum" else 0
            if not budget.reserve("training_parameter_shift", len(batch) * calls_per_sample):
                status = "BUDGET_CENSORED"
                failure_reason = f"circuit cap reached before update {update}"
                break
            loss, gradients, observed_calls = hae_loss_and_gradients(
                params, U_train[batch], mode
            )
            if observed_calls != calls_per_sample:
                raise RuntimeError(
                    f"HAE circuit accounting mismatch: reserved {calls_per_sample}, "
                    f"observed {observed_calls}"
                )
            if not np.isfinite(loss) or not all(np.all(np.isfinite(g)) for g in gradients.values()):
                status = "FAILED"
                failure_reason = f"non-finite loss/gradient at update {update}"
                break
            _adam_update(params, gradients, state, update, learning_rates)
            final_update = update
            if not _all_finite(params):
                status = "FAILED"
                failure_reason = f"non-finite parameter at update {update}"
                break

            if update % CHECK_EVERY == 0:
                validation_calls = (
                    len(legitimate_validation) if mode == "hybrid_quantum" else 0
                )
                if not budget.reserve("validation", validation_calls):
                    status = "BUDGET_CENSORED"
                    failure_reason = f"circuit cap reached before validation at update {update}"
                    break
                validation_loss = hae_reconstruction_loss(
                    params, U_validation[legitimate_validation], mode
                )
                if not np.isfinite(validation_loss):
                    status = "FAILED"
                    failure_reason = (
                        f"non-finite legitimate-validation reconstruction loss "
                        f"at update {update}"
                    )
                    break
                history.append(
                    {
                        "update": update,
                        "train_reconstruction_loss": float(loss),
                        "legitimate_validation_reconstruction_loss": float(validation_loss),
                        "validation_rows": int(len(legitimate_validation)),
                        "validation_fraud_labels_used": 0,
                        "circuit_forwards_after_check": budget.total,
                    }
                )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_step = update
                    best_params = _copy_params(params)
                    checks_without_improvement = 0
                else:
                    checks_without_improvement += 1
                    if checks_without_improvement >= PATIENCE_CHECKS:
                        status = "EARLY_STOPPED"
                        break
    except Exception as exc:  # retain frozen failed seeds instead of replacing them
        status = "FAILED"
        failure_reason = f"{type(exc).__name__}: {exc}"

    if best_params is None:
        return np.full(len(U_test), np.nan), np.full(len(U_validation), np.nan), {
            "seed": seed,
            "status": status if status != "COMPLETED" else "FAILED",
            "failure_reason": failure_reason or "no finite validation checkpoint",
            "final_update": final_update,
            "best_checkpoint_update": None,
            "validation_history": history,
            "parameter_counts": _parameter_counts_hae(mode),
            "circuit_budget": budget.metadata(),
            "wall_seconds": time.time() - started,
        }

    params = best_params
    if status == "FAILED":
        return np.full(len(U_test), np.nan), np.full(len(U_validation), np.nan), {
            "seed": seed,
            "status": "FAILED",
            "failure_reason": failure_reason,
            "final_update": final_update,
            "best_checkpoint_update": best_step,
            "best_legitimate_validation_reconstruction_loss": best_validation_loss,
            "validation_history": history,
            "parameter_counts": _parameter_counts_hae(mode),
            "checkpoint_parameters_before_failure": _json_params(params),
            "circuit_budget": budget.metadata(),
            "wall_seconds": time.time() - started,
        }
    train_latent_calls = len(legitimate_train) if mode == "hybrid_quantum" else 0
    validation_latent_calls = len(U_validation) if mode == "hybrid_quantum" else 0
    test_latent_calls = len(U_test) if mode == "hybrid_quantum" else 0
    score = np.full(len(U_test), np.nan)
    validation_score = np.full(len(U_validation), np.nan)
    latent_meta = None
    if not budget.reserve("isolation_forest_train_latents", train_latent_calls):
        status = "BUDGET_CENSORED"
        failure_reason = "circuit cap reached before legitimate-train latent extraction"
    else:
        try:
            from sklearn.ensemble import IsolationForest

            latent_train = hae_latents(params, U_train[legitimate_train], mode)
        except Exception as exc:
            status = "FAILED"
            failure_reason = f"train-latent {type(exc).__name__}: {exc}"
    if status not in {"FAILED", "BUDGET_CENSORED"}:
        if not budget.reserve("threshold_validation_latents", validation_latent_calls):
            status = "BUDGET_CENSORED"
            failure_reason = "circuit cap reached before full-validation latent extraction"
        else:
            try:
                latent_validation = hae_latents(params, U_validation, mode)
            except Exception as exc:
                status = "FAILED"
                failure_reason = f"validation-latent {type(exc).__name__}: {exc}"
    if status not in {"FAILED", "BUDGET_CENSORED"}:
        if not budget.reserve("test_latents", test_latent_calls):
            status = "BUDGET_CENSORED"
            failure_reason = "circuit cap reached before full-test latent extraction"
        else:
            try:
                latent_test = hae_latents(params, U_test, mode)
            except Exception as exc:
                status = "FAILED"
                failure_reason = f"test-latent {type(exc).__name__}: {exc}"
    if status not in {"FAILED", "BUDGET_CENSORED"}:
        try:
            forest = IsolationForest(
                n_estimators=300,
                max_samples=min(8192, len(latent_train)),
                contamination="auto",
                random_state=seed,
                n_jobs=4,
            ).fit(latent_train)
            validation_score = -forest.score_samples(latent_validation)
            score = -forest.score_samples(latent_test)
            latent_meta = {
                "fit_rows": int(len(latent_train)),
                "fit_fraud_labels_used": 0,
                "latent_width": int(latent_train.shape[1]),
                "n_estimators": 300,
                "max_samples": int(forest.max_samples_),
                "contamination": "auto",
                "random_state": seed,
                "tree_nodes": _count_tree_nodes(forest),
            }
        except Exception as exc:
            status = "FAILED"
            failure_reason = f"IsolationForest {type(exc).__name__}: {exc}"

    return score, validation_score, {
        "seed": seed,
        "status": status,
        "failure_reason": failure_reason,
        "final_update": final_update,
        "best_checkpoint_update": best_step,
        "best_legitimate_validation_reconstruction_loss": best_validation_loss,
        "validation_history": history,
        "parameter_counts": _parameter_counts_hae(mode),
        "checkpoint_parameters": _json_params(params),
        "downstream_isolation_forest": latent_meta,
        "test_metrics": _metrics(
            y_test, score, y_validation, validation_score
        ),
        "circuit_budget": budget.metadata(),
        "wall_seconds": time.time() - started,
    }


def _ensemble_or_nan(
    seed_scores: list[np.ndarray],
    seed_statuses: list[str],
) -> tuple[np.ndarray, str]:
    eligible = {"COMPLETED", "EARLY_STOPPED"}
    if (
        len(seed_scores) != 5
        or len(seed_statuses) != 5
        or not all(status in eligible for status in seed_statuses)
        or not all(np.all(np.isfinite(score)) for score in seed_scores)
    ):
        length = len(seed_scores[0]) if seed_scores else 0
        return np.full(length, np.nan), "INCOMPLETE_FROZEN_SEED_SET"
    return np.mean(np.vstack(seed_scores), axis=0), "COMPLETE_FIVE_SEED_MEAN"


def run_qnn_family(pipe: dict, score_bundle: dict[str, np.ndarray]) -> dict:
    U_train, U_validation, U_test = (pipe["U"][name] for name in ("tr", "va", "te"))
    y_train, y_validation, y_test = (
        pipe["y_split"][name] for name in ("tr", "va", "te")
    )
    result = {}
    for mode in ("trainable_quantum", "frozen_quantum", "dense_head"):
        seed_scores = []
        seed_validation_scores = []
        rows = []
        for seed in QNN_SEEDS:
            score, validation_score, row = train_qnn_seed(
                U_train,
                y_train,
                U_validation,
                y_validation,
                U_test,
                y_test,
                seed,
                mode,
            )
            seed_scores.append(score)
            seed_validation_scores.append(validation_score)
            rows.append(row)
            score_bundle[f"qnn_{mode}_seed{seed}"] = score
            score_bundle[f"qnn_{mode}_validation_seed{seed}"] = validation_score
            log(f"QNN {mode} seed {seed}: {row['status']}")
        ensemble, ensemble_status = _ensemble_or_nan(
            seed_scores, [row["status"] for row in rows]
        )
        validation_ensemble, validation_ensemble_status = _ensemble_or_nan(
            seed_validation_scores, [row["status"] for row in rows]
        )
        if validation_ensemble_status != ensemble_status:
            raise RuntimeError("QNN validation/test ensemble status mismatch")
        score_bundle[f"qnn_{mode}_ensemble"] = ensemble
        score_bundle[f"qnn_{mode}_validation_ensemble"] = validation_ensemble
        result[mode] = {
            "seeds": list(QNN_SEEDS),
            "per_seed": rows,
            "ensemble_rule": "arithmetic mean of all five frozen seed score arrays",
            "ensemble_status": ensemble_status,
            "ensemble_test_metrics": _metrics(
                y_test, ensemble, y_validation, validation_ensemble
            ),
            "parameter_counts": _parameter_counts_qnn(mode),
            "implied_inference_measurements": _qnn_measurement_ledger(
                mode, len(y_validation), len(y_test)
            ),
        }
    return result


def run_hae_family(pipe: dict, score_bundle: dict[str, np.ndarray]) -> dict:
    U_train, U_validation, U_test = (pipe["U"][name] for name in ("tr", "va", "te"))
    y_train, y_validation, y_test = (
        pipe["y_split"][name] for name in ("tr", "va", "te")
    )
    result = {}
    for mode in ("hybrid_quantum", "direct4_classical", "width8_classical"):
        seed_scores = []
        seed_validation_scores = []
        rows = []
        for seed in HAE_SEEDS:
            score, validation_score, row = train_hae_seed(
                U_train,
                y_train,
                U_validation,
                y_validation,
                U_test,
                y_test,
                seed,
                mode,
            )
            seed_scores.append(score)
            seed_validation_scores.append(validation_score)
            rows.append(row)
            score_bundle[f"hae_{mode}_seed{seed}"] = score
            score_bundle[f"hae_{mode}_validation_seed{seed}"] = validation_score
            log(f"HAE {mode} seed {seed}: {row['status']}")
        ensemble, ensemble_status = _ensemble_or_nan(
            seed_scores, [row["status"] for row in rows]
        )
        validation_ensemble, validation_ensemble_status = _ensemble_or_nan(
            seed_validation_scores, [row["status"] for row in rows]
        )
        if validation_ensemble_status != ensemble_status:
            raise RuntimeError("HAE validation/test ensemble status mismatch")
        score_bundle[f"hae_{mode}_ensemble"] = ensemble
        score_bundle[f"hae_{mode}_validation_ensemble"] = validation_ensemble
        result[mode] = {
            "seeds": list(HAE_SEEDS),
            "per_seed": rows,
            "ensemble_rule": "arithmetic mean of all five frozen seed score arrays",
            "ensemble_status": ensemble_status,
            "ensemble_test_metrics": _metrics(
                y_test, ensemble, y_validation, validation_ensemble
            ),
            "parameter_counts": _parameter_counts_hae(mode),
            "implied_inference_measurements": _hae_measurement_ledger(
                mode, int(np.sum(y_train == 0)), len(y_validation), len(y_test)
            ),
        }
    return result


# ---------------------------------------------------------------------------
# Shared paired Poisson bootstrap and frozen decision summaries
# ---------------------------------------------------------------------------


def _weighted_ap_sorted(
    y_sorted: np.ndarray,
    weights_sorted: np.ndarray,
    group_ends: np.ndarray,
) -> np.ndarray:
    positive_weight = weights_sorted * y_sorted[None, :]
    cumulative_positive = np.cumsum(positive_weight, axis=1, dtype=float)
    cumulative_all = np.cumsum(weights_sorted, axis=1, dtype=float)
    total_positive = cumulative_positive[:, -1]
    positive_at_group = cumulative_positive[:, group_ends]
    all_at_group = cumulative_all[:, group_ends]
    precision = np.divide(
        positive_at_group,
        all_at_group,
        out=np.zeros_like(positive_at_group),
        where=all_at_group > 0,
    )
    group_positive = np.diff(
        np.column_stack([np.zeros(len(weights_sorted)), positive_at_group]), axis=1
    )
    return np.divide(
        np.sum(precision * group_positive, axis=1),
        total_positive,
        out=np.full(len(weights_sorted), np.nan),
        where=total_positive > 0,
    )


def paired_simultaneous_bootstrap(
    y: np.ndarray,
    score_map: dict[str, np.ndarray],
    comparisons: list[tuple[str, str]],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """Identical Poisson(1) row weights and conservative 98.75% intervals."""

    from sklearn.metrics import average_precision_score

    finite_scores = {
        name: np.asarray(score, float)
        for name, score in score_map.items()
        if np.all(np.isfinite(score))
    }
    orders = {
        name: np.argsort(-score, kind="mergesort") for name, score in finite_scores.items()
    }
    group_ends = {}
    for name, order in orders.items():
        sorted_score = finite_scores[name][order]
        group_ends[name] = np.flatnonzero(
            np.r_[sorted_score[1:] != sorted_score[:-1], True]
        )
    points = {
        name: float(average_precision_score(y, score)) for name, score in finite_scores.items()
    }
    draws: dict[str, list[float]] = {name: [] for name in finite_scores}
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
            values = _weighted_ap_sorted(y[order], weights[:, order], group_ends[name])
            draws[name].extend(values[np.isfinite(values)].tolist())
        made = min(len(values) for values in draws.values())
    arrays = {name: np.asarray(values[:n_boot]) for name, values in draws.items()}
    alpha = 1.0 - SIMULTANEOUS_LEVEL
    quantiles = (alpha / 2.0, 1.0 - alpha / 2.0)
    output = {
        "bootstrap": "paired Poisson(1) test-row weights",
        "seed": seed,
        "n_boot": n_boot,
        "interval": "98.75% two-sided simultaneous interval (Bonferroni-equivalent for four co-primary families)",
        "point_auprc": points,
        "auprc_intervals": {
            name: [float(value) for value in np.quantile(samples, quantiles)]
            for name, samples in arrays.items()
        },
        "comparisons": {},
        "omitted_nonfinite_score_arrays": sorted(set(score_map) - set(finite_scores)),
    }
    for left, right in comparisons:
        key = f"{left}_minus_{right}"
        if left not in arrays or right not in arrays:
            output["comparisons"][key] = {
                "status": "NOT_AVAILABLE",
                "reason": "one or both five-seed score arrays are incomplete",
            }
            continue
        delta = arrays[left] - arrays[right]
        point = points[left] - points[right]
        interval = [float(value) for value in np.quantile(delta, quantiles)]
        if abs(point) <= 0.01 or interval[0] <= 0.0 <= interval[1]:
            decision = "PRACTICAL_TIE"
        elif interval[0] > 0.0:
            decision = "LEFT_WINS"
        else:
            decision = "RIGHT_WINS"
        output["comparisons"][key] = {
            "status": "AVAILABLE",
            "left": left,
            "right": right,
            "point": float(point),
            "interval": interval,
            "decision": decision,
            "practical_tie_margin": 0.01,
        }
    return output


# ---------------------------------------------------------------------------
# Data-free synthetic referees (default execution path)
# ---------------------------------------------------------------------------


def _finite_difference(
    objective: Callable[[], float],
    target: np.ndarray,
    index,
    epsilon: float = 1e-6,
) -> float:
    original = float(target[index])
    target[index] = original + epsilon
    plus = objective()
    target[index] = original - epsilon
    minus = objective()
    target[index] = original
    return float((plus - minus) / (2.0 * epsilon))


def _scaled_error(analytic: float, numeric: float) -> float:
    return float(abs(analytic - numeric) / max(1.0, abs(analytic), abs(numeric)))


def run_self_checks() -> dict:
    rng = np.random.default_rng(20260831)

    # QNN norm, elementary-angle shifts, and all 16 variable chain rules.
    q_inputs = rng.normal(scale=0.4, size=(3, 9))
    q_weights = rng.normal(scale=0.3, size=7)
    q_angles = qnn_gate_angles(q_inputs, q_weights)
    q_state = qnn_state_from_angles(q_angles)
    q_logits, q_input_jac, q_weight_jac, q_calls = qnn_head_jacobians(
        q_inputs, q_weights, train_quantum=True
    )
    assert q_weight_jac is not None
    epsilon = 1e-6
    q_gate_error = 0.0
    for gate in range(QNN_GATE_ANGLES):
        plus = q_angles.copy()
        minus = q_angles.copy()
        plus[:, gate] += epsilon
        minus[:, gate] -= epsilon
        numeric = (qnn_head_from_angles(plus) - qnn_head_from_angles(minus)) / (2 * epsilon)
        shifted_plus = q_angles.copy()
        shifted_minus = q_angles.copy()
        shifted_plus[:, gate] += np.pi / 2
        shifted_minus[:, gate] -= np.pi / 2
        analytic = 0.5 * (
            qnn_head_from_angles(shifted_plus) - qnn_head_from_angles(shifted_minus)
        )
        q_gate_error = max(q_gate_error, float(np.max(np.abs(analytic - numeric))))
    q_chain_error = 0.0
    for variable in range(9):
        plus = q_inputs.copy()
        minus = q_inputs.copy()
        plus[:, variable] += epsilon
        minus[:, variable] -= epsilon
        numeric = (qnn_head(plus, q_weights) - qnn_head(minus, q_weights)) / (2 * epsilon)
        q_chain_error = max(
            q_chain_error, float(np.max(np.abs(q_input_jac[:, :, variable] - numeric)))
        )
    for variable in range(7):
        plus = q_weights.copy()
        minus = q_weights.copy()
        plus[variable] += epsilon
        minus[variable] -= epsilon
        numeric = (qnn_head(q_inputs, plus) - qnn_head(q_inputs, minus)) / (2 * epsilon)
        q_chain_error = max(
            q_chain_error, float(np.max(np.abs(q_weight_jac[:, :, variable] - numeric)))
        )

    # HAE norm and eight elementary-angle/chain-rule derivatives.
    h_encoded = rng.uniform(-0.8, 0.8, size=(3, 4))
    h_theta = rng.normal(scale=0.3, size=4)
    h_angles = hae_gate_angles(h_encoded, h_theta)
    h_state = hae_state_from_angles(h_angles)
    h_latent, h_encoded_jac, h_theta_jac, h_calls = hae_bottleneck_jacobians(
        h_encoded, h_theta
    )
    h_gate_error = 0.0
    for gate in range(HAE_GATE_ANGLES):
        plus = h_angles.copy()
        minus = h_angles.copy()
        plus[:, gate] += epsilon
        minus[:, gate] -= epsilon
        numeric = (hae_bottleneck_from_angles(plus) - hae_bottleneck_from_angles(minus)) / (
            2 * epsilon
        )
        shifted_plus = h_angles.copy()
        shifted_minus = h_angles.copy()
        shifted_plus[:, gate] += np.pi / 2
        shifted_minus[:, gate] -= np.pi / 2
        analytic = 0.5 * (
            hae_bottleneck_from_angles(shifted_plus)
            - hae_bottleneck_from_angles(shifted_minus)
        )
        h_gate_error = max(h_gate_error, float(np.max(np.abs(analytic - numeric))))
    h_chain_error = 0.0
    for variable in range(4):
        plus = h_encoded.copy()
        minus = h_encoded.copy()
        plus[:, variable] += epsilon
        minus[:, variable] -= epsilon
        numeric = (hae_bottleneck(plus, h_theta) - hae_bottleneck(minus, h_theta)) / (
            2 * epsilon
        )
        h_chain_error = max(
            h_chain_error, float(np.max(np.abs(h_encoded_jac[:, :, variable] - numeric)))
        )
    for variable in range(4):
        plus = h_theta.copy()
        minus = h_theta.copy()
        plus[variable] += epsilon
        minus[variable] -= epsilon
        numeric = (hae_bottleneck(h_encoded, plus) - hae_bottleneck(h_encoded, minus)) / (
            2 * epsilon
        )
        h_chain_error = max(
            h_chain_error, float(np.max(np.abs(h_theta_jac[:, :, variable] - numeric)))
        )

    # End-to-end QNN gradients with fixed synthetic dropout masks.
    Xq = rng.uniform(0.05, 0.95, size=(6, 8))
    yq = np.array([0, 0, 0, 1, 1, 1])
    qparams = _qnn_mode_params(830, "trainable_quantum")
    mask1, mask2 = _dropout_masks(830, 1, len(Xq))
    qloss, qgrads, _ = qnn_loss_and_gradients(
        qparams, Xq, yq, "trainable_quantum", mask1, mask2
    )
    q_gradient_errors = {}
    for name, index in (("W1", (0, 0)), ("W2", (0, 0)), ("q", (6,))):
        numeric = _finite_difference(
            lambda: qnn_loss_and_gradients(
                qparams, Xq, yq, "trainable_quantum", mask1, mask2
            )[0],
            qparams[name],
            index,
        )
        analytic = float(qgrads[name][index])
        q_gradient_errors[f"{name}{index}"] = {
            "analytic": analytic,
            "finite_difference": numeric,
            "scaled_error": _scaled_error(analytic, numeric),
        }
    dense_params = _qnn_mode_params(830, "dense_head")
    dense_loss, dense_grads, _ = qnn_loss_and_gradients(
        dense_params, Xq, yq, "dense_head", mask1, mask2
    )
    dense_numeric = _finite_difference(
        lambda: qnn_loss_and_gradients(
            dense_params, Xq, yq, "dense_head", mask1, mask2
        )[0],
        dense_params["Wh"],
        (0, 0),
    )
    dense_gradient_error = _scaled_error(float(dense_grads["Wh"][0, 0]), dense_numeric)

    # End-to-end HAE hybrid and direct classical gradients.
    Xh = rng.uniform(0.05, 0.95, size=(5, 8))
    hparams = _hae_mode_params(850, "hybrid_quantum")
    hloss, hgrads, _ = hae_loss_and_gradients(hparams, Xh, "hybrid_quantum")
    h_gradient_errors = {}
    for name, index in (("We", (0, 0)), ("Wd", (0, 0)), ("q", (0,))):
        numeric = _finite_difference(
            lambda: hae_loss_and_gradients(hparams, Xh, "hybrid_quantum")[0],
            hparams[name],
            index,
        )
        analytic = float(hgrads[name][index])
        h_gradient_errors[f"{name}{index}"] = {
            "analytic": analytic,
            "finite_difference": numeric,
            "scaled_error": _scaled_error(analytic, numeric),
        }
    direct_params = _hae_mode_params(850, "direct4_classical")
    direct_loss, direct_grads, _ = hae_loss_and_gradients(
        direct_params, Xh, "direct4_classical"
    )
    direct_numeric = _finite_difference(
        lambda: hae_loss_and_gradients(direct_params, Xh, "direct4_classical")[0],
        direct_params["We"],
        (0, 0),
    )
    direct_gradient_error = _scaled_error(float(direct_grads["We"][0, 0]), direct_numeric)

    tolerances = {
        "norm": 5e-13,
        "gate_parameter_shift": 2e-8,
        "variable_chain_rule": 2e-8,
        "end_to_end_gradient": 2e-6,
    }
    maximum_q_gradient_error = max(
        value["scaled_error"] for value in q_gradient_errors.values()
    )
    maximum_h_gradient_error = max(
        value["scaled_error"] for value in h_gradient_errors.values()
    )
    passed = bool(
        _max_norm_residual(q_state) <= tolerances["norm"]
        and _max_norm_residual(h_state) <= tolerances["norm"]
        and q_gate_error <= tolerances["gate_parameter_shift"]
        and h_gate_error <= tolerances["gate_parameter_shift"]
        and q_chain_error <= tolerances["variable_chain_rule"]
        and h_chain_error <= tolerances["variable_chain_rule"]
        and maximum_q_gradient_error <= tolerances["end_to_end_gradient"]
        and dense_gradient_error <= tolerances["end_to_end_gradient"]
        and maximum_h_gradient_error <= tolerances["end_to_end_gradient"]
        and direct_gradient_error <= tolerances["end_to_end_gradient"]
        and np.all(np.isfinite(q_logits))
        and np.all(np.isfinite(h_latent))
        and np.isfinite(qloss)
        and np.isfinite(dense_loss)
        and np.isfinite(hloss)
        and np.isfinite(direct_loss)
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "data_loaded": False,
        "tolerances": tolerances,
        "qnn": {
            "max_norm_residual": _max_norm_residual(q_state),
            "max_gate_parameter_shift_error": q_gate_error,
            "max_16_variable_chain_rule_error": q_chain_error,
            "calls_per_train_sample": q_calls,
            "end_to_end_gradient_checks": q_gradient_errors,
            "dense_twin_head_gradient_scaled_error": dense_gradient_error,
        },
        "hae": {
            "max_norm_residual": _max_norm_residual(h_state),
            "max_gate_parameter_shift_error": h_gate_error,
            "max_8_variable_chain_rule_error": h_chain_error,
            "calls_per_train_sample": h_calls,
            "end_to_end_gradient_checks": h_gradient_errors,
            "direct4_encoder_gradient_scaled_error": direct_gradient_error,
        },
    }
    if not passed:
        raise AssertionError(json.dumps(result, indent=2, sort_keys=True))
    return result


def implementation_freeze_metadata() -> dict:
    return {
        "scope": "only protocol sections 3.3 and 3.4 items 2-3",
        "execution_guard": "ULB data and output paths are touched only with explicit --run",
        "qnn": {
            "front_end": [
                "Dense(8,32), ReLU, He-normal, kernel L2=0.01",
                "inverted Dropout(0.3)",
                "Dense(32,9), ReLU, He-normal, kernel L1=L2=0.001",
                "inverted Dropout(0.3)",
            ],
            "head": "verbatim public Deloitte/AWS three-qubit seven-weight circuit",
            "gate_sequence": [
                "RY(x0) q0; RY(x1) q1; RY(x2) q2",
                "Rot(w0*x3,w1*x4,w2*x5) q1 as RZ,RY,RZ",
                "Rot(w3*x6,w4*x7,w5*x8) q2 as RZ,RY,RZ",
                "CNOT 1->2; RY(w6) q2; CNOT 0->2; CNOT 1->2",
                "measure Z(q0), Z(q2) as two logits",
            ],
            "differentiation": {
                "dense_outputs": 9,
                "trainable_quantum_weights": 7,
                "differentiable_variables": QNN_DIFFERENTIABLE_VARIABLES,
                "elementary_rotation_angles": QNN_GATE_ANGLES,
                "rule": "exact +/-pi/2 shift per elementary angle plus product-angle chain rule",
                "calls_per_train_sample_trainable_head": 1 + 2 * QNN_GATE_ANGLES,
                "calls_per_train_sample_frozen_head": 1 + 2 * (QNN_GATE_ANGLES - 1),
            },
            "loss": "two-logit cross-entropy; total batch weight 0.5 per class",
            "regularization": {
                "first_dense_kernel_l2": QNN_L2_FIRST,
                "second_dense_kernel_l1": QNN_L1_SECOND,
                "second_dense_kernel_l2": QNN_L2_SECOND,
                "biases": "unregularized",
            },
            "initialization": {
                "dense_kernels": "seeded He normal",
                "biases": "zero",
                "quantum_weights": "seeded normal standard deviation 0.1",
            },
            "twins": ["frozen_quantum", "dense_head"],
        },
        "hae": {
            "hybrid_architecture": "tanh 8->4; RY(pi*h); CNOT ring; four trainable RY; local Z; linear 4->8",
            "ring_order": ["0->1", "1->2", "2->3", "3->0"],
            "differentiation": {
                "elementary_rotation_angles": HAE_GATE_ANGLES,
                "rule": "exact +/-pi/2 shift per elementary angle with pi chain factor on encoded tanh outputs",
                "calls_per_train_sample": 1 + 2 * HAE_GATE_ANGLES,
            },
            "loss": "mean squared reconstruction error over rows and eight features",
            "regularization": "none",
            "initialization": "seeded Xavier normal classical kernels; zero biases; quantum normal sd=0.1",
            "twins": ["direct4_classical", "width8_classical"],
            "isolation_forest": {
                "n_estimators": 300,
                "max_samples": "min(8192,n_train_legitimate)",
                "contamination": "auto",
                "random_state": "model seed",
                "fit_rows": "all legitimate training rows only",
            },
        },
        "training": {
            "qnn_seeds": list(QNN_SEEDS),
            "hae_seeds": list(HAE_SEEDS),
            "updates": UPDATES,
            "legitimate_rows_per_update": BATCH_LEGIT,
            "qnn_fraud_rows_per_update": "all 295",
            "checkpoint_every_updates": CHECK_EVERY,
            "patience_checks": PATIENCE_CHECKS,
            "tie_rule": "strict improvement only; earliest exact tie retained",
            "adam": {
                "beta1": ADAM_BETA1,
                "beta2": ADAM_BETA2,
                "epsilon": ADAM_EPS,
                "qnn_dense_learning_rate": QNN_DENSE_LR,
                "qnn_quantum_learning_rate": QNN_QUANTUM_LR,
                "hae_classical_learning_rate": HAE_CLASSICAL_LR,
                "hae_quantum_learning_rate": HAE_QUANTUM_LR,
            },
        },
        "compute_accounting": {
            "unit": "one exact statevector forward for one sample",
            "per_seed_cap": MAX_CIRCUIT_FORWARDS,
            "full_300_update_upper_bounds_at_frozen_split": {
                "qnn_trainable_quantum": 5_189_077,
                "qnn_frozen_quantum": 4_781_677,
                "hae_hybrid_quantum": 3_095_857,
                "classical_twins": 0,
            },
            "included": [
                "unshifted train forwards",
                "all plus/minus parameter shifts",
                "validation scoring",
                "HAE legitimate-train latent extraction",
                "HAE full-validation threshold latent extraction",
                "test scoring/latent extraction",
            ],
            "exact_statevector_is": "simulation ceiling, not shot-matched or hardware evidence",
            "implied_measurements": {
                "qnn": "two commuting local-Z observables in one setting per transaction",
                "hae": "four commuting local-Z observables in one setting per transaction",
                "shots": "NOT_RUN; no finite-shot equivalent is inferred",
            },
        },
    }


# ---------------------------------------------------------------------------
# Explicit data-bearing execution
# ---------------------------------------------------------------------------


def _load_reference_scores(path: Path, y_test: np.ndarray) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"frozen v1 score artifact is required: {path}")
    data = np.load(path)
    required = {
        "S1_analytic": "n8_S1_analytic",
        "S2_L1_seed500": "n8_S2_L1_seed500",
        "XGB_subset": "n8_XGB_subset",
    }
    if "n8_y" not in data.files or not np.array_equal(np.asarray(data["n8_y"], int), y_test):
        raise RuntimeError("v1 reference test labels do not exactly match the frozen v2 split")
    missing = [key for key in required.values() if key not in data.files]
    if missing:
        raise KeyError(f"missing frozen reference arrays: {missing}")
    return {name: np.asarray(data[key], float) for name, key in required.items()}


def run_audit(output: Path, scores_output: Path, references_path: Path) -> None:
    import sklearn

    output, scores_output, references_path, data_path = _resolve_locked_paths(
        output, scores_output, references_path
    )
    source_hashes_at_start = {
        "script": sha256(SCRIPT_FILE),
        "protocol": sha256(PROTOCOL_FILE),
        "references": sha256(references_path),
        "ulb_data": sha256(data_path),
    }
    self_checks = run_self_checks()
    pipe = prepare_hsbc(8, seed=0)
    y_train = pipe["y_split"]["tr"]
    y_validation = pipe["y_split"]["va"]
    y_test = pipe["y_split"]["te"]
    observed = {
        "train_rows": len(y_train),
        "validation_rows": len(y_validation),
        "test_rows": len(y_test),
        "train_frauds": int(np.sum(y_train)),
        "validation_frauds": int(np.sum(y_validation)),
        "test_frauds": int(np.sum(y_test)),
        "features": int(pipe["U"]["tr"].shape[1]),
    }
    expected = {
        "train_rows": 170884,
        "validation_rows": 56961,
        "test_rows": 56962,
        "train_frauds": 295,
        "validation_frauds": 98,
        "test_frauds": 99,
        "features": 8,
    }
    if observed != expected:
        raise RuntimeError(f"frozen split mismatch: observed={observed}, expected={expected}")
    if tuple(pipe["feature_names"]) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"frozen feature mismatch: observed={pipe['feature_names']}, "
            f"expected={list(EXPECTED_FEATURES)}"
        )

    references = _load_reference_scores(references_path, y_test)
    score_bundle: dict[str, np.ndarray] = {"y_test": y_test}
    for name, score in references.items():
        score_bundle[f"reference_{name}"] = score

    qnn = run_qnn_family(pipe, score_bundle)
    hae = run_hae_family(pipe, score_bundle)

    comparison_scores = {
        "S1_analytic": references["S1_analytic"],
        "S2_L1_seed500": references["S2_L1_seed500"],
        "XGB_subset": references["XGB_subset"],
        "QNN_trainable_quantum": score_bundle["qnn_trainable_quantum_ensemble"],
        "QNN_frozen_quantum": score_bundle["qnn_frozen_quantum_ensemble"],
        "QNN_dense_head": score_bundle["qnn_dense_head_ensemble"],
        "HAE_hybrid_quantum": score_bundle["hae_hybrid_quantum_ensemble"],
        "HAE_direct4_classical": score_bundle["hae_direct4_classical_ensemble"],
        "HAE_width8_classical": score_bundle["hae_width8_classical_ensemble"],
    }
    comparisons = [
        ("S2_L1_seed500", "QNN_trainable_quantum"),
        ("QNN_trainable_quantum", "QNN_dense_head"),
        ("QNN_trainable_quantum", "QNN_frozen_quantum"),
        ("S1_analytic", "HAE_hybrid_quantum"),
        ("HAE_hybrid_quantum", "HAE_direct4_classical"),
        ("HAE_hybrid_quantum", "HAE_width8_classical"),
    ]
    bootstrap = paired_simultaneous_bootstrap(y_test, comparison_scores, comparisons)

    source_hashes_at_end = {
        "script": sha256(SCRIPT_FILE),
        "protocol": sha256(PROTOCOL_FILE),
        "references": sha256(references_path),
        "ulb_data": sha256(data_path),
    }
    if source_hashes_at_end != source_hashes_at_start:
        raise RuntimeError(
            "source/protocol/reference artifact changed during execution; refusing to write outcomes: "
            f"start={source_hashes_at_start}, end={source_hashes_at_end}"
        )

    payload = {
        "schema": "hsbc-hybrid-qml-v2",
        "status": "OUTCOMES_COMPUTED_BY_EXPLICIT_RUN",
        "protocol_freeze": PROTOCOL,
        "protocol_sha256": source_hashes_at_start["protocol"],
        "script_sha256": source_hashes_at_start["script"],
        "implementation_freeze": implementation_freeze_metadata(),
        "self_checks": self_checks,
        "data": {
            **observed,
            "split_seed": 0,
            "feature_names": pipe["feature_names"],
            "validation_and_test_resampled": False,
            "feature_screen_uses_labels": True,
            "one_class_scope": "conditional on the shared supervised feature screen",
        },
        "qnn": qnn,
        "hybrid_autoencoder": hae,
        "paired_bootstrap": bootstrap,
        "reference_scores_artifact": str(references_path),
        "reference_scores_sha256": source_hashes_at_start["references"],
        "ulb_data_artifact": str(data_path),
        "ulb_data_sha256": source_hashes_at_start["ulb_data"],
        "scores_artifact": str(scores_output),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "wall_seconds": time.time() - T0,
        "claim_boundary": (
            "Only the named preregistered instantiations are compared. A QML win is not "
            "quantum advantage without its matched classical twin and resource/shot accounting."
        ),
    }
    _stage_and_publish_pair(output, scores_output, score_bundle, payload)
    log(f"wrote {output}")
    log(f"wrote {scores_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen HSBC v2 hybrid QNN and hybrid-autoencoder rivals"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="explicitly authorize loading ULB data, training all frozen seeds, and writing outcomes",
    )
    parser.add_argument("--output", default=DEFAULT_JSON)
    parser.add_argument("--scores-output", default=DEFAULT_NPZ)
    parser.add_argument("--references", default=DEFAULT_REFERENCES)
    args = parser.parse_args()

    if not args.run:
        checks = run_self_checks()
        print(json.dumps(checks, indent=2, sort_keys=True))
        print("Synthetic checks passed. No HSBC/ULB data loaded and no artifacts written.")
        print("Use --run only after the pre-outcome implementation freeze is accepted.")
        return

    run_audit(Path(args.output), Path(args.scores_output), Path(args.references))


if __name__ == "__main__":
    main()
