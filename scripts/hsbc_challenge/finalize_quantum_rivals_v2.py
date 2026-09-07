#!/usr/bin/env python3
"""Deterministically combine the three frozen HSBC quantum-rival score bundles.

This is a post-outcome reporting step only.  It applies the already frozen
2,000-draw paired Poisson bootstrap and 98.75% interval/practical-tie rule to
an exhaustive, predeclared set of stack/rival and quantum/classical-twin pairs.
It does not train, select, tune, or load the source transaction table.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from audit_hybrid_qml_v2 import paired_simultaneous_bootstrap


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]
RUN_DIR = REPO_ROOT / "runs/hsbc_challenge/audit_v2"
PROTOCOL = REPO_ROOT / "docs/hsbc_challenge_quantum_rivals_audit_v2.md"
LOCK = RUN_DIR / "preoutcome_source_lock_v2.json"
KERNEL_JSON = RUN_DIR / "quantum_kernels_v2.json"
KERNEL_NPZ = RUN_DIR / "quantum_kernels_v2.npz"
VQA_JSON = RUN_DIR / "vqa_qae_v2.json"
VQA_NPZ = RUN_DIR / "vqa_qae_v2.npz"
HYBRID_JSON = RUN_DIR / "hybrid_qml_v2.json"
HYBRID_NPZ = RUN_DIR / "hybrid_qml_v2.npz"
OUTPUT = RUN_DIR / "quantum_rivals_v2_combined.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _array(bundle, key: str, expected_length: int) -> np.ndarray:
    value = np.asarray(bundle[key], float)
    if value.shape != (expected_length,) or not np.all(np.isfinite(value)):
        raise RuntimeError(f"invalid aligned score array {key}: {value.shape}")
    return value


def main() -> None:
    if Path.cwd().resolve() != REPO_ROOT:
        raise RuntimeError(f"run from repository root {REPO_ROOT}")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")

    lock = _read_json(LOCK)
    kernel_meta = _read_json(KERNEL_JSON)
    vqa_meta = _read_json(VQA_JSON)
    hybrid_meta = _read_json(HYBRID_JSON)
    if kernel_meta["status"] != "COMPLETE" or vqa_meta["status"] != "COMPLETE":
        raise RuntimeError("kernel and VQA/QAE families must be COMPLETE")
    if hybrid_meta["status"] != "OUTCOMES_COMPUTED_BY_EXPLICIT_RUN":
        raise RuntimeError("hybrid family is not a completed explicit run")

    expected_protocol_hash = lock["protocol"]["sha256"]
    if sha256(PROTOCOL) != expected_protocol_hash:
        raise RuntimeError("protocol changed after the pre-outcome freeze")
    for meta in (kernel_meta, vqa_meta, hybrid_meta):
        if meta["protocol_sha256"] != expected_protocol_hash:
            raise RuntimeError("outcome artifact does not reference the frozen protocol")

    embedded_score_hashes = {
        KERNEL_NPZ: kernel_meta["artifacts"]["scores_sha256"],
        VQA_NPZ: vqa_meta["scores_sha256"],
        HYBRID_NPZ: hybrid_meta["scores_sha256"],
    }
    for path, expected in embedded_score_hashes.items():
        if sha256(path) != expected:
            raise RuntimeError(f"score artifact hash mismatch: {path}")

    with (
        np.load(KERNEL_NPZ, allow_pickle=False) as kernel,
        np.load(VQA_NPZ, allow_pickle=False) as vqa,
        np.load(HYBRID_NPZ, allow_pickle=False) as hybrid,
    ):
        y = np.asarray(kernel["y_test"], int)
        if y.shape != (56_962,) or int(np.sum(y)) != 99:
            raise RuntimeError("frozen full-test labels/counts are wrong")
        if not np.array_equal(y, np.asarray(vqa["n8_y"], int)):
            raise RuntimeError("VQA/QAE test rows do not match the kernel rows")
        if not np.array_equal(y, np.asarray(hybrid["y_test"], int)):
            raise RuntimeError("hybrid test rows do not match the kernel rows")

        scores = {
            "S1_analytic": _array(kernel, "S1_analytic", len(y)),
            "S2_L1_seed500": _array(kernel, "S2_L1_seed500", len(y)),
            "XGB_subset": _array(kernel, "XGB_subset", len(y)),
            "product_fidelity_QSVC": _array(kernel, "product_fidelity_QSVC", len(y)),
            "classical_RBF_SVC_twin": _array(kernel, "classical_RBF_SVC_twin", len(y)),
            "ring_IQP_fidelity_Nystrom": _array(
                kernel, "ring_IQP_fidelity_Nystrom", len(y)
            ),
            "classical_RBF_Nystrom_twin": _array(
                kernel, "classical_RBF_Nystrom_twin", len(y)
            ),
            "projected_ring_IQP_SVC": _array(
                kernel, "projected_ring_IQP_SVC", len(y)
            ),
            "ring_IQP_fidelity_Nystrom_OCSVM": _array(
                kernel, "ring_IQP_fidelity_Nystrom_OCSVM", len(y)
            ),
            "classical_RBF_Nystrom_OCSVM_twin": _array(
                kernel, "classical_RBF_Nystrom_OCSVM_twin", len(y)
            ),
            "VQC_entangling_mean5": _array(
                vqa, "vqc_entangling_ensemble_mean5", len(y)
            ),
            "VQC_no_entanglement_mean5": _array(
                vqa, "vqc_no_entanglement_ensemble_mean5", len(y)
            ),
            "balanced_logistic": _array(vqa, "balanced_logistic_control", len(y)),
            "trash_QAE_mean5": _array(vqa, "qae_trash_ensemble_mean5", len(y)),
            "QNN_trainable_quantum": _array(
                hybrid, "qnn_trainable_quantum_ensemble", len(y)
            ),
            "QNN_frozen_quantum": _array(
                hybrid, "qnn_frozen_quantum_ensemble", len(y)
            ),
            "QNN_dense_head": _array(hybrid, "qnn_dense_head_ensemble", len(y)),
            "HAE_hybrid_quantum": _array(
                hybrid, "hae_hybrid_quantum_ensemble", len(y)
            ),
            "HAE_direct4_classical": _array(
                hybrid, "hae_direct4_classical_ensemble", len(y)
            ),
            "HAE_width8_classical": _array(
                hybrid, "hae_width8_classical_ensemble", len(y)
            ),
        }

    comparisons = [
        ("S2_L1_seed500", "product_fidelity_QSVC"),
        ("product_fidelity_QSVC", "classical_RBF_SVC_twin"),
        ("S2_L1_seed500", "ring_IQP_fidelity_Nystrom"),
        ("ring_IQP_fidelity_Nystrom", "classical_RBF_Nystrom_twin"),
        ("S2_L1_seed500", "projected_ring_IQP_SVC"),
        ("projected_ring_IQP_SVC", "classical_RBF_SVC_twin"),
        ("S1_analytic", "ring_IQP_fidelity_Nystrom_OCSVM"),
        ("ring_IQP_fidelity_Nystrom_OCSVM", "classical_RBF_Nystrom_OCSVM_twin"),
        ("S2_L1_seed500", "VQC_entangling_mean5"),
        ("VQC_entangling_mean5", "balanced_logistic"),
        ("VQC_entangling_mean5", "VQC_no_entanglement_mean5"),
        ("S1_analytic", "trash_QAE_mean5"),
        ("S2_L1_seed500", "QNN_trainable_quantum"),
        ("QNN_trainable_quantum", "QNN_dense_head"),
        ("QNN_trainable_quantum", "QNN_frozen_quantum"),
        ("S1_analytic", "HAE_hybrid_quantum"),
        ("HAE_hybrid_quantum", "HAE_direct4_classical"),
        ("HAE_hybrid_quantum", "HAE_width8_classical"),
        ("S2_L1_seed500", "classical_RBF_Nystrom_twin"),
        ("S2_L1_seed500", "balanced_logistic"),
        ("S2_L1_seed500", "QNN_dense_head"),
        ("S2_L1_seed500", "XGB_subset"),
    ]
    bootstrap = paired_simultaneous_bootstrap(y, scores, comparisons)
    payload = {
        "schema": "hsbc-quantum-rivals-v2-combined",
        "status": "COMPLETE",
        "analysis_scope": (
            "post-outcome deterministic application of frozen section-5 rules; "
            "no training, tuning, selection, or source-table loading"
        ),
        "test_rows": len(y),
        "test_frauds": int(np.sum(y)),
        "test_resampled": False,
        "protocol_sha256": expected_protocol_hash,
        "preoutcome_lock_sha256": sha256(LOCK),
        "source_artifacts": {
            str(path.relative_to(REPO_ROOT)): sha256(path)
            for path in (KERNEL_JSON, KERNEL_NPZ, VQA_JSON, VQA_NPZ, HYBRID_JSON, HYBRID_NPZ)
        },
        "paired_bootstrap": bootstrap,
        "reporting_cautions": [
            "Named implementations only; no family-wide winner claim.",
            "All one-class rows remain conditional on the supervised feature screen.",
            "Exact statevector results are not shot-matched or hardware advantage evidence.",
            "The trash-QAE runner has no standalone classical-AE twin; the hybrid-AE runner supplies separate matched bottleneck twins.",
        ],
    }
    staged = OUTPUT.with_name(f".{OUTPUT.name}.{os.getpid()}.tmp")
    staged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    checked = _read_json(staged)
    if checked["status"] != "COMPLETE" or checked["test_rows"] != 56_962:
        staged.unlink(missing_ok=True)
        raise RuntimeError("staged combined report failed verification")
    os.replace(staged, OUTPUT)
    print(json.dumps({"status": "COMPLETE", "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
