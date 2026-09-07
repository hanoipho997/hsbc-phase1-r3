#!/usr/bin/env python3
"""Read-only structural diagnostic for Engine-A's frozen C-f noise head.

This script never loads the IEEE-CIS rows or the sealed test outputs.  It
inspects only the already-fitted joblib artifact and records the standardized
logistic coefficients used by the score-replacement noise control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


DEFAULT_HEAD = Path(
    "runs/hsbc_challenge/engine_a_v1/arms_v1/head_C_f.joblib"
)
DEFAULT_OUTPUT = Path(
    "runs/hsbc_challenge/audit_report_v2/engine_a_noise_head_diagnostic_v2.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    frozen = joblib.load(args.head)
    scaler = frozen["scaler"]
    classifier = frozen["clf"]
    coefficients = np.asarray(classifier.coef_, dtype=float).reshape(-1)
    if coefficients.shape != (9,):
        raise RuntimeError(f"expected nine C-f coefficients, got {coefficients.shape}")

    noise_coefficients = coefficients[1:]
    result = {
        "status": "COMPLETE_READ_ONLY_ARTIFACT_DIAGNOSTIC",
        "sealed_test_rows_loaded": False,
        "source_artifact": str(args.head),
        "source_artifact_sha256": sha256(args.head),
        "feature_order": ["backbone_logit"]
        + [f"keyed_noise_{index}" for index in range(8)],
        "coefficient_coordinate": "after the frozen StandardScaler",
        "standardized_coefficients": coefficients.tolist(),
        "standardized_backbone_logit_coefficient": float(coefficients[0]),
        "standardized_noise_coefficients": noise_coefficients.tolist(),
        "standardized_noise_block_l2": float(np.linalg.norm(noise_coefficients)),
        "intercept": float(np.asarray(classifier.intercept_).reshape(-1)[0]),
        "scaler_mean": np.asarray(scaler.mean_, dtype=float).tolist(),
        "scaler_scale": np.asarray(scaler.scale_, dtype=float).tolist(),
        "head_C": float(frozen["C"]),
        "validation_pool_auprc": float(frozen["val_pool_auprc"]),
        "interpretation_boundary": (
            "Nonzero keyed-noise coefficients prove that C-f is not a monotone "
            "function of the backbone logit alone. They do not identify which "
            "combination of overfit, reranking, score-boundary mismatch, and "
            "temporal drift caused the sealed-test loss."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
