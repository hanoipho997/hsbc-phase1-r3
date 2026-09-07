#!/usr/bin/env python3
"""Post-outcome polarity diagnostic for catastrophic v2 VQC/QAE scores.

This script does not replace or retune the preregistered primary scores. It
checks whether simply reversing each frozen ensemble ordering exposes signal,
which is relevant only to the strawman/family-claim boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]
SCORES = REPO_ROOT / "runs/hsbc_challenge/audit_v2/vqa_qae_v2.npz"
RESULT = REPO_ROOT / "runs/hsbc_challenge/audit_v2/vqa_qae_v2.json"
EXPECTED = {
    SCORES: "78c80cf8671a948602ac03f3ebc8ac2087752ecf20948f8c184f1bc625301084",
    RESULT: "4f018a1f78d8cc94604d3fc41a8bd94972043285575704b7414d2636f667750a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "auprc": float(average_precision_score(y, score)),
        "auc_roc": float(roc_auc_score(y, score)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="runs/hsbc_challenge/audit_report_v2/rival_score_polarity_diagnostic_v2.json",
    )
    args = parser.parse_args()
    if Path.cwd().resolve() != REPO_ROOT:
        raise RuntimeError(f"run from repository root {REPO_ROOT}")
    for path, expected in EXPECTED.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"source-lock mismatch: {path}: {observed}")
    output = (REPO_ROOT / args.output).resolve(strict=False)
    allowed = (REPO_ROOT / "runs/hsbc_challenge/audit_report_v2").resolve()
    if output.parent != allowed or output.suffix != ".json":
        raise ValueError(f"output must be a JSON direct child of {allowed}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    with np.load(SCORES, allow_pickle=False) as bundle:
        y = np.asarray(bundle["n8_y"], int)
        methods = {
            "VQC_entangling_mean5": np.asarray(
                bundle["vqc_entangling_ensemble_mean5"], float
            ),
            "trash_QAE_mean5": np.asarray(
                bundle["qae_trash_ensemble_mean5"], float
            ),
        }
    if y.shape != (56_962,) or int(y.sum()) != 99:
        raise RuntimeError("unexpected full-test rows")
    rows = {}
    for name, score in methods.items():
        if score.shape != y.shape or not np.all(np.isfinite(score)):
            raise RuntimeError(f"invalid score: {name}")
        rows[name] = {
            "preregistered_polarity": metrics(y, score),
            "reversed_polarity_posthoc": metrics(y, -score),
        }
    payload = {
        "schema": "hsbc-rival-score-polarity-diagnostic-v2",
        "status": "COMPLETE_POSTHOC_DIAGNOSTIC",
        "test_rows": int(len(y)),
        "test_frauds": int(y.sum()),
        "test_resampled_or_balanced": False,
        "methods": rows,
        "claim_boundary": (
            "The preregistered polarity remains the official primary result. "
            "Reversed scores are a post-test strawman diagnostic only: they may "
            "show that a catastrophic primary score does not bound family capacity."
        ),
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): digest for path, digest in EXPECTED.items()
        },
        "runner_sha256": sha256(SCRIPT),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    staged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if json.loads(staged.read_text())["status"] != "COMPLETE_POSTHOC_DIAGNOSTIC":
        staged.unlink(missing_ok=True)
        raise RuntimeError("staged output failed validation")
    os.replace(staged, output)
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
