"""Compare isolated HSBC fit_v0 reproductions with the source-locked artifacts.

The original experiment scripts write to a relative ``runs/.../fit_v0`` path.
Run them from an isolated working directory, then point this script at that
directory.  Wall-clock fields are deliberately ignored; all other fields are
compared recursively and all source/reproduction SHA-256 digests are retained.

Example
-------
.venv/bin/python scripts/hsbc_challenge/audit_reproduction_compare_v1.py \
  --reproduced-root /private/tmp/hsbc_audit_repro_20260830/runs/hsbc_challenge/fit_v0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


FILES = (
    "exp_b_scorer.json",
    "exp_b2_addendum.json",
    "exp_c_dequantization.json",
    "exp_d_alignment.json",
    "exp_e_depth.json",
    "exp_f_bottlenecks.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(a, b, path="$", differences=None):
    if differences is None:
        differences = []
    if path.endswith(".wall_seconds"):
        return differences
    if type(a) is not type(b):
        differences.append({"path": path, "source": a, "reproduced": b,
                            "reason": "type"})
        return differences
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key == "wall_seconds":
                continue
            if key not in a or key not in b:
                differences.append({"path": f"{path}.{key}",
                                    "reason": "missing-key"})
            else:
                compare(a[key], b[key], f"{path}.{key}", differences)
    elif isinstance(a, list):
        if len(a) != len(b):
            differences.append({"path": path, "source_len": len(a),
                                "reproduced_len": len(b), "reason": "length"})
        for idx, (left, right) in enumerate(zip(a, b)):
            compare(left, right, f"{path}[{idx}]", differences)
    elif isinstance(a, float):
        if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14):
            differences.append({"path": path, "source": a, "reproduced": b,
                                "absolute_error": abs(a - b), "reason": "float"})
    elif a != b:
        differences.append({"path": path, "source": a, "reproduced": b,
                            "reason": "value"})
    return differences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path,
                        default=Path("runs/hsbc_challenge/fit_v0"))
    parser.add_argument("--reproduced-root", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("runs/hsbc_challenge/audit_v1/"
                                     "baseline_reproduction_v1.json"))
    args = parser.parse_args()

    rows = {}
    for name in FILES:
        source = args.source_root / name
        reproduced = args.reproduced_root / name
        if not reproduced.exists():
            rows[name] = {"status": "MISSING_REPRODUCTION",
                          "source_sha256": sha256(source)}
            continue
        left = json.loads(source.read_text())
        right = json.loads(reproduced.read_text())
        differences = compare(left, right)
        rows[name] = {
            "status": "MATCH" if not differences else "MISMATCH",
            "comparison": "recursive; wall_seconds excluded; floats rtol=1e-12 atol=1e-14",
            "source_sha256": sha256(source),
            "reproduced_sha256": sha256(reproduced),
            "differences": differences,
        }

    result = {
        "protocol": "isolated exact-script rerun using repo .venv and linked source NPZ",
        "source_root": str(args.source_root.resolve()),
        "reproduced_root": str(args.reproduced_root.resolve()),
        "all_match": all(row["status"] == "MATCH" for row in rows.values()),
        "files": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
