#!/usr/bin/env python3
"""Run the CPU model checks and record exact implementation source hashes."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import unittest

import torch

import test_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("validation.json"))
    args = parser.parse_args()
    # No device discovery or CUDA probes are needed for CPU qualification.
    torch.set_num_threads(1)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_models.PlainModelsTest)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), end="")
    if not result.wasSuccessful():
        raise SystemExit(1)
    directory = Path(__file__).resolve().parent
    record = {
        "record_kind": "gh_moonshot_plain_reference_validation", "schema_version": 1,
        "status": "passed", "tested_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": {
            "git_parent_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip(),
            "files_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                             for name in ("model.py", "test_models.py", "validate.py")},
            "immutable_seed_sha256": "9f7ffcfc31f3ee6dc292fffee6c4608e5859c01da777d9e50caadd0b3a263045",
        },
        "runtime": {"torch": torch.__version__, "device": "cpu", "threads": 1,
                    "tested_dtypes": ["float32", "float64"]},
        "models": [
            {"class": "MatrixPatchField", "callable_path": "plain/model.py:MatrixPatchField.forward",
             "shape": "(patches,width,width)", "tested_shapes": [[2, 2, 2], [2, 3, 3]],
             "derivatives": ["input VJP", "left/right/decay parameter VJP"]},
            {"class": "SylvesterFlow", "callable_path": "plain/model.py:SylvesterFlow.forward",
             "shape": "(width,width)", "tested_shapes": [[2, 2], [3, 3]],
             "derivatives": ["input VJP", "left/right parameter VJP", "scalar dt VJP/JVP"],
             "flow_conditions": "fixed autonomous L,R; torch.matrix_exp numerical evaluation"},
        ],
        "checks": {"tests_run": result.testsRun, "log": stream.getvalue(),
                   "learning": test_models.LEARNING_RESULTS},
        "argv": [os.sys.executable, "-B", "experiments/moonshot-parallel-v1/plain/validate.py"],
        "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "native_execution": "not_run", "cuda_execution": "not_run", "benchmark": "not_run",
        "biological_validation": "not_run",
        "unsupported_or_unqualified": ["float16/bfloat16/complex", "batched-cell inputs",
            "time-varying/nonlinear exact flow", "native CE/CelleraTorch adapter",
            "CUDA execution", "higher-order derivative qualification", "performance claims"],
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "tests_run": result.testsRun,
                      "learning": test_models.LEARNING_RESULTS}, indent=2))


if __name__ == "__main__":
    main()
