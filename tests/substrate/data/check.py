#!/usr/bin/env python3
"""Run actual predecessor cohort adapter and replay tests on CPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests/substrate/data',
           '-p', 'test_*.py', '-v']
subprocess.run(command, cwd=ROOT, check=True,
               env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / 'python'), str(ROOT)]),
                        CUDA_VISIBLE_DEVICES=''))
paths = ['docs/learning/pilot-evidence/pilot.records.csv',
         'experiments/celltag_reprogramming/missing-inputs.toml',
         'experiments/celltag_reprogramming/data.py',
         'experiments/celltag_reprogramming/data_adapter/__init__.py',
         'python/glasshelix/data/cohort.py', 'python/glasshelix/learning.py',
         'tests/substrate/data/test_adapter.py']
print(json.dumps({'status': 'passed', 'scope': 'actual retained fixture audit and adapter replay',
                  'biological_status': 'not_run', 'command': command,
                  'input_and_source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                              for p in paths}}, indent=2))
