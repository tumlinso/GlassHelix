#!/usr/bin/env python3
"""Actual existing model owners through the common scientific client."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests/substrate/models',
           '-p', 'test_*.py', '-v']
subprocess.run(command, cwd=ROOT, check=True,
               env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / 'python'), str(ROOT)]),
                        CUDA_VISIBLE_DEVICES=''))
paths = ['experiments/moonshot-parallel-v1/plain/model.py',
         'experiments/moonshot-parallel-v1/actor/model.py',
         'experiments/celltag_reprogramming/pilot.py', 'python/glasshelix/learning.py',
         'python/glasshelix/models/client.py', 'tests/substrate/models/test_client.py']
print(json.dumps({'status': 'passed', 'scope': 'existing CPU framework model clients',
                  'native_cuda': 'not_run', 'biology': 'not_run', 'command': command,
                  'input_and_source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                              for p in paths}}, indent=2))
