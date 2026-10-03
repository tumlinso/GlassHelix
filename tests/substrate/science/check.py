#!/usr/bin/env python3
"""Real host scientific contracts and predecessor evidence adapter checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument('--ce-include', type=Path, required=True)
args = parser.parse_args()
checks = []
with tempfile.TemporaryDirectory(prefix='gh-science-') as directory:
    executable = Path(directory) / 'contracts'
    compile_command = ['g++', '-std=c++20', '-O2', '-UNDEBUG', '-I', str(ROOT / 'include'),
                       '-I', str(args.ce_include), str(ROOT / 'tests/substrate/science/specification.cc'),
                       '-o', str(executable)]
    subprocess.run(compile_command, cwd=ROOT, check=True)
    subprocess.run([str(executable)], cwd=ROOT, check=True)
    checks.append(compile_command)
env = dict(os.environ, PYTHONPATH=str(ROOT / 'python'), CUDA_VISIBLE_DEVICES='')
command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests/substrate/science',
           '-p', 'test_*.py', '-v']
subprocess.run(command, cwd=ROOT, env=env, check=True)
checks.append(command)
sources = ['include/GlassHelix/experiment/specification.hh',
           'python/glasshelix/experiment/specification.py', 'python/glasshelix/experiment/__init__.py',
           'python/glasshelix/learning.py', 'include/GlassHelix/inference/inference.hh',
           'tests/substrate/science/specification.cc', 'tests/substrate/science/test_specification.py']
print(json.dumps({'status': 'passed', 'scope': 'host scientific contracts and replay',
                  'biological_status': 'not_run', 'commands': checks,
                  'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                    for p in sources}}, indent=2))
