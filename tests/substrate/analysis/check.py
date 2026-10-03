#!/usr/bin/env python3
"""Real CE host differential consumer and actual supplied refactoring clients."""
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
parser.add_argument('--cuda-include', type=Path, required=True)
parser.add_argument('--ce-library', type=Path, action='append', required=True)
args = parser.parse_args()
with tempfile.TemporaryDirectory(prefix='gh-analysis-') as directory:
    executable = Path(directory) / 'contracts'
    compile_command = ['g++', '-std=c++20', '-O2', '-UNDEBUG', '-ffunction-sections',
        '-fdata-sections', '-Wl,--gc-sections', '-I', str(ROOT / 'include'),
        '-I', str(args.ce_include), '-I', str(args.cuda_include),
        str(ROOT / 'tests/substrate/analysis/requests.cc'), *[str(p) for p in args.ce_library],
        '-o', str(executable)]
    subprocess.run(compile_command, cwd=ROOT, check=True)
    subprocess.run([str(executable)], cwd=ROOT, check=True)
command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests/substrate/analysis',
           '-p', 'test_*.py', '-v']
subprocess.run(command, cwd=ROOT, check=True,
               env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / 'python'), str(ROOT)]),
                        CUDA_VISIBLE_DEVICES=''))
paths = ['include/GlassHelix/analysis/requests.hh',
         'include/GlassHelix/interrogation/alternatives.hh',
         'include/GlassHelix/interrogation/differentiation.hh',
         'experiments/moonshot-parallel-v1/refactor/model.py',
         'python/glasshelix/analysis/requests.py', 'python/glasshelix/refactoring/proposal.py',
         'tests/substrate/analysis/requests.cc', 'tests/substrate/analysis/test_analysis.py']
print(json.dumps({'status': 'passed', 'scope': 'real CE host JVP/VJP and supplied framework refactoring',
                  'native_epoch_publication': 'not_run', 'biology': 'not_run',
                  'commands': [compile_command, command],
                  'ce_archive_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in args.ce_library},
                  'input_and_source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                              for p in paths}}, indent=2))
