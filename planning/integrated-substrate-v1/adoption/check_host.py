#!/usr/bin/env python3
"""Bounded host check against published CE headers; no GPU invocation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument('--ce-include', type=Path, required=True)
parser.add_argument('--ce-library', type=Path, action='append', default=[])
args = parser.parse_args()
if not (args.ce_include / 'Cellerator/compute/operation/native_foundation_contract.hh').is_file():
    parser.error('--ce-include must contain the published Cellerator headers')
with tempfile.TemporaryDirectory(prefix='gh-is1-host-') as temporary:
    outputs = []
    for name, sources in (
        ('metadata', ['planning/integrated-substrate-v1/adoption/host_contract.cc']),
        ('assembly', ['tests/native_foundation/system/s01.cc', 'src/system/assembly.cc']),
        ('inference', ['tests/native_foundation/inference/inference.cc']),
    ):
        executable = Path(temporary) / name
        command = ['g++', '-std=c++20', '-O2', '-UNDEBUG', '-ffunction-sections',
                   '-fdata-sections', '-Wl,--gc-sections', '-I', str(ROOT / 'include'),
                   '-I', str(args.ce_include), *[str(ROOT / p) for p in sources],
                   *[str(p) for p in args.ce_library],
                   '-o', str(executable)]
        subprocess.run(command, check=True)
        subprocess.run([str(executable)], check=True)
        outputs.append({'check': name, 'command': command,
                        'sources': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                    for p in sources}})
    print(json.dumps({'status': 'passed', 'scope': 'host contracts and existing host consumers',
                      'ce_include': str(args.ce_include),
                      'ce_libraries': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in args.ce_library},
                      'checks': outputs}, indent=2))
