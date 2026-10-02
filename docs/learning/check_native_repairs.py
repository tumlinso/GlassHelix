#!/usr/bin/env python3
"""Build/run affected native GH regressions; no CUDA device execution."""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
ce_prefix = Path(os.environ.get('GH_CE_HEADER_PREFIX', '/home/tumlinson/Software/cellerator-ml2-cpp'))
cuda = Path(os.environ.get('GH_CUDA_ROOT', '/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9'))
with tempfile.TemporaryDirectory(prefix='gh-ml2-repair-') as directory:
    for name, source, libs in (
        ('diagnostics', 'tests/native_foundation/interrogation/diagnostics.cc', ['-llapack', '-lblas']),
        ('inference', 'tests/native_foundation/inference/inference.cc', []),
    ):
        executable = Path(directory) / name
        command = ['g++', '-std=c++20', '-O2', '-I'+str(root/'include'),
                   '-I'+str(ce_prefix/'include'), '-I'+str(cuda/'include'),
                   str(root/source), *libs, '-o', str(executable)]
        subprocess.run(command, check=True)
        subprocess.run([str(executable)], check=True)
        print(f'{name}: actual native regression passed', flush=True)
