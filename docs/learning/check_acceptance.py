#!/usr/bin/env python3
"""Accept retained exact source-bound GPU work and affected current native tests."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

root=Path(__file__).resolve().parents[2]
receipt=json.loads((root/'docs/learning/accept-evidence/ce-train-receipt.json').read_text())
assert receipt['task']=='CE-ML2-TRAIN' and receipt['status']=='complete'
native=Path('/home/tumlinson/Software/cellerator-ml2-python/lib/libcellera_torch_mechanism.so')
qualified={r['path']:r['sha256'] for r in receipt['artifacts']}
assert qualified[str(native)]==hashlib.sha256(native.read_bytes()).hexdigest()
for command in (
    ['python3','-B','docs/learning/check_native_repairs.py'],
    ['python3','-B','docs/learning/verify_learn_evidence.py','--require-native'],
    ['python3','-B','docs/learning/verify_pilot_evidence.py'],
    ['env','CUDA_VISIBLE_DEVICES=','/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python','-B','-m','unittest','discover','-s','tests/learning','-p','test_learning.py','-v'],
):
    subprocess.run(command,cwd=root,check=True)
print(json.dumps({'toolkit':'qualified_fixture','client':'qualified_fixture','biology':'not_run','performance':'evaluated_not_promoted','source_bound_gpu':'retained_and_verified'}))
