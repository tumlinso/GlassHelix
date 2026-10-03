#!/usr/bin/env python3
"""Check the accepted native core; this is not an installed-provider receipt."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

RUNTIME = Path('/home/tumlinson/.local/share/project-control/candidates/call-audit-256m-20260926')
FINGERPRINT = '15521b1576eab3f0f8aec70519cd2d06f52396db6d48b3a681d5bfb9e6a3464d'
for key in ('PROJECT_CONTROL_RELEASE_MANIFEST', 'PROJECT_CONTROL_RELEASE_DIGEST',
            'CODING_WORKFLOW_SKILLS_ROOT', 'CODING_WORKFLOW_RUNTIME_FINGERPRINT'):
    os.environ.pop(key, None)
os.environ['PROJECT_CONTROL_SKILLS_ROOT'] = str(RUNTIME / 'runtime-skills')
os.environ['PROJECT_CONTROL_TODO_RUNTIME_FINGERPRINT'] = FINGERPRINT
os.environ['PYTHONPATH'] = '/home/tumlinson/project-control/src'
if Path(sys.prefix) != RUNTIME:
    os.execve(str(RUNTIME / 'bin/python'), [str(RUNTIME / 'bin/python'), '-B',
                                         str(Path(__file__).resolve()), *sys.argv[1:]], os.environ)
sys.path.insert(0, '/home/tumlinson/project-control/src')
from project_control.workflow_binding import runtime_identity
assert runtime_identity().fingerprint == FINGERPRINT
from todo_orchestrator.semantic import SemanticReader
from todo_orchestrator.service import Service

parser = argparse.ArgumentParser()
parser.add_argument('--ce-root', type=Path, default=Path('/home/tumlinson/Cellerator'))
parser.add_argument('--ce-sdk', type=Path, default=Path('/tmp/ce-is1-sdk-b'))
parser.add_argument('--ce-core-commit', default='41952561a1fa9b2aab7d6d563fae2e2006b0850e')
args = parser.parse_args()
task_id = 'CE-IS1-MERGE-B'
gate_id = 'CE-IS1-MERGE-B-10-NATIVE'
evidence_id = 'b738953a-3596-4d51-a4e0-718200c2089a'
state = SemanticReader(args.ce_root).state(task_id=task_id)
task = next(t for t in state['tasks'] if t['id'] == task_id)
assert task['effective_state'] == 'done', 'CE native core is unfinished'
service = Service(args.ce_root, read_only=True)
gate = next(g for g in service.gate_list(task_id) if g['id'] == gate_id)
assert gate['required'] and gate['status'] == 'passed' and gate['valid'], 'CE native gate is invalid'
tables = service.export()['state']['tables']
completed = next(t for t in tables['tasks'] if t['id'] == task_id)
assert completed['completion_revision'] == 7872
assert completed['completion_git_head'] == completed['completion_commit'] == args.ce_core_commit
evidence = next(e for e in tables['evidence'] if e['id'] == evidence_id)
metadata = json.loads(evidence['metadata_json'])
assert evidence['gate_id'] == gate_id and evidence['status'] == 'passed' and evidence['revision'] == 7871
assert metadata['returncode'] == 0 and metadata['inputs']['recorded_git_head'] == args.ce_core_commit
assert metadata['input_fingerprint'] == gate['input_fingerprint'], 'CE gate evidence changed'
subprocess.run(['git', '-C', str(args.ce_root), 'merge-base', '--is-ancestor',
                args.ce_core_commit, 'HEAD'], check=True)
config = (args.ce_sdk / 'lib/cmake/Cellerator/CelleratorConfig.cmake').read_text()
revision = re.search(r'set\(Cellerator_BUILD_SOURCE_REVISION\s+"([^"]+)"\)', config)
assert revision and revision.group(1) == args.ce_core_commit, 'CE SDK source revision differs'
print(json.dumps({'status': 'passed', 'task': task_id, 'gate': gate_id,
                  'evidence': evidence_id, 'core_commit': args.ce_core_commit,
                  'completion_revision': 7872, 'validation_revision': 7871,
                  'authority_revision': state['revision'], 'sdk': str(args.ce_sdk)}, sort_keys=True))
