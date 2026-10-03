#!/usr/bin/env python3
"""Bounded installed CPU composition and actual CE host refactoring gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ce-sdk', type=Path, required=True)
parser.add_argument('--ce-core-commit', required=True)
parser.add_argument('--output', type=Path)
parser.add_argument('--framework-python', type=Path, default=Path(os.environ.get(
    'GLASSHELIX_FRAMEWORK_PYTHON',
    '/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python')),
    help='Existing CPU Torch interpreter used by accepted MODELS checks')
args = parser.parse_args()
args.ce_sdk = args.ce_sdk.resolve()
commands = []
output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='gh-is1-pipeline-evidence-'))
output.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONPATH', None)

def execute(command):
    command = list(map(str, command))
    commands.append(command)
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError('command failed: ' + repr(command) + '\n' + result.stdout + result.stderr)
    return result.stdout

with tempfile.TemporaryDirectory(prefix='gh-is1-pipeline-build-') as directory:
    build = Path(directory)
    native_build, gh_build, prefix = build / 'native', build / 'gh', build / 'prefix'
    execute(['cmake', '-S', ROOT / 'tests/substrate/pipeline', '-B', native_build,
             '-DCellerator_DIR=' + str(args.ce_sdk / 'lib/cmake/Cellerator'),
             '-DCE_EXPECTED_REVISION=' + args.ce_core_commit])
    execute(['cmake', '--build', native_build, '-j', '2'])
    # Existing supported packaging route. The consumer runs isolated from source python/.
    execute(['cmake', '-S', ROOT / 'cmake/substrate', '-B', gh_build,
             '-DCMAKE_INSTALL_PREFIX=' + str(prefix), '-DCMAKE_INSTALL_LIBDIR=lib',
             '-DGLASSHELIX_REQUIRE_INTEGRATED=ON'])
    execute(['cmake', '--build', gh_build, '-j', '2'])
    execute(['cmake', '--install', gh_build])
    manifest = json.loads((prefix / 'share/glasshelix-substrate/install-manifest.json').read_text())
    for relative, expected in manifest['python_source_sha256'].items():
        path = prefix / 'lib/python' / Path(relative).relative_to('python')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('installed toolkit hash differs: ' + relative)
    consumer = json.loads(execute([args.framework_python, '-I', '-B',
        ROOT / 'examples/substrate/scientific_tool/run.py', '--gh-prefix', prefix,
        '--native-refactor', native_build / 'native_refactor', '--ce-core-commit', args.ce_core_commit,
        '--output', output]))
    native_hash = hashlib.sha256((native_build / 'native_refactor').read_bytes()).hexdigest()
sources = [ROOT / 'experiments/celltag_reprogramming/pilot.py',
           ROOT / 'experiments/celltag_reprogramming/data.py',
           ROOT / 'experiments/celltag_reprogramming/data_adapter/__init__.py',
           ROOT / 'experiments/moonshot-parallel-v1/plain/model.py',
           ROOT / 'experiments/moonshot-parallel-v1/actor/model.py']
for owned in ('experiments/celltag_reprogramming/integrated',
              'examples/substrate/scientific_tool', 'tests/substrate/pipeline'):
    sources.extend(path for path in (ROOT / owned).rglob('*') if path.is_file())
receipt = {'status': 'passed', 'commands': commands, 'consumer': consumer,
    'gh_install_manifest': manifest, 'ce_core_commit': args.ce_core_commit,
    'native_consumer_sha256': native_hash,
    'ce_config_sha256': hashlib.sha256((args.ce_sdk / 'lib/cmake/Cellerator/CelleratorConfig.cmake').read_bytes()).hexdigest(),
    'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(set(sources))},
    'artifact_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(output.iterdir()) if path.is_file()},
    'biological_status': 'not_run', 'native_cuda': 'not_run'}
(output / 'gate.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
print(json.dumps({'status': 'passed', 'evidence': str(output / 'gate.json'),
                  'summary': str(output / 'summary.json'), 'capabilities':
                  ['installed GH CPU fit/evaluate', 'model and optimizer checkpoint recomputation',
                   'actual CE adaptive publication'], 'biology': 'not_run', 'native_cuda': 'not_run'}))
