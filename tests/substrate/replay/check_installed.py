#!/usr/bin/env python3
"""Enforced actual installed scientific run/evaluation/replay after source merge."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=Path, required=True)
args = parser.parse_args()
launcher = args.prefix / 'bin/glasshelix-substrate'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
env.pop('PYTHONPATH', None)
with tempfile.TemporaryDirectory(prefix='gh-installed-replay-') as directory:
    root = Path(directory)
    def cli(*arguments, check=True):
        return subprocess.run([sys.executable, str(launcher), *map(str, arguments)], cwd=root,
                              env=env, check=check, text=True, capture_output=True)
    cap = json.loads(cli('capabilities').stdout)
    if cap['run'] != 'available' or not all(cap['components'].values()):
        raise RuntimeError('accepted integrated installation is incomplete')
    plan = {'driver': 'joint_product_reference', 'coefficient': .5, 'coefficient_id': 'law',
            'observable_id': 'response', 'provider_evidence_id': 'installed-framework-client',
            'specification': {'experiment_id': 'installed-client', 'model_id': 'joint-model',
                'quantities': [{'quantity_id': 'state', 'role': 'state', 'extent': 2, 'units': 'latent'},
                               {'quantity_id': 'law', 'role': 'parameter', 'extent': 1, 'units': 'declared'},
                               {'quantity_id': 'response', 'role': 'measurement', 'extent': 1, 'units': 'declared'}],
                'hypotheses': [{'hypothesis_id': 'h1', 'mechanism_id': 'joint-product',
                                'joint_state': [1., 2.], 'parameter_ids': ['law'], 'weight': .5},
                               {'hypothesis_id': 'h2', 'mechanism_id': 'joint-product',
                                'joint_state': [2., 1.], 'parameter_ids': ['law'], 'weight': .5}],
                'source_id': 'fixture', 'audit_evidence_id': 'audit', 'input_manifest_ids': ['inputs'],
                'source_kind': 'synthetic_fixture'},
            'provenance': {'model_id': 'joint-model', 'model_revision': '1',
                'gh_source_commit': 'installed-client-fixture', 'ce_source_commit': 'framework-reference',
                'numerical_policy': 'torch_cpu_f64_reference', 'structure_epoch': 1,
                'state_generation': 1, 'parameter_generation': 1, 'evidence_ids': ['fixture-inputs']}}
    plan_path = root / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    result_path, replay_path = root / 'result.json', root / 'replay.json'
    cli('run', '--plan', plan_path, '--output', result_path)
    result = json.loads(result_path.read_text())['result']
    if ([p['values'] for p in result['predictions']] != [[1.], [1.]] or
            result['biological_status'] != 'not_run' or
            [h['joint_state'] for h in result['hypotheses']] != [[1., 2.], [2., 1.]]):
        raise RuntimeError('installed actual model result differs from the declared fixture')
    observations = {'observable_id': 'response', 'values': [1.], 'present': [True],
                    'evidence_id': 'held-out', 'modality': 'RNA', 'split': 'test',
                    'time': 1., 'source_id': 'fixture'}
    observation_path, score_path = root / 'observations.json', root / 'scores.json'
    observation_path.write_text(json.dumps(observations))
    cli('evaluate', '--result', result_path, '--observations', observation_path, '--output', score_path)
    scores = json.loads(score_path.read_text())
    if len(scores['scores']) != 2 or any(s['masked_loss'] != 0 for s in scores['scores']):
        raise RuntimeError('installed masked evaluation differs')
    cli('replay', '--result', result_path, '--output', replay_path)
    if result_path.read_bytes() != replay_path.read_bytes():
        raise RuntimeError('whole-hypothesis replay changed the result')
    tampered = json.loads(result_path.read_text())
    tampered['result']['interpretation_scope'] = 'globally_identified_mechanism'
    (root / 'invalid.json').write_text(json.dumps(tampered))
    if cli('replay', '--result', root / 'invalid.json', '--output', replay_path, check=False).returncode == 0:
        raise RuntimeError('unsupported identification survived installed replay')
    # The implementation packages must come from the tested install, and the
    # supported command must not import experimental moonshot modules.
    code = '''
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
import glasshelix.cli, glasshelix.experiment, glasshelix.data, glasshelix.models
import glasshelix.analysis, glasshelix.refactoring
paths = [m.__file__ for m in (glasshelix.cli, glasshelix.experiment, glasshelix.data,
                            glasshelix.models, glasshelix.analysis, glasshelix.refactoring)]
if not all(pathlib.Path(p).is_relative_to(sys.argv[1]) for p in paths):
    raise RuntimeError("a source-tree module leaked into installed qualification")
if any("moonshot" in name for name in sys.modules):
    raise RuntimeError("experiment imports leaked into core installation")
print(json.dumps(paths))
'''
    imports = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(args.prefix / 'lib/python')],
                             cwd=root, check=True, text=True, capture_output=True, env=env)
    print(json.dumps({'status': 'passed', 'scope': 'installed actual framework run/evaluate/replay',
                      'biological_status': 'not_run', 'installed_imports': json.loads(imports.stdout)}))
