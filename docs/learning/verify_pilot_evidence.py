#!/usr/bin/env python3
"""Pure verification of the retained, controller-run synthetic PILOT evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
GH_LEARNING = Path('/tmp/gh-ml2-installed-sdk/lib/glasshelix/learning.py')
CT_PACKAGE = Path('/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/lib/python3.13/site-packages/celleratorch')
CT_NATIVE = Path('/home/tumlinson/Software/cellerator-ml2-python/lib/libcellera_torch_mechanism.so')
MARKER = 'GH_ML2_PILOT_EVIDENCE='
PHASES = ['CPU-to-CUDA conversion', 'activity generation', 'relation forward',
          'masked objective', 'backward', 'Adam/guarded publication', 'synchronize', 'scalar transfer']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(), parse_constant=reject_constant)


def reject_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def number(value, label, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f'invalid {label}')
    return value


def producer(directory):
    path = directory / 'ce-bio-manifest.json'
    manifest, receipt = load(path), load(directory / 'ce-bio-receipt.json')
    require(manifest['task'] == receipt['task'] == 'CE-ML2-BIO'
            and receipt['status'] == 'passed', 'unqualified CE BIO producer')
    require(receipt['evidence_sha256']['manifest.json'] == sha(path), 'CE BIO manifest binding mismatch')
    require(manifest['environment']['installed_package'] == str(CT_PACKAGE), 'installed CT package mismatch')
    for installed in (CT_PACKAGE / '__init__.py', CT_PACKAGE / 'mechanism.py',
                      CT_PACKAGE / 'biology.py', CT_NATIVE):
        require(manifest['input_sha256'][str(installed)] == sha(installed),
                f'installed producer changed: {installed.name}')
    require(manifest['native_library']['path'] == str(CT_NATIVE)
            and manifest['native_library']['sha256'] == sha(CT_NATIVE), 'native producer mismatch')


def controller(directory, result_path, manifest_path, sources):
    receipt = load(directory / 'controller.json')
    require(receipt['classification'] == 'decision-result' and receipt['status'] == 'succeeded'
            and receipt['ok'] is True and type(receipt['returncode']) is int
            and receipt['returncode'] == 0, 'controller execution failed')
    uuid.UUID(receipt['evidence_id'])
    for field, path in (('stdout_sha256', directory / 'controller-stdout.txt'),
                        ('stderr_sha256', directory / 'controller-stderr.txt'),
                        ('result_sha256', result_path), ('manifest_sha256', manifest_path)):
        require(receipt[field] == sha(path), f'controller {field} mismatch')
    require(receipt['source_hashes'] == sources, 'controller source identity mismatch')
    require(not re.search(r'"(?:lease_token|token|workflow_handle)"\s*:', json.dumps(receipt)),
            'controller receipt contains private authorization material')
    markers = [line[len(MARKER):] for line in (directory / 'controller-stdout.txt').read_text().splitlines()
               if line.startswith(MARKER)]
    require(len(markers) == 1, 'exactly one controller stdout evidence marker required')
    marker = json.loads(markers[0], parse_constant=reject_constant)
    require(marker['result_sha256'] == sha(result_path)
            and marker['manifest_sha256'] == sha(manifest_path), 'stdout artifact binding mismatch')
    return receipt['evidence_id'], marker


def costs(record):
    require(record['timed_phases'] == PHASES, 'complete iteration timing phases changed')
    for key in ('model_setup_seconds', 'optimizer_setup_seconds', 'setup_seconds', 'five_complete_iterations_seconds',
                'first_training_iteration_after_equivalence_seconds',
                'setup_plus_first_training_iteration_seconds',
                'setup_plus_five_iterations_seconds', 'mean_complete_iteration_seconds'):
        number(record[key], key, positive=True)
    require(math.isclose(record['setup_seconds'], record['model_setup_seconds'] + record['optimizer_setup_seconds'],
                         rel_tol=1e-10)
            and math.isclose(record['setup_plus_five_iterations_seconds'],
                         record['setup_seconds'] + record['five_complete_iterations_seconds'], rel_tol=1e-10)
            and math.isclose(record['mean_complete_iteration_seconds'] * 5,
                             record['five_complete_iterations_seconds'], rel_tol=1e-10)
            and math.isclose(record['setup_plus_first_training_iteration_seconds'],
                             record['setup_seconds'] + record['first_training_iteration_after_equivalence_seconds'],
                             rel_tol=1e-10), 'timing interval arithmetic mismatch')
    for key in ('peak_allocated_bytes', 'baseline_allocated_bytes',
                'peak_increment_bytes', 'peak_reserved_bytes'):
        require(type(record[key]) is int and record[key] >= 0, f'invalid memory field: {key}')
    require(record['peak_allocated_bytes'] >= record['baseline_allocated_bytes']
            and record['peak_increment_bytes'] == record['peak_allocated_bytes'] - record['baseline_allocated_bytes']
            and record['peak_reserved_bytes'] >= record['peak_allocated_bytes'], 'memory accounting mismatch')
    require(record['resident_models'] == 'both candidate graphs and optimizer states; memory is process allocation',
            'memory scope changed')


def verify(directory):
    result_path, manifest_path = directory / 'pilot.json', directory / 'pilot.manifest.json'
    records_path = directory / 'pilot.records.csv'
    result, manifest = load(result_path), load(manifest_path)
    require(result['status'] == 'passed' and result['fixture'] is True
            and result['biological_status'] == 'not_run', 'synthetic PILOT scope or status mismatch')
    require(result['protocol'] == 'blind_population_law'
            and result['scientific_scope'] == 'cross-sectional RNA population response; no individual trajectories or causal claim',
            'scientific scope changed')
    require(manifest['format'] == 'celltag-pilot-evidence-v1'
            and manifest['fixture'] is True and manifest['biological_status'] == 'not_run'
            and manifest['protocol'] == result['protocol'], 'manifest scope mismatch')
    source_dir = ROOT / 'experiments/celltag_reprogramming'
    sources = {name: sha(source_dir / (name + '.py')) for name in ('pilot', 'test_pilot', 'data', 'reference')}
    sources['learning'] = sha(ROOT / 'python/glasshelix/learning.py')
    require(result['source_hashes'] == manifest['source_hashes'] == sources, 'scoped source map mismatch')
    require(manifest['pilot_source_sha256'] == result['pilot_source_sha256'] == sources['pilot']
            and manifest['test_source_sha256'] == sources['test_pilot']
            and manifest['data_source_sha256'] == sources['data']
            and manifest['reference_source_sha256'] == sources['reference'], 'scoped source identity mismatch')
    require(manifest['installed_gh_sha256'] == result['installed_gh_sha256']
            == sha(GH_LEARNING) == sources['learning'], 'installed GH learning identity mismatch')
    require(manifest['source_commit'] == result['source_commit']
            and re.fullmatch('[0-9a-f]{40}', result['source_commit']), 'source commit identity mismatch')
    require(manifest['records_sha256'] == result['source_sha256'] == sha(records_path), 'records identity mismatch')
    require(result['manifest_sha256'] == sha(manifest_path), 'result manifest binding mismatch')
    # Recorded paths are checked as metadata and never used to choose hash inputs.
    require(manifest['records_path'] == str(records_path.resolve())
            and result['manifest_path'] == str(manifest_path.resolve()), 'artifact path metadata mismatch')
    require(number(result['cutoff'], 'cutoff') == 12, 'fixture cutoff changed')
    expected_permissions = {'RNA_only': True, 'ATAC_withheld': True, 'clone_ID_predictor': False,
                            'clone_split_isolation_audited': True,
                            'training_only_feature_manifest': True, 'training_time_cutoff': 12}
    require(result['permissions'] == manifest['permissions'] == expected_permissions, 'evidence permissions changed')
    features = result['feature_manifest']
    require(features == manifest['feature_manifest']
            and features['inputs'] == ['fixture_rna_x0', 'fixture_rna_x1', 'fixture_rna_x2']
            and features['targets'] == ['fixture_rna_y0', 'fixture_rna_y1'], 'fixture feature identity changed')
    feature_payload = {key: value for key, value in features.items() if key != 'sha256'}
    require(features['sha256'] == hashlib.sha256(json.dumps(feature_payload, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest(), 'feature manifest hash mismatch')
    require(set(result['controls']) == {'structured_torch', 'snapshot_only', 'flexible_same_information'},
            'required controls missing')
    for name, control in result['controls'].items():
        number(control['train_loss'], f'{name} training loss')
        number(control['test_loss'], f'{name} held-out loss')
        number(control['CPU_fit_and_evaluation_seconds'], f'{name} CPU cost', positive=True)
        expected_inputs = features['inputs'] + ([] if name == 'snapshot_only' else ['released_time_label'])
        require(control['permitted_inputs'] == expected_inputs, f'{name} information control mismatch')
        evidence = control['evidence_manifest']['evidence']
        require(evidence == manifest['fit_evidence'][name] and evidence['modality'] == 'RNA'
                and evidence['split'] == 'train' and evidence['source_id'] == sha(records_path),
                f'{name} fit evidence permission mismatch')
        require(len(evidence['evidence_ids']) == len(evidence['times']) == len(evidence['mask']) == 8
                and all(0 <= number(t, 'evidence time') <= 12 for t in evidence['times']),
                f'{name} evidence count or cutoff mismatch')
        require(all(len(row) == 2 and all(type(value) is bool for value in row) for row in evidence['mask'])
                and sum(sum(row) for row in evidence['mask']) == 15, 'masked observation support changed')
        evidence_payload = {key: value for key, value in evidence.items() if key != 'sha256'}
        require(evidence['sha256'] == hashlib.sha256(json.dumps(evidence_payload, sort_keys=True,
                allow_nan=False).encode()).hexdigest(), 'fit evidence hash mismatch')
    # Independently check retained clone split metadata without importing Torch.
    clones = {}
    with records_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            clone = row['clone_id']
            if clone:
                require(clone not in clones or clones[clone] == row['split'], 'retained clone split leakage')
                clones[clone] = row['split']
    native = result['native_comparison']
    require(native == manifest['native_comparison'] and native['status'] == 'passed'
            and native['batch_sizes'] == [8, 3], 'native comparison or batch coverage mismatch')
    for check in ('native_coefficients_trained', 'same_graph_forward_and_objective_vjp', 'same_graph_adam_update'):
        require(native[check] is True, f'native check failed: {check}')
    before, fitted = native['initial_native_coefficients'], native['fitted_native_coefficients']
    require(len(before) == len(fitted) == 5 and before != fitted, 'native coefficients were not fitted')
    for value in before + fitted:
        require(type(value) in (int, float) and math.isfinite(value), 'nonfinite native coefficient')
    require(native['disposition'] == 'evaluated_not_promoted'
            and native['promotion_limit'] == 'small synthetic fixture; no general speedup or biological claim',
            'native promotion scope changed')
    require(native['producer_package_sha256'] == sha(CT_PACKAGE / '__init__.py')
            and native['native_library_sha256'] == sha(CT_NATIVE), 'native runtime identity mismatch')
    for name in ('native', 'torch'):
        require(number(native['final_loss'][name], 'final loss') < number(native['initial_loss'][name], 'initial loss'),
                f'{name} training did not improve')
        costs(native['costs'][name])
    require(math.isclose(native['final_loss']['native'], native['final_loss']['torch'], rel_tol=3e-4, abs_tol=3e-5),
            'native and Torch final objectives differ')
    producer(directory)
    evidence_id, marker = controller(directory, result_path, manifest_path, sources)
    require(marker['source_hashes'] == sources and marker['pilot_source_sha256'] == sources['pilot']
            and marker['installed_gh_sha256'] == sources['learning']
            and marker['records_sha256'] == sha(records_path)
            and marker['source_commit'] == result['source_commit']
            and marker['result_path'] == str(result_path.resolve())
            and marker['manifest_path'] == str(manifest_path.resolve()), 'stdout source binding mismatch')
    return {'status': 'pass', 'biological_status': 'not_run', 'controller_evidence_id': evidence_id,
            'source_hashes': sources, 'native_disposition': native['disposition']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, default=ROOT / 'docs/learning/pilot-evidence')
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.evidence_dir), sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f'PILOT evidence rejected: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
