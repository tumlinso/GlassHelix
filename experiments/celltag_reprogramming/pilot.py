"""Bounded blind RNA population-response pilot; fixtures are synthetic only."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch import nn

try:
    from .data import FIELDS, audit, canonical_hash, read_records
    from .reference import factorized, masked_mse
except ImportError:
    from data import FIELDS, audit, canonical_hash, read_records
    from reference import factorized, masked_mse

SRC = (0, 0, 1, 2, 1)
DST = (0, 0, 0, 1, 1)
FIXTURE_FEATURES = {'inputs': ['fixture_rna_x0', 'fixture_rna_x1', 'fixture_rna_x2'],
                    'targets': ['fixture_rna_y0', 'fixture_rna_y1']}


def installed_learning(prefix):
    prefix = Path(prefix).resolve()
    sys.path.insert(0, str(prefix / 'lib'))
    module = importlib.import_module('glasshelix.learning')
    if not Path(module.__file__).resolve().is_relative_to(prefix / 'lib'):
        raise RuntimeError('pilot requires GH from the declared installed SDK')
    return module


def clone_abundance(rows):
    """Sampled cell abundance with mixed clones retained; no fate probability."""
    cells = defaultdict(set)
    for row in rows:
        if row['regime'] == 'common' and row['modality'] == 'RNA' and row['qc_pass']:
            cells[(row['clone_id'], row['time_label'], row['replicate'])].add((row['sample_id'], row['cell_id']))
    totals = defaultdict(int)
    for (_, label, replicate), members in cells.items():
        totals[(label, replicate)] += len(members)
    return [{'clone_id': clone, 'time_label': label, 'replicate': replicate,
             'sampled_cells': len(members), 'sampled_abundance': len(members) / totals[(label, replicate)],
             'interpretation': 'sampled abundance; growth/survival/sampling unresolved'}
            for (clone, label, replicate), members in sorted(cells.items())]


def prepare(path, manifest, cutoff=12):
    if not math.isfinite(cutoff):
        raise ValueError('training cutoff must be finite')
    inputs, targets = manifest['inputs'], manifest['targets']
    if (len(inputs) != 3 or len(targets) != 2 or len(set(inputs + targets)) != 5):
        raise ValueError('this bounded pilot requires three distinct input and two target RNA features')
    report = audit(path, inputs + targets)
    rows, _ = read_records(path)
    eligible_features = {row['feature_id'] for row in rows if row['modality'] == 'RNA'
                         and row['split'] == 'train' and row['regime'] == 'common'
                         and row['qc_pass'] and row['value'] is not None
                         and math.isfinite(float(row['time_label'])) and float(row['time_label']) <= cutoff}
    if not set(inputs + targets).issubset(eligible_features):
        raise ValueError('frozen features require observed training RNA through the cutoff')
    grouped = defaultdict(dict)
    for row in rows:
        if row['modality'] == 'RNA' and row['regime'] == 'common' and row['qc_pass']:
            key = (row['sample_id'], row['cell_id'], row['replicate'])
            grouped[key][row['feature_id']] = row
    result = {}
    excluded = defaultdict(int)
    for split in ('train', 'validation', 'test'):
        xx, yy, masks, ids, times = [], [], [], [], []
        for key, features in sorted(grouped.items()):
            row = next(iter(features.values()))
            if row['split'] != split:
                continue
            label_time = float(row['time_label'])
            if not math.isfinite(label_time):
                raise ValueError('released time labels must be finite; labels are not recoded')
            if split == 'train' and label_time > cutoff:
                excluded['training_after_cutoff'] += 1
                continue
            if any(feature not in features or features[feature]['value'] is None for feature in inputs):
                excluded['missing_input_cells'] += 1
                continue
            target_values = [features.get(feature, {}).get('value') for feature in targets]
            mask = [value is not None for value in target_values]
            if not any(mask):
                excluded['no_observed_target_cells'] += 1
                continue
            xx.append([features[feature]['value'] for feature in inputs] + [label_time / 12.])
            yy.append([0. if value is None else value for value in target_values])
            masks.append(mask)
            ids.append('|'.join(key))
            times.append(label_time)
        result[split] = {'inputs': torch.tensor(xx, dtype=torch.float32).reshape(-1, 4),
                         'targets': torch.tensor(yy, dtype=torch.float32).reshape(-1, 2),
                         'mask': torch.tensor(masks, dtype=torch.bool).reshape(-1, 2),
                         'evidence_ids': tuple(ids), 'times': tuple(times)}
    if not len(result['train']['inputs']) or not len(result['test']['inputs']):
        raise ValueError('pilot requires eligible training and clone-isolated test RNA observations')
    result['audit'] = report
    result['feature_manifest'] = dict(manifest, sha256=canonical_hash(manifest))
    result['exclusions'] = dict(excluded)
    result['clone_abundance'] = clone_abundance(rows)
    return result


class PopulationRelation(nn.Module):
    def __init__(self, *, native=False, device='cpu', initial=None):
        super().__init__()
        self.source_generator = nn.Linear(1, 3, device=device)
        self.target_generator = nn.Linear(1, 2, device=device)
        with torch.no_grad():
            self.source_generator.weight.copy_(torch.tensor([[.4], [-.2], [.1]], device=device))
            self.source_generator.bias.copy_(torch.tensor([.2, .1, .3], device=device))
            self.target_generator.weight.copy_(torch.tensor([[.3], [-.1]], device=device))
            self.target_generator.bias.copy_(torch.tensor([.4, .3], device=device))
        initial = torch.tensor([.1] * 5, device=device) if initial is None else initial.to(device)
        self.native = native
        if native:
            from celleratorch import Axis, Identity
            from celleratorch.biology import SharedSupportRelation, SharedSupportSpec
            def axis(base, size):
                return Axis(*(Identity(base + i, 1000 + base + i) for i in range(4)), size)
            spec = SharedSupportSpec(axis(10, 3), axis(20, 2), axis(30, 5),
                                     tuple(Identity(100 + i, 200 + i) for i in range(5)), SRC, DST)
            self.relation = SharedSupportRelation(spec, initial, max_batch=8, max_live_forwards=8)
        else:
            self.weights = nn.Parameter(initial.clone())
        self.register_buffer('src', torch.tensor(SRC, device=device))
        self.register_buffer('dst', torch.tensor(DST, device=device))

    @property
    def coefficients(self):
        return self.relation.coefficients if self.native else self.weights

    def forward(self, inputs):
        x, context = inputs[:, :3], inputs[:, 3:]
        s = self.source_generator(context).sigmoid()
        a = self.target_generator(context).sigmoid()
        if self.native:
            return self.relation(x, s, a)
        return factorized(self.weights, x, s, a, self.dst, self.src)


class FlexibleControl(nn.Module):
    def __init__(self, snapshot=False):
        super().__init__()
        self.snapshot = snapshot
        with torch.random.fork_rng():
            torch.manual_seed(29)
            self.network = nn.Sequential(nn.Linear(3 if snapshot else 4, 12), nn.Tanh(), nn.Linear(12, 2))

    def forward(self, inputs):
        return self.network(inputs[:, :3] if self.snapshot else inputs)


def fixture_records(path):
    """Write clearly named synthetic cells; never substitute for released data."""
    generator = torch.Generator().manual_seed(71)
    x = torch.rand((11, 3), generator=generator) + .2
    labels = ['3', '11', '12', '3', '11', '12', '3', '12', '3', '11', '12']
    context = torch.tensor([float(label) / 12 for label in labels]).reshape(-1, 1)
    truth = PopulationRelation(initial=torch.tensor([.6, -.2, .3, .8, -.1]))
    with torch.no_grad():
        y = truth(torch.cat((x, context), 1))
    rows = []
    for i in range(11):
        split = 'train' if i < 8 else 'test'
        for j, feature in enumerate(FIXTURE_FEATURES['inputs'] + FIXTURE_FEATURES['targets']):
            value = float(x[i, j]) if j < 3 else float(y[i, j - 3])
            rows.append(dict(sample_id=f'fixture_sample_{labels[i]}', cell_id=f'fixture_cell_{i}',
                             clone_id=f'fixture_{split}_clone_{i // 2}', time_label=labels[i], modality='RNA',
                             replicate=f'fixture_rep_{1 + i % 2}', regime='common', split=split,
                             feature_id=feature, value='' if i == 0 and j == 4 else value))
        rows.append(dict(rows[-1], modality='ATAC', feature_id='fixture_withheld_peak', value=99999))
    rows.append(dict(rows[0], sample_id='fixture_perturbation', cell_id='fixture_p',
                     clone_id='fixture_p', regime='Zfp281_perturbation', value=99999))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def train_controls(problem, gh):
    train, test = problem['train'], problem['test']
    evidence = gh.EvidenceView(train['targets'], train['mask'], train['evidence_ids'], 'RNA', 'train',
                               train['times'], problem['audit']['source_sha256'])
    results = {}
    for name, model in (('structured_torch', PopulationRelation()),
                        ('snapshot_only', FlexibleControl(snapshot=True)),
                        ('flexible_same_information', FlexibleControl())):
        start = time.perf_counter()
        fitted = gh.fit_global(model, train['inputs'], evidence, gh.EvidencePermission(),
                               hypothesis_id=f'fixture-blind-population-{name}',
                               domain_ids=tuple(problem['feature_manifest']['inputs'] + problem['feature_manifest']['targets']),
                               steps=120, lr=.03)
        with torch.no_grad():
            test_loss = float(masked_mse(model(test['inputs']), test['targets'], test['mask']))
        results[name] = {'train_loss': fitted.manifest['final_loss'], 'test_loss': test_loss,
                         'CPU_fit_and_evaluation_seconds': time.perf_counter() - start,
                         'permitted_inputs': problem['feature_manifest']['inputs'] + ([] if name == 'snapshot_only' else ['released_time_label']),
                         'evidence_manifest': fitted.manifest}
    return results


def gpu_compare(problem):
    """Same graph, inputs, initialization, loss and Adam; complete cost interval."""
    from celleratorch import guarded_step
    import celleratorch
    if len(problem['train']['inputs']) > 8 or len(problem['test']['inputs']) > 8:
        raise ValueError('bounded native pilot capacity is eight observations per split')
    models, setup = {}, {}
    for name, native in (('torch', False), ('native', True)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        models[name] = PopulationRelation(native=native, device='cuda')
        torch.cuda.synchronize()
        setup[name] = time.perf_counter() - start
    # Objective VJP qualification before updates, including tail batch three.
    for split in ('train', 'test'):
        data = problem[split]
        gradients, outputs = {}, {}
        for name, model in models.items():
            xx = data['inputs'].cuda().requires_grad_()
            output = model(xx)
            loss = masked_mse(output, data['targets'].cuda(), data['mask'].cuda())
            outputs[name] = output.detach()
            planes = (xx, model.coefficients, *model.source_generator.parameters(), *model.target_generator.parameters())
            gradients[name] = torch.autograd.grad(loss, planes)
        torch.testing.assert_close(outputs['native'], outputs['torch'], rtol=2e-5, atol=2e-6)
        for actual, expected in zip(gradients['native'], gradients['torch']):
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    optimizers, optimizer_setup = {}, {}
    model_setup = dict(setup)
    for name, model in models.items():
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizers[name] = torch.optim.Adam(model.parameters(), lr=.03)
        torch.cuda.synchronize()
        optimizer_setup[name] = time.perf_counter() - start
        setup[name] += optimizer_setup[name]
    def iteration(name):
        data, model, optimizer = problem['train'], models[name], optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        # Explicit CPU-to-device conversion belongs to every complete iteration.
        inputs, target, mask = data['inputs'].cuda(), data['targets'].cuda(), data['mask'].cuda()
        loss = masked_mse(model(inputs), target, mask)
        loss.backward()
        if name == 'native':
            if not guarded_step(model, optimizer):
                raise RuntimeError('native pilot update was rejected')
        else:
            optimizer.step()
        torch.cuda.synchronize()
        return float(loss.detach().cpu())
    before = models['native'].coefficients.detach().clone()
    initial, final, costs = {}, {}, {}
    for name in models:
        torch.cuda.synchronize()
        first_start = time.perf_counter()
        initial[name] = iteration(name)
        first_seconds = time.perf_counter() - first_start
        for _ in range(119):
            final[name] = iteration(name)
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(5):
            final[name] = iteration(name)
        elapsed = time.perf_counter() - start
        costs[name] = {'setup_seconds': setup[name], 'five_complete_iterations_seconds': elapsed,
                       'model_setup_seconds': model_setup[name], 'optimizer_setup_seconds': optimizer_setup[name],
                       'first_training_iteration_after_equivalence_seconds': first_seconds,
                       'setup_plus_first_training_iteration_seconds': setup[name] + first_seconds,
                       'setup_plus_five_iterations_seconds': setup[name] + elapsed,
                       'mean_complete_iteration_seconds': elapsed / 5,
                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'baseline_allocated_bytes': baseline_allocated,
                       'peak_increment_bytes': torch.cuda.max_memory_allocated() - baseline_allocated,
                       'resident_models': 'both candidate graphs and optimizer states; memory is process allocation',
                       'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                       'timed_phases': ['CPU-to-CUDA conversion', 'activity generation', 'relation forward',
                                        'masked objective', 'backward', 'Adam/guarded publication', 'synchronize', 'scalar transfer']}
    if torch.equal(before, models['native'].coefficients):
        raise AssertionError('native coefficients were not trained')
    if final['native'] >= initial['native']:
        raise AssertionError('native pilot failed to lower its training loss')
    torch.testing.assert_close(models['native'].coefficients, models['torch'].coefficients, rtol=3e-4, atol=3e-5)
    for left, right in zip((*models['native'].source_generator.parameters(), *models['native'].target_generator.parameters()),
                           (*models['torch'].source_generator.parameters(), *models['torch'].target_generator.parameters())):
        torch.testing.assert_close(left, right, rtol=3e-4, atol=3e-5)
    with torch.no_grad():
        xx = problem['test']['inputs'].cuda()
        torch.testing.assert_close(models['native'](xx), models['torch'](xx), rtol=3e-4, atol=3e-5)
    return {'status': 'passed', 'initial_loss': initial, 'final_loss': final,
            'batch_sizes': [len(problem['train']['inputs']), len(problem['test']['inputs'])],
            'native_coefficients_trained': True, 'same_graph_forward_and_objective_vjp': True,
            'same_graph_adam_update': True,
            'initial_native_coefficients': before.cpu().tolist(),
            'fitted_native_coefficients': models['native'].coefficients.detach().cpu().tolist(),
            'costs': costs, 'disposition': 'evaluated_not_promoted',
            'cost_comparison': 'native_slower' if costs['native']['setup_plus_five_iterations_seconds'] >= costs['torch']['setup_plus_five_iterations_seconds'] else 'native_faster_in_bounded_fixture',
            'promotion_limit': 'small synthetic fixture; no general speedup or biological claim',
            'producer_package_sha256': hashlib.sha256(Path(celleratorch.__file__).read_bytes()).hexdigest(),
            'native_library_sha256': hashlib.sha256(Path(os.environ['CELLERATORCH_NATIVE_LIBRARY']).read_bytes()).hexdigest()}


def run(records, manifest, *, fixture=False, gpu=False, cutoff=12, gh_prefix='/tmp/gh-ml2-installed-sdk'):
    gh = installed_learning(gh_prefix)
    problem = prepare(records, manifest, cutoff)
    controls = train_controls(problem, gh)
    comparison = gpu_compare(problem) if gpu else {'status': 'not_run', 'reason': 'requires controller GPU lease'}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    source_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                     {'pilot': Path(__file__), 'test_pilot': Path(__file__).with_name('test_pilot.py'),
                      'data': Path(__file__).with_name('data.py'), 'reference': Path(__file__).with_name('reference.py'),
                      'learning': Path(gh.__file__)}.items()}
    return {'status': 'passed', 'protocol': 'blind_population_law', 'fixture': fixture,
            'source_hashes': source_hashes,
            'biological_status': 'not_run' if fixture else 'bounded_processed_input_fit',
            'scientific_scope': 'cross-sectional RNA population response; no individual trajectories or causal claim',
            'source_commit': commit, 'pilot_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'installed_gh_sha256': hashlib.sha256(Path(gh.__file__).read_bytes()).hexdigest(),
            'cutoff': cutoff, 'feature_manifest': problem['feature_manifest'],
            'source_sha256': problem['audit']['source_sha256'], 'coverage': problem['audit']['coverage'],
            'audit_exclusions': problem['audit']['training_exclusion_counts'], 'pilot_exclusions': problem['exclusions'],
            'clone_abundance': problem['clone_abundance'], 'controls': controls, 'native_comparison': comparison,
            'permissions': {'RNA_only': True, 'ATAC_withheld': True, 'clone_ID_predictor': False,
                            'clone_split_isolation_audited': True,
                            'training_only_feature_manifest': True, 'training_time_cutoff': cutoff},
            'unresolved': ['effective modules are not identified molecular regulation',
                           'duplicate endpoint coefficients identify only their sum',
                           'weight/activity scale transformations can preserve the response',
                           'population abundance mixes growth, survival, transitions and sampling',
                           'finite fits do not prove identifiability']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--fixture', action='store_true')
    source.add_argument('--records', type=Path)
    parser.add_argument('--feature-manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cutoff', type=float, default=12)
    parser.add_argument('--gpu', action='store_true', help='Requires root GPU lease; never substitutes a CPU result')
    parser.add_argument('--gh-prefix', default='/tmp/gh-ml2-installed-sdk')
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.fixture:
        records = args.output.with_suffix('.records.csv')
        fixture_records(records)
        result = run(records, FIXTURE_FEATURES, fixture=True, gpu=args.gpu,
                     cutoff=args.cutoff, gh_prefix=args.gh_prefix)
    else:
        if not args.records.is_file() or not args.feature_manifest or not args.feature_manifest.is_file():
            parser.error('biological_status=not_run: processed records and frozen feature manifest are required')
        result = run(args.records, json.loads(args.feature_manifest.read_text()), gpu=args.gpu,
                     cutoff=args.cutoff, gh_prefix=args.gh_prefix)
        records = args.records
    manifest_path = args.output.with_suffix('.manifest.json')
    manifest = {'format': 'celltag-pilot-evidence-v1', 'fixture': result['fixture'],
                'biological_status': result['biological_status'], 'protocol': result['protocol'],
                'source_commit': result['source_commit'],
                'pilot_source_sha256': result['pilot_source_sha256'],
                'source_hashes': result['source_hashes'],
                'data_source_sha256': result['source_hashes']['data'],
                'reference_source_sha256': result['source_hashes']['reference'],
                'installed_gh_sha256': result['installed_gh_sha256'],
                'records_path': str(records.resolve()), 'records_sha256': result['source_sha256'],
                'feature_manifest': result['feature_manifest'], 'permissions': result['permissions'],
                'test_source_sha256': hashlib.sha256(Path(__file__).with_name('test_pilot.py').read_bytes()).hexdigest(),
                'fit_evidence': {name: value['evidence_manifest']['evidence'] for name, value in result['controls'].items()},
                'native_comparison': result['native_comparison']}
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n')
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    result['manifest_path'], result['manifest_sha256'] = str(manifest_path.resolve()), manifest_sha
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    marker = {'result_path': str(args.output.resolve()), 'manifest_path': str(manifest_path.resolve()),
              'result_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'manifest_sha256': manifest_sha, 'records_sha256': result['source_sha256'],
              'pilot_source_sha256': result['pilot_source_sha256'], 'installed_gh_sha256': result['installed_gh_sha256'],
              'source_hashes': result['source_hashes'],
              'source_commit': result['source_commit']}
    print('GH_ML2_PILOT_EVIDENCE=' + json.dumps(marker, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
