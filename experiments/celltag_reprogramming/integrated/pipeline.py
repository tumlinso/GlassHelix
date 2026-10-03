"""Synthetic composition of accepted experiment, data, model and science owners."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from torch import nn

from glasshelix.learning import EvidenceView, EvidencePermission, fit_global, masked_loss
from glasshelix.models import ModelClient, ModelDeclaration, load_predecessor
from glasshelix.experiment import ExperimentSpecification, Quantity, Provenance, ScientificResult
from glasshelix.refactoring import StructuralProposal
from experiments.celltag_reprogramming import pilot
from experiments.celltag_reprogramming.data_adapter import load_cohort, replay_cohort

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SnapshotBatch(nn.Module):
    """Explicit fixed fixture state/readout map; each predecessor still owns math."""
    def __init__(self, client):
        super().__init__()
        self.client = client

    def state(self, row):
        family = self.client.declaration.family
        return row.reshape(1, 2, 2) if family == 'matrix_patch' else row.reshape(2, 2)

    def forward(self, inputs):
        rows = []
        for row in inputs:
            state = self.state(row)
            family = self.client.declaration.family
            value = self.client(state, .2) if family == 'sylvester_flow' else self.client(state)
            # Supplied readouts over anonymous coordinates; RNA axes are not inferred.
            rows.append(self.client.observe(value) if family == 'actor_port'
                        else value.reshape(4)[:2])
        return torch.stack(rows)


def model(name):
    if name == 'population_relation':
        owner = pilot.PopulationRelation().double()
        shape, domains, family, batched = (4,), ('fixture-input',), name, True
    else:
        family, batched = name, False
        source = ROOT / ('experiments/moonshot-parallel-v1/actor/model.py' if name == 'actor_port'
                         else 'experiments/moonshot-parallel-v1/plain/model.py')
        kind = load_predecessor(source, name, expected_sha256=sha(source))
        owner = (kind(2, 2, 1, ([0, 1], [1, 0])) if name == 'actor_port'
                 else kind(1, 2) if name == 'matrix_patch' else kind(2)).double()
        shape = (1, 2, 2) if name == 'matrix_patch' else (2, 2)
        domains = ('fixture-actor-0', 'fixture-actor-1') if name == 'actor_port' else ('fixture-matrix',)
    source = Path(pilot.__file__) if name == 'population_relation' else source
    declaration = ModelDeclaration(name, name + '-whole', name + '-mechanism', family, shape,
        domains, tuple((n, name + ':' + n) for n, _ in owner.named_parameters()), sha(source), batched)
    client = ModelClient(owner, declaration)
    return client if batched else SnapshotBatch(client)


def evidence(data, split, source):
    return EvidenceView(data['targets'].double(), data['mask'], data['evidence_ids'], 'RNA', split,
                        data['times'], source)


def run(output, native_refactor, ce_commit, gh_commit):
    """CPU fit/evaluate/restore/recompute; no download or GPU route."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = output / 'fixture.records.csv'
    pilot.fixture_records(records)
    problem = pilot.prepare(records, pilot.FIXTURE_FEATURES)
    cohort = load_cohort(records, source_id=problem['audit']['source_sha256'],
                         audit_evidence_id='fixture-audit',
                         features=pilot.FIXTURE_FEATURES['inputs'] + pilot.FIXTURE_FEATURES['targets'])
    (output / 'cohort.json').write_text(cohort.dumps())
    assert replay_cohort(cohort.dumps(), records).dumps() == cohort.dumps()
    # The DATA owner retains clone partitions and masks; verify the pilot agrees.
    train = problem['train']; test = problem['test']
    time_map = {s.time_label: float(s.time_label) for s in cohort.view('train').snapshots}
    time_map.update({s.time_label: float(s.time_label) for s in cohort.view('test').snapshots})
    target_features = pilot.FIXTURE_FEATURES['targets']
    for split in ('train', 'test'):
        converted = cohort.view(split, features=target_features).to_evidence(time_map, 'released_label')
        torch.testing.assert_close(converted.values, problem[split]['targets'].double(), rtol=1e-6, atol=1e-7)
        assert torch.equal(converted.observed, problem[split]['mask'])
    assert not set(train['evidence_ids']) & set(test['evidence_ids'])
    train_view = evidence(train, 'train', cohort.source.source_id)
    test_view = evidence(test, 'test', cohort.source.source_id)
    permission = EvidencePermission()
    permission_rejections = []
    for forbidden in (test_view, replace(train_view, modality='ATAC')):
        try:
            permission.authorize(forbidden, 'global_fit')
        except PermissionError:
            permission_rejections.append(forbidden.split + ':' + forbidden.modality)
        else:
            raise AssertionError('held-out/modality fitting permission accepted')
    reports, fitted_models = {}, {}
    with torch.random.fork_rng():
        torch.manual_seed(91)
        candidates = {name: model(name) for name in
                      ('population_relation', 'matrix_patch', 'sylvester_flow', 'actor_port')}
        candidates.update(snapshot_only=pilot.FlexibleControl(snapshot=True).double(),
                          flexible_same_information=pilot.FlexibleControl().double())
        for name, candidate in candidates.items():
            fitted = fit_global(candidate, train['inputs'].double(), train_view, permission,
                                hypothesis_id=name + '-whole', domain_ids=('fixture-RNA-response',),
                                steps=80, lr=.02)
            # Predecessor learning metadata names its original product fixture;
            # state the actual current family explicitly instead of inheriting that claim.
            fitted.manifest.update(question='synthetic RNA population response fixture',
                assumptions=['explicit fixture input/state/readout map', 'owner ' + name,
                             'no molecular identity or individual histories'])
            with torch.no_grad():
                prediction = candidate(test['inputs'].double()).clone()
                score = float(masked_loss(prediction, test_view))
            assert torch.isfinite(prediction).all() and fitted.manifest['final_loss'] < fitted.manifest['initial_loss']
            checkpoint = output / (name + '.pt')
            torch.save({'model': candidate.state_dict(), 'optimizer': fitted.optimizer.state_dict()}, checkpoint)
            # Fresh owner with the same scientific declaration; actual disk restore.
            restored = (model(name) if name in fitted_models or name in
                        ('population_relation', 'matrix_patch', 'sylvester_flow', 'actor_port')
                        else pilot.FlexibleControl(snapshot=name == 'snapshot_only').double())
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if isinstance(restored, ModelClient):
                restored.restore(saved['model'])
            elif isinstance(restored, SnapshotBatch):
                child = {key.removeprefix('client.'): value for key, value in saved['model'].items()}
                restored.client.restore(child)
            else:
                restored.load_state_dict(saved['model'])
            optimizer = torch.optim.Adam(restored.parameters(), lr=.02)
            optimizer.load_state_dict(saved['optimizer'])
            with torch.no_grad():
                replayed = restored(test['inputs'].double())
            torch.testing.assert_close(replayed, prediction, rtol=0, atol=0)
            # Resuming one training step from disk preserves optimizer state too.
            for current, opt in ((candidate, fitted.optimizer), (restored, optimizer)):
                opt.zero_grad(set_to_none=True)
                masked_loss(current(train['inputs'].double()), train_view).backward()
                opt.step()
            for left, right in zip(candidate.parameters(), restored.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            reports[name] = {'fit': fitted.manifest, 'test_loss': score,
                             'prediction': prediction.tolist(), 'checkpoint_sha256': sha(checkpoint),
                             'checkpoint_recompute': True, 'optimizer_resume': True}
            fitted_models[name] = restored
            if isinstance(candidate, (ModelClient, SnapshotBatch)):
                client = candidate if isinstance(candidate, ModelClient) else candidate.client
                state = test['inputs'][:1].double() if name == 'population_relation' else candidate.state(test['inputs'][0].double())
                hypothesis = client.to_hypothesis(state, 1.)
                quantities = (Quantity('state', 'state', state.numel(), 'anonymous'),
                    Quantity('RNA-response', 'measurement', 2, 'fixture'),
                    *(Quantity(identity, 'parameter', client.owner.get_parameter(key).numel(), 'declared')
                      for key, identity in client.declaration.parameter_ids))
                specification = ExperimentSpecification(name, name, quantities, (hypothesis,),
                    cohort.source.source_id, 'fixture-audit', (problem['feature_manifest']['sha256'],))
                with torch.no_grad(): current_prediction = candidate(test['inputs'][:1].double())[0]
                result = ScientificResult(specification, (hypothesis,),
                    (client.to_prediction(current_prediction, 'RNA-response'),),
                    Provenance(name, '1', gh_commit, ce_commit, 'torch_cpu_f64', 1, 1, 81,
                               ('fixture-audit',)))
                result_path = output / (name + '.result.json')
                result_path.write_text(result.dumps())
                assert ScientificResult.loads(result_path.read_text()) == result
                torch.testing.assert_close(torch.tensor(result.predictions[0].values, dtype=torch.float64),
                                           restored(test['inputs'][:1].double())[0], rtol=0, atol=0)
    linear = fitted_models['sylvester_flow'].client.owner
    state = test['inputs'][0].double().reshape(2, 2)
    parameters = (*linear.left.detach().reshape(-1).tolist(), *linear.right.detach().reshape(-1).tolist(),
                  *state.reshape(-1).tolist())
    native = subprocess.run([str(native_refactor)], input=' '.join(map(repr, parameters)),
                            text=True, capture_output=True, check=True)
    receipt = json.loads(native.stdout)
    proposal = StructuralProposal('supplied-permutation', 'sylvester_flow-whole', 1, 2,
        tuple('old-' + str(i) for i in range(4)), tuple('new-' + str(i) for i in range(4)),
        'supplied_basis', 'reverse-four', 'conjugate-linear-generator', 'inverse-readout',
        'mapped-state', 'reset_transformed_parameter_moments', 'drain_old_readers',
        ('old-tape',), 'CE-adaptive-sidecar', 'whole state space',
        ('linear autonomous flow only; nonlinear matrix/actor refactoring unqualified',))
    proposal.validate()
    summary = {'status': 'passed', 'biological_status': 'not_run', 'native_cuda': 'not_run',
        'gh_source_commit': gh_commit, 'ce_source_commit': ce_commit,
        'scope': 'synthetic CPU composition and native host linear refactoring',
        'controls': reports, 'permission_rejections': permission_rejections,
        'partition': {'train': list(train['evidence_ids']), 'test': list(test['evidence_ids'])},
        'feature_manifest': problem['feature_manifest'], 'refactoring': receipt,
        'proposal': asdict(proposal), 'records_sha256': sha(records),
        'limits': ['no released dataset or biological fit', 'anonymous fixture coordinate/readout maps',
                   'no nonlinear refactoring or CUDA qualification', 'no performance comparison']}
    (output / 'summary.json').write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return summary
