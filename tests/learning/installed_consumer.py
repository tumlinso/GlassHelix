"""Installed GH/Cellerator training, owner restore and fresh-process replay.

Run --gpu only under the controller's assigned GPU lease. No skip can pass.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch import nn

def make_model():
    from cellerator import (Axis, Identity, Incidence, Mechanism,
                            MechanismSpec, OutputContribution, prepare_mechanism)
    from cellerator.torch import MechanismModule
    def identity(n):
        return Identity(n, n + 1000)
    def axis(base, extent):
        return Axis(identity(base), identity(base + 1), identity(base + 2), identity(base + 3), extent)
    spec = MechanismSpec(axis(101, 2), axis(201, 1), axis(301, 1), (identity(401),),
                         (Mechanism(identity(501), identity(401),
                                    (Incidence(0, identity(601), 0), Incidence(1, identity(602), 1)),
                                    (OutputContribution(0, identity(701), 0, identity(801), 1.0),)),))
    handle = prepare_mechanism(
        spec, np.ascontiguousarray([0.1], dtype=np.float32), device='cuda:0',
        max_batch=8, max_live_forwards=8, precision='f32')
    native = MechanismModule.from_handle(handle)
    before = nn.Linear(2, 2, device='cuda')
    after = nn.Linear(1, 1, device='cuda')
    with torch.no_grad():
        before.weight.copy_(torch.eye(2, device='cuda'))
        before.bias.zero_()
        after.weight.fill_(1.)
        after.bias.zero_()
    for parameter in (*before.parameters(), *after.parameters()):
        parameter.requires_grad_(False)
    return nn.Sequential(before, native, after)


def view(values, *, split, kind='response'):
    tensor = torch.tensor(values, device='cuda', dtype=torch.float32).reshape(-1, 1)
    return EvidenceView(tensor, torch.ones_like(tensor, dtype=torch.bool),
                        tuple(f'synthetic-{kind}-evidence-{i}' for i in range(len(tensor))),
                        'RNA', split, tuple(3. for _ in range(len(tensor))), 'controlled-product-fixture-v1',
                        tuple(f'fixture{i}' for i in range(len(tensor))))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def installed_cellerator_artifacts():
    distribution = metadata.distribution('cellerator')
    artifacts = {}
    artifact_paths = set()
    for relative in distribution.files or ():
        relative = Path(relative)
        if relative.parts[:1] != ('cellerator',):
            continue
        installed = Path(distribution.locate_file(relative)).resolve()
        if installed.is_file():
            artifacts[str(relative)] = sha(installed)
            artifact_paths.add(installed)
    if not artifacts or not any(Path(name).name.startswith('_native') for name in artifacts):
        raise RuntimeError('installed Cellerator package files or core extension are missing')
    return distribution, artifacts, artifact_paths


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def replay(checkpoint, manifest_path, output):
    import cellerator
    from cellerator import _native as cellerator_native
    from cellerator.torch import _torch as cellerator_torch_native
    from cellerator.torch import load_checkpoint
    import cellerator.torch as cellerator_torch_adapter
    distribution, artifacts, artifact_paths = installed_cellerator_artifacts()
    manifest = json.loads(manifest_path.read_text())
    if sha(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('checkpoint identity differs from scientific manifest')
    if sha(learning_module.__file__) != manifest['learning_source_sha256']:
        raise ValueError('installed GH learning source differs from fitted source')
    imported_modules = (cellerator.__file__, cellerator_native.__file__,
                        cellerator_torch_adapter.__file__, cellerator_torch_native.__file__)
    if any(Path(module_path).resolve() not in artifact_paths for module_path in imported_modules):
        raise RuntimeError('imported Cellerator module is not listed in the installed distribution')
    if (sha(__file__) != manifest['consumer_source_sha256']
            or sha(cellerator.__file__) != manifest['cellerator_source_sha256']
            or sha(cellerator_native.__file__) != manifest['native_library_sha256']
            or sha(cellerator_torch_native.__file__) != manifest['torch_adapter_sha256']
            or artifacts != manifest['cellerator_artifacts_sha256']
            or distribution.version != manifest['cellerator_version']):
        raise ValueError('consumer or native producer differs from fitted source')
    model = make_model()
    optimizer = torch.optim.Adam([model[1].coefficients], lr=0.05)
    load_checkpoint(checkpoint, model, optimizer)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    values = manifest['bridge']
    x = torch.tensor([[values['input_x'], values['local_z'] * values['forcing']]], device='cuda', requires_grad=True)
    prediction = model(x)
    prediction.sum().backward()
    # Input second coordinate is z*forcing, so apply the declared chain rule.
    derivative_z = float(x.grad[0, 1]) * values['forcing']
    torch.testing.assert_close(prediction.detach().cpu(), torch.tensor([[values['expected_output']]]), rtol=2e-5, atol=2e-6)
    if abs(derivative_z - values['derivative_z']) > 2e-5:
        raise AssertionError('fresh-process derivative replay differs')
    torch.cuda.synchronize()
    write_json(output, {'ok': True, 'fresh_process': True, 'prediction': float(prediction.detach()[0, 0]),
                        'derivative_z': derivative_z, 'checkpoint_sha256': sha(checkpoint)})


def qualify(output, gh_prefix):
    import cellerator
    from cellerator import _native as cellerator_native
    from cellerator.torch import _torch as cellerator_torch_native
    from cellerator.torch import guarded_step, save_checkpoint
    import cellerator.torch as cellerator_torch_adapter
    distribution, artifacts, artifact_paths = installed_cellerator_artifacts()
    imported_modules = (cellerator.__file__, cellerator_native.__file__,
                        cellerator_torch_adapter.__file__, cellerator_torch_native.__file__)
    if any(Path(module_path).resolve() not in artifact_paths for module_path in imported_modules):
        raise RuntimeError('imported Cellerator module is not listed in the installed distribution')
    model = make_model()
    inputs = torch.tensor([[0.5, 1.], [1., 0.6], [1.5, 0.8]], device='cuda')
    calibration = view((1.7 * inputs[:, 0] * inputs[:, 1]).tolist(), split='train')
    # Compare the complete reduced objective, including input and native k VJPs.
    differentiable_inputs = inputs.detach().clone().requires_grad_()
    prediction = model(differentiable_inputs)
    objective = (prediction - calibration.values).square().mean()
    actual = torch.autograd.grad(objective, (differentiable_inputs, model[1].coefficients))
    ref_inputs = inputs.detach().cpu().double().requires_grad_()
    ref_k = model[1].coefficients.detach().cpu().double().requires_grad_()
    ref_target = calibration.values.cpu().double()
    ref_objective = (ref_k * ref_inputs[:, :1] * ref_inputs[:, 1:2] - ref_target).square().mean()
    expected_gradients = torch.autograd.grad(ref_objective, (ref_inputs, ref_k))
    for observed, expected in zip(actual, expected_gradients):
        torch.testing.assert_close(observed.detach().cpu().double(), expected, rtol=2e-5, atol=2e-6)
    from dataclasses import replace
    for forbidden in (replace(calibration, split='test'), replace(calibration, modality='ATAC')):
        try:
            EvidencePermission().authorize(forbidden, 'global_fit')
        except PermissionError:
            pass
        else:
            raise AssertionError('held-out evidence was authorized')
    fitted = fit_global(model, inputs, calibration, EvidencePermission(),
                        hypothesis_id='synthetic-product-7001-v1',
                        domain_ids=('source-axis-101:x,z*forcing', 'target-axis-201:response', 'parameter-axis-301:k'),
                        steps=400, guarded_step=guarded_step)
    if fitted.manifest['final_loss'] > 1e-10:
        raise AssertionError('shared native coefficient did not fit calibration')
    shared_k = float(model[1].coefficients.detach()[0])
    torch.testing.assert_close(model[1].coefficients.detach(), torch.tensor([1.7], device='cuda'), rtol=2e-5, atol=2e-6)
    x = torch.tensor([[1.3]], device='cuda')
    local = condition_local(fitted, x, view([1.7 * 1.3 * 0.8], split='conditioning'),
                            EvidencePermission(cutoff=3),
                            independent_z=view([0.8], split='conditioning', kind='activity'))
    if local['residual'] > 1e-9 or abs(float(local['z'][0, 0]) - 0.8) > 2e-5:
        raise AssertionError('frozen-law local conditioning did not recover observed activity')
    # Native owner checkpoint is authoritative; model state is not copied directly.
    checkpoint = output.with_suffix('.pt')
    save_checkpoint(checkpoint, model, fitted.optimizer)
    z, forcing = float(local['z'][0, 0]), float(local['forcing'][0, 0])
    z_variable = torch.tensor([[z]], device='cuda', requires_grad=True)
    prediction = model(torch.cat((x, z_variable * forcing), dim=1))
    prediction.sum().backward()
    expected, derivative = float(prediction.detach()[0, 0]), float(z_variable.grad[0, 0])
    if abs(expected - shared_k * float(x[0, 0]) * z * forcing) > 2e-6:
        raise AssertionError('composed native forward differs from product')
    if abs(derivative - shared_k * float(x[0, 0]) * forcing) > 2e-6:
        raise AssertionError('composed native derivative differs from product')
    torch.cuda.synchronize()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    bridge = dict(shared_coefficient=shared_k, input_x=float(x[0, 0]), local_z=z,
                  forcing=forcing, expected_output=expected, derivative_z=derivative,
                  observed_mask=1, model_id=7001, model_revision=1, domain_id=101,
                  order_id=102, geometry_id=103, partition_id=104, structure_epoch=1,
                  source_extent=2, target_domain_id=201, target_order_id=202,
                  target_geometry_id=203, target_partition_id=204, target_extent=1,
                  parameter_domain_id=301, parameter_order_id=302,
                  parameter_geometry_id=303, parameter_partition_id=304,
                  parameter_extent=1, coefficient_id=401,
                  hypothesis_label='synthetic-product-7001-v1',
                  evidence_id=8001, structure_id=105, mechanism_id=501, mechanism_label='joint_product',
                  units='arbitrary', scale=1, state_generation=1,
                  parameter_generation=int(model[1]._native.generation), tolerance=2e-5,
                  modality='synthetic', time_units='hours', sampling_id='fixture0', time=3)
    native_library = Path(cellerator_native.__file__)
    manifest = dict(fitted.manifest, source_commit=commit, bridge=bridge,
                    identity_mapping={'hypothesis': {'label': 'synthetic-product-7001-v1', 'system_id': 7001},
                                      'mechanism': {'label': 'joint_product', 'identity': [501, 1501]},
                                      'coefficient': {'label': 'shared_k', 'identity': [401, 1401]},
                                      'source_axis': {'domain': [101, 1101], 'order': [102, 1102],
                                                      'geometry': [103, 1103], 'partition': [104, 1104],
                                                      'slots': ['input_x', 'local_z*forcing']},
                                      'target_axis': {'domain': [201, 1201], 'order': [202, 1202],
                                                      'geometry': [203, 1203], 'partition': [204, 1204],
                                                      'slots': ['response']},
                                      'parameter_axis': {'domain': [301, 1301], 'order': [302, 1302],
                                                         'geometry': [303, 1303], 'partition': [304, 1304],
                                                         'slots': ['shared_k']}},
                    checkpoint_sha256=sha(checkpoint), learning_source_sha256=sha(learning_module.__file__),
                    consumer_source_sha256=sha(__file__), gh_prefix=str(gh_prefix) if gh_prefix else None,
                    cellerator_source_sha256=sha(cellerator.__file__),
                    native_library_sha256=sha(native_library),
                    torch_adapter_sha256=sha(cellerator_torch_native.__file__),
                    cellerator_version=distribution.version,
                    cellerator_artifacts_sha256=artifacts,
                    local_conditioning={key: value for key, value in local.items() if key not in ('z', 'forcing', 'prediction')})
    manifest_path = output.with_suffix('.manifest.json')
    write_json(manifest_path, manifest)
    bridge_path = output.with_suffix('.bridge-values.txt')
    bridge_path.write_text(''.join(f'{key} {value}\n' for key, value in bridge.items())
                           + f"source_sha256 {manifest['learning_source_sha256']}\n"
                           + f"checkpoint_sha256 {manifest['checkpoint_sha256']}\n")
    replay_path = output.with_suffix('.replay.json')
    replay_command = [sys.executable, '-B', str(Path(__file__).resolve()), '--gpu',
                    '--replay', str(checkpoint.resolve()), '--manifest', str(manifest_path.resolve()),
                    '--output', str(replay_path.resolve())]
    if gh_prefix:
        replay_command.extend(['--gh-prefix', str(gh_prefix.resolve())])
    subprocess.run(replay_command, check=True, env=os.environ.copy())
    write_json(output, {'ok': True, 'status': 'passed', 'native_joint_product_trained': True,
                        'checks': dict(native_training=True, global_fit=True, frozen_conditioning=True,
                                       complete_objective_gradient=True, checkpoint_fresh_process=True,
                                       evidence_permissions=True),
                        'biological_status': 'not_run',
                        'source_hashes': {'learning': manifest['learning_source_sha256'],
                                          'consumer': manifest['consumer_source_sha256']},
                        'checkpoint_sha256': manifest['checkpoint_sha256'],
                        'producer_package_sha256': manifest['cellerator_source_sha256'],
                        'native_library_sha256': manifest['native_library_sha256'],
                        'torch_adapter_sha256': manifest['torch_adapter_sha256'],
                        'cellerator_version': manifest['cellerator_version'],
                        'cellerator_artifacts_sha256': manifest['cellerator_artifacts_sha256'],
                        'bridge': bridge, 'identity_mapping': manifest['identity_mapping'],
                        'source_commit': commit, 'initial_loss': fitted.manifest['initial_loss'],
                        'final_loss': fitted.manifest['final_loss'], 'shared_coefficient': shared_k,
                        'global_frozen': True, 'local_residual': local['residual'],
                        'checkpoint': str(checkpoint), 'manifest': str(manifest_path),
                        'bridge_values': str(bridge_path), 'fresh_process_replay': str(replay_path),
                        'scope': 'synthetic installed consumer; biological validation not_run'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--gh-prefix', type=Path, help='Installed GH SDK prefix; imports PREFIX/lib')
    args = parser.parse_args()
    if not args.gpu:
        parser.error('--gpu requires controller authorization')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.gh_prefix:
        sys.path.insert(0, str(args.gh_prefix.resolve() / 'lib'))
    global learning_module, EvidencePermission, EvidenceView, condition_local, fit_global
    learning_module = importlib.import_module('glasshelix.learning')
    if args.gh_prefix and not Path(learning_module.__file__).resolve().is_relative_to(args.gh_prefix.resolve() / 'lib'):
        raise RuntimeError('GH learning import is outside the installed prefix')
    EvidencePermission = learning_module.EvidencePermission
    EvidenceView = learning_module.EvidenceView
    condition_local = learning_module.condition_local
    fit_global = learning_module.fit_global
    if args.replay:
        if not args.manifest:
            parser.error('--replay requires --manifest')
        replay(args.replay, args.manifest, args.output)
    else:
        qualify(args.output, args.gh_prefix)


if __name__ == '__main__':
    main()
