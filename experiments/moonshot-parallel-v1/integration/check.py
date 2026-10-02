#!/usr/bin/env python3
"""Small actual installed-native consumer gate; --record alone writes evidence."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CE_COMMIT = '1688030c27bd942f6ac3bf40312ba06472c99dbf'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    receipt_dir = HERE.parent / 'receipt'
    manifest_path = receipt_dir / 'install-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    require(manifest['source_commit'] == CE_COMMIT, 'unexpected Cellerator pin')
    subprocess.run([sys.executable, '-B', str(receipt_dir / 'verify_consumer_evidence.py')], check=True)
    installed = Path(manifest['install_target'])
    sys.path.insert(0, str(installed))
    os.environ['CELLERATOR_PRODUCT2_LIBRARY'] = manifest['native_library']
    import torch
    from celleratorch import moonshot_product
    from model import NativeProductReadout
    import examples
    require(Path(moonshot_product.__file__).resolve() == (installed / 'celleratorch/moonshot_product.py').resolve(), 'wrong installed adapter')
    files = ['check.py', 'model.py', 'examples.py', 'capability.json',
             '../plain/model.py', '../actor/model.py', '../refactor/model.py']
    before_sources = {f: sha(HERE / f) for f in files}
    torch.set_num_threads(1)
    torch.manual_seed(7)
    checks = []

    def close(a, b, atol=2e-6, rtol=2e-6):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)

    def make(coefficients=None):
        return NativeProductReadout(('h0', 'h1', 'h2'), ('o0', 'o1'),
            ('c0', 'c1', 'c2', 'c3'), torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 1, 1]),
            torch.tensor([0.4, -0.1, 0., 0.7]) if coefficients is None else coefficients)

    def oracle(h, k):
        out = torch.zeros(2, dtype=h.dtype)
        return out.index_add(0, torch.tensor([0, 0, 1, 1]), k * h[torch.tensor([0, 0, 1, 2])])

    model = make()
    h = torch.tensor([1.2, -0.7, 0.5], requires_grad=True)
    close(model(h), oracle(h, model.coefficients))
    checks.append('forward_repeated_contributions')
    cotangent = torch.tensor([0.3, -0.8])
    gx, gk = torch.autograd.grad((model(h) * cotangent).sum(), (h, model.coefficients))
    hr = h.detach().clone().requires_grad_()
    kr = model.coefficients.detach().clone().requires_grad_()
    rx, rk = torch.autograd.grad((oracle(hr, kr) * cotangent).sum(), (hr, kr))
    close(gx, rx); close(gk, rk)
    require(abs(float(gk[2])) > 0, 'zero coefficient lost its gradient')
    checks.append('independent_input_and_coefficient_vjp_zero_coefficient')
    dx = torch.tensor([0.1, -0.2, 0.3]); dk = torch.tensor([-0.2, 0.1, 0.4, -0.1])
    y, tangent = model.jvp(h.detach(), dx, dk)
    close(y, oracle(h.detach(), model.coefficients.detach()))
    close(tangent, oracle(dx, model.coefficients.detach()) + oracle(h.detach(), dk))
    eps = 1e-3
    fd = (oracle(h.detach() + eps * dx, model.coefficients.detach() + eps * dk)
          - oracle(h.detach() - eps * dx, model.coefficients.detach() - eps * dk)) / (2 * eps)
    close(tangent, fd, atol=8e-5, rtol=8e-4)
    checks.append('full_readout_jvp_independent_and_finite_difference')
    target = torch.tensor([0.9, -0.4]); optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    def step(m, opt):
        opt.zero_grad(); loss = (m(h.detach()) - target).square().sum(); loss.backward(); opt.step(); return float(loss.detach())
    first = step(model, optimizer)
    for _ in range(19):
        last = step(model, optimizer)
    require(last < first * 0.25, 'native coefficients did not reduce loss')
    checks.append('native_coefficients_actual_loss_reduction')
    state = copy.deepcopy(model.state_dict()); optim_state = copy.deepcopy(optimizer.state_dict())
    restored = make(); restored.load_state_dict(state)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.04)
    restored_optimizer.load_state_dict(optim_state)
    a = step(model, optimizer); b = step(restored, restored_optimizer)
    close(model.coefficients, restored.coefficients, atol=0, rtol=0)
    require(a == b, 'checkpoint next-step loss changed')
    checks.append('checkpoint_adam_next_step_equivalence')
    for bad in (
        lambda: NativeProductReadout(('h0', 'h0'), ('o0',), ('c0',), torch.tensor([0]), torch.tensor([0]), torch.ones(1)),
        lambda: NativeProductReadout(('h0',), ('o0',), ('c0',), torch.tensor([1]), torch.tensor([0]), torch.ones(1)),
        lambda: NativeProductReadout(('h0',), ('o0',), ('c0',), torch.tensor([0]), torch.tensor([1]), torch.ones(1)),
        lambda: make().load_state_dict({**state, '_extra_state': {'invalid': True}}),
    ):
        try:
            bad()
        except (ValueError, RuntimeError, KeyError):
            pass
        else:
            raise ValueError('invalid identity/topology admitted')
    corrupt = copy.deepcopy(state)
    corrupt['native.a'][0] = 2
    unchanged = make()
    before = unchanged.coefficients.detach().clone()
    try:
        unchanged.load_state_dict(corrupt)
    except RuntimeError:
        close(unchanged.coefficients, before, atol=0, rtol=0)
    else:
        raise ValueError('checkpoint different topology admitted')
    for assign in (False, True):
        for bad_coefficients in (state['native.coefficients'].double(),
                                 torch.full_like(state['native.coefficients'], float('nan')),
                                 state['native.coefficients'][:-1]):
            corrupt = copy.deepcopy(state)
            corrupt['native.coefficients'] = bad_coefficients
            unchanged = make()
            snapshot = copy.deepcopy(unchanged.state_dict())
            try:
                unchanged.load_state_dict(corrupt, assign=assign)
            except (ValueError, RuntimeError):
                for key, before in snapshot.items():
                    after = unchanged.state_dict()[key]
                    if isinstance(before, torch.Tensor):
                        close(after, before, atol=0, rtol=0)
                    else:
                        require(after == before, 'checkpoint metadata changed after rejection')
            else:
                raise ValueError('invalid checkpoint coefficients admitted')
    checks.append('identity_topology_and_checkpoint_rejection')
    plain = examples.plain_example()
    refactor = examples.refactorable_example()
    require(plain['loss_after'] < plain['loss_before'] * 0.05, 'plain native readout did not fit')
    require(plain['trajectory_gradient_max_error'] < 2e-6, 'native/Torch trajectory gradients differ')
    require(plain['field_gradient_l1'] > 0, 'native readout did not backpropagate into field')
    require(refactor['initial_prediction_error'] < 2e-6, 'refactoring/regrowth changed initial observation')
    require(refactor['initial_outgoing_gradient_l1'] > 0, 'regrowth coefficient gradient lost')
    require(refactor['hidden_gradient_l1_by_step'][0] == 0 and refactor['hidden_gradient_l1_by_step'][1] > 0,
            'regrowth hidden gradients did not become live after coefficient update')
    checks.extend(['plain_matrix_patch_trajectory_native_readout', 'supplied_refactoring_residual_actor_native_readout'])
    require(before_sources == {f: sha(HERE / f) for f in files}, 'source changed during execution')
    require(sha(manifest['native_library']) == manifest['native_library_sha256'], 'native library changed during execution')
    result = {'schema_version': 1, 'status': 'passed', 'checks': checks,
              'source_sha256': {f: sha(HERE / f) for f in files},
              'cellerator_source_commit': CE_COMMIT,
              'install_manifest_sha256': sha(manifest_path),
              'adapter_sha256': sha(moonshot_product.__file__),
              'native_library': manifest['native_library'],
              'native_library_sha256': sha(manifest['native_library']),
              'device': 'cpu', 'precision': 'float32', 'biology': 'not_run',
              'argv': [sys.executable, '-B', str(HERE / 'check.py')] + (['--record'] if args.record else []),
              'examples': {'plain': plain, 'refactorable': refactor},
              'coefficient_fit': {'initial_loss': first, 'final_loss': last}}
    evidence = HERE / 'evidence/consumer-checks.json'
    if args.record:
        evidence.parent.mkdir(exist_ok=True)
        evidence.write_text(json.dumps(result, indent=2) + '\n')
    else:
        previous = json.loads(evidence.read_text())
        for key in ('source_sha256', 'cellerator_source_commit', 'install_manifest_sha256', 'adapter_sha256', 'native_library_sha256', 'checks'):
            require(previous[key] == result[key], 'recorded evidence differs: ' + key)
    print(json.dumps(result, sort_keys=True))

if __name__ == '__main__':
    main()
