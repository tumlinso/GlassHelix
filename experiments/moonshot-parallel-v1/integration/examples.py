"""Runnable synthetic plain and supplied-refactoring native GH consumers."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import torch


def _load(leaf, name):
    path = Path(__file__).resolve().parent.parent / leaf / 'model.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Each import loads the existing accepted owner, rather than copying its model.
plain = _load('plain', 'gh_native_plain')
actor = _load('actor', 'gh_native_actor')
refactor = _load('refactor', 'gh_native_refactor')
_spec = importlib.util.spec_from_file_location('gh_native_readout', Path(__file__).with_name('model.py'))
_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model)
NativeProductReadout = _model.NativeProductReadout


def plain_example(steps=12):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError('positive training steps required')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        field = plain.MatrixPatchField(1, 2)
        state = torch.tensor([[[0.8, -0.3], [0.4, 0.6]]], dtype=torch.float32)
        for _ in range(3):
            state = state + 0.05 * field(state)
        readout = NativeProductReadout(
            ('patch0:r0:c0', 'patch0:r0:c1', 'patch0:r1:c0', 'patch0:r1:c1'),
            ('synthetic:measurement0',), ('term0', 'term1', 'term2'),
            torch.tensor([0, 0, 3]), torch.tensor([0, 0, 0]),
            torch.tensor([0.1, -0.05, 0.2]))
        hidden = state.flatten()
        owners = (field.left, field.right, field.decay, readout.coefficients)
        native_gradients = torch.autograd.grad(readout(hidden).sum(), owners, retain_graph=True)
        reference = (readout.coefficients * hidden[torch.tensor([0, 0, 3])]).sum()
        reference_gradients = torch.autograd.grad(reference, owners)
        gradient_error = max(float((a - b).abs().max())
                             for a, b in zip(native_gradients, reference_gradients))
        field_gradient = float(native_gradients[0].abs().sum())
        hidden = hidden.detach()
        optimizer = torch.optim.SGD(readout.parameters(), lr=0.2)
        target = torch.tensor([1.25])
        before = float((readout(hidden) - target).square().sum().detach())
        for _ in range(steps):
            optimizer.zero_grad()
            loss = (readout(hidden) - target).square().sum()
            loss.backward()
            optimizer.step()
        after = float((readout(hidden) - target).square().sum().detach())
        primal, tangent = readout.jvp(hidden, torch.ones_like(hidden),
                                     torch.zeros_like(readout.coefficients))
        return {'example': 'plain_matrix_patch_native_readout', 'biology': 'not_run',
                'dtype': 'float32', 'device': 'cpu', 'trajectory_steps': 3,
                'loss_before': before, 'loss_after': after, 'training_steps': steps,
                'field_gradient_l1': field_gradient, 'trajectory_gradient_max_error': gradient_error,
                'prediction': primal.tolist(),
                'hidden_jvp': tangent.tolist(), 'source_ids': list(readout.source_ids),
                'observation_ids': list(readout.observation_ids),
                'contribution_ids': list(readout.contribution_ids)}


def refactorable_example(steps=2):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 2:
        raise ValueError('at least two steps required for live hidden gradients')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        old = refactor.LinearModel(torch.tensor([[-0.1, 0.0], [0.0, -0.2]]),
                                   torch.tensor([[0.7, -0.2]]))
        migration = refactor.LinearRefactoring.conjugate(
            old, torch.tensor([[1.0, 0.25], [0.0, 1.0]]),
            ('old:s0', 'old:s1'), ('new:q0', 'new:q1'), tolerance=1e-5)
        old_state = torch.tensor([0.8, -0.3])
        state = migration.initialize_state(old_state)
        branch = actor.RegrowthBranch(2, 2, 1)
        residual = refactor.ResidualActor(migration.model, branch, 'anonymous:regrowth0', 2)
        readout = NativeProductReadout(
            ('anonymous:regrowth0:h0', 'anonymous:regrowth0:h1'), ('synthetic:measurement0',),
            ('new-outgoing0', 'new-outgoing1'), torch.tensor([0, 1]), torch.tensor([0, 0]),
            torch.zeros(2))
        def predict():
            return residual.base.observe(state) + readout(residual.hidden(state))
        initial_error = float((predict() - old.observe(old_state)).abs().max().detach())
        optimizer = torch.optim.SGD([readout.coefficients, branch.inside.weight], lr=0.3)
        initial_outgoing_gradient = None
        hidden_gradients = []
        for step in range(steps):
            optimizer.zero_grad()
            loss = (predict() - torch.tensor([1.2])).square().sum()
            loss.backward()
            hidden_gradients.append(float(branch.inside.weight.grad.abs().sum()))
            if step == 0:
                initial_outgoing_gradient = float(readout.coefficients.grad.abs().sum())
            optimizer.step()
        return {'example': 'supplied_refactoring_native_regrowth', 'biology': 'not_run',
                'dtype': 'float32', 'device': 'cpu', 'migration_domain': migration.receipt.domain,
                'old_coordinates': list(migration.receipt.old_coordinates),
                'new_coordinates': list(migration.receipt.new_coordinates),
                'initial_prediction_error': initial_error,
                'initial_outgoing_gradient_l1': initial_outgoing_gradient,
                'hidden_gradient_l1_by_step': hidden_gradients,
                'final_prediction': predict().detach().tolist(),
                'source_ids': list(readout.source_ids), 'observation_ids': list(readout.observation_ids)}


if __name__ == '__main__':
    print(json.dumps({'plain': plain_example(), 'refactorable': refactorable_example()}, indent=2))
