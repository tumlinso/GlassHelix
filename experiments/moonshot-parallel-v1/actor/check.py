"""CPU derivative and learning witnesses; synthetic engineering evidence only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.func import functional_call
from model import ActorPortField, RegrowthBranch


def gradcheck(model, state, observation=False):
    parameters = dict(model.named_parameters())
    names = list(parameters)
    values = tuple(p.detach().clone().requires_grad_() for p in parameters.values())
    def action(x, *weights):
        replacement = dict(zip(names, weights))
        result = functional_call(model, replacement, (x,))
        if observation:
            # Forward-state observation exercises readout via functional_call too.
            return result, (result * replacement['readout']).sum(-1)
        return result
    assert torch.autograd.gradcheck(action, (state.requires_grad_(), *values))
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = ActorPortField(3, 2, 2, ([0, 0, 1, 2], [1, 1, 2, 0])).double()
    state = torch.randn(3, 2, dtype=torch.float64) * .2
    checked = gradcheck(model, state.clone(), observation=True)
    assert model.observe(state).shape == (3,)
    torch.testing.assert_close(model.observe(state), (state * model.readout).sum(-1))
    ports = torch.einsum('ih,ihp->ip', state, model.encoder)
    received = torch.zeros_like(ports)
    for i, (source, destination) in enumerate(zip(model.source, model.destination)):
        received[destination] += model.edge[i] * ports[source]
    expected = torch.tanh(torch.einsum('ih,ihk->ik', state, model.local)) - state
    expected += torch.einsum('ip,iph->ih', received, model.decoder)
    torch.testing.assert_close(model(state), expected, rtol=0, atol=0)
    assert all(torch.equal(model(state), model(state)) for _ in range(5))
    empty = ActorPortField(3, 2, 2, ([], [])).double()
    torch.testing.assert_close(empty(state), torch.tanh(torch.einsum(
        'ih,ihk->ik', state, empty.local)) - state)
    noncontiguous = torch.randn(2, 3, dtype=torch.float64).t()
    gradcheck(model, noncontiguous, observation=True)
    invalid_edges = [([0.5], [1]), ([0], [3]), ([0, 1], [1]), ([[0]], [[1]])]
    for edges in invalid_edges:
        try:
            ActorPortField(3, 2, 2, edges)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid topology accepted')
    for action in (lambda: model(torch.ones(2, 2)),
                   lambda: model.observe(torch.ones(2, 2)),
                   lambda: RegrowthBranch(0, 2, 2)):
        try:
            action()
        except ValueError:
            pass
        else:
            raise AssertionError('invalid dimension accepted')
    branch = RegrowthBranch(2, 3, 2).double()
    branch_checked = gradcheck(branch, state.clone())
    assert torch.count_nonzero(branch.inside.weight) > 0
    assert torch.equal(branch(state), torch.zeros_like(state))
    branch(state).sum().backward()
    outgoing_norm = branch.outside.weight.grad.norm().item()
    assert outgoing_norm > 0
    assert torch.count_nonzero(branch.inside.weight.grad) == 0
    with torch.no_grad():
        branch.outside.weight.add_(-.1 * branch.outside.weight.grad)
    branch.zero_grad(set_to_none=True)
    branch(state).square().sum().backward()
    feature_norm = branch.inside.weight.grad.norm().item()
    assert feature_norm > 0
    branch_learn = RegrowthBranch(2, 4, 2).double()
    inputs = torch.randn(24, 2, dtype=torch.float64) * .3
    target = torch.tanh(inputs @ torch.tensor([[.8, -.3], [.2, .6]], dtype=torch.float64))
    optimizer = torch.optim.Adam(branch_learn.parameters(), lr=.03)
    first_regrowth = (branch_learn(inputs) - target).square().mean().item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = (branch_learn(inputs) - target).square().mean()
        loss.backward()
        optimizer.step()
    last_regrowth = (branch_learn(inputs) - target).square().mean().item()
    assert last_regrowth < first_regrowth * .1
    teacher = ActorPortField(3, 2, 2, ([0, 0, 1, 2], [1, 1, 2, 0])).double()
    student = ActorPortField(3, 2, 2, ([0, 0, 1, 2], [1, 1, 2, 0])).double()
    def rollout(field):
        x = state.detach()
        history = []
        for _ in range(4):
            x = x + .05 * field(x)
            history.append(x)
        return torch.stack(history)
    with torch.no_grad():
        trajectory = rollout(teacher)
        observed = teacher.observe(trajectory[-1])
    def actor_loss():
        predicted = rollout(student)
        return (predicted - trajectory).square().mean() + (
            student.observe(predicted[-1]) - observed).square().mean()
    first_actor = actor_loss().item()
    optimizer = torch.optim.Adam(student.parameters(), lr=.02)
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = actor_loss()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in student.parameters())
        optimizer.step()
    last_actor = actor_loss().item()
    assert last_actor < first_actor * .1
    leaf = Path(__file__).resolve().parent
    receipt = {
        'status': 'passed', 'device': 'cpu', 'dtype': 'float64',
        'torch_version': torch.__version__,
        'models': {'ActorPortField': {'callable': 'actor/model.py:ActorPortField',
                   'state_shape': [3, 2], 'port_shape': [3, 2], 'edges': 4,
                   'parameter_gradchecks': checked},
                   'RegrowthBranch': {'callable': 'actor/model.py:RegrowthBranch',
                   'input_shape': [3, 2], 'capacity': 3, 'output_shape': [3, 2],
                   'parameter_gradchecks': branch_checked}},
        'checks': ['input_and_parameter_gradcheck', 'noncontiguous_input_gradcheck',
                   'repeated_edges_sum', 'repeated_cpu_forward_determinism',
                   'empty_topology', 'invalid_shapes_and_topology',
                   'zero_initial_regrowth_effect', 'live_outgoing_gradient',
                   'subsequent_feature_gradient', 'actor_rollout_learning', 'regrowth_learning'],
        'outgoing_gradient_norm': outgoing_norm, 'subsequent_feature_gradient_norm': feature_norm,
        'learning': {'actor': [first_actor, last_actor], 'regrowth': [first_regrowth, last_regrowth],
                     'steps': 80},
        'source_sha256': {name: hashlib.sha256((leaf / name).read_bytes()).hexdigest()
                          for name in ('model.py', 'check.py')},
        'seed_sha256': hashlib.sha256((leaf.parent / 'seeds/glasshelix/actor_fields.py').read_bytes()).hexdigest(),
        'invocation': "CUDA_VISIBLE_DEVICES='' /home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python experiments/moonshot-parallel-v1/actor/check.py --receipt experiments/moonshot-parallel-v1/actor/receipt.json",
        'unsupported': ['CUDA qualification', 'native CE execution', 'arbitrary nonlinear basis conjugacy',
                        'biological identification', 'performance claims'],
        'meaning': 'Synthetic engineering witnesses on whole supplied models; no biological evidence.'
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + '\n'
    if args.receipt:
        args.receipt.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
