import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import unittest

import torch

from glasshelix.models import ModelClient, ModelDeclaration, load_predecessor
from glasshelix.learning import ProductHypothesis
from experiments.celltag_reprogramming.pilot import PopulationRelation

ROOT = Path(__file__).resolve().parents[3]
PLAIN = ROOT / 'experiments/moonshot-parallel-v1/plain/model.py'
ACTOR = ROOT / 'experiments/moonshot-parallel-v1/actor/model.py'


def owner(path, family, *args):
    kind = load_predecessor(path, family, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return kind(*args).double()


def bind(module, family, shape, domains=('declared-state',), batched=False):
    declaration = ModelDeclaration('model', 'whole-hypothesis', 'mechanism', family,
                                   shape, domains,
                                   tuple((name, 'parameter:' + name) for name, _ in module.named_parameters()),
                                   'actual-predecessor-source', batched)
    return ModelClient(module, declaration)


class ModelClients(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        torch.set_num_threads(1)

    def test_plain_actual_owner_forward_vjp_and_time(self):
        for family, shape, module, args in (
            ('matrix_patch', (2, 2, 2), owner(PLAIN, 'matrix_patch', 2, 2), ()),
            ('sylvester_flow', (2, 2), owner(PLAIN, 'sylvester_flow', 2),
             (torch.tensor(.3, dtype=torch.float64, requires_grad=True),)),
        ):
            client = bind(module, family, shape)
            state = torch.randn(shape, dtype=torch.float64, requires_grad=True)
            expected = module(state, *args)
            actual = client(state, *args)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            inputs = (state, *tuple(module.parameters()), *args)
            reference = torch.autograd.grad(expected.sum(), inputs, retain_graph=True)
            observed = torch.autograd.grad(actual.sum(), inputs)
            for first, second in zip(reference, observed):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            self.assertEqual(client.capability('input_vjp'), 'framework_owned')
            self.assertEqual(client.capability('higher_order'), 'unqualified')
            with self.assertRaises(NotImplementedError):
                client.observe(state)

    def test_actor_repeated_edges_private_readout_and_topology(self):
        module = owner(ACTOR, 'actor_port', 3, 2, 1, ([0, 0, 1], [1, 1, 2]))
        client = bind(module, 'actor_port', (3, 2), ('actor-a', 'actor-b', 'actor-c'))
        state = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(client(state), module(state), rtol=0, atol=0)
        torch.testing.assert_close(client.observe(state), module.observe(state), rtol=0, atol=0)
        grads = torch.autograd.grad(client(state).sum() + client.observe(state).sum(),
                                    (state, *module.parameters()))
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
        self.assertEqual(module.source.tolist(), [0, 0, 1])
        self.assertIn('private actor', client.coordinate_meaning)
        module.source[0] = 2
        with self.assertRaisesRegex(RuntimeError, 'topology changed'):
            client(state)

    def test_actual_ml2_joint_and_population_clients(self):
        for family, module, shape in (
            ('joint_product', ProductHypothesis(), (2,)),
            ('population_relation', PopulationRelation(native=False).double(), (4,)),
        ):
            client = bind(module, family, shape, batched=True)
            state = torch.randn(3, *shape, dtype=torch.float64, requires_grad=True)
            torch.testing.assert_close(client(state), module(state), rtol=0, atol=0)
            gradients = torch.autograd.grad(client(state).sum(), (state, *module.parameters()))
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
            self.assertEqual(client.capability('native_cuda'), 'unqualified')

    def test_checkpoint_replay_preserves_scientific_and_topology_ids(self):
        module = owner(ACTOR, 'actor_port', 2, 2, 1, ([0, 0], [1, 1]))
        client = bind(module, 'actor_port', (2, 2), ('a', 'b'))
        state = torch.randn(2, 2, dtype=torch.float64)
        expected = client(state).detach().clone()
        checkpoint = copy.deepcopy(client.state_dict())
        with torch.no_grad():
            module.edge.add_(.3)
        client.restore(checkpoint)
        torch.testing.assert_close(client(state), expected, rtol=0, atol=0)
        for mutate in ('metadata', 'topology', 'shape'):
            bad = copy.deepcopy(checkpoint)
            if mutate == 'metadata':
                bad['_extra_state']['declaration']['hypothesis_id'] = 'other-branch'
            elif mutate == 'topology':
                bad['owner.source'][0] = 1
            else:
                bad['owner.edge'] = torch.ones(3, dtype=torch.float64)
            before = copy.deepcopy(client.state_dict())
            with self.assertRaises(RuntimeError):
                client.restore(bad)
            for key, value in before.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(client.state_dict()[key], value, rtol=0, atol=0)

    def test_source_identity_shape_and_parameter_admission(self):
        with self.assertRaisesRegex(ValueError, 'source differs'):
            load_predecessor(PLAIN, 'matrix_patch', expected_sha256='0' * 64)
        module = owner(PLAIN, 'matrix_patch', 1, 2)
        client = bind(module, 'matrix_patch', (1, 2, 2))
        with self.assertRaises(ValueError):
            client(torch.zeros(1, 2, 3))
        with self.assertRaises(ValueError):
            client([1, 2])
        with self.assertRaises(ValueError):
            ModelClient(module, replace(client.declaration, parameter_ids=()))


if __name__ == '__main__':
    unittest.main()
