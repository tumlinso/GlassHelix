"""Required combined consumer check after accepted SCIENCE/MODELS source merge."""
import hashlib
from pathlib import Path
import unittest

import torch

from glasshelix.experiment import (ExperimentSpecification, Quantity, Provenance,
                                  ScientificResult)
from glasshelix.models import ModelClient, ModelDeclaration, load_predecessor

ROOT = Path(__file__).resolve().parents[3]


class ScienceModelComposition(unittest.TestCase):
    def test_actual_actor_client_whole_hypothesis_prediction_replay(self):
        source = ROOT / 'experiments/moonshot-parallel-v1/actor/model.py'
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        kind = load_predecessor(source, 'actor_port', expected_sha256=source_hash)
        hypotheses, predictions, parameters = [], [], []
        for label, state_values in (('a', (1., 2.)), ('b', (8., 9.))):
            owner = kind(1, 2, 1, ([0], [0])).double()
            declaration = ModelDeclaration('model', 'hypothesis-' + label, 'mechanism-' + label,
                'actor_port', (1, 2), ('actor-' + label,),
                tuple((name, label + ':' + name) for name, _ in owner.named_parameters()), source_hash)
            client = ModelClient(owner, declaration)
            state = torch.tensor([state_values], dtype=torch.float64)
            hypotheses.append(client.to_hypothesis(state, .5))
            predictions.append(client.to_prediction(client.observe(state), 'reporter'))
            parameters.extend(Quantity(identity, 'parameter', owner.get_parameter(name).numel(),
                                       'declared') for name, identity in declaration.parameter_ids)
        specification = ExperimentSpecification('experiment', 'model',
            (Quantity('hidden', 'state', 2, 'latent'), Quantity('reporter', 'measurement', 1, 'declared'),
             *parameters), tuple(hypotheses), 'fixture', 'audit', ('input-manifest',))
        provenance = Provenance('model', '1', 'accepted-gh-merge', 'framework-owner',
                                'cpu_float64', 1, 1, 1, ('fixture-evidence',))
        result = ScientificResult(specification, tuple(hypotheses), tuple(predictions), provenance)
        replay = ScientificResult.loads(result.dumps())
        self.assertEqual(replay, result)
        self.assertEqual(replay.hypotheses[0].joint_state, (1., 2.))
        self.assertEqual(replay.hypotheses[1].joint_state, (8., 9.))
        self.assertEqual(replay.biological_status, 'not_run')


if __name__ == '__main__':
    unittest.main()
