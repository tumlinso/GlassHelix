from dataclasses import replace
import json
import unittest

import torch

from glasshelix.experiment import (ExperimentSpecification, JointHypothesis, Quantity,
                                  Prediction, Provenance, ScientificResult)
from glasshelix.learning import EvidencePermission, EvidenceView, masked_loss


class ScientificContracts(unittest.TestCase):
    def setUp(self):
        self.hypotheses = (JointHypothesis('h1', 'm1', (1., 2.), ('law',), .5),
                           JointHypothesis('h2', 'm2', (8., 9.), ('law',), .5))
        self.spec = ExperimentSpecification(
            'experiment', 'model', (Quantity('hidden', 'state', 2, 'latent'),
                                    Quantity('law', 'parameter', 1, 'declared'),
                                    Quantity('rna', 'measurement', 2, 'counts')),
            self.hypotheses, 'fixture', 'audit', ('inputs',))
        self.provenance = Provenance('model', '1', 'gh-source', 'ce-source', 'fp32',
                                     1, 2, 3, ('e1', 'e2'))
        self.result = ScientificResult(
            self.spec, self.hypotheses,
            (Prediction('h1', 'rna', (0., 1.)), Prediction('h2', 'rna', (0., 8.))),
            self.provenance, ('e1',))
        self.evidence = EvidenceView(torch.tensor([[0., float('nan')], [2., 3.]]),
                                     torch.tensor([[True, False], [True, True]]),
                                     ('e1', 'e2'), 'RNA', 'conditioning', (0., 1.), 'fixture')

    def test_mask_and_detection_remain_distinct(self):
        detection = torch.tensor([[False, True], [True, True]])
        manifest = self.spec.authorize(self.evidence, EvidencePermission(cutoff=1.),
                                       'conditioning', 'rna', detection)
        observation = manifest['observation']
        self.assertEqual(observation['observed_values'], [0., 2., 3.])
        self.assertEqual(observation['mask'][0], [True, False])
        self.assertEqual(observation['detection_support'][0], [False, True])
        self.assertEqual(manifest['sampling_unit'], 'population_snapshot')
        prediction = torch.tensor([[0., 100.], [2., 3.]])
        self.assertEqual(masked_loss(prediction, self.evidence).item(), 0.)

    def test_existing_permission_blocks_future_and_heldout(self):
        with self.assertRaises(PermissionError):
            self.spec.authorize(self.evidence, EvidencePermission(cutoff=.5), 'conditioning', 'rna')
        with self.assertRaises(PermissionError):
            self.spec.authorize(replace(self.evidence, split='test'), EvidencePermission(cutoff=2),
                                'conditioning', 'rna')
        with self.assertRaises(PermissionError):
            self.spec.authorize(replace(self.evidence, modality='ATAC'), EvidencePermission(cutoff=2),
                                'conditioning', 'rna')

    def test_source_shape_and_observed_nonfinite_rejection(self):
        for evidence in (replace(self.evidence, source_id='different'),
                         replace(self.evidence, values=torch.zeros(2, 3)),
                         replace(self.evidence, observed=torch.ones(2, 2, dtype=torch.bool)),
                         replace(self.evidence, evidence_ids=('', 'e2'))):
            with self.assertRaises(ValueError):
                self.spec.authorize(evidence, EvidencePermission(cutoff=2), 'conditioning', 'rna')
        with self.assertRaises(ValueError):
            self.spec.authorize(self.evidence, EvidencePermission(cutoff=2), 'conditioning', 'rna',
                                torch.zeros(2, 2))

    def test_replay_retains_whole_hypotheses_and_nonidentifiability(self):
        replay = ScientificResult.loads(self.result.dumps())
        self.assertEqual(replay, self.result)
        self.assertEqual(replay.hypotheses[0].joint_state, (1., 2.))
        self.assertEqual(replay.hypotheses[1].joint_state, (8., 9.))
        self.assertEqual(replay.hypotheses[0].parameter_ids, ('law',))
        self.assertEqual(replay.interpretation_scope, 'supplied_alternatives')
        self.assertEqual(replay.biological_status, 'not_run')

    def test_incomplete_mixed_and_duplicate_alternatives_rejected(self):
        for hypotheses in ((self.hypotheses[0],),
                           (replace(self.hypotheses[0], mechanism_id='m2'), self.hypotheses[1]),
                           (replace(self.hypotheses[0], joint_state=(1., 9.)), self.hypotheses[1]),
                           (self.hypotheses[0], self.hypotheses[0])):
            with self.assertRaises(ValueError):
                replace(self.result, hypotheses=hypotheses).validate()
        with self.assertRaises(ValueError):
            replace(self.result, predictions=(self.result.predictions[0],)).validate()

    def test_scientifically_unsupported_claims_rejected(self):
        for field, value in (('interpretation_scope', 'globally_identified_mechanism'),
                             ('interpretation_scope', 'individual_history'),
                             ('biological_status', 'qualified')):
            with self.assertRaises(ValueError):
                replace(self.result, **{field: value}).validate()
        with self.assertRaises(ValueError):
            replace(self.result, conditioning_evidence_ids=('withheld',)).validate()

    def test_replay_rejects_changed_whole_identity_and_unknown_schema(self):
        payload = json.loads(self.result.dumps())
        payload['result']['hypotheses'][0]['mechanism_id'] = 'm2'
        with self.assertRaises(ValueError):
            ScientificResult.loads(json.dumps(payload))
        payload['schema'] = 2
        with self.assertRaises(ValueError):
            ScientificResult.loads(json.dumps(payload))


if __name__ == '__main__':
    unittest.main()
