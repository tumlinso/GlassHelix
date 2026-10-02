"""CPU qualification of bounded scientific permissions and inference."""
import unittest

import torch

from glasshelix.learning import (EvidencePermission, EvidenceView,
                                ProductHypothesis, condition_local, fit_global,
                                evidence_manifest)


def evidence(values, *, split='train', modality='RNA', times=None, mask=None, kind='response'):
    values = torch.tensor(values, dtype=torch.float64).reshape(-1, 1)
    return EvidenceView(values, torch.ones_like(values, dtype=torch.bool) if mask is None else mask,
                        tuple(f'synthetic-{kind}-{i}' for i in range(len(values))), modality, split,
                        tuple([3.] * len(values) if times is None else times), 'synthetic-fixture-v1')


class LearningTests(unittest.TestCase):
    def fit(self):
        x = torch.tensor([[0.5, 1.0], [1., 0.6], [1.5, 0.8]], dtype=torch.float64)
        target = evidence((1.7 * x[:, 0] * x[:, 1]).tolist())
        return fit_global(ProductHypothesis(), x, target, EvidencePermission(),
                          hypothesis_id='synthetic-product-v1', domain_ids=('input-x', 'local-z', 'response'))

    def test_unknown_shared_coefficient_is_fitted(self):
        fitted = self.fit()
        self.assertLess(fitted.manifest['final_loss'], 1e-12)
        self.assertGreater(fitted.manifest['initial_loss'], 0.1)
        torch.testing.assert_close(fitted.hypothesis.coefficients.detach(),
                                   torch.tensor([1.7], dtype=torch.float64), rtol=1e-6, atol=1e-7)

    def test_holdouts_and_atac_are_denied(self):
        permission = EvidencePermission(cutoff=3)
        for view in (evidence([1], split='test'), evidence([1], modality='ATAC')):
            with self.assertRaises(PermissionError):
                permission.authorize(view, 'global_fit')
        with self.assertRaises(PermissionError):
            permission.authorize(evidence([1], split='test'), 'conditioning')

    def test_prospective_cutoff_and_retrospective_meaning(self):
        late = evidence([1], split='conditioning', times=[12])
        with self.assertRaises(PermissionError):
            EvidencePermission(cutoff=3).authorize(late, 'conditioning')
        EvidencePermission(mode='retrospective').authorize(late, 'conditioning')
        with self.assertRaises(PermissionError):
            EvidencePermission().authorize(evidence([1], split='conditioning'), 'conditioning')

    def test_frozen_local_fit_retains_unresolved_product(self):
        fitted = self.fit()
        before = fitted.hypothesis.coefficients.detach().clone()
        x = torch.tensor([[1.3]], dtype=torch.float64)
        view = evidence([1.7 * 1.3 * 0.8], split='conditioning')
        result = condition_local(fitted, x, view, EvidencePermission(cutoff=3))
        self.assertLess(result['residual'], 1e-12)
        self.assertEqual(result['ambiguity'], 'z*forcing only')
        torch.testing.assert_close(before, fitted.hypothesis.coefficients)
        self.assertFalse(fitted.hypothesis.coefficients.requires_grad)
        alternatives = torch.tensor([[1.3, 0.4 * 2], [1.3, 0.8 * 1]], dtype=torch.float64)
        outputs = fitted.hypothesis(alternatives)
        torch.testing.assert_close(outputs[0], outputs[1])

    def test_independent_observation_distinguishes_activity_and_forcing(self):
        fitted = self.fit()
        x = torch.tensor([[1.3]], dtype=torch.float64)
        response = evidence([1.7 * 1.3 * 0.8], split='conditioning')
        measured_z = evidence([0.8], split='conditioning', kind='activity')
        result = condition_local(fitted, x, response, EvidencePermission(cutoff=3), independent_z=measured_z)
        torch.testing.assert_close(result['z'], torch.tensor([[0.8]], dtype=torch.float64), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(result['forcing'], torch.ones_like(x), rtol=1e-6, atol=1e-7)
        self.assertIsNotNone(result['independent_z_evidence'])
        self.assertNotEqual((0.4 - 0.8) ** 2, (0.8 - 0.8) ** 2)

    def test_missing_targets_and_manifest_identity(self):
        mask = torch.tensor([[True], [False]])
        view = evidence([0., float('nan')], mask=mask)
        view.validate()
        manifest = evidence_manifest(view)
        self.assertEqual(manifest['observed_values'], [0.])
        self.assertEqual(len(manifest['sha256']), 64)
        self.assertEqual(manifest['sha256'], evidence_manifest(view)['sha256'])
        with self.assertRaises(ValueError):
            evidence([1.], mask=torch.tensor([[False]])).validate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
