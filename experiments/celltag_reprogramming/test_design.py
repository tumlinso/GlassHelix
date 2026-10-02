"""Exact CPU double checks for the selected shared-support construction."""
from __future__ import annotations

import unittest

import torch

try:
    from .reference import dense, factorized, masked_mse, materialized
except ImportError:
    from reference import dense, factorized, masked_mse, materialized


class SharedSupportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(47)
        self.dst = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 1, 0])
        self.src = torch.tensor([0, 2, 1, 2, 0, 4, 3, 4, 1, 0])
        self.args = tuple(torch.randn(shape, generator=generator,
                                      dtype=torch.float64, requires_grad=True)
                          for shape in [(10,), (3, 5), (3, 5), (3, 4)])
        self.target = torch.randn((3, 4), generator=generator, dtype=torch.float64)
        self.mask = torch.tensor([[True, False, True, True],
                                  [False, True, True, False],
                                  [True, True, False, True]])

    def close(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)

    def output(self, implementation, args=None):
        return implementation(*(self.args if args is None else args), self.dst, self.src)

    def loss(self, implementation, args=None):
        return masked_mse(self.output(implementation, args), self.target, self.mask)

    def test_three_forward_realizations(self):
        expected = self.output(factorized)
        for implementation in (materialized, dense):
            self.close(self.output(implementation), expected)

    def test_all_trainable_gradients_match_two_references(self):
        expected = torch.autograd.grad(self.loss(factorized), self.args)
        for implementation in (materialized, dense):
            actual = torch.autograd.grad(self.loss(implementation), self.args)
            for actual_gradient, expected_gradient in zip(actual, expected):
                self.close(actual_gradient, expected_gradient)

    def test_masked_scalar_finite_difference_gradcheck(self):
        self.assertTrue(torch.autograd.gradcheck(
            lambda *args: self.loss(factorized, args), self.args,
            eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_analytic_gradients_and_single_mean_scaling(self):
        w, x, s, a = self.args
        y = self.output(factorized)
        cotangent = torch.where(self.mask, 2 * (y - self.target) / self.mask.sum(), 0)
        q = s * x
        v = q.new_zeros(a.shape).index_add(1, self.dst, q[:, self.src] * w)
        u = a * cotangent
        transposed = q.new_zeros(q.shape).index_add(1, self.src, u[:, self.dst] * w)
        expected = ((u[:, self.dst] * q[:, self.src]).sum(0),
                    s * transposed, x * transposed, cotangent * v)
        actual = torch.autograd.grad(masked_mse(y, self.target, self.mask), self.args)
        for observed, reference in zip(actual, expected):
            self.close(observed, reference)

    def test_duplicate_edge_identity_retained(self):
        w, x, s, a = self.args
        gradients = torch.autograd.grad(self.loss(factorized), self.args)[0]
        self.assertEqual(gradients.shape, (10,))
        self.close(gradients[0], gradients[9])
        self.close(gradients[2], gradients[8])
        changed = w.detach().clone()
        changed[0] += 0.75
        expected_delta = torch.zeros_like(a).index_add(
            1, self.dst[:1], 0.75 * (s * x)[:, self.src[:1]] * a[:, self.dst[:1]])
        self.close(self.output(factorized, (changed, x, s, a)) - self.output(factorized),
                   expected_delta)
        self.close(changed[9], w[9])

    def test_repeated_parameter_use_adds_gradients_without_update(self):
        before = tuple(argument.detach().clone() for argument in self.args)
        w, x, s, a = self.args
        second_args = (w, 0.7 * x + 0.1, 1.2 * s, -0.8 * a)
        first = self.loss(factorized)
        second = self.loss(factorized, second_args)
        expected = torch.autograd.grad(first + second, self.args)
        self.loss(factorized).backward()
        self.loss(factorized, (w, 0.7 * x + 0.1, 1.2 * s, -0.8 * a)).backward()
        for argument, gradient, original in zip(self.args, expected, before):
            self.close(argument.grad, gradient)
            self.close(argument, original)

    def test_zeros_retain_derivative_support(self):
        # A zero in each trainable plane must not imply structural exclusion.
        for zero_plane in range(4):
            args = tuple(torch.full_like(argument, 0.0 if i == zero_plane else 1.0,
                                         requires_grad=True)
                         for i, argument in enumerate(self.args))
            gradient = torch.autograd.grad(self.output(factorized, args).sum(), args)[zero_plane]
            self.assertTrue(bool(torch.all(gradient > 0)))
            for implementation in (materialized, dense):
                self.close(self.output(factorized, args), self.output(implementation, args))
                expected = torch.autograd.grad(self.output(implementation, args).sum(), args)[zero_plane]
                self.close(gradient, expected)

    def test_edge_permutation_preserves_parameter_gradient_identity(self):
        w, x, s, a = self.args
        order = torch.tensor([9, 4, 0, 6, 2, 8, 3, 1, 7, 5])
        y = factorized(w[order], x, s, a, self.dst[order], self.src[order])
        self.close(y, self.output(factorized))
        expected = torch.autograd.grad(self.loss(factorized), self.args)
        actual = torch.autograd.grad(masked_mse(y, self.target, self.mask), self.args)
        for observed, reference in zip(actual, expected):
            self.close(observed, reference)

    def test_domain_permutation_requires_index_and_observation_remap(self):
        w, x, s, a = self.args
        source_order = torch.tensor([2, 4, 0, 3, 1])
        target_order = torch.tensor([2, 0, 3, 1])
        y = factorized(w, x[:, source_order], s[:, source_order], a[:, target_order],
                       target_order.argsort()[self.dst], source_order.argsort()[self.src])
        self.close(y, self.output(factorized)[:, target_order])
        remapped_loss = masked_mse(y, self.target[:, target_order], self.mask[:, target_order])
        self.close(remapped_loss, self.loss(factorized))
        expected = torch.autograd.grad(self.loss(factorized), self.args)
        actual = torch.autograd.grad(remapped_loss, self.args)
        for observed, reference in zip(actual, expected):
            self.close(observed, reference)

    def test_observed_zero_counts_and_missing_nan_is_excluded(self):
        prediction = torch.tensor([[2., 100., 3.]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[0., float('nan'), 1.]], dtype=torch.float64)
        mask = torch.tensor([[True, False, True]])
        loss = masked_mse(prediction, target, mask)
        self.close(loss, torch.tensor(4., dtype=torch.float64))
        gradient, = torch.autograd.grad(loss, prediction)
        self.close(gradient, torch.tensor([[2., 0., 2.]], dtype=torch.float64))

    def test_invalid_measurement_masks_are_explicit(self):
        with self.assertRaisesRegex(ValueError, 'at least one observed'):
            masked_mse(self.target, self.target, torch.zeros_like(self.mask))
        with self.assertRaisesRegex(TypeError, 'boolean'):
            masked_mse(self.target, self.target, self.mask.double())
        with self.assertRaisesRegex(ValueError, 'equal shapes'):
            masked_mse(self.target, self.target, self.mask[:1])

    def test_two_sided_activity_cannot_fit_general_context_edges(self):
        # For a nonzero 2x2 base W, any ratio to diag(a) W diag(s)
        # has rank <= 1. This desired context ratio has determinant 1.
        base = torch.tensor([[2., 3.], [4., 5.]], dtype=torch.float64)
        ratio = torch.tensor([[1., 1.], [1., 2.]], dtype=torch.float64)
        alternative = base * ratio
        required_ratio = alternative / base
        self.close(torch.linalg.det(required_ratio), torch.tensor(1., dtype=torch.float64))
        self.assertEqual(int(torch.linalg.matrix_rank(required_ratio)), 2)
        outer_gates = torch.tensor([0.7, 1.2], dtype=torch.float64)[:, None] * torch.tensor(
            [0.5, 1.3], dtype=torch.float64)[None, :]
        self.close(torch.linalg.det(outer_gates), torch.tensor(0., dtype=torch.float64))


if __name__ == '__main__':
    unittest.main(verbosity=2)
