"""Bounded CPU checks of algebra, derivatives, admission, and synthetic fitting."""
from __future__ import annotations

import unittest

import torch
from torch.func import functional_call

from model import MatrixPatchField, SylvesterFlow

LEARNING_RESULTS: dict[str, dict] = {}


class PlainModelsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_patch_seed_algebra_and_noncontiguous_input(self):
        model = MatrixPatchField(2, 3).double()
        state = torch.randn(2, 3, 3, dtype=torch.double).transpose(-1, -2)
        self.assertFalse(state.is_contiguous())
        expected = torch.stack([
            torch.tanh(model.left[i] @ state[i]) @ model.right[i]
            - torch.nn.functional.softplus(model.decay[i]) * state[i]
            for i in range(2)
        ])
        torch.testing.assert_close(model(state), expected, rtol=1e-12, atol=1e-12)

    def test_patch_input_and_all_parameter_gradcheck(self):
        model = MatrixPatchField(2, 2).double()
        state = torch.randn(2, 2, 2, dtype=torch.double, requires_grad=True)
        names, parameters = zip(*model.named_parameters())
        def call(x, *values):
            return functional_call(model, dict(zip(names, values)), (x,))
        self.assertTrue(torch.autograd.gradcheck(call, (state, *parameters)))

    def test_flow_input_generators_and_time_gradcheck(self):
        model = SylvesterFlow(2).double()
        with torch.no_grad():
            model.left.copy_(torch.tensor([[-.2, .3], [-.1, .15]], dtype=torch.double))
            model.right.copy_(torch.tensor([[.1, -.2], [.4, -.3]], dtype=torch.double))
        state = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        dt = torch.tensor(.27, dtype=torch.double, requires_grad=True)
        def call(x, left, right, time):
            return functional_call(model, {"left": left, "right": right}, (x, time))
        self.assertTrue(torch.autograd.gradcheck(call, (state, model.left, model.right, dt)))

    def test_flow_fixed_generators_semigroup_identity_and_generator(self):
        model = SylvesterFlow(3).double()
        with torch.no_grad():
            model.left.copy_(torch.randn(3, 3, dtype=torch.double) * .2)
            model.right.copy_(torch.randn(3, 3, dtype=torch.double) * .2)
        state = torch.randn(3, 3, dtype=torch.double).transpose(0, 1)
        torch.testing.assert_close(model(state, 0.), state, rtol=0, atol=0)
        torch.testing.assert_close(model(model(state, .31), .17), model(state, .48),
                                   rtol=1e-11, atol=1e-12)
        dt = torch.tensor(0., dtype=torch.double, requires_grad=True)
        _, time_jvp = torch.autograd.functional.jvp(lambda t: model(state, t), dt,
                                                    torch.ones_like(dt))
        torch.testing.assert_close(time_jvp, model.left @ state + state @ model.right,
                                   rtol=1e-12, atol=1e-12)
        first = model(state, .2).detach()
        with torch.no_grad():
            model.left.add_(.1 * torch.eye(3, dtype=torch.double))
        self.assertGreater((model(state, .2) - first).abs().max().item(), 1e-6)

    def test_shape_dtype_and_dimension_contracts(self):
        for make in (lambda: MatrixPatchField(0), lambda: MatrixPatchField(2, -1),
                     lambda: MatrixPatchField(1.5), lambda: SylvesterFlow(True)):
            with self.assertRaises(ValueError):
                make()
        for model, correct in ((MatrixPatchField(2, 3).double(), (2, 3, 3)),
                               (SylvesterFlow(3).double(), (3, 3))):
            args = () if isinstance(model, MatrixPatchField) else (.2,)
            with self.assertRaises(ValueError):
                model(torch.zeros((1, *correct), dtype=torch.double), *args)
            with self.assertRaises(ValueError):
                model(torch.zeros(correct, dtype=torch.float32), *args)
            with self.assertRaises(ValueError):
                model(torch.zeros(correct, dtype=torch.int64), *args)
        model = SylvesterFlow(3).double()
        x = torch.zeros(3, 3, dtype=torch.double)
        for bad in (float('nan'), float('inf'), True, [.2], torch.tensor([.2]),
                    torch.tensor(.2, dtype=torch.float32)):
            with self.assertRaises(ValueError):
                model(x, bad)

    def test_float32_forward_and_gradients(self):
        for model, state, args in ((MatrixPatchField(2, 3), torch.randn(2, 3, 3), ()),
                                   (SylvesterFlow(3), torch.randn(3, 3), (.3,))):
            state.requires_grad_(True)
            model(state, *args).square().sum().backward()
            for gradient in (state.grad, *(p.grad for p in model.parameters())):
                self.assertIsNotNone(gradient)
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                self.assertGreater(gradient.abs().sum().item(), 0.)

    def test_synthetic_learning_both_families(self):
        for family in ("patch", "flow"):
            torch.manual_seed(29)
            make = (lambda: MatrixPatchField(2, 3).double()) if family == "patch" else (
                lambda: SylvesterFlow(3).double())
            teacher, student = make(), make()
            if family == "flow":
                with torch.no_grad():
                    teacher.left.add_(torch.randn_like(teacher.left) * .15)
                    teacher.right.add_(torch.randn_like(teacher.right) * .15)
            shape = (2, 3, 3) if family == "patch" else (3, 3)
            inputs = [torch.randn(shape, dtype=torch.double) * .3 for _ in range(4)]
            def predict(model, x):
                if family == "flow":
                    return model(x, .4)
                history = []
                for _ in range(3):
                    x = x + .05 * model(x)
                    history.append(x)
                return torch.stack(history)
            with torch.no_grad():
                targets = [predict(teacher, x) for x in inputs]
            def loss():
                return torch.stack([(predict(student, x) - y).square().mean()
                                    for x, y in zip(inputs, targets)]).mean()
            initial = loss().item()
            optimizer = torch.optim.Adam(student.parameters(), lr=.025)
            for _ in range(120):
                optimizer.zero_grad(set_to_none=True)
                current = loss()
                self.assertTrue(bool(torch.isfinite(current)))
                current.backward()
                for p in student.parameters():
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(bool(torch.isfinite(p.grad).all()))
                optimizer.step()
            final = loss().item()
            self.assertLess(final, initial * .05)
            LEARNING_RESULTS[family] = {"initial_loss": initial, "final_loss": final,
                                       "steps": 120, "optimizer": "torch.optim.Adam"}


if __name__ == "__main__":
    unittest.main()
