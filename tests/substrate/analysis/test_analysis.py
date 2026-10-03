from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

from glasshelix.analysis import PrimalIdentity, Perturbation, ResponseRequest, execute_response
from glasshelix.refactoring import StructuralProposal, SuppliedMigration

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'experiments/moonshot-parallel-v1/refactor/model.py'
spec = importlib.util.spec_from_file_location('gh_analysis_supplied_refactor', SOURCE)
predecessor = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = predecessor
spec.loader.exec_module(predecessor)


def proposal(demonstration, kind='supplied_basis'):
    r = demonstration.receipt
    return StructuralProposal('proposal', 'whole-hypothesis', r.old_epoch, r.new_epoch,
        r.old_coordinates, r.new_coordinates, kind, 'supplied-state-map', 'supplied-parameter-map',
        'supplied-readout-map', 'supplied-initialization', 'reset_transformed_parameter_moments',
        'drain_old_readers', ('old-projection',), 'supplied-map-evidence', r.domain,
        ('anonymous coordinates', 'finite precision', 'not biological identification'))


class AnalysisClients(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(97)
        torch.set_num_threads(1)
        self.old = predecessor.LinearModel(
            torch.tensor([[-.4, .2], [-.1, -.7]], dtype=torch.float64),
            torch.tensor([[.8, -.3]], dtype=torch.float64))
        transform = torch.tensor([[1., .4], [-.2, 1.]], dtype=torch.float64)
        self.demo = predecessor.LinearRefactoring.conjugate(self.old, transform,
                                                            ('s0', 's1'), ('q0', 'q1'))

    def test_actual_framework_response_boundary_and_generation(self):
        state = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        direction = torch.randn_like(state)
        request = ResponseRequest('whole-hypothesis', 'vector_field', 'jvp',
            Perturbation('state', 'state', 'direction', 'declared'), (3, 2), 'response', 'framework-source')
        primal = PrimalIdentity(1, 2, 3, 4, 5)
        providers = {('whole-hypothesis', 'vector_field', 'jvp'):
                     lambda: torch.func.jvp(self.old.forward, (state,), (direction,))[1]}
        response = execute_response(request, primal, primal, providers)
        torch.testing.assert_close(response.values, self.old(direction))
        request = replace(request, differentiated_object='observation', action='vjp')
        providers[('whole-hypothesis', 'observation', 'vjp')] = lambda: torch.autograd.grad(
            self.old.observe(state).sum(), state)[0]
        response = execute_response(request, primal, primal, providers)
        torch.testing.assert_close(response.values, self.old.readout.expand(3, 2))
        with self.assertRaises(ValueError):
            execute_response(request, primal, replace(primal, state_generation=3), providers)
        with self.assertRaises(NotImplementedError):
            execute_response(replace(request, differentiated_object='implemented_rollout'), primal, primal, providers)
        with self.assertRaises(NotImplementedError):
            execute_response(replace(request, whole_hypothesis_id='other-branch'), primal, primal, providers)

    def test_actual_supplied_basis_state_readout_and_flow(self):
        bound = SuppliedMigration(proposal(self.demo), self.demo)
        old_state = torch.randn(4, 2, dtype=torch.float64)
        new_state = bound.initialize_state(old_state)
        torch.testing.assert_close(bound.lift_state(new_state), old_state)
        torch.testing.assert_close(bound.migrated_model.observe(new_state), self.old.observe(old_state))
        for time in (0., .3, 1.):
            torch.testing.assert_close(bound.migrated_model.observe(bound.migrated_model.flow(new_state, time)),
                                       self.old.observe(self.old.flow(old_state, time)))
        receipt = bound.receipt()
        self.assertEqual(receipt['proposal']['optimizer_policy'], 'reset_transformed_parameter_moments')
        self.assertEqual(receipt['native_epoch_publication'], 'not_run')

    def test_actual_invariant_quotient_and_outside_domain(self):
        old = predecessor.LinearModel(torch.diag(torch.tensor([-.3, -.8], dtype=torch.float64)),
                                      torch.tensor([[1., 2.]], dtype=torch.float64))
        embedding = torch.tensor([[1.], [0.]], dtype=torch.float64)
        projection = embedding.T
        demonstration = predecessor.LinearRefactoring(old, embedding, projection,
            ('s0', 's1'), ('q0',), quotient=True)
        bound = SuppliedMigration(proposal(demonstration, 'invariant_quotient'), demonstration)
        state = torch.tensor([[2., 0.]], dtype=torch.float64)
        migrated = bound.initialize_state(state)
        torch.testing.assert_close(bound.migrated_model.observe(migrated), old.observe(state))
        with self.assertRaises(ValueError):
            bound.initialize_state(torch.tensor([[2., 1.]], dtype=torch.float64))

    def test_proposal_rejects_inferred_basis_invalid_epochs_and_optimizer_transport(self):
        original = proposal(self.demo)
        for invalid in (replace(original, interpretation='identified_molecule'),
                        replace(original, new_epoch=1),
                        replace(original, readout_migration_id=''),
                        replace(original, uncertainty=())):
            with self.assertRaises(ValueError):
                SuppliedMigration(invalid, self.demo)
        with self.assertRaises(NotImplementedError):
            SuppliedMigration(replace(original, optimizer_policy='transport_diagonal_Adam'), self.demo)
        with self.assertRaises(ValueError):
            SuppliedMigration(replace(original, new_coordinates=('other-q0', 'other-q1')), self.demo)

    def test_predecessor_reader_guard_remains_a_prototype_owner(self):
        guard = predecessor.EpochGuard()
        token = guard.acquire()
        with self.assertRaises(RuntimeError):
            guard.publish()
        guard.release(token)
        self.assertEqual(guard.publish(), 2)
        with self.assertRaises(RuntimeError):
            guard.release(token)


if __name__ == '__main__':
    unittest.main()
