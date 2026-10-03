"""Evidence and scientific policy for actual supplied refactoring demonstrations.

The prototype/framework or CE owner executes state/readout maps. This module
does not discover a basis, transform optimizer moments or publish native epochs.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StructuralProposal:
    proposal_id: str
    whole_hypothesis_id: str
    old_epoch: int
    new_epoch: int
    old_coordinates: tuple[str, ...]
    new_coordinates: tuple[str, ...]
    kind: str
    state_migration_id: str
    parameter_migration_id: str
    readout_migration_id: str
    initialization_id: str
    optimizer_policy: str
    reader_policy: str
    invalidated_projection_ids: tuple[str, ...]
    evidence_id: str
    applicable_domain: str
    uncertainty: tuple[str, ...]
    interpretation: str = 'supplied_anonymous_coordinates'

    def validate(self):
        if any(not isinstance(x, str) or not x for x in (
                self.proposal_id, self.whole_hypothesis_id, self.state_migration_id,
                self.parameter_migration_id, self.readout_migration_id, self.initialization_id,
                self.evidence_id, self.applicable_domain)):
            raise ValueError('proposal requires whole hypothesis, migration and evidence identities')
        if (type(self.old_epoch) is not int or type(self.new_epoch) is not int
                or self.old_epoch < 1 or self.new_epoch != self.old_epoch + 1):
            raise ValueError('explicit successor structure epochs required')
        for ids in (self.old_coordinates, self.new_coordinates, self.invalidated_projection_ids,
                    self.uncertainty):
            if (not isinstance(ids, tuple) or not ids or len(set(ids)) != len(ids)
                    or any(not isinstance(x, str) or not x for x in ids)):
                raise ValueError('immutable coordinate, invalidation and uncertainty declarations required')
        if self.kind not in ('supplied_basis', 'invariant_quotient'):
            raise NotImplementedError('this seam qualifies supplied linear demonstrations only')
        if self.interpretation != 'supplied_anonymous_coordinates':
            raise ValueError('a supplied basis does not identify a biological mechanism')
        if self.optimizer_policy != 'reset_transformed_parameter_moments':
            raise NotImplementedError('general basis changes do not justify diagonal Adam moment transport')
        if self.reader_policy != 'drain_old_readers':
            raise NotImplementedError('retaining old native epochs requires an owner-specific publication contract')


class SuppliedMigration:
    """Bind the existing LinearRefactoring witness to scientific declarations."""
    def __init__(self, proposal: StructuralProposal, demonstration):
        proposal.validate()
        receipt = demonstration.receipt
        domain = ('whole state space' if proposal.kind == 'supplied_basis'
                  else 'supplied invariant manifold')
        if (receipt.old_epoch != proposal.old_epoch or receipt.new_epoch != proposal.new_epoch
                or receipt.old_coordinates != proposal.old_coordinates
                or receipt.new_coordinates != proposal.new_coordinates
                or receipt.domain != domain or proposal.applicable_domain != domain):
            raise ValueError('supplied migration differs from the scientific proposal')
        self.proposal = proposal
        self.demonstration = demonstration

    def initialize_state(self, old_state):
        return self.demonstration.initialize_state(old_state)

    def lift_state(self, new_state):
        return self.demonstration.lift_state(new_state)

    @property
    def migrated_model(self):
        return self.demonstration.model

    def receipt(self):
        return {'proposal': asdict(self.proposal), 'numeric_owner': 'supplied framework demonstration',
                'native_epoch_publication': 'not_run', 'optimizer_reset': 'caller required',
                'biological_status': 'not_run'}
