"""Scientific request admission; providers own all derivative arithmetic."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PrimalIdentity:
    structure_epoch: int
    state_generation: int
    parameter_generation: int
    forcing_generation: int
    activity_generation: int

    def validate(self):
        if any(type(x) is not int or x < 0 for x in
               (self.structure_epoch, self.state_generation, self.parameter_generation,
                self.forcing_generation, self.activity_generation)):
            raise ValueError('explicit nonnegative structure/value identities required')


@dataclass(frozen=True)
class Perturbation:
    quantity_id: str
    role: str
    direction_id: str
    units: str

    def validate(self):
        if (self.role not in ('state', 'parameter', 'forcing', 'activity')
                or any(not isinstance(x, str) or not x for x in
                       (self.quantity_id, self.direction_id, self.units))):
            raise ValueError('supplied perturbation identity, role and units required')


@dataclass(frozen=True)
class ResponseRequest:
    whole_hypothesis_id: str
    differentiated_object: str
    action: str
    perturbation: Perturbation
    output_shape: tuple[int, ...]
    evidence_id: str
    provider_evidence_id: str

    def validate(self):
        self.perturbation.validate()
        if (self.differentiated_object not in
                ('vector_field', 'discrete_step', 'observation', 'implemented_rollout')
                or self.action not in ('jvp', 'vjp', 'selected_response')
                or any(not isinstance(x, str) or not x for x in
                       (self.whole_hypothesis_id, self.evidence_id, self.provider_evidence_id))
                or not isinstance(self.output_shape, tuple)
                or any(type(x) is not int or x <= 0 for x in self.output_shape)):
            raise ValueError('typed response object/action/shape and evidence required')


@dataclass(frozen=True)
class Response:
    request: ResponseRequest
    primal: PrimalIdentity
    values: object
    uncertainty_scope: str = 'local supplied primal; global identification unresolved'


def execute_response(request: ResponseRequest, saved: PrimalIdentity,
                     live: PrimalIdentity, provider_actions):
    """Admit one exact object/action then call its CE/framework provider.

    Providers and callers retain buffers, saved-forward lifetimes and streams.
    A vector-field action does not qualify an observation or rollout derivative.
    """
    request.validate()
    saved.validate()
    live.validate()
    if saved != live:
        raise ValueError('response primal epoch or generation is stale')
    action = provider_actions.get((request.whole_hypothesis_id, request.differentiated_object, request.action))
    if action is None:
        raise NotImplementedError('requested object/action lacks a qualified provider')
    values = action()
    import torch
    if (not isinstance(values, torch.Tensor) or tuple(values.shape) != request.output_shape
            or not bool(torch.isfinite(values).all())):
        raise ValueError('provider response differs from the declared output or is nonfinite')
    return Response(request, saved, values)
