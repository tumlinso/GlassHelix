"""Compose the existing model owners without copying their mathematics."""
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
from pathlib import Path

import torch
from torch import nn


_FAMILIES = {'matrix_patch': 'MatrixPatchField', 'sylvester_flow': 'SylvesterFlow',
             'actor_port': 'ActorPortField'}


def load_predecessor(source_path, family, *, expected_sha256):
    """Load an explicitly supplied, hash-bound predecessor class.

    No repository-relative fallback and no native-provider substitution occurs.
    Installation must ship or explicitly reference the accepted source component.
    """
    path = Path(source_path)
    if family not in _FAMILIES:
        raise ValueError('unknown predecessor model family')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError('predecessor model source differs from the accepted hash')
    module_spec = importlib.util.spec_from_file_location(f'glasshelix_predecessor_{actual}', path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, _FAMILIES[family])


@dataclass(frozen=True)
class ModelDeclaration:
    model_id: str
    hypothesis_id: str
    mechanism_id: str
    family: str
    state_shape: tuple[int, ...]
    domain_ids: tuple[str, ...]
    parameter_ids: tuple[tuple[str, str], ...]
    provider_evidence_id: str
    batched: bool = False

    def validate(self):
        if any(not isinstance(x, str) or not x for x in
               (self.model_id, self.hypothesis_id, self.mechanism_id, self.provider_evidence_id)):
            raise ValueError('scientific, whole-hypothesis and provider identities required')
        if self.family not in (*_FAMILIES, 'joint_product', 'population_relation', 'native_owner'):
            raise ValueError('unknown scientific model family')
        if (not isinstance(self.state_shape, tuple) or not self.state_shape
                or any(type(x) is not int or x <= 0 for x in self.state_shape)):
            raise ValueError('positive declared state dimensions required')
        if (not isinstance(self.domain_ids, tuple) or not self.domain_ids
                or any(not isinstance(x, str) or not x for x in self.domain_ids)
                or len(set(self.domain_ids)) != len(self.domain_ids)):
            raise ValueError('unique immutable scientific domain IDs required')
        if (not isinstance(self.parameter_ids, tuple)
                or any(not isinstance(p, tuple) or len(p) != 2
                       or any(not isinstance(x, str) or not x for x in p) for p in self.parameter_ids)
                or len({p[0] for p in self.parameter_ids}) != len(self.parameter_ids)
                or len({p[1] for p in self.parameter_ids}) != len(self.parameter_ids)):
            raise ValueError('unique named parameter owner identities required')
        if self.family == 'actor_port' and len(self.domain_ids) != self.state_shape[0]:
            raise ValueError('actor-local state requires one supplied domain identity per actor')
        if type(self.batched) is not bool or (self.batched and self.family in _FAMILIES):
            raise ValueError('one-cell predecessor families do not declare a batch axis')


class ModelClient(nn.Module):
    """One supplied whole model and its declared scientific coordinate domain.

    The child module owns arithmetic, autograd and parameter storage. The client
    adds identity/admission and rejects undeclared topology/checkpoint changes.
    """
    def __init__(self, owner: nn.Module, declaration: ModelDeclaration):
        super().__init__()
        declaration.validate()
        if not isinstance(owner, nn.Module):
            raise ValueError('existing framework/native module owner required')
        if set(dict(owner.named_parameters())) != {p[0] for p in declaration.parameter_ids}:
            raise ValueError('declared parameter identities differ from owner parameters')
        self.owner = owner
        self.declaration = declaration
        self._topology = {}
        if declaration.family == 'actor_port':
            for name in ('source', 'destination'):
                value = getattr(owner, name, None)
                if not isinstance(value, torch.Tensor):
                    raise ValueError('actor owner must expose its declared topology buffers')
                self._topology[name] = value.detach().cpu().clone()

    @property
    def coordinate_meaning(self):
        return {'matrix_patch': 'internal matrix layout; no inferred shared biological axes',
                'sylvester_flow': 'declared matrix state with fixed autonomous generators',
                'actor_port': 'private actor coordinates communicating through explicit ports',
                'joint_product': 'declared joint input response; no identified causal mechanism',
                'population_relation': 'shared effective RNA response; population snapshots are not cell histories',
                'native_owner': 'caller-declared scientific roles over native-owned execution'}[self.declaration.family]

    def capability(self, action):
        if self.declaration.family == 'native_owner':
            return 'requires_producer_receipt'
        return ('framework_owned' if action in ('forward', 'input_vjp', 'parameter_vjp')
                else 'unqualified')

    def _admit(self, state):
        if not isinstance(state, torch.Tensor):
            raise ValueError('state requires an owner-compatible tensor')
        shape = tuple(state.shape[1:] if self.declaration.batched else state.shape)
        if shape != self.declaration.state_shape:
            raise ValueError('input differs from the supplied scientific state shape')
        for name, expected in self._topology.items():
            if not torch.equal(getattr(self.owner, name).detach().cpu(), expected):
                raise RuntimeError('actor topology changed without a new scientific declaration')

    def forward(self, state, *args, **kwargs):
        self._admit(state)
        return self.owner(state, *args, **kwargs)

    def observe(self, state):
        self._admit(state)
        if self.declaration.family != 'actor_port':
            raise NotImplementedError('an explicit scientific observation map is required')
        return self.owner.observe(state)

    def get_extra_state(self):
        return {'schema': 1, 'declaration': asdict(self.declaration),
                'topology': {k: v.tolist() for k, v in self._topology.items()}}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError('checkpoint scientific declaration differs')

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.declaration.family == 'native_owner':
            raise NotImplementedError('native restore requires its sanctioned owner publication route')
        if state_dict.get(prefix + '_extra_state') != self.get_extra_state():
            raise RuntimeError('checkpoint scientific declaration differs')
        for name, expected in self._topology.items():
            value = state_dict.get(prefix + 'owner.' + name)
            if not isinstance(value, torch.Tensor) or not torch.equal(value.detach().cpu(), expected):
                raise RuntimeError('checkpoint actor topology differs')
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def restore(self, checkpoint):
        if self.declaration.family == 'native_owner':
            raise NotImplementedError('native restore requires its sanctioned owner publication route')
        expected = self.state_dict()
        if set(checkpoint) != set(expected):
            raise RuntimeError('checkpoint keys differ from the owner')
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                incoming = checkpoint[key]
                if (not isinstance(incoming, torch.Tensor) or incoming.shape != value.shape
                        or incoming.dtype != value.dtype or incoming.device != value.device
                        or (incoming.is_floating_point() and not bool(torch.isfinite(incoming).all()))):
                    raise RuntimeError('checkpoint tensor representation differs or is nonfinite')
        return self.load_state_dict(checkpoint, strict=True)

    def to_hypothesis(self, state, weight):
        """Join with accepted SCIENCE declarations after the source merge barrier."""
        from ..experiment import JointHypothesis
        self._admit(state)
        if self.declaration.batched and state.shape[0] != 1:
            raise ValueError('one whole individual hypothesis requires exactly one supplied batch row')
        return JointHypothesis(self.declaration.hypothesis_id, self.declaration.mechanism_id,
                               tuple(state.detach().cpu().reshape(-1).tolist()),
                               tuple(p[1] for p in self.declaration.parameter_ids), weight)

    def to_prediction(self, observed_output, observable_id):
        """Caller supplies a scientifically chosen readout, never an inferred one."""
        from ..experiment import Prediction
        if not isinstance(observed_output, torch.Tensor) or not bool(torch.isfinite(observed_output).all()):
            raise ValueError('finite caller-supplied observation output required')
        return Prediction(self.declaration.hypothesis_id, observable_id,
                          tuple(observed_output.detach().cpu().reshape(-1).tolist()))
