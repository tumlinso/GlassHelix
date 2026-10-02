"""Explicit GH observation lowering through installed CE product packets.

CPU FP32, one hidden vector, first-order derivatives. Supplied coordinate and
contribution identities are declarations; synthetic fitting establishes no biology.
"""
from __future__ import annotations

import torch
from torch import nn
from celleratorch.moonshot_product import Product2Module, product2_jvp


def _ids(values, name):
    values = tuple(values)
    if not values or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
        raise ValueError(f'{name} requires unique nonempty supplied strings')
    return values


def _indices(value, count, extent, name):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or value.device.type != 'cpu' or value.ndim != 1:
        raise ValueError(f'{name} requires a CPU int64 vector')
    if value.numel() != count or bool(((value < 0) | (value >= extent)).any()):
        raise ValueError(f'{name} extent or bounds mismatch')
    return value.detach().clone()


class NativeProductReadout(nn.Module):
    """Sum identified native contributions into supplied observation slots.

    A linear contribution k*h is lowered to k*h*1. Repeated source and
    observation indices remain separate contributions with distinct IDs.
    Restore a checkpoint into the same declared topology between backward passes;
    external optimizers own coefficient updates and their own checkpoint state.
    """
    def __init__(self, source_ids, observation_ids, contribution_ids,
                 source_indices, observation_indices, coefficients):
        super().__init__()
        self.source_ids = _ids(source_ids, 'source_ids')
        self.observation_ids = _ids(observation_ids, 'observation_ids')
        self.contribution_ids = _ids(contribution_ids, 'contribution_ids')
        count = len(self.contribution_ids)
        if (not isinstance(coefficients, torch.Tensor) or coefficients.device.type != 'cpu'
                or coefficients.dtype != torch.float32 or coefficients.shape != (count,)
                or not bool(torch.isfinite(coefficients).all())):
            raise ValueError('coefficients require a finite CPU FP32 contribution vector')
        source = _indices(source_indices, count, len(self.source_ids), 'source_indices')
        destination = _indices(observation_indices, count, len(self.observation_ids), 'observation_indices')
        self.native = Product2Module(coefficients, source,
                                    torch.full_like(source, len(self.source_ids)))
        self.register_buffer('observation_indices', destination)
        self._source = tuple(source.tolist())
        self._destination = tuple(destination.tolist())

    @property
    def coefficients(self):
        return self.native.coefficients

    def get_extra_state(self):
        return {'source_ids': self.source_ids, 'observation_ids': self.observation_ids,
                'contribution_ids': self.contribution_ids, 'source_indices': self._source,
                'observation_indices': self._destination, 'schema': 1}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError('checkpoint coordinate or contribution declarations differ')

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Reject incompatible declarations before loading this owner's parameters.
        expected = {'native.a': self.native.a, 'native.b': self.native.b,
                    'observation_indices': self.observation_indices}
        if state_dict.get(prefix + '_extra_state') != self.get_extra_state():
            raise RuntimeError('checkpoint coordinate or contribution declarations differ')
        for key, value in expected.items():
            incoming = state_dict.get(prefix + key)
            if (not isinstance(incoming, torch.Tensor) or incoming.dtype != torch.int64
                    or incoming.device.type != 'cpu' or not torch.equal(incoming, value)):
                raise RuntimeError('checkpoint topology differs')
        coefficients = state_dict.get(prefix + 'native.coefficients')
        if (not isinstance(coefficients, torch.Tensor)
                or coefficients.device.type != 'cpu'
                or coefficients.dtype != torch.float32
                or coefficients.shape != (len(self.contribution_ids),)
                or not bool(torch.isfinite(coefficients).all())):
            raise RuntimeError('checkpoint coefficients require the declared finite CPU FP32 vector')
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _admit(self, hidden):
        if (not isinstance(hidden, torch.Tensor) or hidden.device.type != 'cpu'
                or hidden.dtype != torch.float32 or hidden.shape != (len(self.source_ids),)):
            raise ValueError('hidden requires the declared CPU FP32 source vector')
        if (tuple(self.native.a.tolist()) != self._source
                or tuple(self.native.b.tolist()) != (len(self.source_ids),) * len(self.contribution_ids)
                or tuple(self.observation_indices.tolist()) != self._destination):
            raise RuntimeError('declared topology changed')
        return torch.cat((hidden, hidden.new_ones(1)))

    def _reduce(self, contributions):
        return contributions.new_zeros(len(self.observation_ids)).index_add(
            0, self.observation_indices, contributions)

    def forward(self, hidden):
        return self._reduce(self.native(self._admit(hidden)))

    def jvp(self, hidden, hidden_direction, coefficient_direction):
        x = self._admit(hidden)
        self._admit(hidden_direction)
        dx = torch.cat((hidden_direction, hidden_direction.new_zeros(1)))
        primal, tangent = product2_jvp(x, self.coefficients, self.native.a,
                                      self.native.b, dx, coefficient_direction)
        return self._reduce(primal), self._reduce(tangent)
