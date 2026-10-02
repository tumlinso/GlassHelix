"""Experimental actor-local Torch models with explicitly supplied topology."""
from __future__ import annotations

import operator
import torch
from torch import nn


def _dimension(value):
    value = operator.index(value)
    if value <= 0:
        raise ValueError('positive dimensions required')
    return value


class ActorPortField(nn.Module):
    """Private hidden coordinates communicate through shared explicit ports.

    Actor index represents a supplied predictive factor, with no biological
    identity claim. Repeated directed edges each contribute to the destination.
    Generic basis changes do not commute with coordinatewise tanh.
    """

    def __init__(self, actors: int, width: int, ports: int, edges):
        super().__init__()
        actors, width, ports = map(_dimension, (actors, width, ports))
        source, destination = edges
        vectors = []
        for indices in (source, destination):
            raw = torch.as_tensor(indices)
            if raw.ndim != 1 or (raw.numel() and raw.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
            )):
                raise ValueError('edges require one-dimensional integer indices')
            vectors.append(raw.to(dtype=torch.long, device='cpu').clone())
        source, destination = vectors
        if source.shape != destination.shape or bool((
            (source < 0) | (source >= actors) |
            (destination < 0) | (destination >= actors)
        ).any()):
            raise ValueError('invalid declared edges')
        self.register_buffer('source', source)
        self.register_buffer('destination', destination)
        self.encoder = nn.Parameter(torch.randn(actors, width, ports) * width**-0.5)
        self.decoder = nn.Parameter(torch.randn(actors, ports, width) * ports**-0.5)
        self.local = nn.Parameter(torch.randn(actors, width, width) * width**-0.5)
        self.edge = nn.Parameter(torch.full((source.numel(),), 0.1))
        self.readout = nn.Parameter(torch.randn(actors, width) * width**-0.5)

    def _check_state(self, state):
        if state.shape != self.encoder.shape[:2]:
            raise ValueError('one-cell actor/slot shape mismatch')

    def forward(self, state):
        self._check_state(state)
        messages = torch.einsum('ih,ihp->ip', state, self.encoder)
        received = torch.zeros_like(messages).index_add(
            0, self.destination, self.edge[:, None] * messages[self.source]
        )
        local = torch.tanh(torch.einsum('ih,ihk->ik', state, self.local)) - state
        return local + torch.einsum('ip,iph->ih', received, self.decoder)

    def observe(self, state):
        self._check_state(state)
        return (state * self.readout).sum(-1)


class RegrowthBranch(nn.Module):
    """Nonzero features and zero outgoing weights give zero initial effect.

    Output-weight gradients remain live at zero effect. Feature gradients are
    initially zero and become available once outgoing weights begin learning.
    Leading state dimensions are supported; the last axis is input_width.
    """

    def __init__(self, input_width: int, capacity: int, output_width: int):
        super().__init__()
        input_width, capacity, output_width = map(
            _dimension, (input_width, capacity, output_width)
        )
        self.inside = nn.Linear(input_width, capacity, bias=False)
        self.outside = nn.Linear(capacity, output_width, bias=False)
        nn.init.zeros_(self.outside.weight)

    def forward(self, state):
        if state.ndim == 0 or state.shape[-1] != self.inside.in_features:
            raise ValueError('regrowth input width mismatch')
        return self.outside(torch.tanh(self.inside(state)))
