"""Torch reference models for one cell; no native execution or biological claim."""
from __future__ import annotations

import math
import numbers

import torch
from torch import nn


def _positive_dimension(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _check_state(state: torch.Tensor, parameter: torch.Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(state, torch.Tensor) or tuple(state.shape) != shape:
        raise ValueError(f"one-cell state must have shape {shape}")
    if state.dtype not in (torch.float32, torch.float64):
        raise ValueError("reference state requires float32 or float64")
    if state.dtype != parameter.dtype or state.device != parameter.device:
        raise ValueError("state dtype and device must match model parameters")


class MatrixPatchField(nn.Module):
    """One-cell nonlinear field; patches are internal coordinates, not cells.

    Trainable decay is passed through softplus. No shared biological meaning is
    assigned to corresponding coordinates in different patches.
    """

    def __init__(self, patches: int, width: int = 16):
        super().__init__()
        patches = _positive_dimension(patches, "patches")
        width = _positive_dimension(width, "width")
        scale = width ** -0.5
        self.left = nn.Parameter(torch.randn(patches, width, width) * scale)
        self.right = nn.Parameter(torch.randn(patches, width, width) * scale)
        self.decay = nn.Parameter(torch.zeros(patches, 1, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        _check_state(state, self.left, tuple(self.left.shape))
        return (torch.tanh(self.left @ state) @ self.right
                - torch.nn.functional.softplus(self.decay) * state)


class SylvesterFlow(nn.Module):
    """Flow of dX/dt = L X + X R for fixed autonomous generators L and R.

    Torch matrix_exp provides a numerical evaluation of the exact formula.
    The semigroup property requires the same generators throughout the interval.
    This does not solve a nonlinear field or time-varying generators. Each call
    recomputes exponentials, preserving current parameter and time derivatives.
    """

    def __init__(self, width: int = 16):
        super().__init__()
        width = _positive_dimension(width, "width")
        self.left = nn.Parameter(-0.1 * torch.eye(width))
        self.right = nn.Parameter(-0.1 * torch.eye(width))

    def forward(self, x: torch.Tensor, dt: float | torch.Tensor) -> torch.Tensor:
        _check_state(x, self.left, tuple(self.left.shape))
        if isinstance(dt, torch.Tensor):
            if dt.ndim != 0 or dt.dtype != x.dtype or dt.device != x.device:
                raise ValueError("dt must be scalar with the state dtype and device")
            if not bool(torch.isfinite(dt)):
                raise ValueError("dt must be finite")
        elif isinstance(dt, bool) or not isinstance(dt, numbers.Real) or not math.isfinite(dt):
            raise ValueError("dt must be a finite real scalar")
        return torch.matrix_exp(dt * self.left) @ x @ torch.matrix_exp(dt * self.right)
