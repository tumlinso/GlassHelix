"""Small Torch reference for a shared-support effective relation.

Finite real-valued inputs use shapes w=(E,), x=s=(B,N), a=(B,M).
Each position on the edge axis is a logical parameter identity, even when
several edges have the same endpoints. No numerical zero removes an edge.
This CPU algebra reference supplies no native, performance or biology evidence.
"""
from __future__ import annotations

import torch
from torch import Tensor


def factorized(w: Tensor, x: Tensor, s: Tensor, a: Tensor,
               dst: Tensor, src: Tensor) -> Tensor:
    """Compute y = a * (W @ (s * x)) by sum assembly on declared edges."""
    q = s * x
    v = q.new_zeros(a.shape).index_add(1, dst, q[:, src] * w)
    return a * v


def materialized(w: Tensor, x: Tensor, s: Tensor, a: Tensor,
                 dst: Tensor, src: Tensor) -> Tensor:
    """Materialize B x E instance edge values as an exact reference."""
    instance_edges = w * s[:, src] * a[:, dst]
    return x.new_zeros(a.shape).index_add(1, dst, instance_edges * x[:, src])


def dense(w: Tensor, x: Tensor, s: Tensor, a: Tensor,
          dst: Tensor, src: Tensor) -> Tensor:
    """Dense reference; duplicate edges add without merging parameter owners."""
    m, n = a.shape[1], x.shape[1]
    matrix = w.new_zeros(m * n).index_add(0, dst * n + src, w).view(m, n)
    return a * ((s * x) @ matrix.T)


def masked_mse(prediction: Tensor, target: Tensor, observed: Tensor) -> Tensor:
    """Mean squared residual over observed entries, with one scalar reduction.

    The boolean mask declares measurement availability independently of value.
    Missing targets may contain NaN; observed targets must be finite. A batch
    with no observed targets has no defined mean and is rejected explicitly.
    """
    if prediction.shape != target.shape or observed.shape != target.shape:
        raise ValueError('prediction, target and observed must have equal shapes')
    if observed.dtype != torch.bool:
        raise TypeError('observed must be a boolean measurement mask')
    if not bool(observed.any()):
        raise ValueError('masked mean requires at least one observed target')
    return (prediction[observed] - target[observed]).square().mean()
