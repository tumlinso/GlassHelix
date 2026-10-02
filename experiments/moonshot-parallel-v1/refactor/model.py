"""Externally supplied linear refactorings; no latent identity discovery."""
from dataclasses import dataclass
import torch
from torch import nn


class LinearModel(nn.Module):
    """Column-vector law ds/dt=A s, represented by batched row inputs."""
    def __init__(self, law, readout):
        super().__init__()
        if (law.ndim != 2 or law.shape[0] != law.shape[1] or law.shape[0] == 0
                or readout.ndim != 2 or readout.shape[1] != law.shape[0]
                or law.dtype != readout.dtype or law.device != readout.device
                or not law.is_floating_point()
                or not torch.isfinite(law).all() or not torch.isfinite(readout).all()):
            raise ValueError('invalid linear model')
        self.law = nn.Parameter(law.detach().clone())
        self.readout = nn.Parameter(readout.detach().clone())

    def forward(self, state):
        return state @ self.law.T

    def observe(self, state):
        return state @ self.readout.T

    def flow(self, state, time):
        return state @ torch.matrix_exp(time * self.law).T


@dataclass(frozen=True)
class Migration:
    old_epoch: int
    new_epoch: int
    old_coordinates: tuple[str, ...]
    new_coordinates: tuple[str, ...]
    label: str
    domain: str
    optimizer_policy: str = 'reset transformed parameter moments; retain untouched owners'


class EpochGuard:
    """Prototype controller guard: caller releases old backward tapes before publish."""
    def __init__(self, epoch=1):
        self.epoch = epoch
        self._next = 0
        self._live = set()

    def acquire(self):
        self._next += 1
        token = (self.epoch, self._next)
        self._live.add(token)
        return token

    def release(self, token):
        if token not in self._live or token[0] != self.epoch:
            raise RuntimeError('invalid or stale tape')
        self._live.remove(token)

    def publish(self):
        if self._live:
            raise RuntimeError('finish old backward before refactoring')
        self.epoch += 1
        return self.epoch


class LinearRefactoring:
    """Fixed caller maps plus a detached migrated model and explicit receipt.

    Quotient correspondence holds only on s=E q, including tangent directions.
    Caller must retain that constraint if subsequently training the new law.
    """
    def __init__(self, old, embedding, projection, old_coordinates,
                 new_coordinates, old_epoch=1, tolerance=1e-11, quotient=False):
        n = old.law.shape[0]
        if (embedding.ndim != 2 or embedding.shape[0] != n
                or embedding.shape[1] == 0 or projection.shape != embedding.T.shape
                or len(old_coordinates) != n or len(new_coordinates) != embedding.shape[1]
                or len(set(old_coordinates)) != len(old_coordinates)
                or len(set(new_coordinates)) != len(new_coordinates)
                or old_epoch < 1 or tolerance <= 0
                or any(x.dtype != old.law.dtype or x.device != old.law.device
                       or not torch.isfinite(x).all() for x in (embedding, projection))):
            raise ValueError('invalid supplied migration')
        if not quotient and embedding.shape != (n, n):
            raise ValueError('invertible transform must be square')
        identity = torch.eye(embedding.shape[1], dtype=old.law.dtype, device=old.law.device)
        new_law = projection @ old.law.detach() @ embedding
        if not torch.allclose(projection @ embedding, identity, atol=tolerance, rtol=0):
            raise ValueError('projection is not a left inverse')
        if not torch.allclose(old.law.detach() @ embedding, embedding @ new_law,
                              atol=tolerance, rtol=0):
            raise ValueError('supplied subspace is not invariant')
        self.embedding = embedding.detach().clone()
        self.projection = projection.detach().clone()
        self.tolerance = tolerance
        self.model = LinearModel(new_law, old.readout.detach() @ embedding)
        self.receipt = Migration(old_epoch, old_epoch + 1, tuple(old_coordinates),
                                 tuple(new_coordinates), 'equivalent lowering',
                                 'supplied invariant manifold' if quotient else 'whole state space')

    @classmethod
    def conjugate(cls, old, transform, old_coordinates, new_coordinates, **kwargs):
        if transform.ndim != 2 or transform.shape != old.law.shape:
            raise ValueError('invalid coordinate transform')
        try:
            inverse = torch.linalg.inv(transform)
        except RuntimeError as exc:
            raise ValueError('singular coordinate transform') from exc
        return cls(old, inverse, transform, old_coordinates, new_coordinates, **kwargs)

    def initialize_state(self, old_state):
        new = old_state @ self.projection.T
        if not torch.allclose(new @ self.embedding.T, old_state,
                              atol=self.tolerance, rtol=0):
            raise ValueError('old state lies outside declared migration domain')
        return new

    def lift_state(self, new_state):
        return new_state @ self.embedding.T

    def reduce_adjoint(self, old_adjoint):
        # Physical replicas contribute a sum, rather than an average.
        return old_adjoint @ self.embedding


class ResidualActor(nn.Module):
    """New anonymous actor adds a hidden row, with zero initial readout effect."""
    def __init__(self, base, branch, actor_id, epoch):
        super().__init__()
        if not actor_id or epoch < 1:
            raise ValueError('explicit actor ID and positive epoch required')
        self.base = base
        self.branch = branch
        self.actor_id = actor_id
        self.epoch = epoch

    def hidden(self, state):
        return torch.tanh(self.branch.inside(state))

    def forward(self, state):
        return self.base.observe(state) + self.branch.outside(self.hidden(state))
