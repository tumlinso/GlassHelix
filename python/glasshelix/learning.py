"""Bounded scientific fitting over ordinary Torch and owner-backed modules.

This layer declares evidence permissions and unknowns; optimizers and automatic
derivatives remain Torch/CelleraTorch facilities. Synthetic product models do
not establish biological mechanism or general identifiability.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import torch
from torch import nn


@dataclass(frozen=True)
class EvidenceView:
    values: torch.Tensor
    observed: torch.Tensor
    evidence_ids: tuple[str, ...]
    modality: str
    split: str
    times: tuple[float, ...]
    source_id: str
    correspondence_ids: tuple[str, ...] | None = None

    def validate(self):
        if self.observed.dtype != torch.bool or self.values.shape != self.observed.shape:
            raise ValueError('observation values require an equal-shaped boolean mask')
        if len(self.evidence_ids) != self.values.shape[0] or len(self.times) != len(self.evidence_ids):
            raise ValueError('evidence IDs and times must identify each observation row')
        if len(set(self.evidence_ids)) != len(self.evidence_ids) or not self.source_id:
            raise ValueError('unique evidence IDs and source identity are required')
        if any(not math.isfinite(time) for time in self.times):
            raise ValueError('observation times must be finite')
        if self.correspondence_ids is not None:
            if (not isinstance(self.correspondence_ids, tuple)
                    or len(self.correspondence_ids) != len(self.evidence_ids)
                    or len(set(self.correspondence_ids)) != len(self.correspondence_ids)
                    or any(not identity for identity in self.correspondence_ids)):
                raise ValueError('correspondence IDs must be an immutable unique identity for each row')
        if not bool(self.observed.any()) or not bool(torch.isfinite(self.values[self.observed]).all()):
            raise ValueError('at least one finite observed value is required')


@dataclass(frozen=True)
class EvidencePermission:
    mode: str = 'prospective'
    cutoff: float | None = None
    modalities: tuple[str, ...] = ('RNA',)

    def authorize(self, evidence: EvidenceView, purpose: str):
        evidence.validate()
        if self.mode not in ('prospective', 'retrospective'):
            raise ValueError('mode must be prospective or retrospective')
        expected_split = {'global_fit': 'train', 'conditioning': 'conditioning'}
        if purpose not in expected_split or evidence.split != expected_split[purpose]:
            raise PermissionError('held-out targets cannot enter fitting or conditioning')
        if evidence.modality not in self.modalities:
            raise PermissionError('modality is outside the declared evidence permission')
        if purpose == 'conditioning' and self.mode == 'prospective':
            if self.cutoff is None or not math.isfinite(self.cutoff) or any(time > self.cutoff for time in evidence.times):
                raise PermissionError('prospective conditioning requires evidence through the cutoff')


def masked_loss(prediction, evidence):
    if prediction.shape != evidence.values.shape:
        raise ValueError('prediction and evidence shapes differ')
    return (prediction[evidence.observed] - evidence.values[evidence.observed]).square().mean()


def evidence_manifest(view):
    payload = {'evidence_ids': list(view.evidence_ids), 'source_id': view.source_id,
               'modality': view.modality, 'split': view.split, 'times': list(view.times),
               'mask': view.observed.detach().cpu().tolist(),
               'correspondence_ids': None if view.correspondence_ids is None else list(view.correspondence_ids),
               'observed_values': view.values[view.observed].detach().cpu().tolist()}
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    return dict(payload, sha256=hashlib.sha256(encoded).hexdigest())


class ProductHypothesis(nn.Module):
    """Torch reference r=k*x*z, with one shared unknown law coefficient."""
    def __init__(self, initial=0.1, *, dtype=torch.float64):
        super().__init__()
        self.coefficients = nn.Parameter(torch.tensor([initial], dtype=dtype))

    def forward(self, inputs):
        return self.coefficients * inputs[:, :1] * inputs[:, 1:2]


@dataclass
class FittedExplanation:
    hypothesis: nn.Module
    manifest: dict
    optimizer: torch.optim.Optimizer

    def freeze(self):
        for parameter in self.hypothesis.parameters():
            parameter.requires_grad_(False)
        self.manifest['global_frozen'] = True
        return self


def fit_global(hypothesis, inputs, evidence, permission, *, hypothesis_id,
               domain_ids, steps=400, lr=0.05, guarded_step=None):
    permission.authorize(evidence, 'global_fit')
    if not hypothesis_id or not domain_ids:
        raise ValueError('hypothesis and scientific domain identities are required')
    parameters = [parameter for parameter in hypothesis.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    with torch.no_grad():
        initial_loss = float(masked_loss(hypothesis(inputs), evidence))
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        masked_loss(hypothesis(inputs), evidence).backward()
        if guarded_step is None:
            optimizer.step()
        elif not guarded_step(hypothesis, optimizer):
            raise RuntimeError('native owner rejected the training update')
    with torch.no_grad():
        final_loss = float(masked_loss(hypothesis(inputs), evidence))
    return FittedExplanation(hypothesis, {
        'format': 'glasshelix-fitted-explanation-v1',
        'hypothesis_id': hypothesis_id, 'domain_ids': list(domain_ids),
        'question': 'synthetic shared law calibration and local conditioning',
        'evidence': evidence_manifest(evidence), 'initial_loss': initial_loss,
        'input_sha256': hashlib.sha256(json.dumps(inputs.detach().cpu().tolist(), allow_nan=False).encode()).hexdigest(),
        'final_loss': final_loss, 'global_frozen': False,
        'scope': 'controlled synthetic qualification; biological validation not_run',
        'assumptions': ['joint product response', 'one shared law coefficient',
                        'local activity and effective forcing are distinct unknowns'],
    }, optimizer)


def condition_local(fitted, x, evidence, permission, *, initial_z=0.5,
                    initial_forcing=1.0, independent_z=None, steps=800, lr=0.03):
    """Freeze the shared law; fit local z and forcing to permitted observations.

    Without an independent z observation, only z*forcing is identified. The
    returned optimizer solution is one explanation, not a uniqueness proof.
    """
    permission.authorize(evidence, 'conditioning')
    if independent_z is not None:
        permission.authorize(independent_z, 'conditioning')
        if evidence.correspondence_ids is None or independent_z.correspondence_ids is None:
            raise ValueError('independent observation requires explicit row correspondence IDs')
        if evidence.correspondence_ids != independent_z.correspondence_ids:
            raise ValueError('independent observation row correspondence or order differs')
        if evidence.source_id != independent_z.source_id:
            raise ValueError('independent observation source identity differs')
        if evidence.times != independent_z.times:
            raise ValueError('independent observation row times differ')
        if independent_z.values.shape != x.shape:
            raise ValueError('independent observation shape differs from local unknowns')
    fitted.freeze()
    global_before = [parameter.detach().clone() for parameter in fitted.hypothesis.parameters()]
    z = nn.Parameter(torch.full_like(x, initial_z))
    forcing = nn.Parameter(torch.full_like(x, initial_forcing))
    optimizer = torch.optim.Adam([z, forcing], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = fitted.hypothesis(torch.cat((x, z * forcing), dim=1))
        loss = masked_loss(prediction, evidence)
        if independent_z is not None:
            loss = loss + masked_loss(z, independent_z)
        loss.backward()
        optimizer.step()
    for parameter, before in zip(fitted.hypothesis.parameters(), global_before):
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    with torch.no_grad():
        prediction = fitted.hypothesis(torch.cat((x, z * forcing), dim=1))
    return {'z': z.detach(), 'forcing': forcing.detach(), 'prediction': prediction,
            'evidence': evidence_manifest(evidence),
            'independent_z_evidence': None if independent_z is None else evidence_manifest(independent_z),
            'mode': permission.mode, 'cutoff': permission.cutoff,
            'residual': float(masked_loss(prediction, evidence)),
            'ambiguity': 'z*forcing only' if independent_z is None else 'independent z observed; finite optimizer fit',
            'global_frozen': True}
