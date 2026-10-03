"""Immutable projection of a predecessor audit into reusable split views.

The experiment audit owns record normalization, leakage checks and feature
selection. This layer preserves its outcomes and explicit observability limits.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True)
class Source:
    source_id: str
    kind: str
    audit_evidence_id: str
    records_sha256: str
    audit_implementation_sha256: str
    provenance_manifest_ids: tuple[str, ...] = ()

    def validate(self):
        if (not self.source_id or not self.audit_evidence_id
                or self.kind not in ('synthetic_fixture', 'audited_celltag')
                or not _sha(self.records_sha256) or not _sha(self.audit_implementation_sha256)
                or not isinstance(self.provenance_manifest_ids, tuple)
                or any(not x for x in self.provenance_manifest_ids)):
            raise ValueError('explicit source, audit and input identities required')
        if self.kind == 'audited_celltag' and not self.provenance_manifest_ids:
            raise ValueError('real CellTag source requires independent input provenance references')


@dataclass(frozen=True)
class Record:
    sample_id: str
    cell_id: str
    clone_id: str
    time_label: str
    modality: str
    replicate: str
    regime: str
    split: str
    feature_id: str
    value: float | None
    qc_pass: bool
    qc_reason: str


@dataclass(frozen=True)
class Snapshot:
    sample_id: str
    cell_id: str
    clone_id: str
    replicate: str
    time_label: str
    values: tuple[float | None, ...]
    # Donor identity is not supplied by the current standardized record format.
    donor_id: str | None = None

    @property
    def evidence_id(self):
        # Structured identity avoids delimiter collisions in caller-supplied IDs.
        return json.dumps([self.sample_id, self.cell_id, self.replicate], separators=(',', ':'))

    @property
    def present(self):
        return tuple(value is not None for value in self.values)


@dataclass(frozen=True)
class SnapshotView:
    source: Source
    split: str
    modality: str
    features: tuple[str, ...]
    snapshots: tuple[Snapshot, ...]
    sampling_unit: str = 'population_snapshot'
    correspondence_ids: None = None

    def to_evidence(self, time_mapping, time_units, dtype=None):
        """Explicit conversion for the existing learning evidence owner.

        Original split and modality survive conversion. No numeric conversion of
        released labels, conditioning split override or paired-cell IDs is inferred.
        """
        import torch
        from ..learning import EvidenceView
        dtype = torch.float64 if dtype is None else dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('observation adapter supports explicit float32 or float64 conversion')
        if not time_units or not self.snapshots:
            raise ValueError('nonempty snapshots and explicit time units required')
        times = []
        for snapshot in self.snapshots:
            if snapshot.time_label not in time_mapping:
                raise ValueError('every released time label requires an explicit mapping')
            time = time_mapping[snapshot.time_label]
            if type(time) not in (int, float) or not math.isfinite(time):
                raise ValueError('mapped time must be finite')
            times.append(float(time))
        evidence = EvidenceView(
            torch.tensor([[0. if value is None else value for value in row.values]
                          for row in self.snapshots], dtype=dtype),
            torch.tensor([row.present for row in self.snapshots], dtype=torch.bool),
            tuple(row.evidence_id for row in self.snapshots), self.modality, self.split,
            tuple(times), self.source.source_id, correspondence_ids=None)
        evidence.validate()
        return evidence

    def evidence_manifest(self, time_mapping, time_units, dtype=None):
        from ..learning import evidence_manifest
        evidence = self.to_evidence(time_mapping, time_units, dtype)
        return {'observation': evidence_manifest(evidence), 'time_units': time_units,
                'original_time_labels': [s.time_label for s in self.snapshots],
                'time_mapping': {label: time_mapping[label] for label in sorted(
                    {s.time_label for s in self.snapshots})},
                'features': list(self.features), 'tensor_dtype': str(evidence.values.dtype),
                'sampling_unit': self.sampling_unit, 'correspondence_ids': None}


class AuditedCohort:
    """Own immutable normalized records and the exact predecessor audit receipt."""
    def __init__(self, records, audit, source: Source):
        source.validate()
        self._source = source
        self._records = tuple(Record(**row) for row in records)
        self._audit_json = json.dumps(audit, sort_keys=True, allow_nan=False)
        raw = [asdict(row) for row in self._records]
        if (source.records_sha256 != audit.get('source_sha256')
                or canonical_hash(raw) != audit.get('canonical_records_sha256')
                or len(raw) != audit.get('unique_records')):
            raise ValueError('records or source differ from predecessor audit')
        manifest = audit.get('feature_manifest', {})
        features = manifest.get('features', [])
        if (not features or features != sorted(set(features))
                or manifest.get('modality') != 'RNA' or manifest.get('split') != 'train'
                or manifest.get('regime') != 'common'):
            raise ValueError('frozen training RNA feature manifest required')
        self._features = tuple(features)
        training = audit.get('training_records', [])
        if canonical_hash(training) != manifest.get('training_records_sha256'):
            raise ValueError('training audit hash differs')
        source_rows = {canonical_hash(row) for row in raw}
        for row in training:
            if (canonical_hash(row) not in source_rows or row['modality'] != 'RNA'
                    or row['split'] != 'train' or row['regime'] != 'common' or not row['qc_pass']
                    or row['value'] is None or row['feature_id'] not in self.features):
                raise ValueError('training receipt contains unauthorized source records')

    @property
    def source(self):
        return self._source

    @property
    def features(self):
        return self._features

    @property
    def records(self):
        return self._records

    @property
    def audit(self):
        return json.loads(self._audit_json)

    @property
    def limitations(self):
        return {'biological_status': 'not_run', 'donor_identity': 'unavailable',
                'individual_trajectories': 'unobserved', 'modality_pairing': 'unverified',
                'clone_meaning': 'released membership metadata; no inferred parent-child links'}

    def view(self, split, modality='RNA', features=None):
        if split not in ('train', 'validation', 'test') or modality not in ('RNA', 'ATAC'):
            raise ValueError('unknown split or modality')
        if split == 'train' and modality != 'RNA':
            raise PermissionError('ATAC is independent validation evidence')
        if modality == 'RNA':
            features = self.features if features is None else tuple(features)
            if not set(features) <= set(self.features):
                raise ValueError('RNA views must use frozen training features')
        elif features is None:
            raise ValueError('ATAC validation requires an explicit independent feature declaration')
        else:
            features = tuple(features)
        if not features or len(set(features)) != len(features) or any(not x for x in features):
            raise ValueError('unique nonempty feature declarations required')
        grouped = {}
        for row in self.records:
            if (row.split != split or row.modality != modality or row.regime != 'common'
                    or not row.qc_pass):
                continue
            key = (row.sample_id, row.cell_id, row.replicate)
            group = grouped.setdefault(key, {'identity': row, 'values': {}})
            if row.feature_id in features:
                group['values'][row.feature_id] = row.value
        snapshots = []
        for key in sorted(grouped):
            group = grouped[key]
            row = group['identity']
            snapshots.append(Snapshot(row.sample_id, row.cell_id, row.clone_id, row.replicate,
                                      row.time_label, tuple(group['values'].get(f) for f in features)))
        return SnapshotView(self.source, split, modality, features, tuple(snapshots))

    def dumps(self):
        body = {'source': asdict(self.source), 'records': [asdict(r) for r in self.records],
                'audit': self.audit, 'limitations': self.limitations}
        return json.dumps({'schema': 1, 'body': body, 'sha256': canonical_hash(body)},
                          sort_keys=True, allow_nan=False)

    @classmethod
    def loads(cls, text):
        def reject(value):
            raise ValueError(f'nonfinite replay value: {value}')
        payload = json.loads(text, parse_constant=reject)
        if payload.get('schema') != 1 or set(payload) != {'schema', 'body', 'sha256'}:
            raise ValueError('unsupported cohort replay schema')
        body = payload['body']
        if canonical_hash(body) != payload['sha256']:
            raise ValueError('cohort replay hash differs')
        source = dict(body['source'])
        source['provenance_manifest_ids'] = tuple(source['provenance_manifest_ids'])
        result = cls(body['records'], body['audit'], Source(**source))
        if body['limitations'] != result.limitations:
            raise ValueError('cohort replay changed observability restrictions')
        return result
