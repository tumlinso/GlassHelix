"""Thin adapter to the real predecessor audit; no replacement preprocessing."""
import hashlib
from pathlib import Path

from glasshelix.data import AuditedCohort, Source
from .. import data as predecessor


def load_cohort(records_path, *, source_id, audit_evidence_id,
                source_kind='synthetic_fixture', features=None, provenance_manifest_ids=()):
    path = Path(records_path)
    # Missing external inputs fail at their actual path. No fixture substitution.
    report = predecessor.audit(path, features)
    records, _ = predecessor.read_records(path)
    source = Source(source_id, source_kind, audit_evidence_id, report['source_sha256'],
                    hashlib.sha256(Path(predecessor.__file__).read_bytes()).hexdigest(),
                    tuple(provenance_manifest_ids))
    return AuditedCohort(records, report, source)


def replay_cohort(text, records_path):
    retained = AuditedCohort.loads(text)
    selection = retained.audit['feature_manifest']['selection']
    features = retained.features if selection == 'explicit_whitelist' else None
    source = retained.source
    rebuilt = load_cohort(records_path, source_id=source.source_id,
                          audit_evidence_id=source.audit_evidence_id,
                          source_kind=source.kind, features=features,
                          provenance_manifest_ids=source.provenance_manifest_ids)
    if rebuilt.dumps() != retained.dumps():
        raise ValueError('source input or predecessor audit changed since the retained cohort')
    return rebuilt
