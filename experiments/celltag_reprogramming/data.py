"""Deterministic audit of released-label records; no inferred trajectories."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

FIELDS = ('sample_id', 'cell_id', 'clone_id', 'time_label', 'modality',
          'replicate', 'regime', 'split', 'feature_id', 'value')
JOIN = ('sample_id', 'cell_id', 'modality', 'replicate', 'feature_id')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read_records(path):
    """Read long-form CSV; exact duplicate rows collapse, conflicts fail."""
    unique, cell_metadata, clones = {}, {}, {}
    duplicates = 0
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError('CSV requires columns: ' + ','.join(FIELDS))
        for line, raw in enumerate(reader, 2):
            if None in raw or any(raw.get(field) is None for field in FIELDS):
                raise ValueError(f'malformed CSV row {line}')
            row = {field: raw[field] for field in FIELDS}
            for field in FIELDS[:-1]:
                if field != 'clone_id' and not row[field].strip():
                    raise ValueError(f'empty {field} on row {line}')
            if row['split'] not in ('train', 'validation', 'test'):
                raise ValueError(f'unknown split on row {line}')
            if row['modality'] not in ('RNA', 'ATAC'):
                raise ValueError(f'unknown modality on row {line}')
            row['value'] = None if row['value'].strip() == '' else float(row['value'])
            if row['value'] is not None and not math.isfinite(row['value']):
                raise ValueError(f'nonfinite observed value on row {line}')
            qc = raw.get('qc_pass', 'true')
            if qc not in ('true', 'false'):
                raise ValueError(f'qc_pass must be true or false on row {line}')
            row['qc_pass'] = qc == 'true'
            row['qc_reason'] = raw.get('qc_reason', '')
            key = tuple(row[field] for field in JOIN)
            if key in unique:
                if unique[key] != row:
                    raise ValueError(f'conflicting duplicate join key {key}')
                duplicates += 1
                continue
            # Consistent metadata does not assert same-cell paired measurements.
            cell_key = (row['sample_id'], row['cell_id'], row['replicate'])
            metadata = tuple(row[field] for field in ('clone_id', 'time_label', 'regime', 'split'))
            if cell_key in cell_metadata and cell_metadata[cell_key] != metadata:
                raise ValueError(f'conflicting cell metadata {cell_key}')
            cell_metadata[cell_key] = metadata
            clone = row['clone_id']
            if clone:
                if clone in clones and clones[clone] != row['split']:
                    raise ValueError(f'clone split leakage: {clone}')
                clones[clone] = row['split']
            unique[key] = row
    return [unique[key] for key in sorted(unique)], duplicates


def audit(path, features=None):
    """Freeze observed training RNA features in the nominal common regime.

    Clone IDs are audit metadata only. ATAC, validation/test values, perturbation
    and QC failures never select features or enter the returned training rows.
    Empty values are missing observations; a measured zero remains observed.
    """
    path = Path(path)
    rows, duplicates = read_records(path)
    candidates = [row for row in rows if row['split'] == 'train'
                  and row['modality'] == 'RNA' and row['regime'] == 'common'
                  and row['qc_pass'] and row['value'] is not None]
    candidate_keys = {tuple(row[field] for field in JOIN) for row in candidates}
    available = {row['feature_id'] for row in candidates}
    selected = sorted(available if features is None else set(features))
    if not selected or not set(selected).issubset(available):
        raise ValueError('feature selection must be nonempty and observed in eligible training RNA')
    training = [row for row in candidates if row['feature_id'] in selected]
    exclusions, missingness = Counter(), Counter()
    coverage = defaultdict(lambda: {'cells': set(), 'records': 0, 'missing': 0})
    for row in rows:
        key = tuple(row[field] for field in ('clone_id', 'time_label', 'modality', 'replicate'))
        entry = coverage[key]
        entry['cells'].add((row['sample_id'], row['cell_id']))
        entry['records'] += 1
        entry['missing'] += row['value'] is None
        if row['value'] is None:
            missingness[row['modality']] += 1
        if row['regime'] != 'common':
            exclusions['non_common_regime'] += 1
        if not row['qc_pass']:
            exclusions['qc_failure'] += 1
        if row['modality'] != 'RNA':
            exclusions['withheld_ATAC'] += 1
        if row['split'] != 'train':
            exclusions['held_out_split'] += 1
        if row['value'] is None:
            exclusions['missing_value'] += 1
        if tuple(row[field] for field in JOIN) in candidate_keys and row['feature_id'] not in selected:
            exclusions['unselected_feature'] += 1
    coverage_rows = []
    for key, entry in sorted(coverage.items()):
        coverage_rows.append(dict(zip(('clone_id', 'time_label', 'modality', 'replicate'), key),
                                  cells=len(entry['cells']), records=entry['records'], missing=entry['missing']))
    return {
        'schema_version': 1,
        'scope': 'standardized record audit; input provenance determines biological status',
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'canonical_records_sha256': canonical_hash(rows),
        'unique_records': len(rows), 'identical_duplicates_collapsed': duplicates,
        'time_labels': sorted({row['time_label'] for row in rows}),
        'coverage': coverage_rows,
        'missing_values_by_modality': dict(sorted(missingness.items())),
        'missing_clone_records': sum(not row['clone_id'] for row in rows),
        'qc_failure_reasons': dict(sorted(Counter(row['qc_reason'] or 'unspecified'
                                                 for row in rows if not row['qc_pass']).items())),
        'training_exclusion_counts': dict(sorted(exclusions.items())),
        'exclusion_counts_overlap': True,
        'feature_manifest': {'features': selected, 'modality': 'RNA', 'split': 'train',
                             'regime': 'common', 'selection': 'explicit_whitelist' if features is not None else 'observed_training_features',
                             'training_records_sha256': canonical_hash(training)},
        'training_records': training,
        'assumptions': ['clone IDs are globally unique in this standardized input',
                        'clone calls and split provenance require independent authorization',
                        'no time interpolation, modality pairing or parent-child links inferred'],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature', action='append', help='Training RNA whitelist; repeat for each feature')
    args = parser.parse_args(argv)
    report = audit(args.records, args.feature)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'unique_records': report['unique_records'],
                      'training_records': len(report['training_records']),
                      'source_sha256': report['source_sha256']}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
