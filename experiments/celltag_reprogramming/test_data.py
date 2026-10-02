"""Synthetic fixture qualification only; these rows are not CellTag evidence."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

try:
    from .data import FIELDS, audit, main, read_records
except ImportError:
    from data import FIELDS, audit, main, read_records


def fixture(**changes):
    row = dict(sample_id='fixture_sample', cell_id='fixture_cell',
               clone_id='fixture_clone', time_label='11', modality='RNA',
               replicate='fixture_rep1', regime='common', split='train',
               feature_id='fixture_gene', value='0', qc_pass='true', qc_reason='')
    row.update(changes)
    return row


class RecordAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='celltag-synthetic-fixture-')
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'fixture_records.csv'

    def write(self, rows):
        with self.path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=(*FIELDS, 'qc_pass', 'qc_reason'))
            writer.writeheader()
            writer.writerows(rows)
        return self.path

    def test_deterministic_join_and_duplicate_collapse(self):
        rows = [fixture(), fixture(feature_id='fixture_gene2', value='3'), fixture()]
        report = audit(self.write(rows))
        shuffled = audit(self.write(list(reversed(rows))))
        self.assertEqual(report['canonical_records_sha256'], shuffled['canonical_records_sha256'])
        self.assertEqual(report['feature_manifest'], shuffled['feature_manifest'])
        self.assertEqual(report['identical_duplicates_collapsed'], 1)
        self.assertEqual(report['unique_records'], 2)

    def test_conflicting_duplicate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'conflicting duplicate'):
            read_records(self.write([fixture(), fixture(value='4')]))

    def test_clone_split_isolation_and_cell_metadata(self):
        with self.assertRaisesRegex(ValueError, 'clone split leakage'):
            audit(self.write([fixture(), fixture(sample_id='fixture_other', cell_id='fixture_other', split='test')]))
        with self.assertRaisesRegex(ValueError, 'conflicting cell metadata'):
            audit(self.write([fixture(), fixture(feature_id='fixture_other', time_label='12')]))

    def test_training_selection_excludes_atac_perturbation_and_holdouts(self):
        rows = [fixture(), fixture(modality='ATAC', feature_id='fixture_peak'),
                fixture(sample_id='fixture_perturbation', cell_id='fixture_p', clone_id='fixture_p',
                        regime='Zfp281_perturbation', feature_id='fixture_perturbation_only'),
                fixture(sample_id='fixture_test', cell_id='fixture_t', clone_id='fixture_t',
                        split='test', feature_id='fixture_test_only')]
        report = audit(self.write(rows))
        self.assertEqual(report['feature_manifest']['features'], ['fixture_gene'])
        self.assertEqual(len(report['training_records']), 1)
        for forbidden in ('fixture_peak', 'fixture_perturbation_only', 'fixture_test_only'):
            with self.assertRaisesRegex(ValueError, 'eligible training RNA'):
                audit(self.path, [forbidden])
        rows[-1]['value'] = '999999'
        changed = audit(self.write(rows))
        self.assertEqual(report['feature_manifest'], changed['feature_manifest'])
        self.assertNotEqual(report['source_sha256'], changed['source_sha256'])

    def test_labels_coverage_missingness_and_zero_observation(self):
        rows = [fixture(), fixture(sample_id='fixture_later', cell_id='fixture_later', time_label='12',
                                   feature_id='fixture_missing', value=''),
                fixture(modality='ATAC', feature_id='fixture_peak', value='')]
        report = audit(self.write(rows))
        self.assertEqual(report['time_labels'], ['11', '12'])
        self.assertEqual(report['missing_values_by_modality'], {'ATAC': 1, 'RNA': 1})
        self.assertEqual(report['training_records'][0]['value'], 0)
        self.assertEqual(len(report['coverage']), 3)
        self.assertTrue(all(entry['cells'] == 1 for entry in report['coverage']))

    def test_qc_exclusion_and_feature_whitelist(self):
        rows = [fixture(), fixture(feature_id='fixture_second', value='2'),
                fixture(feature_id='fixture_failed', qc_pass='false', qc_reason='fixture_low_quality')]
        report = audit(self.write(rows), ['fixture_gene'])
        self.assertEqual(report['feature_manifest']['selection'], 'explicit_whitelist')
        self.assertEqual(report['qc_failure_reasons'], {'fixture_low_quality': 1})
        self.assertEqual(report['training_exclusion_counts']['unselected_feature'], 1)
        self.assertEqual(len(report['training_records']), 1)

    def test_invalid_input_and_no_eligible_training_data(self):
        for changes in ({'value': 'nan'}, {'modality': 'OTHER'}, {'split': 'other'}, {'qc_pass': 'maybe'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                audit(self.write([fixture(**changes)]))
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            audit(self.write([fixture(value='')]))
        self.path.write_text('sample_id,value\nfixture,1\n')
        with self.assertRaisesRegex(ValueError, 'requires columns'):
            audit(self.path)

    def test_cli_emits_valid_hashed_audit(self):
        self.write([fixture()])
        output = Path(self.directory.name) / 'audit.json'
        self.assertEqual(main(['--records', str(self.path), '--output', str(output),
                               '--feature', 'fixture_gene']), 0)
        report = json.loads(output.read_text())
        self.assertEqual(report['unique_records'], 1)
        self.assertEqual(len(report['source_sha256']), 64)
        self.assertEqual(report['feature_manifest']['features'], ['fixture_gene'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
