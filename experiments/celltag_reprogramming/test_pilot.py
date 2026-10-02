"""Synthetic fixtures only; CPU tests never exercise the native GPU path."""
import csv
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

try:
    from .data import FIELDS, read_records
    from .pilot import (FIXTURE_FEATURES, FlexibleControl, clone_abundance,
                        fixture_records, main, prepare, run)
except ImportError:
    from data import FIELDS, read_records
    from pilot import (FIXTURE_FEATURES, FlexibleControl, clone_abundance,
                       fixture_records, main, prepare, run)


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='celltag-pilot-test-synthetic-')
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'fixture.csv'
        fixture_records(self.path)

    def change_rows(self, transform):
        with self.path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        transform(rows)
        with self.path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_inputs_exclude_clone_ids_atac_and_original_labels_survive(self):
        original = prepare(self.path, FIXTURE_FEATURES)
        def change(rows):
            for row in rows:
                row['clone_id'] = 'renamed_' + row['clone_id']
                if row['modality'] == 'ATAC':
                    row['value'] = '0'
        self.change_rows(change)
        changed = prepare(self.path, FIXTURE_FEATURES)
        for split in ('train', 'test'):
            torch.testing.assert_close(original[split]['inputs'], changed[split]['inputs'])
            torch.testing.assert_close(original[split]['targets'], changed[split]['targets'])
        self.assertEqual(original['audit']['time_labels'], ['11', '12', '3'])
        self.assertEqual(len(original['train']['inputs']), 8)
        self.assertEqual(len(original['test']['inputs']), 3)
        self.assertEqual(original['train']['inputs'].shape[1], 4)
        self.assertEqual(int((~original['train']['mask']).sum()), 1)

    def test_clone_split_leakage_rejected(self):
        def change(rows):
            train_clone = next(row['clone_id'] for row in rows if row['split'] == 'train')
            for row in rows:
                if row['split'] == 'test':
                    row['clone_id'] = train_clone
        self.change_rows(change)
        with self.assertRaisesRegex(ValueError, 'clone split leakage'):
            prepare(self.path, FIXTURE_FEATURES)

    def test_cutoff_and_feature_authorization(self):
        problem = prepare(self.path, FIXTURE_FEATURES, cutoff=3)
        self.assertTrue(all(time <= 3 for time in problem['train']['times']))
        self.assertGreater(problem['exclusions']['training_after_cutoff'], 0)
        bad = dict(FIXTURE_FEATURES, inputs=['fixture_withheld_peak', *FIXTURE_FEATURES['inputs'][1:]])
        with self.assertRaises(ValueError):
            prepare(self.path, bad)
        with self.assertRaisesRegex(ValueError, 'finite'):
            prepare(self.path, FIXTURE_FEATURES, cutoff=float('nan'))

    def test_snapshot_and_flexible_controls_receive_declared_information(self):
        inputs = torch.tensor([[.2, .3, .4, .25], [.2, .3, .4, 1.]])
        snapshot, flexible = FlexibleControl(True), FlexibleControl(False)
        torch.testing.assert_close(snapshot(inputs)[0], snapshot(inputs)[1])
        self.assertFalse(torch.equal(flexible(inputs)[0], flexible(inputs)[1]))

    def test_abundance_preserves_cells_and_does_not_emit_fate_probabilities(self):
        rows, _ = read_records(self.path)
        abundance = clone_abundance(rows)
        self.assertEqual(sum(entry['sampled_cells'] for entry in abundance), 11)
        self.assertTrue(all('sampled_abundance' in entry for entry in abundance))
        self.assertTrue(all('fate_probability' not in entry for entry in abundance))
        self.assertTrue(all('survival' in entry['interpretation'] for entry in abundance))

    def test_cpu_fixture_controls_and_explicit_missing_input(self):
        output = self.path.with_suffix('.json')
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(['--fixture', '--output', str(output)]), 0)
        result = json.loads(output.read_text())
        marker = json.loads(stdout.getvalue().split('GH_ML2_PILOT_EVIDENCE=', 1)[1])
        self.assertEqual(marker['result_sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
        manifest = Path(marker['manifest_path'])
        self.assertEqual(marker['manifest_sha256'], hashlib.sha256(manifest.read_bytes()).hexdigest())
        self.assertEqual(set(marker['source_hashes']), {'pilot', 'test_pilot', 'data', 'reference', 'learning'})
        self.assertEqual(result['biological_status'], 'not_run')
        self.assertEqual(result['native_comparison']['status'], 'not_run')
        self.assertEqual(set(result['controls']), {'structured_torch', 'snapshot_only', 'flexible_same_information'})
        self.assertFalse(result['permissions']['clone_ID_predictor'])
        self.assertLess(result['controls']['structured_torch']['train_loss'], .001)
        with self.assertRaises(SystemExit) as raised, redirect_stderr(io.StringIO()):
            main(['--records', str(self.path.with_name('unavailable.csv')),
                  '--output', str(self.path.with_suffix('.json'))])
        self.assertEqual(raised.exception.code, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
