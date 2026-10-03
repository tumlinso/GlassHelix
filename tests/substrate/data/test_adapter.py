import csv
import json
from pathlib import Path
import tempfile
import unittest

from glasshelix.data import AuditedCohort
from glasshelix.learning import EvidencePermission
from experiments.celltag_reprogramming import data as predecessor
from experiments.celltag_reprogramming.data_adapter import load_cohort, replay_cohort

ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT / 'docs/learning/pilot-evidence/pilot.records.csv'


class CohortAdapter(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'records.csv'
        self.rows = [
            ['s1', 'c1', 'clone-train', 'day-03', 'RNA', 'r1', 'common', 'train', 'gene', '0'],
            ['s1', 'c1', 'clone-train', 'day-03', 'RNA', 'r1', 'common', 'train', 'missing', ''],
            ['s2', 'c2', 'clone-train-2', 'day-03', 'RNA', 'r1', 'common', 'train', 'missing', ''],
            ['s3', 'c3', 'clone-validation', 'day-14', 'RNA', 'r2', 'common', 'validation', 'gene', '9'],
            ['s3', 'c3', 'clone-validation', 'day-14', 'RNA', 'r2', 'common', 'validation', 'heldout', '7'],
            ['s3', 'c3', 'clone-validation', 'day-14', 'ATAC', 'r2', 'common', 'validation', 'peak', '4'],
            ['s4', 'c4', 'clone-perturbed', 'day-14', 'RNA', 'r2', 'Zfp281_perturbation', 'train', 'perturbation', '5'],
        ]
        self.write(self.rows)

    def write(self, rows):
        with self.path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(predecessor.FIELDS)
            writer.writerows(rows)

    def load(self, **kwargs):
        return load_cohort(self.path, source_id='fixture', audit_evidence_id='audit', **kwargs)

    def test_existing_pilot_cohort_reused_and_replayed(self):
        cohort = load_cohort(PILOT, source_id='retained-pilot-fixture', audit_evidence_id='GH-ML2-PILOT')
        self.assertEqual(cohort.audit, predecessor.audit(PILOT))
        self.assertEqual(cohort.limitations['biological_status'], 'not_run')
        replay = replay_cohort(cohort.dumps(), PILOT)
        self.assertEqual(replay.dumps(), cohort.dumps())
        self.assertTrue(cohort.view('train').snapshots)
        self.assertEqual({s.time_label for s in cohort.view('train').snapshots},
                         {r.time_label for r in cohort.records if r.split == 'train' and r.regime == 'common'})

    def test_training_only_features_zero_missing_and_units(self):
        cohort = self.load()
        self.assertEqual(cohort.features, ('gene',))
        view = cohort.view('train')
        self.assertEqual(view.snapshots[0].values, (0.,))
        self.assertEqual(view.snapshots[0].present, (True,))
        self.assertEqual(view.snapshots[1].values, (None,))
        self.assertEqual(view.snapshots[1].present, (False,))
        self.assertEqual(view.snapshots[0].clone_id, 'clone-train')
        self.assertIsNone(view.snapshots[0].donor_id)
        evidence = view.to_evidence({'day-03': 3}, 'days')
        self.assertEqual(evidence.observed.tolist(), [[True], [False]])
        self.assertEqual(evidence.split, 'train')
        self.assertEqual(evidence.times, (3., 3.))
        self.assertIsNone(evidence.correspondence_ids)
        EvidencePermission().authorize(evidence, 'global_fit')
        manifest = view.evidence_manifest({'day-03': 3}, 'days')
        self.assertEqual(manifest['time_units'], 'days')
        self.assertEqual(manifest['original_time_labels'], ['day-03', 'day-03'])
        self.assertEqual(manifest['observation']['observed_values'], [0.])

    def test_independent_validation_and_no_pairing(self):
        cohort = self.load()
        with self.assertRaises(PermissionError):
            cohort.view('train', 'ATAC', ('peak',))
        with self.assertRaises(ValueError):
            cohort.view('validation', 'RNA', ('heldout',))
        with self.assertRaises(ValueError):
            cohort.view('validation', 'ATAC')
        atac = cohort.view('validation', 'ATAC', ('peak',))
        evidence = atac.to_evidence({'day-14': 14}, 'days')
        self.assertIsNone(atac.correspondence_ids)
        self.assertEqual(atac.sampling_unit, 'population_snapshot')
        self.assertEqual(evidence.modality, 'ATAC')
        with self.assertRaises(PermissionError):
            EvidencePermission().authorize(evidence, 'global_fit')
        with self.assertRaises(ValueError):
            cohort.view('train').to_evidence({}, 'days')

    def test_clone_leakage_and_conflicting_join_use_original_audit(self):
        changed = [list(row) for row in self.rows]
        changed[3][2] = 'clone-train'
        self.write(changed)
        with self.assertRaisesRegex(ValueError, 'clone split leakage'):
            self.load()
        changed = [list(row) for row in self.rows]
        changed.append(list(changed[0]))
        changed[-1][-1] = '5'
        self.write(changed)
        with self.assertRaisesRegex(ValueError, 'conflicting duplicate'):
            self.load()

    def test_replay_hash_source_and_observability_restrictions(self):
        cohort = self.load()
        text = cohort.dumps()
        self.assertEqual(AuditedCohort.loads(text).view('train'), cohort.view('train'))
        payload = json.loads(text)
        payload['body']['records'][0]['value'] = 4
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            AuditedCohort.loads(json.dumps(payload))
        changed = [list(row) for row in self.rows]
        changed[0][-1] = '1'
        self.write(changed)
        with self.assertRaisesRegex(ValueError, 'changed'):
            replay_cohort(text, self.path)

    def test_explicit_whitelist_and_missing_external_input(self):
        cohort = self.load(features=('gene',))
        self.assertEqual(replay_cohort(cohort.dumps(), self.path).dumps(), cohort.dumps())
        with self.assertRaises(ValueError):
            self.load(features=('heldout',))
        with self.assertRaises(ValueError):
            self.load(source_kind='audited_celltag')
        with self.assertRaises(FileNotFoundError):
            load_cohort(self.path.with_name('missing.csv'), source_id='missing', audit_evidence_id='audit')


if __name__ == '__main__':
    unittest.main()
