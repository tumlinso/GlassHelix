#!/usr/bin/env python3
"""Package linter regressions only; no Todo/Git/repository changes."""
import copy
import json
from pathlib import Path
import unittest
from check_package import lint_plan, PackageError
ROOT=Path(__file__).resolve().parents[1]
class PackageTests(unittest.TestCase):
    def setUp(self):self.plan=json.loads((ROOT/'machine/glasshelix.todo-plan.json').read_text())
    def reject(self, mutate):
        mutate(self.plan)
        with self.assertRaises(PackageError):lint_plan(self.plan,'GH-L1')
    def test_valid_plan(self):self.assertEqual(lint_plan(self.plan,'GH-L1')['outcomes'],5)
    def test_duplicate_id(self):self.reject(lambda p:p['tasks'].append(copy.deepcopy(p['tasks'][1])))
    def test_foreign_dependency(self):self.reject(lambda p:p['tasks'][1]['depends_on'].append({'type':'task','task_id':'CE-L1-ENABLE'}))
    def test_epic_dependency(self):self.reject(lambda p:p['tasks'][1]['depends_on'].append({'type':'task','task_id':'GH-L1-0000'}))
    def test_path_escape(self):self.reject(lambda p:p['tasks'][1]['scope']['exclusive_paths'].append('../cellerator'))
    def test_pretend_completion(self):self.reject(lambda p:p['tasks'][1].update({'status':'done'}))
    def test_coordinator_cannot_replace_writable_lane(self):self.reject(lambda p:p['runs'][0]['lanes'][0].update({'role':'coordinator'}))
    def test_cycle(self):self.reject(lambda p:p['tasks'][1]['depends_on'].append({'type':'task','task_id':'GH-L1-LEARN'}))
    def test_bad_queue(self):self.reject(lambda p:p['runs'][0]['lanes'][0]['tasks'].reverse())
    def test_placeholder_gate(self):self.reject(lambda p:p['tasks'][1].update({'gates':[{'id':'fake','type':'command','argv':['true']}]}))
if __name__=='__main__':unittest.main()
