#!/usr/bin/env python3
"""Explicit installed GH toolkit plus accepted experiment sources, CPU only."""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gh-prefix', type=Path, required=True)
parser.add_argument('--native-refactor', type=Path, required=True)
parser.add_argument('--ce-core-commit', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[3]
installed = args.gh_prefix.resolve() / 'lib/python'
sys.path[:0] = [str(installed), str(ROOT)]
import glasshelix.learning
import glasshelix.models
import glasshelix.experiment
import glasshelix.data
import glasshelix.refactoring
import torch
torch.set_num_threads(2)
modules = (glasshelix.learning, glasshelix.models, glasshelix.experiment,
           glasshelix.data, glasshelix.refactoring)
if not all(Path(module.__file__).resolve().is_relative_to(installed) for module in modules):
    raise RuntimeError('pipeline requires the declared installed scientific toolkit')
manifest = json.loads((args.gh_prefix / 'share/glasshelix-substrate/install-manifest.json').read_text())
from experiments.celltag_reprogramming.integrated.pipeline import run
summary = run(args.output, args.native_refactor.resolve(), args.ce_core_commit, manifest['source_revision'])
print(json.dumps({'status': summary['status'], 'evidence': str(args.output.resolve() / 'summary.json'),
                  'framework_python': sys.executable, 'torch_version': torch.__version__,
                  'device': 'cpu', 'torch_threads': torch.get_num_threads(),
                  'installed_modules': [module.__file__ for module in modules]}))
