#!/usr/bin/env python3
"""Fresh public contract installation; enforce integrated replay when requested."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument('--ce-include', type=Path, required=True)
parser.add_argument('--require-integrated', action='store_true')
args = parser.parse_args()
commands = []
with tempfile.TemporaryDirectory(prefix='gh-substrate-install-') as directory:
    root = Path(directory)
    prefix, build, consumer = root / 'prefix', root / 'build', root / 'consumer'
    def run(command):
        commands.append(list(map(str, command)))
        result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(result.stdout)
    run(['cmake', '-S', str(ROOT / 'cmake/substrate'), '-B', str(build),
         '-DCMAKE_INSTALL_PREFIX=' + str(prefix), '-DCMAKE_INSTALL_LIBDIR=lib',
         '-DGLASSHELIX_REQUIRE_INTEGRATED=' + ('ON' if args.require_integrated else 'OFF')])
    run(['cmake', '--build', str(build), '-j', '2'])
    run(['cmake', '--install', str(build)])
    run(['cmake', '-S', str(ROOT / 'tests/substrate/install'), '-B', str(consumer),
         '-DCMAKE_PREFIX_PATH=' + str(prefix), '-DGLASSHELIX_CELLERATOR_INCLUDE=' + str(args.ce_include)])
    run(['cmake', '--build', str(consumer), '-j', '2'])
    run([str(consumer / 'installed_contracts')])
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    capabilities = subprocess.run([sys.executable, str(prefix / 'bin/glasshelix-substrate'), 'capabilities'],
                                  cwd=root, env=env, check=True, text=True, capture_output=True)
    capability = json.loads(capabilities.stdout)
    manifest = json.loads((prefix / 'share/glasshelix-substrate/install-manifest.json').read_text())
    for relative, expected in manifest['python_source_sha256'].items():
        installed = prefix / 'lib/python' / Path(relative).relative_to('python')
        if hashlib.sha256(installed.read_bytes()).hexdigest() != expected:
            raise RuntimeError('installed source differs from the configure-time manifest')
    if manifest['native_available']:
        raise RuntimeError('baseline installer falsely advertises an unbuilt native component')
    native_source = root / 'native-request'
    native_source.mkdir()
    (native_source / 'CMakeLists.txt').write_text(
        'cmake_minimum_required(VERSION 3.28)\nproject(MissingNative NONE)\n'
        'find_package(GlassHelixSubstrate CONFIG REQUIRED COMPONENTS native)\n')
    unsupported = subprocess.run(['cmake', '-S', str(native_source), '-B', str(root / 'native-build'),
        '-DCMAKE_PREFIX_PATH=' + str(prefix)], cwd=root, text=True, capture_output=True)
    if unsupported.returncode == 0:
        raise RuntimeError('unbuilt native component incorrectly satisfies a required consumer')
    if args.require_integrated:
        run([sys.executable, '-B', str(ROOT / 'tests/substrate/replay/check_installed.py'),
             '--prefix', str(prefix)])
    print(json.dumps({'status': 'passed', 'scope': 'fresh installed actual public contracts',
                      'integrated_requested': args.require_integrated, 'capabilities': capability,
                      'commands': commands, 'install_manifest': manifest}, indent=2))
