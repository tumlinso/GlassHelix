#!/usr/bin/env python3
"""Read-only verification of receipt, installed bytes, and consumer provenance."""
import hashlib
import json
import pathlib
import subprocess


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    here = pathlib.Path(__file__).resolve().parent
    receipt = json.loads((here / 'ce-native-acceptance.json').read_text())
    manifest_path = here / 'install-manifest.json'
    require(sha(manifest_path) == receipt['install_manifest_sha256'], 'install manifest changed')
    manifest = json.loads(manifest_path.read_text())
    result_path = here / 'consumer-result.json'
    result = json.loads(result_path.read_text())
    require(result['status'] == 'passed' and len(result['checks']) == 8, 'consumer qualification absent')
    require(result['source_commit'] == manifest['source_commit'] == receipt['cellerator_source_commit'], 'source identities differ')
    source = pathlib.Path(manifest['source_root'])
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    require(actual == manifest['source_commit'], 'Cellerator HEAD changed')
    require(result['consumer_sha256'] == receipt['consumer_sha256'] == sha(here / 'installed_consumer.py'), 'executed consumer source changed')
    require(result['install_manifest_sha256'] == sha(manifest_path), 'consumer used different manifest')
    installed = pathlib.Path(manifest['install_target'])
    for relative, digest in manifest['package_files_sha256'].items():
        require(sha(installed / relative) == digest, 'installed package changed: ' + relative)
        require(sha(source / 'components/CelleraTorch/python' / relative) == digest, 'package source changed: ' + relative)
    require(pathlib.Path(result['adapter_file']).resolve() == (installed / 'celleratorch/moonshot_product.py').resolve(), 'consumer imported different adapter')
    require(result['adapter_sha256'] == sha(result['adapter_file']), 'consumer adapter changed')
    require(result['native_library'] == manifest['native_library'], 'consumer used different native library')
    require(result['native_library_sha256'] == manifest['native_library_sha256'] == sha(manifest['native_library']), 'native library changed')
    require(result['device'] == 'cpu' and result['precision'] == 'float32', 'unqualified framework device/precision')
    for relative, digest in receipt['producer_source_sha256'].items():
        require(sha(source / relative) == digest, 'producer source changed: ' + relative)
    require(sha(source / 'experiments/moonshot-parallel-v1/integration/capability.json') == receipt['producer_capability_sha256'], 'producer capability changed')
    for check in receipt['checks']:
        if check['project'] == 'glasshelix':
            require(check['sha256'] == sha(result_path), 'consumer result changed')
    print('Installed CPU product2 consumer source, package, library and result hashes passed')


if __name__ == '__main__':
    main()
