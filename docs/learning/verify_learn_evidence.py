#!/usr/bin/env python3
"""Verify retained installed LEARN evidence without importing Torch or using GPUs.

Source hashes permit unrelated documentation commits. Evidence paths never select
the source, installed package, native library, or SDK files being hashed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
GH_PREFIX = Path('/tmp/gh-ml2-installed-sdk')
CT_PACKAGE = Path('/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/lib/python3.13/site-packages/celleratorch/__init__.py')
CT_NATIVE = Path('/home/tumlinson/Software/cellerator-ml2-python/lib/libcellera_torch_mechanism.so')
CE_SDK = Path('/home/tumlinson/Cellerator/build-ml2-cpp')
BRIDGE_WORK = Path('/tmp/gh-ml2-native-fitted')
CHECKS = ('native_training', 'global_fit', 'frozen_conditioning',
          'complete_objective_gradient', 'checkpoint_fresh_process', 'evidence_permissions')
SCIENTIFIC_FIELDS = {
    'model_id': 7001, 'model_revision': 1, 'domain_id': 101, 'order_id': 102,
    'geometry_id': 103, 'partition_id': 104, 'structure_id': 105,
    'structure_epoch': 1, 'evidence_id': 8001, 'source_extent': 2,
    'target_extent': 1, 'target_domain_id': 201, 'target_order_id': 202,
    'target_geometry_id': 203, 'target_partition_id': 204,
    'parameter_domain_id': 301, 'parameter_order_id': 302,
    'parameter_geometry_id': 303, 'parameter_partition_id': 304,
    'parameter_extent': 1, 'coefficient_id': 401, 'mechanism_label': 'joint_product',
    'mechanism_id': 501, 'hypothesis_label': 'synthetic-product-7001-v1',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f'nonfinite JSON constant: {value}')))


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f'{label} must be finite')
    return value


def close(actual, expected, label, tolerance=2e-5):
    finite(actual, label)
    finite(expected, label)
    require(math.isclose(actual, expected, rel_tol=tolerance, abs_tol=2e-6),
            f'{label} differs from fitted value')


def mapping():
    result = {'hypothesis': {'label': 'synthetic-product-7001-v1', 'system_id': 7001},
              'mechanism': {'label': 'joint_product', 'identity': [501, 1501]},
              'coefficient': {'label': 'shared_k', 'identity': [401, 1401]}}
    for name, base, slots in (('source_axis', 101, ['input_x', 'local_z*forcing']),
                              ('target_axis', 201, ['response']),
                              ('parameter_axis', 301, ['shared_k'])):
        result[name] = {key: [base + offset, base + offset + 1000]
                        for offset, key in enumerate(('domain', 'order', 'geometry', 'partition'))}
        result[name]['slots'] = slots
    return result


def verify_ce_bio(directory):
    manifest_path = directory / 'ce-bio-manifest.json'
    manifest = load(manifest_path)
    receipt = load(directory / 'ce-bio-receipt.json')
    require(manifest['task'] == receipt['task'] == 'CE-ML2-BIO'
            and receipt['status'] == 'passed', 'CE BIO producer was not qualified')
    require(receipt['evidence_sha256']['manifest.json'] == sha(manifest_path),
            'CE BIO receipt does not bind retained manifest')
    require(manifest['environment']['installed_package'] == str(CT_PACKAGE.parent),
            'CE BIO installed package location differs')
    # Hash only these pinned installed files; manifest input paths are lookup
    # keys and cannot select arbitrary files to satisfy producer identity.
    for path in (CT_PACKAGE, CT_PACKAGE.parent / 'mechanism.py',
                 CT_PACKAGE.parent / 'biology.py', CT_NATIVE):
        require(manifest['input_sha256'][str(path)] == sha(path),
                f'CE BIO installed producer changed: {path.name}')
    require(manifest['native_library']['path'] == str(CT_NATIVE)
            and manifest['native_library']['sha256'] == sha(CT_NATIVE),
            'CE BIO native library identity differs')


def verify_controller(path, directory, result_path, manifest_path, sources):
    controller = load(path)
    require(controller['classification'] == 'decision-result', 'controller classification')
    require(controller['status'] == 'succeeded' and controller['ok'] is True
            and type(controller['returncode']) is int and controller['returncode'] == 0,
            'controller run did not succeed')
    uuid.UUID(controller['evidence_id'])
    for field, artifact in (('stdout_sha256', directory / 'controller-stdout.txt'),
                            ('stderr_sha256', directory / 'controller-stderr.txt'),
                            ('result_sha256', result_path), ('manifest_sha256', manifest_path)):
        require(controller[field] == sha(artifact), f'controller {field} mismatch')
    require(controller['source_hashes'] == sources, 'controller source identity mismatch')
    # Public retained receipts must never include the original opaque lease token.
    require(not re.search(r'"(?:lease_token|token|workflow_handle)"\s*:', json.dumps(controller)),
            'controller receipt contains private authorization material')
    return controller['evidence_id']


def verify_native(path, directory, result, manifest):
    native = load(path)
    require(native['status'] == 'pass' and native['kind'] == 'fitted_installed_ct_bridge',
            'native receipt is not a fitted installed bridge pass')
    require(native['biology'] == 'not_run' and native['gh_worktree_dirty'] is False,
            'native scope or source cleanliness mismatch')
    for field, artifact in (('values_sha256', directory / 'installed-consumer.bridge-values.txt'),
                            ('checkpoint_sha256', directory / 'installed-consumer.pt'),
                            ('producer_receipt_sha256', directory / 'installed-consumer.json'),
                            ('executable_sha256', BRIDGE_WORK / 'consumer-build/gh_ml2_native_bridge'),
                            ('installed_gh_config_sha256', GH_PREFIX / 'lib/cmake/GlassHelix/GlassHelixConfig.cmake')):
        require(native[field] == sha(artifact), f'native {field} mismatch')
    require(native['installed_gh_prefix'] == str(GH_PREFIX), 'native GH prefix mismatch')
    require(native['cellerator_package'] == str(CE_SDK), 'native CE package mismatch')
    require(native['gh_source_commit'] == result['source_commit'] == manifest['source_commit'],
            'native GH fitted source identity mismatch')
    metadata = CE_SDK / 'CelleratorBuildIdentity.cmake'
    revision = re.search(r'set\(Cellerator_BUILD_SOURCE_REVISION "([0-9a-f]{40})"\)', metadata.read_text())
    require(revision is not None and native['ce_source_commit'] == revision.group(1),
            'native CE source identity mismatch')
    require(native['cellerator_identity_metadata_sha256'] == {
        str(p): sha(p) for p in CE_SDK.glob('*Identity.cmake')}, 'native CE identity metadata mismatch')
    targets = (CE_SDK / 'CelleratorNativeTargets.cmake').read_text()
    archives = {}
    for target in ('native_numeric', 'local_differential', 'indexed_mechanism', 'executable_program'):
        match = re.search(r'set_target_properties\(Cellerator::' + target
                          + r' PROPERTIES\s+IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "[^"]*"\s+'
                          + r'IMPORTED_LOCATION_RELEASE "([^"]+)"', targets)
        require(match is not None, f'CE archive target absent: {target}')
        archive = Path(match.group(1))
        require(archive.resolve().is_relative_to(CE_SDK.resolve()), 'CE archive outside pinned SDK')
        archives[str(archive)] = sha(archive)
    require(native['cellerator_archive_sha256'] == archives, 'native CE archive identity mismatch')
    frozen = {key: str(value) for key, value in SCIENTIFIC_FIELDS.items()}
    require(native['frozen_scientific_identity'] == frozen,
            'native frozen scientific identity mismatch')
    close(native['output'], manifest['bridge']['expected_output'], 'native output')
    close(native['derivative_z'], manifest['bridge']['derivative_z'], 'native derivative')
    for source in sorted((ROOT / 'include/GlassHelix').rglob('*.hh')):
        installed = GH_PREFIX / 'include' / source.relative_to(ROOT / 'include')
        require(sha(installed) == sha(source), f'installed GH header differs: {source.name}')


def verify(directory, controller_path, native_path, require_native):
    result_path = directory / 'installed-consumer.json'
    manifest_path = directory / 'installed-consumer.manifest.json'
    result, manifest = load(result_path), load(manifest_path)
    replay = load(directory / 'installed-consumer.replay.json')
    require(result['ok'] is True and result['status'] == 'passed'
            and result['native_joint_product_trained'] is True, 'installed consumer did not pass')
    require(set(result['checks']) == set(CHECKS)
            and all(result['checks'][name] is True for name in CHECKS), 'required checks did not all pass')
    require(result['biological_status'] == 'not_run', 'biological evidence must remain not_run')
    sources = {'learning': sha(ROOT / 'python/glasshelix/learning.py'),
               'consumer': sha(ROOT / 'tests/learning/installed_consumer.py')}
    require(result['source_hashes'] == sources, 'result source hashes differ from current scoped source')
    require(manifest['learning_source_sha256'] == sources['learning']
            and manifest['consumer_source_sha256'] == sources['consumer'], 'manifest source mismatch')
    require(manifest['gh_prefix'] == str(GH_PREFIX)
            and sha(GH_PREFIX / 'lib/glasshelix/learning.py') == sources['learning'],
            'installed GH learning identity mismatch')
    for field, manifest_field, installed in (
            ('producer_package_sha256', 'celleratorch_source_sha256', CT_PACKAGE),
            ('native_library_sha256', 'native_library_sha256', CT_NATIVE)):
        require(result[field] == manifest[manifest_field] == sha(installed), f'installed {field} mismatch')
    verify_ce_bio(directory)
    checkpoint_hash = sha(directory / 'installed-consumer.pt')
    require(result['checkpoint_sha256'] == manifest['checkpoint_sha256']
            == replay['checkpoint_sha256'] == checkpoint_hash, 'checkpoint identity mismatch')
    require(replay['ok'] is True and replay['fresh_process'] is True, 'fresh process replay failed')
    require(result['global_frozen'] is True and manifest['global_frozen'] is True
            and manifest['local_conditioning']['global_frozen'] is True, 'global law not frozen')
    for record in (result, manifest):
        require(0 <= finite(record['final_loss'], 'final loss') <= 1e-10, 'global fit tolerance')
        require(finite(record['initial_loss'], 'initial loss') > record['final_loss'], 'global fit did not improve')
    residual = finite(result['local_residual'], 'local residual')
    require(0 <= residual <= 1e-9 and residual == manifest['local_conditioning']['residual'],
            'local residual tolerance or identity mismatch')
    require(result['identity_mapping'] == manifest['identity_mapping'] == mapping(), 'scientific axis mapping mismatch')
    require(result['bridge'] == manifest['bridge'], 'result and manifest bridge differ')
    require(result['source_commit'] == manifest['source_commit']
            and re.fullmatch('[0-9a-f]{40}', result['source_commit']), 'fitted source commit mismatch')
    bridge = manifest['bridge']
    require(all(bridge[key] == value for key, value in SCIENTIFIC_FIELDS.items()),
            'fitted bridge scientific identity changed')
    require(manifest['format'] == 'glasshelix-fitted-explanation-v1'
            and manifest['hypothesis_id'] == SCIENTIFIC_FIELDS['hypothesis_label']
            and manifest['domain_ids'] == ['source-axis-101:x,z*forcing',
                                          'target-axis-201:response', 'parameter-axis-301:k'],
            'fitted manifest scientific identity changed')
    for key in ('state_generation', 'parameter_generation'):
        require(type(bridge[key]) is int and bridge[key] > 0, f'invalid {key}')
    require(bridge['observed_mask'] == 1 and bridge['units'] == 'arbitrary'
            and bridge['modality'] == 'synthetic' and bridge['scale'] == 1,
            'bridge observation scope changed')
    for key in ('shared_coefficient', 'input_x', 'local_z', 'forcing', 'expected_output', 'derivative_z'):
        finite(bridge[key], key)
    require(0 < finite(bridge['tolerance'], 'tolerance') <= 2e-5, 'bridge tolerance changed')
    close(result['shared_coefficient'], bridge['shared_coefficient'], 'shared coefficient')
    close(bridge['shared_coefficient'], 1.7, 'fitted coefficient')
    close(bridge['local_z'], .8, 'local activity')
    close(bridge['expected_output'], bridge['shared_coefficient'] * bridge['input_x']
          * bridge['local_z'] * bridge['forcing'], 'composed forward')
    close(bridge['derivative_z'], bridge['shared_coefficient'] * bridge['input_x']
          * bridge['forcing'], 'composed derivative')
    close(replay['prediction'], bridge['expected_output'], 'replay prediction')
    close(replay['derivative_z'], bridge['derivative_z'], 'replay derivative')
    fields = {}
    for line in (directory / 'installed-consumer.bridge-values.txt').read_text().splitlines():
        key, value = line.split(maxsplit=1)
        require(key not in fields, f'duplicate bridge key: {key}')
        fields[key] = value
    expected_fields = {key: str(value) for key, value in bridge.items()}
    expected_fields.update(source_sha256=sources['learning'], checkpoint_sha256=checkpoint_hash)
    require(fields == expected_fields, 'bridge text differs from fitted manifest')
    evidence_id = verify_controller(controller_path, directory, result_path, manifest_path, sources)
    if require_native:
        require(native_path is not None, 'native receipt required')
    if native_path is not None:
        verify_native(native_path, directory, result, manifest)
    return {'status': 'pass', 'biological_status': 'not_run', 'source_hashes': sources,
            'checkpoint_sha256': checkpoint_hash, 'controller_evidence_id': evidence_id,
            'native_verified': native_path is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, default=ROOT / 'docs/learning/learn-evidence')
    parser.add_argument('--controller', type=Path)
    parser.add_argument('--native-receipt', type=Path)
    parser.add_argument('--require-native', action='store_true')
    args = parser.parse_args()
    controller = args.controller or args.evidence_dir / 'controller.json'
    native = args.native_receipt
    if args.require_native and native is None:
        native = args.evidence_dir / 'native-bridge.json'
    try:
        print(json.dumps(verify(args.evidence_dir, controller, native, args.require_native), sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f'LEARN evidence rejected: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
