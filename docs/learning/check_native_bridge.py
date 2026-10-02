#!/usr/bin/env python3
"""Build a public installed GH/CE bridge and verify fitted TRAIN values."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, log):
    result = subprocess.run([str(x) for x in command], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with log.open("a") as stream:
        stream.write("$ " + " ".join(str(x) for x in command) + "\n" + result.stdout)
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}; see {log}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--values", type=pathlib.Path, help="actual fitted bridge-values.txt")
    source.add_argument("--fixture", action="store_true", help="controlled CPU fixture only")
    parser.add_argument("--cellerator-dir", type=pathlib.Path, required=True)
    parser.add_argument("--cuda-root", type=pathlib.Path,
                        default=pathlib.Path("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9"))
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument("--install-prefix", type=pathlib.Path,
                        help="installed GH prefix shared with installed Python consumer")
    parser.add_argument("--producer-receipt", type=pathlib.Path,
                        help="installed CT fit receipt; mandatory for fitted acceptance")
    parser.add_argument("--checkpoint", type=pathlib.Path,
                        help="actual fitted checkpoint; mandatory for fitted acceptance")
    args = parser.parse_args()
    if args.values and (not args.producer_receipt or not args.checkpoint):
        parser.error("--values requires --producer-receipt and --checkpoint")
    work = (args.work_dir or pathlib.Path(tempfile.mkdtemp(prefix="gh-ml2-native-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    log = work / "build.log"
    revision = run(["git", "-C", ROOT, "rev-parse", "HEAD"], log).strip()
    # Changed public code must have a real source commit before final acceptance.
    dirty = run(["git", "-C", ROOT, "status", "--porcelain", "--",
                 "include", "src", "cmake", "CMakeLists.txt", "bindings", "python",
                 "tests/learning/native_bridge", "docs/learning/check_native_bridge.py"], log).strip()
    if dirty and args.values:
        raise RuntimeError("fitted acceptance requires committed public GH/bridge source")
    values = args.values.resolve() if args.values else work / "fixture-values.txt"
    if args.fixture:
        values.write_text("""shared_coefficient 2
input_x 3
local_z 4
forcing 0.5
expected_output 12
derivative_z 3
observed_mask 1
model_id 7001
model_revision 1
mechanism_id 501
mechanism_label joint_product
coefficient_id 401
parameter_extent 1
hypothesis_label synthetic-product-7001-v1
domain_id 101
order_id 102
geometry_id 103
partition_id 104
source_extent 2
target_domain_id 201
target_order_id 202
target_geometry_id 203
target_partition_id 204
target_extent 1
parameter_domain_id 301
parameter_order_id 302
parameter_geometry_id 303
parameter_partition_id 304
structure_id 105
structure_epoch 1
evidence_id 8001
state_generation 1
parameter_generation 1
units arbitrary
scale 1
tolerance 0.00002
modality synthetic
time_units hours
sampling_id fixture0
time 0
""")
    fields = dict(line.split() for line in values.read_text().splitlines()
                  if line and not line.startswith("#"))
    frozen = {"model_id": "7001", "model_revision": "1", "domain_id": "101",
              "order_id": "102", "geometry_id": "103", "partition_id": "104",
              "structure_id": "105", "structure_epoch": "1", "evidence_id": "8001",
              "source_extent": "2", "target_extent": "1", "target_domain_id": "201",
              "target_order_id": "202", "target_geometry_id": "203", "target_partition_id": "204",
              "parameter_domain_id": "301", "parameter_order_id": "302",
              "parameter_geometry_id": "303", "parameter_partition_id": "304",
              "parameter_extent": "1", "coefficient_id": "401", "mechanism_label": "joint_product",
              "mechanism_id": "501", "hypothesis_label": "synthetic-product-7001-v1"}
    if any(fields.get(key) != value for key, value in frozen.items()):
        raise RuntimeError("fitted artifact changed frozen scientific identity")
    checkpoint_hash = sha(args.checkpoint) if args.checkpoint else None
    if args.values and fields.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError("bridge values do not bind the supplied fitted checkpoint")
    prefix = args.install_prefix.resolve() if args.install_prefix else work / "installed"
    gh_build, consumer_build = work / "gh-build", work / "consumer-build"
    run(["cmake", "-S", ROOT, "-B", gh_build, "-DBUILD_TESTING=OFF",
         f"-DCellerator_DIR={args.cellerator_dir.resolve()}",
         f"-DCUDAToolkit_ROOT={args.cuda_root}", f"-DCMAKE_INSTALL_PREFIX={prefix}"], log)
    run(["cmake", "--build", gh_build, "--parallel", "2"], log)
    run(["cmake", "--install", gh_build], log)
    run(["cmake", "-S", ROOT / "tests/learning/native_bridge", "-B", consumer_build,
         f"-DGlassHelix_DIR={prefix / 'lib/cmake/GlassHelix'}",
         f"-DCellerator_DIR={args.cellerator_dir.resolve()}",
         f"-DCUDAToolkit_ROOT={args.cuda_root}", f"-DGH_BRIDGE_SOURCE_COMMIT={revision}"], log)
    run(["cmake", "--build", consumer_build, "--parallel", "2"], log)
    executable = consumer_build / "gh_ml2_native_bridge"
    receipt = json.loads(run([executable, values], log))
    if receipt["gh_source_commit"] != revision:
        raise RuntimeError("consumer GH build identity mismatch")
    archives = [pathlib.Path(x) for x in
                (consumer_build / "sdk-archives.txt").read_text().splitlines()]
    sdk_hashes = {str(path): sha(path) for path in archives}
    sdk_metadata = {str(path): sha(path) for path in
                    args.cellerator_dir.glob("*Identity.cmake")}
    if args.values:
        producer = json.loads(args.producer_receipt.read_text())
        if (producer.get("status") != "passed" or producer.get("ok") is not True or
                producer.get("native_joint_product_trained") is not True or
                producer.get("global_frozen") is not True or
                producer.get("checkpoint_sha256") != checkpoint_hash or
                producer.get("source_commit") != revision):
            raise RuntimeError("producer does not qualify fitted source/checkpoint/training")
        checks = ("native_training", "global_fit", "frozen_conditioning",
                  "complete_objective_gradient", "checkpoint_fresh_process", "evidence_permissions")
        if any(producer.get("checks", {}).get(key) is not True for key in checks):
            raise RuntimeError("producer has an unqualified required training check")
        if any(fields.get(key) != str(value) for key, value in producer["bridge"].items()):
            raise RuntimeError("bridge values differ from producer receipt")
        installed_learning_hash = sha(prefix / "lib/glasshelix/learning.py")
        if (producer["source_hashes"]["learning"] != installed_learning_hash or
                fields.get("source_sha256") != installed_learning_hash):
            raise RuntimeError("producer learned module differs from installed GH package")
        mapping = producer["identity_mapping"]
        for axis, base in (("source_axis", 101), ("target_axis", 201), ("parameter_axis", 301)):
            for offset, key in enumerate(("domain", "order", "geometry", "partition")):
                if mapping[axis][key] != [base + offset, base + offset + 1000]:
                    raise RuntimeError("producer changed frozen full axis identity")
        if (mapping["mechanism"]["identity"] != [501, 1501] or
                mapping["coefficient"]["identity"] != [401, 1401] or
                mapping["hypothesis"] != {"label": "synthetic-product-7001-v1", "system_id": 7001}):
            raise RuntimeError("producer changed frozen hypothesis/mechanism/coefficient mapping")
    receipt.update(kind="controlled_cpu_fixture" if args.fixture else "fitted_installed_ct_bridge",
                   gh_worktree_dirty=bool(dirty), values_sha256=sha(values),
                   executable_sha256=sha(executable), checkpoint_sha256=checkpoint_hash,
                   producer_receipt_sha256=sha(args.producer_receipt) if args.producer_receipt else None,
                   frozen_scientific_identity=frozen, work_dir=str(work),
                   executable=str(executable), values=str(values), build_log=str(log),
                   cellerator_package=str(args.cellerator_dir.resolve()), biology="not_run")
    receipt.update(cellerator_archive_sha256=sdk_hashes,
                   cellerator_identity_metadata_sha256=sdk_metadata,
                   installed_gh_prefix=str(prefix),
                   installed_gh_config_sha256=sha(prefix / "lib/cmake/GlassHelix/GlassHelixConfig.cmake"))
    output = work / "receipt.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "pass", "receipt": str(output), "executable": str(executable)}))


if __name__ == "__main__":
    main()
