#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
build=${1:-"$root/build/nf1-access"}
ce_prefix=${CELLERATOR_PREFIX:-/tmp/nf1a-cellerator-response-install}
mkdir -p "$build"
cxx=${CXX:-c++}
python=${PYTHON:-python3.13}
python_config="${python}-config"
native_link=(-I"$ce_prefix/include" -L"$ce_prefix/lib" -lcellerator_local_differential -lcellerator_operation_core -lcellerator_relation_semantics)
"$cxx" -std=c++20 -I"$root/include" "$root/src/artifacts/nf1_replay.cc" "$root/tools/native_foundation/nf1_replay_cli.cc" "${native_link[@]}" -o "$build/nf1_replay_cli"
"$cxx" -std=c++20 -I"$root/include" "$root/src/artifacts/nf1_replay.cc" "$root/tests/native_foundation/artifacts/nf1_replay_test.cc" "${native_link[@]}" -o "$build/nf1_replay_test"
"$build/nf1_replay_test" "$build/replay.nf1" "$build/nf1_replay_cli"
"$cxx" -std=c++20 -shared -fPIC $($python_config --includes) -I"$ce_prefix/include" "$root/bindings/python/nf1_module.cc" -L"$ce_prefix/lib" -lcellerator_local_differential -lcellerator_operation_core -lcellerator_relation_semantics -o "$build/glasshelix_nf1$($python_config --extension-suffix)"
PYTHONPATH="$build" "$python" "$root/tests/native_foundation/python/test_nf1.py"
