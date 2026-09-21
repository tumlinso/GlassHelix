#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
build=${1:-"$root/build/nf1-access"}
install=${2:-"/tmp/nf1a-glasshelix-access-script-install"}
ce_prefix=${CELLERATOR_PREFIX:-/tmp/nf1a-cellerator-qualified-install}
python=${PYTHON:-python3.13}

cmake -S "$root" -B "$build" \
  -DCMAKE_PREFIX_PATH="$ce_prefix" \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_INSTALL_PREFIX="$install"
cmake --build "$build" --target gh_nf1a_replay_test gh_nf1a_replay_cli glasshelix_nf1 -j2
ctest --test-dir "$build" --output-on-failure -R '^(gh_nf1a_replay|gh_nf1a_python)$'
cmake --build "$build" --target install -j2

consumer_build="$build/installed-consumer"
cmake -S "$root/tests/native_foundation/artifacts/installed_consumer" \
  -B "$consumer_build" -DCMAKE_PREFIX_PATH="$install;$ce_prefix"
cmake --build "$consumer_build" -j2
"$consumer_build/nf1_installed_consumer" \
  "$build/installed-replay.nf1" "$install/bin/gh_nf1a_replay_cli"
PYTHONPATH="$install/lib" "$python" "$root/tests/native_foundation/python/test_nf1.py"
