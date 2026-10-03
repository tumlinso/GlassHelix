# Installed acceptance

Baseline gate:

```sh
/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -B \
  tests/substrate/install/check.py \
  --ce-include /home/tumlinson/Software/cellerator-ml2-cpp/include
```

The check configures/builds/installs into a fresh temporary prefix, consumes the
exported public contract target against actual published CE headers, executes
the whole-candidate contract consumer, verifies every installed Python source
hash and rejects a required unbuilt native component. Baseline CLI reports the
missing integrated packages and native extension honestly.

After source integration, append `--require-integrated`. This enforces package
presence and calls `tests/substrate/replay/check_installed.py` against the fresh
installation. The combined consumer runs the existing framework product model,
scores held-out observations with the actual masked loss, round-trips whole
hypotheses, and rejects unsupported identification claims. It checks installed
module paths and excludes moonshot imports.

Accepted inputs are the owned CMake/CLI/consumer paths, existing public GH headers
and `python/glasshelix/learning.py`, and the explicitly supplied published CE header
directory. Combined qualification additionally needs the accepted merged
`python/glasshelix/{experiment,data,models,analysis,refactoring}` sources.

This check qualifies the public contracts and supported CPU framework entry point.
It does not qualify native numerical assembly, native epoch publication, biological
fitting or performance. Raw historical benchmark omissions stay omitted.
