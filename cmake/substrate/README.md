# Granular substrate installation

The standalone installer exports `GlassHelixSubstrate::contracts` and installs
the actual public GH headers. Contract consumers supply the real published CE
header/component usage requirements. It exports no substitute CE target.

Optional native assembly requires a qualified installed Cellerator package with
`native_foundation` and `indexed_mechanism` targets. Optional analysis requires
the actual `local_differential` target and CUDAToolkit usage requirements.
Their exported targets are loaded only when those components are requested.
Missing native components fail required-component lookup.

```sh
cmake -S cmake/substrate -B build-substrate -DCMAKE_INSTALL_PREFIX=/chosen/prefix
cmake --build build-substrate -j 2
cmake --install build-substrate
```

`GLASSHELIX_SUBSTRATE_INSTALL_FRAMEWORK=OFF` installs only public contracts.
`GLASSHELIX_SUBSTRATE_BUILD_NATIVE=ON` adds native assembly using `find_package`.
`GLASSHELIX_SUBSTRATE_BUILD_ANALYSIS=ON` adds the CE differential interface target.
These native routes are not qualified by the contract-only installation check.

The optional Python packages and launcher live at `lib/python` and
`bin/glasshelix-substrate`. The installation manifest records exact Python source
hashes and the configure-time GH commit. The launcher locates its own prefix;
no source-tree or moonshot import is needed for the supported driver.

CLI commands are `capabilities`, `run --plan ... --output ...`,
`evaluate --result ... --observations ... --output ...`, and
`replay --result ... --output ...`. The current runnable driver is the actual
`learning.ProductHypothesis` through `models.ModelClient`, with a caller-supplied
whole-hypothesis specification/coefficient and explicit CPU float64 reference
policy. Evaluation uses the existing masked objective on declared held-out
observations. Replay validates the typed whole-hypothesis result and scientific
scope. Native local arithmetic is an optional separately installed extension;
`native-capabilities` fails clearly when it is absent. Dependency discovery is
reported separately from numerical/scientific qualification.

`GLASSHELIX_REQUIRE_INTEGRATED=ON` rejects an installation missing accepted
SCIENCE/DATA/MODELS/ANALYSIS/refactoring sources. The baseline leaf checkout contains
only ADOPT and therefore qualifies public contract installation now. Root must
run the integrated installed consumer after its source merge:

```sh
/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -B \
  tests/substrate/install/check.py \
  --ce-include /home/tumlinson/Software/cellerator-ml2-cpp/include \
  --require-integrated
```

That required check executes fresh installed run/evaluation/replay and verifies
installed package paths, retained whole states and rejection of unsupported
global identification. It is submitted unrun until the accepted source merge.
No benchmark is launched or historical missing raw evidence reconstructed.

Root CMake integration proposal: add an explicit option for the substrate
installer and `add_subdirectory(cmake/substrate)` after existing foundation target
definitions. The standalone installer remains available when root native/replay
dependencies are unavailable. Existing numerical targets and managed workflow
files remain under their owners; this leaf makes no root CMake edit.
