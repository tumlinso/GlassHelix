# Build, use and validate

Start with one supported path rather than every historical target. Commands below must be qualified against the source version in the [current snapshot](../status/current.md).

```sh
# Requires CMake 3.28+, C++20, CUDAToolkit, LAPACK and a Cellerator
# install exporting native_foundation and indexed_mechanism.
cmake -S . \
  -B build-docs \
  -DCMAKE_PREFIX_PATH="${CELLERATOR_PREFIX}" \
  -DBUILD_TESTING=OFF
cmake --build build-docs \
  --target glasshelix_foundation \
  -j 4
```

This builds the `glasshelix_foundation` library from `src/system/assembly.cc` against the installed Cellerator components. With `BUILD_TESTING=ON`, configuration also enables CUDA and the native-foundation checks, including CUDA tests and the complete-program executable. Inspect that build's test list with `ctest --test-dir build-docs -N`; run relevant checks only with the matching toolchain and resources. Optional replay/Python integration is configured when its source files are present. Recheck this route against the qualified Cellerator package and source revision before relying on it.

## Optional Python and Torch integration

Install the Cellerator Python distribution in the same environment used by
GlassHelix. Torch integration is optional and imported from
`cellerator.torch`; typed identities, axes and mechanism declarations come from
`cellerator`. The adapter loads native extensions from the installed package, so
consumers do not need a repository path or shared-library environment variable.

```sh
python -m pip install "/path/to/Cellerator[torch]"
python -c "import cellerator; import cellerator.torch"
```

The CPU product2 adapter is available without a CUDA mechanism build. Indexed
mechanism modules require the installed CUDA-enabled native mechanism target.
See the [Cellerator Python binding guide](https://github.com/tumlinso/Cellerator/blob/main/docs/development/python-bindings.md)
for the current package build and validation commands.

## Work on a bounded component

Follow the [source map](source-map.md), inspect its nearest guidance and tests, and keep an example/reference beside the behavior it exercises. A documentation change does not require a full hardware campaign. A source move does require updating build/install/import/include references and rerunning the affected targets.

For measured claims follow [results](../results/index.md). Use assigned resources; record command, source, toolchain, input, precision, outputs, warmup/repeats and timed phases. Keep profiler diagnostics separate from benchmark timings. Preserve the original evidence when rendering new documentation.

## Code and documentation boundaries

Durable design goes in `docs/design/`, practical instructions here, dated summaries in `docs/status/`, and evidence in `docs/results/` linked to original records. Use experiments for unpromoted mechanisms. Generated Todo state and historical notes are not substitute architecture.
