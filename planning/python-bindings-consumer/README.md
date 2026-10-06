# Cellerator Python consumer migration

This bounded GH consumer repair is tracked by `GH-PYBIND-CONSUMERS` in `GH-PYBIND-RUN-V1` (refreshed task revision 509). Root retains workflow lifecycle and final acceptance. The sibling Cellerator implementation task is `CE-PYBIND-ABSORB`; Cellerator source edits and installed-package acceptance remain owned by its root controller.

## Consumer boundary

- Import `Identity`, `Axis`, `Incidence`, `OutputContribution`, `Mechanism`, `MechanismSpec`, and `SharedSupportSpec` from `cellerator`.
- Import `MechanismModule`, `SharedSupportRelation`, `guarded_step`, checkpoint helpers, `Product2Module`, and product2 JVP from `cellerator.torch`.
- Resolve and fingerprint native artifacts through `importlib.metadata.distribution("cellerator")` and the imported `cellerator._native` and, for CUDA mechanisms, `cellerator.torch._torch` modules. Consumer scripts do not set native-library environment variables.
- Each maintained installed consumer checks that the resolved imported module files are members of `distribution.files`, using `distribution.locate_file(...)`, before recording provenance. The integration CPU product2 gate includes all packaged native libraries in its artifact hashes.
- The published core API is `prepare_mechanism(spec, contiguous_numpy_fp32, device=..., max_batch=..., max_live_forwards=..., precision=...)`; it returns the native handle and retains `handle.spec`. Mechanism consumers now pass that handle to `MechanismModule.from_handle(handle)`.
- Shared-support consumers lower `SharedSupportSpec` with `spec.mechanism_spec()`, create one native handle with `prepare_mechanism`, and wrap that same owner using `SharedSupportRelation.from_handle(spec, handle)`. Convenience constructors remain available and delegate to the core constructor.

## Verification status

Static source migration checks passed: Python `compile(...)` on the installed consumer, pilot, integration check, and both evidence verifiers; `git diff --check`; and a maintained-source scan for legacy package imports and loader environment variables. The history snapshots of the old receipt consumer and verifier are byte-for-byte equal to their `HEAD` originals (SHA-256 `87a094735c05a45de8ed4659b59c1d14d2c9257daef3aaf93f3430e31583a0aa` and `49b72ede0317b03a456a523a48c8b6c7956eda3b83027b88e73e8e8f60f973fd`, respectively). Existing receipt JSON, logs, and hashes were left unchanged; they remain evidence for the old component-era producer and are not acceptance evidence for the absorbed package.

A current GlassHelix SDK prefix was built and installed at `/tmp/gh-bindings-installed-sdk` from worktree HEAD `550df30a247ef9c23c7e772c3492189cac840d8c` using Cellerator package `/home/tumlinson/Cellerator/build-ml2-cpp` (producer revision `009a5579c1facec8c3bce3c36f6764c9f49ab97c`). The build used `BUILD_TESTING=OFF`, compiled the host C++ library and Python module, and installed `glasshelix.learning`; it did not execute the bridge, tests, CUDA, or benchmark code. Recreate the SDK-only install with:

```sh
cmake -S /home/tumlinson/GlassHelix -B /tmp/gh-bindings-sdk-build \
  -DBUILD_TESTING=OFF \
  -DCellerator_DIR=/home/tumlinson/Cellerator/build-ml2-cpp \
  -DCUDAToolkit_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9 \
  -DCMAKE_INSTALL_PREFIX=/tmp/gh-bindings-installed-sdk
cmake --build /tmp/gh-bindings-sdk-build --parallel 2
cmake --install /tmp/gh-bindings-sdk-build
```

## Final consumer validation (2026-10-06)

- The installed CPU product2 consumer check passed with `--record`; its final run output is `/tmp/gh-bindings-cpu-consumer-final.json` (SHA-256 `73ee844573c77b8644ee59f5ae41dd3a6ad8c7fd468ef1b93c5a171ecf5f1ee1`). The maintained current receipt is `experiments/moonshot-parallel-v1/integration/evidence/consumer-checks-cellerator.json`.
- The root-managed final CUDA consumer controller passed: evidence `83e65de4-dbec-4263-8c1e-d843d057e245`, foreground run `70174f1d-8f43-4467-a98f-56d1938f66d4`. This covers the native-owner training/gradient/guarded update/checkpoint replay consumer and shared-support adapter pilot on the declared synthetic fixtures.
- Durable final JSON outputs and manifests are under `planning/python-bindings-consumer/validation/`. Their embedded provenance identifies GH source `550df30a247ef9c23c7e772c3492189cac840d8c`, Cellerator package version `0.1.0`, native module SHA-256 `3372ece12178909ef989d725eea12d3135fc204308dd36cc9b6dbe205af8978a`, and Torch adapter SHA-256 `1031aa8db19b9b638d3359d9abccfde51e328bc31e22554ea336f17431ca8edc`. The owner checkpoint SHA-256 is bound in its manifest and replay JSON; the binary is not included in the archived validation bundle.

These checks qualify the bounded synthetic consumers only. Biological validation remains `not_run`; the pilot remains `evaluated_not_promoted`. No biological result or performance promotion is made by this migration note.
