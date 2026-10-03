# Responses and supplied refactoring proposals

`analysis/requests.hh` composes the existing response request, vector-field primal
admission and atomic alternative comparison. Local forward-object JVP/VJP actions
call actual CE `local_jvp/local_vjp`; CE owns all derivative arithmetic and borrowed
spans. Other differentiated objects, evolution programs or selected-output actions
return unsupported rather than acquiring a hidden solver/derivative implementation.
Comparisons retain whole branches, local degeneracy and unresolved global identity;
optimization-run counts never establish probabilities.

`glasshelix.analysis` admits response requests by whole hypothesis, exact
differentiated object/action, supplied perturbation roles/units, evidence and
structure/value generations. A provider map explicitly names the supported
triple; its CE/framework callable owns execution. The seam checks caller-supplied
identities and output shape/finite values. It does not track native lifetimes or
replace the provider's own live-generation admission. Missing actions remain
unsupported. No global identification follows from a local response.

`glasshelix.refactoring` records supplied basis/invariant-quotient proposals,
coordinate identities, old/new epochs, state/parameter/readout/initialization
references, invalidated projections, reader policy and uncertainty. It binds the
actual predecessor `LinearRefactoring` demonstration, forwarding state maps and
its migrated model. Supplied coordinates remain anonymous. General diagonal Adam
moment transport is unsupported; transformed parameter moments require caller
reset. The old reader/tape guard remains the original prototype, not native CE
epoch publication. Native structural publication and biological status remain
`not_run`.

Gate:

```sh
/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -B tests/substrate/analysis/check.py \
  --ce-include /home/tumlinson/Software/cellerator-ml2-cpp/include \
  --cuda-include /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/include \
  --ce-library /home/tumlinson/Cellerator/build-ml2-cpp/src/compute/operation/differential/libcellerator_local_differential.a \
  --ce-library /home/tumlinson/Cellerator/build-ml2-cpp/libcellerator_operation_core.a
```

The native check calls the existing CE host differential archive, exercises JVP,
VJP, stale generation rejection and atomic comparison. Five Python tests execute
actual framework derivatives and predecessor basis/quotient/readout/flow maps,
domain rejection, proposal/optimizer policy and the prototype reader guard. CUDA
headers are needed by the current public CE differential header; no GPU launches
occur. Archive/source hashes are printed; fresh CE package export qualification
remains BUILD/BRIDGE work.

BUILD requests: install the two Python subpackages beside `learning`; export an
optional `GlassHelix::analysis` interface target linking the existing experiment
foundation and `Cellerator::local_differential`, with its declared toolkit usage
requirements. The existing header install covers the new header. Keep native
publication routed to the CE structural transformation owner in later consumers.
