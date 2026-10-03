# GlassHelix foundation adoption

Accepted inspection base: `e932b2d`. The small shared surface keeps current
scientific types and numerical owners. `GlassHelix::experiment` links the existing
foundation, exports the installed headers, and adds a common receipt envelope.
It introduces no evaluator or tensor owner.

Preferred entry points are `experiment/record.hh`, `data/source.hh`,
`models/candidate_set.hh` and `analysis/result.hh`. The candidate seam aliases
the existing `inference::finite_candidates`; complete hypothesis/mechanism/state
alternatives stay intact. Source references carry audit and input manifest IDs.
`audited_celltag` means that an external audit is supplied, not that this header
has audited a dataset. Capability claims remain independent per action and carry
evidence IDs. A fixture cannot be labelled a qualified biological fit.

The receipt admits provenance and claim metadata; it does not verify referenced
files, execute fits or certify claims. DATA/BRIDGE/PIPELINE must establish those
facts with actual consumer evidence.

## Current predecessor frontier

The completed GH-ML2-DATA and GH-ML2-PILOT tasks have usable engineering outcomes.
The actual retained `docs/learning/pilot-evidence/pilot.manifest.json` explicitly
records `biological_status: not_run` and `fixture_rna_*` features. Its controller
receipt binds source `854176133c96291269008fe36b8dc2abeb878ecd` and the input/result
hashes. The present source audit excludes held-out modalities and clone leakage;
it distinguishes measured zero from missing data. Keep that outcome observable
instead of treating task completion as biological qualification. DATA must reuse
the actual cohort or explicitly retain this fixture and missing-input disposition.

Both moonshot packages remain preserved. Their useful native product2 consumer,
matrix/Sylvester and actor/port families are mapped below. Native Torch CPU FP32
product2 evidence does not qualify CUDA tensors or second-order transforms.
The refactor prototype's token/drain and optimizer reset policy is a scientific
proposal, not an already integrated CE structural-publication runtime.

## Source and destination disposition

| Current source | Preferred destination / next owner | Preserved behavior and obligation |
| --- | --- | --- |
| `include/GlassHelix/core/{quantity,system,observation,provenance}.hh` | Keep current types; ADOPT receipt composes them | Scientific roles, axis/epoch identities, units, evidence and generation provenance |
| `include/GlassHelix/system/assembly.hh`, `src/system/assembly.cc` | Keep preferred assembly; GH-IS1-BRIDGE consumes CE components | Ordered arguments/output ownership, numerical policy, registered indexed lowering; no hidden replacement scalar engine |
| `include/GlassHelix/inference/inference.hh` | `models/candidate_set.hh` aliases existing owner; GH-IS1-MODELS | Whole joint candidates, stable hypothesis/mechanism IDs, evidence history and log weights |
| `include/GlassHelix/observation/observation.hh`, `interrogation/*`, `dynamics/*`, `reduction/*` | Keep current entry paths; GH-IS1-ANALYSIS | Observation/likelihood choices remain scientific policy; reusable numerics use CE; derivative and support limits remain explicit |
| `src/artifacts/nf1_replay.cc`, `bindings/python/nf1_module.cc`, `python/glasshelix` | Keep replay/learning route; GH-IS1-BUILD and PIPELINE | Installed typed consumer, evidence permissions/masks/correspondence, logical checkpoints and recorded native bridge |
| `experiments/celltag_reprogramming/{data,reference,pilot}.py`, `real-run.toml`, `missing-inputs.toml` | Existing adapter + proposed `integrated/`; GH-IS1-DATA / PIPELINE | RNA training-only feature selection, original time labels, clone/ATAC validation, sample/cell/replicate provenance, missingness and measured zeros |
| `docs/learning/{learn-evidence,pilot-evidence}`, repair/acceptance verifiers | Retain exact immutable receipts; DATA / BRIDGE / PIPELINE reference IDs | Actual synthetic joint-product training and BIO consumer acceptance; evaluated-not-promoted performance; biology not_run |
| `experiments/moonshot-parallel-v1/plain/model.py` | Reuse via GH-IS1-MODELS adapter | MatrixPatchField and fixed-generator SylvesterFlow; CPU f32/f64 checks, no universal nonlinear solve claim |
| `experiments/moonshot-parallel-v1/actor/model.py` | Reuse via GH-IS1-MODELS | Private actor coordinates, explicit communication ports, distinct repeated edges; regrowth branch identity |
| `experiments/moonshot-parallel-v1/integration/model.py` and installed receipt | Reuse via GH-IS1-BRIDGE / MODELS | Native product contributions, separate source/readout IDs, explicit reduction, topology-checked checkpoint; CE owns product arithmetic |
| `experiments/moonshot-parallel-v1/refactor/model.py` | Scientific proposal adapter in GH-IS1-ANALYSIS | Declared basis migration, old reader policy, new epochs, reset optimizer moments; actual CE migration remains owner work |
| `planning/moonshot-v1/payload`, `planning/moonshot-parallel-v1`, IS1 `seed/`, original ledgers and evidence | Preserve immutable inputs | Witnesses/reference algebra remain available; they are not substituted for native/scientific acceptance |
| `src/memory`, `src/incomplete`, other legacy experiments | Retain in place, outside this bounded adoption | No numerical/ABI/source cleanup was authorized; consumer reconciliation is required before a future move |

## Actual component graph and checks

`GlassHelix::experiment -> GlassHelix::foundation -> GlassHelix::core ->`
`Cellerator::{native_foundation,indexed_mechanism}`. Replay additionally consumes
`Cellerator::local_differential`. Existing experiment packages use installed
CelleraTorch/framework facilities; no Baseplane umbrella or source-tree CE build
is introduced into the GH target graph. Baseplane sequence is optional experiment
input, not a mandatory numerical dependency.

The current root configuration still requires CUDAToolkit/LAPACK and enables CUDA
under BUILD_TESTING. This adoption does not claim a fresh host-only installed
package. GH-IS1-BUILD owns that component qualification after CE publishes the
accepted provider targets. The focused host check uses installed ML2 CE headers
and existing CE host archives; unused indexed CUDA methods are link-discarded.
It exercises the new receipt, existing scientific assembly callback fixture and
existing whole-candidate inference. It neither launches GPU work nor qualifies
the indexed CUDA route, biological fitting, throughput or structural migration.

Gate argv from this repository root:

```sh
python3 -B planning/integrated-substrate-v1/adoption/check_host.py \
  --ce-include /home/tumlinson/Software/cellerator-ml2-cpp/include \
  --ce-library /home/tumlinson/Cellerator/build-ml2-cpp/libcellerator_executable_program.a \
  --ce-library /home/tumlinson/Cellerator/build-ml2-cpp/libcellerator_operation_core.a
```

The output records exact commands, source hashes and library hashes. Root retains
global activation reconciliation, accepted commit publication and task lifecycle.
