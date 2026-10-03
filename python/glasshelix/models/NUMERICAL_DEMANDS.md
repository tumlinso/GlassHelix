# Numerical owner requests from the actual model clients

The root accepted these requests at the foundation seam on 2026-10-03. CE OPS,
MATH and the existing CT/native owners retain the stated numerical scope; the
current framework call paths remain usable while native promotion is qualified.
This owner decision is recorded by the root because it crosses repository scopes.
The current clients call the existing Torch modules directly. They do not
implement substitute kernels, coefficients, preparation or optimizer storage.

| Request | Exact current source/symbol | Consumer need and numerical owner |
| --- | --- | --- |
| GH-MODELS-PATCH | `experiments/moonshot-parallel-v1/plain/model.py:MatrixPatchField.forward` | One-cell `(patches,width,width)` state; `tanh(L @ X) @ R - softplus(decay) * X`; reusable matrix products, activation and their input/parameter VJPs belong to CE/framework providers. Scientific patch layout and decay interpretation remain GH. A native provider must match strided input/dtype constraints and preserve gradients into both maps and decay. |
| GH-MODELS-SYLVESTER | `experiments/moonshot-parallel-v1/plain/model.py:SylvesterFlow.forward` | `exp(dt L) X exp(dt R)` for fixed autonomous generators, with input, generator and time derivatives. CE MATH owns a reusable matrix-exponential/flow realization if promoted; current Torch `matrix_exp` remains the executed provider. Any prepared exponential cache must bind generator generations and `dt`; changing/nonlinear generators require another model. |
| GH-MODELS-PORT | `experiments/moonshot-parallel-v1/actor/model.py:ActorPortField.forward` | Per-actor encoder/decoder/local contractions, directed edge-weighted additive port gathering, and coordinatewise tanh. CE OPS/math owns reusable contraction/routing/reduction and input/parameter VJPs. Distinct repeated edges must contribute and receive cotangents independently, including zero weights. Actor-private hidden coordinates and scientific port roles remain GH. |
| GH-MODELS-READOUT | `experiments/moonshot-parallel-v1/actor/model.py:ActorPortField.observe` | Per-actor weighted readout reduction and its VJPs, implemented by framework arithmetic today. A future CE provider must preserve actor/readout coordinate declarations; arbitrary state outputs must not silently become measurements. |
| GH-MODELS-ML2 | `experiments/celltag_reprogramming/pilot.py:PopulationRelation.forward`, `tests/learning/installed_consumer.py:make_model` | Existing native CE/CT SharedSupportRelation and indexed MechanismModule already own relation/joint-product mathematics, canonical coefficients and sanctioned publication. The ordinary Torch client supplies activity generators/outer layers. Retain repeated support, native saved-forward generations and owner checkpoint restoration; do not create another numerical implementation or optimizer. |

Forward, input VJP and parameter VJP are exercised through the actual current
CPU framework clients. This leaf does not qualify native CUDA, capture,
mixed precision, second-order transforms, throughput or biological interpretation.
Native owner clients require their producer receipt and sanctioned restore path.

Acceptance owner: root; accepted for the existing framework path and bounded
numerical ownership described above. Follow-on implementation owners: CE OPS/MATH and the
existing CT/native producer tasks as scoped by root; no automatic dispatch is
implied by this request file.
