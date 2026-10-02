# Accepted reference model capabilities

[reference-capability.json](reference-capability.json) binds the plain and actor implementations to accepted integration commit `288b345b8a30c0ea1f06ee91b1dcaaaec5979d06`. The aggregation verified the actual source hashes against both producer receipts. Root reported both unchanged test suites passed after merge; this aggregation did not rerun them.

| Model export | Constructor | One-cell input | Qualified checks |
|---|---|---|---|
| `plain/model.py:MatrixPatchField` | `(patches, width=16)` | `(patches,width,width)` | CPU float32/float64; all input/parameter derivatives and small rollout fitting |
| `plain/model.py:SylvesterFlow` | `(width=16)` | `(width,width)`, finite scalar `dt` | CPU float32/float64; input/generator/time derivatives, fixed-generator composition and fitting |
| `actor/model.py:ActorPortField` | `(actors,width,ports,(source,destination))` | `(actors,width)` | CPU float64; input/all-parameter derivatives, repeated directed edges and rollout fitting |
| `actor/model.py:RegrowthBranch` | `(input_width,capacity,output_width)` | `(...,input_width)` | CPU float64; zero initial effect, live output gradients, subsequent feature gradients and fitting |

Import either leaf's `model.py` explicitly using Python's `importlib.util.spec_from_file_location` with a distinct module name, or put its leaf directory on the import path. Instantiate the chosen class, convert it with `.double()` for float64, and call it on a matching-dtype/device state. `ActorPortField.observe(state)` is the actor readout. `SylvesterFlow(state, dt)` evaluates the autonomous linear formula `exp(dt L) state exp(dt R)`; its exact-flow meaning requires fixed generators. Ordinary Torch owns optimization and autograd.

The full invocations and observed losses are in [plain evidence](../plain/validation.json) and [actor evidence](../actor/receipt.json). Their READMEs describe admission and qualification boundaries. Private hidden coordinates, communication ports and supplied topology carry no identified biological meaning. These distinct model families retain whole-model semantics. Native CE execution awaits a real capability receipt and the separately owned GH consumer integration; CUDA, biological validation and throughput measurement are not run.
