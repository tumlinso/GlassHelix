# GlassHelix experimental model contract v1

This agreement unlocks independent GH-MOON-PLAIN and GH-MOON-ACTOR work. It preserves the supplied computational intent; it is an internal experimental seam that can be revised by explicit owner agreement. Root accepts the exact producer commits before dependent work starts.

## Source and ownership

The immutable originals are `../seeds/glasshelix/{actor_fields.py,smoke.py,learn_toy.py}`. `../seeds/manifest.json` binds each file to `planning/moonshot-v1/MANIFEST.json`. Leaf implementations live in `../plain`, `../actor` and `../refactor` respectively. They may read seeds; they never edit them or another leaf. No numerical runtime, optimizer or native scientific ontology is introduced here.

## Plain module

`plain/model.py` exports Torch `nn.Module` classes with the seed signatures:

- `MatrixPatchField(patches: int, width: int = 16)`: positive dimensions; `forward(state)` maps one cell's tensor of shape `(patches, width, width)` to the same shape. Patch index is an internal layout axis. The model has trainable `left`, `right`, and `decay` tensors, and computes `tanh(left @ state) @ right - softplus(decay) * state`.
- `SylvesterFlow(width: int = 16)`: `forward(x, dt)` computes `matrix_exp(dt * left) @ x @ matrix_exp(dt * right)` for a square state. Derivatives reach both trainable generators; caches must respect their value generations and `dt`.

Plain training and checks stay in `plain`. Supported dtype, device and input layouts are stated in the leaf's receipt. Extra examples are permitted within the assigned scope.

## Actor module

`actor/model.py` exports:

- `ActorPortField(actors: int, width: int, ports: int, edges)`: positive dimensions and supplied `(source, destination)` index vectors of equal length, in bounds. `forward(state)` maps `(actors, width)` to the same shape. `observe(state)` returns `(actors,)`. It retains the seed parameter names `encoder`, `decoder`, `local`, `edge`, `readout` and registered topology buffers `source`, `destination`.
- `RegrowthBranch(input_width: int, capacity: int, output_width: int)`: initial outgoing weights are zero while the feature branch remains structurally present. `forward(state)` returns an output of the declared width. Check that outgoing parameters receive gradients at zero contribution and that reused capacity can learn.

Actor hidden coordinates are private. Shared ports represent communication, not globally aligned latent columns. Preserve directed/repeated edges and contribution summation. Topology and refactoring maps are supplied assumptions, not discoveries. Keep observations, latent state, learned parameters and evidence roles distinct.

## Refactoring consumer

After acceptance of the actor producer, refactor imports those exports through `importlib` or a Python package. It cannot copy a changed actor implementation into the actor leaf. It publishes its own explicit supplied state/parameter map, initial function preservation check, and subsequent regrowth-learning witness. Scope the preservation claim to the actual algebra and map being checked. Generic nonlinear coordinate remapping need not preserve a coordinatewise `tanh` model; do not claim that an arbitrary basis change does so.

A newly explicit effective actor is an anonymous predictive factor until independent evidence supports an interpretation. Support overlap, matching derivatives or a supplied equality witness alone cannot identify a molecule. Whole joint state–mechanism candidates remain whole alternatives; no independently mixed-coordinate candidate is substituted.

## Evidence and execution boundary

Each leaf supplies a runnable small check and a JSON receipt recording model class, callable path, shapes, dtype/device, actual derivative actions, exact invocation, observed results, unsupported cases and source identity. Synthetic learning checks may establish engineering behavior; they do not establish biology, identifiability, benchmark wins or native CE execution. Independent Torch modules may be loaded without native CE. Native adapters remain GH-MOON-CONSUMER work and require accepted real CE capability evidence and GH-ML2-ACCEPT completion.

## Active obligations

Root's current adoption context is GH-MOON-RUN-V1/GH-MOON-L-ADOPT/GH-MOON-ADOPT at revision 262. The independently active GH-ML2-RUN-V1 has DESIGN, DATA, REPAIR, LEARN, PILOT, ACCEPT and aggregate obligations. The revision-260 observation reported queued tasks and no claims; this is a dated observation, not ongoing authorization to write their paths. Root must refresh exact active ownership before native integration. This adoption stages seeds and experimental seams only; no controlled DESIGN/LEARN source integration has occurred.

GH-ML2-REPAIR, independent data permissions/audit and pilot obligations, framework learning and same-fitted-component native linkage remain under their owners. The scoped prototypes do not complete those obligations. No task is superseded, reopened, requeued or changed by this agreement. The existing `experiments/baseplane_moonshot` companion remains separately owned.
