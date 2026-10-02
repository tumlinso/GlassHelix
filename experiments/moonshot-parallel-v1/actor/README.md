# Actor-local port prototype

`model.py` exports `ActorPortField(actors, width, ports, edges)` and
`RegrowthBranch(input_width, capacity, output_width)` under the shared
[model contract](../interfaces/model-contract.md). The supplied directed topology
is copied into registered `source` and `destination` buffers. Repeated edges
contribute separately. Each actor has its own encoder, decoder, local transform
and readout; communication occurs through ports. Actor hidden columns have no
assumed alignment or biological identity.

The field accepts one cell's `(actors, width)` state and returns the same shape;
`observe` returns `(actors,)`. The regrowth branch accepts an input with a final
axis of `input_width` and preserves its leading axes. Feature weights are
initialized with nonzero values and outgoing weights with zeros: the initial
branch contributes exactly zero, outgoing gradients are live, and feature
weights begin learning after the outgoing weights move. This preserves a
preexisting prediction when a branch is added. No arbitrary basis conjugacy of
the coordinatewise `tanh` field is asserted.

Run the small CPU witness from the repository root:

```sh
CUDA_VISIBLE_DEVICES='' /home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python experiments/moonshot-parallel-v1/actor/check.py --receipt experiments/moonshot-parallel-v1/actor/receipt.json
```

The check verifies input and all parameter derivatives in float64, including a
noncontiguous input; repeated-edge summation and deterministic CPU evaluation;
empty topology and invalid dimensions/topology; initial zero branch effect;
live outgoing gradients and subsequent feature gradients; and 80-step synthetic
actor-rollout and regrowth fitting. `receipt.json` records exact invocation,
shapes, source hashes and observed values. Use `.to(...)` to convert modules;
state and module floating dtype/device must match. CPU float64 is qualified here.
CUDA, native CE execution, performance, general nonlinear coordinate remapping
and biological interpretation remain unqualified. The training examples concern
whole supplied model alternatives and establish engineering behavior only.
