# Supplied model refactoring witness

`model.py` implements the linear maps supplied in the bootstrap's
`reference/moonref/promotion.py`. States use a final coordinate axis, with column
laws represented as batched row-vector Torch calls. For `q=T s`, the migrated
law is `T A T^-1` and readout is `O T^-1`. `LinearRefactoring.conjugate` returns
the migrated model, fixed maps, and coordinate/epoch receipt. For a quotient,
call `LinearRefactoring(old, E, P, old_ids, new_ids, quotient=True)`. Admission
checks `P E=I` and `A E=E(P A E)`; initialization rejects states outside `s=E q`.
The new law is `P A E`, and readout is `O E`. This equivalence is limited to the
supplied invariant manifold and its tangent domain. Unconstrained later training
of quotient parameters can invalidate the correspondence.

The checked coordinate names are externally supplied labels. The example
reassigns a transformed coordinate to `anonymous-actor:0`, with an explicit
model epoch change from 1 to 2. Replicated state copies use the embedding; their
adjoints reduce by its transpose, adding contributions. No support similarity
or biological identity inference is involved. `EpochGuard` is a prototype
controller helper: acquire a token for each retained old tape, release it after
backward, and publish only when all old tokens are released. It is not wired to
Torch's tape lifecycle or a native CE runtime. The prototype cold-rebuilds the
law/readout and declares transformed optimizer moments reset.

`ResidualActor` adds a named actor at epoch 3 with hidden row `h=tanh(U q)` and
readout `O q+V h`, reusing the accepted `actor/model.py:RegrowthBranch`. Nonzero
seeded incoming weights and zero outgoing weights preserve the initial readout
and give the outgoing weights a live gradient. Later steps give the incoming
weights gradients. The witness freezes the migrated law/readout and creates a
fresh Adam optimizer for the new branch. The state trajectory law stays fixed;
the learned observation function changes. This addition is labeled a new model
family, while the supplied linear maps are equivalent lowering. No approximation
is claimed. The hidden row is an algebraic predictive actor, with no independent
dynamical state estimation claim.

Run from the repository root:

```sh
CUDA_VISIBLE_DEVICES='' /home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -B experiments/moonshot-parallel-v1/refactor/check.py
```

The committed `receipt.json` was generated with the same command plus
`--receipt experiments/moonshot-parallel-v1/refactor/receipt.json`. It records
source hashes, migrations, trajectory errors, selected response Jacobians,
input/law/readout gradient checks, invalid-map/manifold rejection, replica
adjoints, tape guard behavior and live residual fitting. The deterministic CPU
float64 witness fits synthetic targets over 120 steps. Native CE integration,
CUDA qualification, nonlinear discovery, biological validation, general
optimizer-coordinate migration and performance remain unqualified.
