# Design: one learning slice, open scientific representations

## The missing loop

NF1A evaluates supplied systems and supplied candidate alternatives. L1 adds:

```
authorized evidence + admissible family + declared unknowns
    -> native fit -> executable candidates
    -> condition on allowed evidence -> predictions / unresolved alternatives
    -> logical save/reload
```

The five public concerns are **evidence views, hypothesis families, learning problems, inference strategies, fitted explanations**. These are semantic boundaries, not a requirement for five class hierarchies. A small functional API or a few concrete types is preferable to a universal object model. Keep numerical data in existing CE representations and expose capabilities only when executable.

Evidence views preserve axes, time, missingness, individual/population meaning, clone/sibling versus same-cell relations, replicate IDs and input/target permissions. A fitting strategy cannot see unauthorized target-derived preprocessing. A likelihood over sampled observations does not assert intrinsic stochastic fate.

Hypothesis families may be observable-horizon, history-conditioned, augmented-state or mechanistic. Do not require an ODE, Markov closure, a neural encoder, deterministic dynamics or a unique molecular interpretation at the library boundary. Initially implement one modest family and a meaningful alternative/control, rather than every representation.

Unknowns have both scientific roles (state, parameter, forcing, observation/representation calibration) and inference treatment (fixed/fitted/bounded; global/group/individual/time-dependent). Learning a shared law and conditioning a held-out case are distinct operations. A latent fitted against its own future is retrospective, not a forecast.

Fitted results retain whole candidate identity, declared evidence, residual/adequacy diagnostics and explicit search limits. Probabilistic weights belong to a selected strategy; they are not mandatory. A finite candidate archive witnesses disagreements but is not an exhaustive compatible set. No candidate found is not a certified empty model class.

## Numerical seam

Prepare one native evaluation session against a structure and evidence layout. Bind changed values, run the actual forward model/readout/objective, retain needed operands, obtain selected derivatives, and update/publish values through CE. Avoid a GH tape, tensor runtime, hidden GPU dispatcher or duplicate solver. Scientific objective composition belongs in GH; repeated numerical arithmetic and its derivatives belong in CE.

The critical derivative is the **complete objective gradient for shared parameters and local unknowns**, including repeated uses, multiple observation times, observation parameters when fitted, and shared instances. Checking parameter generation is not computing its gradient. Differentiate the discrete calculation actually executed, including RK4 stages only if that model needs them.

Reuse structure preparation across value updates; invalidate stale response handles by actual value/forcing/observation generations. A structure change is not a same-shape value update. Transfers between parameterizations require explicit meaning. Keep logical checkpoints independent of pointers and include source/definition/evidence/numerical identity.

Start with native ownership of resident values and saved primals. A high-level native client or small fit/condition binding is enough; full Python GPU interchange is not a prerequisite. Measure a complete iteration, not a kernel-only surrogate. Keep actual response implementations tied to the forward they differentiate.

## Fast defaults

Use short fixed-structure horizon maps before requiring a rich continuous-time model. Use fixed training-only features before a universal learned observation mapping. Fit a small number of genuine shared/local unknowns with a simple optimizer. Use existing CE parameter/relation/update machinery when appropriate; add only demanded missing operations in CE.

No-change branches require executed capability evidence. Optional CE work is selected by a concrete connected-gradient probe, not guessed from task names. `machine/choices.json` records the decision domains and branch evidence; it is guidance, not a new workflow DSL.

## Independent experiment, cheap authority

The default experiment location is **`experiments/celltag_reprogramming` inside the GH repository**, with its own build/package entry and installed GH/CE dependencies. It must not be included in the core default build/import graph. This deliberately avoids a third authority and illegal sibling-path scopes while preserving separation and later extractability. Moving to a sibling repository is a later explicit ownership decision, not a hidden `../` write.

Dataset constants, downloads, clone joins, biological labels, preprocessing recipes and plots stay there. Generic evidence relations, evaluation, fitting and candidate reporting stay in GH. Reusable numerical operators remain in CE. A second client should not need CellTag code.

## Existing source to use, not replace

GH: `core/{quantity,system,observation,request,provenance}.hh`, `system/assembly.hh`, `dynamics/{evolution,response_plan}.hh`, `observation/observation.hh`, `inference/inference.hh`, and `interrogation/{diagnostics,alternatives}.hh` under `include/GlassHelix`; their current native tests.

CE: `include/Cellerator/parameters.hh`, `compute/operation/{prepared_relation,relation_update,native_foundation_contract}.hh`, `compute/operation/differential/local_arithmetic.hh`, resident numeric operations, segment reductions, and `execution/program/program_v2.h`. Inspect actual capability/export status before selecting reuse; a header or old example alone is not an installed callable path.

The two specific review defects and reproductions are in `references/`. Recheck current source first; keep already-correct work. The long review supplies context, not a mandate to implement every future capability it discusses.

## Stop line

Do not require structural search, amortized inference, universal observation learning, general SDE/DDE solvers, exact genealogy reconstruction, full-cell simulation, sequence grounding, automatic molecular naming, multi-GPU training, new JIT/AD, elaborate optimizers, or a paper-scale biological result. Leave representation/strategy seams open and report the next concrete missing capability after the first installed consumer works.
