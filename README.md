# GlassHelix

GlassHelix is the theory-bearing scientific ML project for inferring and
modeling biological dynamics from incomplete observations. It asks what
description of biological state is sufficient to explain and predict relevant
future behavior, what hidden dynamical or causal structure is identifiable from
observations and interventions, and under what evidence an inferred structure
may legitimately be interpreted as biological mechanism.

This repository is at a foundational bootstrap, not at a settled model or
software architecture. Existing source is archaeological evidence and
experimental material unless this document or a later recorded decision says
otherwise.

## Scientific standard

GlassHelix may represent hypotheses, predictions, uncertainty, and competing
structures. It may make a mechanistic claim only within domains supported by
the observations, interventions, assumptions, and validation actually used.
The following boundaries are authoritative:

- Measurement is not automatically biological state.
- Predictive latent state is not automatically molecular mechanism.
- Predictive accuracy is not automatically mechanistic validity.
- Observability is not identifiability.
- Dynamical equivalence is not biochemical equivalence.
- Population snapshots are not individual trajectories.
- Mechanistic claims are valid only over domains actually supported by evidence.
- Non-identifiability is a legitimate scientific result and must not be hidden
  by selecting one convenient explanation.
- Sequence constrains biological possibilities and mechanisms but does not
  uniquely determine instantaneous cellular state.

The project should ultimately support scientific reasoning about biological
state; evolution, histories, and trajectories of state; population dynamics
where appropriate; latent regulatory variables and state-dependent regulation;
perturbations and interventions; causal and mechanistic inference; uncertainty
and competing compatible mechanisms; developmental transitions; and eventual
sequence-grounded explanation. This list states the scientific reach, not an
implementation architecture.

## Deliberately open questions

GlassHelix does not presently commit to:

- one universal definition of cellular state;
- deterministic or Markov dynamics;
- a smooth manifold or gradient/landscape dynamics;
- a static or Jacobian-defined gene regulatory network;
- individual-state or population-state modeling as the universal formulation;
- one neural-network family, or neural networks as the only inference machinery;
- PyTorch, JAX, or another ML framework as permanent scientific ontology;
- one latent-variable formulation, training paradigm, or causal formalism;
- one serialization representation; or
- one unique recoverable mechanism.

These are open scientific questions, not missing implementation.

## Project boundaries

- **GlassHelix** owns the scientific theory of biological dynamics and
  mechanistic inference.
- **Cellerator** makes biologically meaningful mathematical state and
  transformations calculable and efficiently executable.
- **CellShard** provides distributed execution, storage, delivery, and sharding
  without owning biological semantics.
- **Baseplane** provides sequence grounding and the route from inferred
  biological mechanism toward genomic explanation.

Baseplane grounds biology in sequence. Cellerator makes biological state
calculable. GlassHelix models and interrogates how that state evolves.
CellShard makes the computation scale.

Repository history does not transfer these responsibilities. In particular,
historical sequence, sparse-matrix, storage, CUDA, or distribution experiments
in this tree are not current GlassHelix ownership claims.

## Minimal engineering identity

GlassHelix is a scientific ML library and research system. Python is first-class
because the scientific and ML ecosystem requires it. C++ is first-class and is
the user's strongest systems language. CUDA is first-class for accelerator work
where appropriate. C++20 is the native baseline.

Standard C++20 named modules are preferred for new durable native boundaries
where toolchain interoperability permits. Headers remain legitimate at CUDA,
external-library, compatibility, and legacy boundaries. Existing code need not
be converted to modules during this bootstrap. Cellerator should own lower-level
mathematical execution where its abstractions apply.

No Python binding technology, package layout, ML framework, module graph, or
CUDA architecture has been selected.

## Navigation and work state

- [Repository map](docs/repository-map.md) classifies the existing tree by
  epistemic status. Classification is not endorsement or deletion authority.
- [AGENTS.md](AGENTS.md) records the tool-routing and scope rules for future
  agents.
- `todos.md`, `todo-status.md`, and `todos/` are generated views of the
  todo-orchestrator state. SQLite is operational authority and
  `.todo-orchestrator/state.snapshot.json` is durable recovery state; do not edit
  projections or the snapshot manually.

The initial semantic graph is intentionally tiny: `GH-BOOTSTRAP` covers
repository reconciliation, scientific identity and boundaries, tool indexing,
and legacy classification; `GH-SCIENCE-FOUNDATIONS` preserves the unresolved
foundational scientific questions as future work. It does not authorize model
design. Use project-control for the live overview and frontier.
