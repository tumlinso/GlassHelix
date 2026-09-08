# GH-NF1-O01: Implement partial and linear observation

Status on delivery: **planned**, not claimed or implemented.
Project: `glasshelix`. First-class lane: `GH-NF1-L-O`. Workspace: isolated_merge.
Explicit prerequisites: `GH-NF1-S03`, `GH-NF1-X02`. Earlier entries in the lane queue also apply.

## Objective and implementation direction

Use Cellerator numerical projections with explicit measurement axes, time and modality, separate from the dynamical state schema.

## Acceptance and negative control

1. Selecting a hidden-state subset and a mixed linear readout produce expected values.
2. Reverse reconstruction of full state is not presumed from a forward observation map.

This task is complete only when the relevant implementation or review is available to dependent tasks and the evidence level is stated honestly. A lower-level test does not automatically qualify a full integrated system.

## Read before editing

Read `01_SCOPE_AND_DECISIONS.md`, `02_SCIENTIFIC_AND_NUMERICAL_CONTRACTS.md`, `04_SINGLE_CONTROLLER_AND_PARALLELISM.md`, `05_CROSS_REPOSITORY_RECEIPTS.md` and this task's rows in the machine catalogs. Then inspect the live source, tests and nearest AGENTS rules. The following are starting references, not a closed list:

- `README.md`
- `docs/repository-map.md`

## Initial exclusive write scope

- `include/GlassHelix/observation`
- `src/observation`
- `tests/native_foundation/observation`

These paths coordinate work, not constrain the scientific representation. A needed adjacent owner path may be transferred or added by the controller through authoritative scope operations before edits. Do not clone Cellerator machinery inside GlassHelix to avoid that handoff. Preserve unrelated work and do not alter another planning package or authority tables directly.

## Required qualification

Evidence kind: **executed_host_test**. Required CTest names: `gh_nf1_o01`.

```sh
python3 -B planning/native-foundation-v1/scripts/run_gate.py --group o01
```

Execution bindings are supplied through an external `NF1_EXECUTION_BINDINGS` file. Governance records use the exact path in `machine/acceptance_matrix.json`. No record is prefilled as passed. Test gates require real executable inventory, current source/build evidence and no skips. GPU groups require the actual resource lease and the same shared lock across both projects. Missing hardware blocks GPU qualification, not unrelated ready host tasks.

## Numerical, performance and ownership obligations

Keep scientific identity separate from layout, preserve input snapshot semantics and declare effects. Preparation may allocate bounded workspaces and maps; steady execution must not discover topology or silently canonicalize. Share only what full dependencies and generations permit. Unsupported operations return explicit status rather than a fake result. Qualification should record exact versus tolerance-based behavior and preserve nonfinite semantics. Do not force differentiability, biological naming or a specific learner into this task.

## Autonomy and handoff

Within this objective, make the best constructive choice and continue without per-choice user approval. Coordinate scope or developmental interface changes through the controller, update dependents and rerun affected tests. Commit meaningful work and push intermittently through the authorized workflow. Report changed source, actual reachable capabilities, evidence paths/hashes, numerical restrictions, integrated commit and remaining blockers. Do not mark a hypothetical path complete or start a separate biological experiment.
