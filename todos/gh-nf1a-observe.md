

<!-- todo-orchestrator:v2-managed:start -->
# GH-NF1A-OBSERVE: Deliver observation and finite-candidate evidence updates

Task revision: `185`; current project revision is in `todo-status.md`.

## Objective
Implement distinct observation records and simple declared likelihoods, then propagate and reweight supplied finite candidates without losing joint identity or reporter correlations.

## State
- Lifecycle: `done`
- Execution: `closed`
- Parallel policy: `parallel_safe`
- Result: `implemented`

## Next Action
Read planning/nf1-adaptive-v1/outcomes/OBSERVE.md and relevant source. Choose a short useful implementation loop; preserve actual acceptance.

## Ownership
- `exclusive`: `docs/nf1_adaptive/observation`
- `exclusive`: `include/GlassHelix/inference`
- `exclusive`: `include/GlassHelix/observation`
- `exclusive`: `src/inference`
- `exclusive`: `src/observation`
- `exclusive`: `tests/native_foundation/inference`
- `exclusive`: `tests/native_foundation/observation`

## Dependencies
- `task`: `GH-NF1A-ADOPT`
<!-- todo-orchestrator:v2-managed:end -->
