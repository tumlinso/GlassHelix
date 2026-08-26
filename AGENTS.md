# GlassHelix workspace guidance

Read `README.md` and `docs/repository-map.md` before interpreting source. The
repository is archaeological; file presence is not an architectural decision.

## Tool routing

- Use **project-control** first for read-only inspection of current project
  identity, architecture, decisions, evidence, and frontier.
- Use **coding-workflow** as the authoritative front door for substantial code
  modifications. Claim one task and remain inside its returned scope.
- The todo/semantic orchestration beneath project-control owns project identity,
  tasks, decisions, dependencies, gates, claims, and future work state. SQLite
  is operational authority; generated Markdown and snapshots must not be edited
  as a parallel task system.
- Use scientific literature and web research for external scientific evidence,
  never as repository or project-state authority. Record the evidence domain
  and uncertainty of scientific claims.
- Use CUDA and performance tooling only for an actual accelerator or performance
  task. Hardware results do not decide biological ontology.

## Bootstrap boundaries

- Preserve unknown and unrelated work. Do not reset, clean, overwrite, or
  silently absorb archaeological material.
- Do not infer the future model, API, packages, bindings, ML framework, module
  graph, serialization, dataset schema, CUDA design, or detailed roadmap from
  legacy code.
- Keep GlassHelix, Cellerator, CellShard, and Baseplane ownership as stated in
  `README.md`.
- Treat non-identifiability and deliberately undecided foundations as scientific
  state, not implementation debt.
