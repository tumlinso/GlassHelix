# GlassHelix: prepared migration decisions

This is a source-informed shortlist, not a complete file inventory. `tools/inventory.py` produces the full tracked/untracked listing at execution time. Entries from an old map remain **existence/consumer checks** until inspected locally. No source move is scheduled automatically.

| Observed material | Intended destination/status | Action and guard |
|---|---|---|
| README.md | Human entry | Remove global “archaeological” status; describe actual native foundation separately from open learning/science. |
| AGENTS.md | Agent route | Replace bootstrap-only constraints with current operational guidance; preserve non-identifiability and ownership rules. |
| docs/repository-map.md | Historical bootstrap record + replacement source map | Its statement about root CMake is stale. Preserve the 2026-08-26 reconciliation history; current map comes from current source. |
| docs/nf1_adaptive/performance/GH_ACCEPT_COMPLETE_PROGRAM.md | Results methodology / historical record | Keep lifecycle and unsupported-route boundaries. Export portable raw CSV rather than linking public readers into Todo runtime. |
| planning/native-foundation-v1; planning/nf1-adaptive-v1; planning/learning-v* | Historical/active planning | Label status and replacement; don't copy every task into architecture. ML2 may still change supported learning. |
| include/GlassHelix; src/system; src/artifacts; tests/native_foundation | Current native foundation | Keep good semantic grouping. Preserve actual Cellerator package component dependencies. |
| python/glasshelix; bindings/python | Current supported language boundary | Keep Python/backend roles explicit; no binding-framework rewrite. |
| legacy src/models, src/incomplete, Classifier, low-level helpers named in old map | Existence/consumer review needed | Verify current existence before proposing moves. If truly unused, move one coherent group to archive/legacy with manifest, not silently delete. |
| extern gitlinks | Historical boundary evidence | Inspect actual Git/submodule consumers; do not delete or rewrite siblings as part of a docs pass. |
| tests/docs with literal README text assumptions | Living validation | Redirect semantic checks to the right page, not blanket-remove them. |

## Default source decision

Documentation reorganization first. Keep the current component structure unless the one bounded cluster described above materially improves navigation. Record final paths in `inputs/moves.json`; update bindings after moves. Preserve immutable evidence paths or provide an explicit old→new resolution map. Never change code semantics merely to make the tree look tidy.
