# Repository map

This map prevents historical and experimental material from being mistaken for
current GlassHelix architecture. Status describes epistemic standing, not code
quality, deletion permission, or a permanent disposition.

## Current and useful foundation

- `README.md`, `AGENTS.md`, and this map are the authoritative bootstrap
  orientation.
- `.todo-orchestrator/` contains project identity and durable recovery state.
  `todos.md`, `todo-status.md`, and `todos/` are generated projections of the
  live todo authority.
- `.gitmodules` and the `extern/` gitlinks are current Git history, but their
  presence is not a current architecture decision or a transfer of sibling
  project ownership.

No pre-bootstrap model source is designated as current GlassHelix architecture.
Some utilities may later provide useful evidence, but reuse requires a scoped
review and an explicit decision.

## Experimental

- `src/models/GlassEtch/` is an early proposed CUDA-centric model. Its extensive
  model comment is a historical hypothesis, not current scientific doctrine;
  the `glasshelix::glassetch` namespace and selected prototype entry point do
  not implement the described system.
- `Classifier/` and `data/preprocess/` contain exploratory Python preprocessing
  and classification scripts with local assumptions and hard-coded paths.
- `src/tests/network/`, `src/main.cc`, and the root `CMakeLists.txt` reflect
  prototype build and network experiments. They do not define a validated
  library surface.

## Historical

- `archive/` is explicitly historical, including early Python model sketches
  and a backup build file.
- `.idea/` is retained IDE metadata from earlier work.
- The pre-bootstrap dirty tree is preserved exactly on local branch
  `preserve/pre-bootstrap-8ebf69f-20260826` at commit `c4b8c30`. It contains the
  `cuda_helpers.cuh` include-guard change, removal of `sigmoid.cuh`, and the
  empty-file move detected by Git from `colab.ipynb` to `matrixmarket.hh`. These
  changes were not judged, discarded, or folded into the bootstrap.

## Overlaps sibling ownership

The following material is legacy GlassHelix-tree code whose subject now overlaps
a sibling project's boundary. It is not automatically useless, but it must not
be used to reclaim that responsibility for GlassHelix:

- `src/memory/`, sparse-matrix and low-level CUDA helpers under `src/include/`,
  and related tests overlap Cellerator's mathematical execution responsibility.
- `src/Sequence.hh` and other direct sequence representations overlap
  Baseplane's sequence-grounding responsibility.
- historical sharding/distribution links, including the `extern/VitriNode`
  gitlink, do not replace CellShard's current execution, storage, delivery, and
  sharding responsibility.
- `extern/Cellerator` records an earlier integration approach; the sibling
  Cellerator project remains the authority for its own implementation.

Disposition, migration, or removal of these paths remains unresolved.

## Unknown or unclassified

- `src/incomplete/` is explicitly incomplete and includes graph, tokenizer,
  transformer, binding, sparse-reader, and prototype fragments. No future API,
  binding system, neural architecture, or module graph may be inferred from it.
- `src/models/GlassHelix/GlassHelix.hh` is an empty placeholder, not an
  architectural boundary.
- The actual buildability and scientific relevance of individual legacy tests
  and utilities remain unclassified. This bootstrap does not repair or validate
  them.

## Reconciliation record

On 2026-08-26, local `main` at `8ebf69f` was confirmed 11 commits behind
`origin/main`. The four pre-existing dirty paths were preserved on the safety
branch above. Main was then fast-forwarded without conflict to remote commit
`ce9eb73`. No source experiment was deleted, migrated, reformatted, or promoted
to current architecture during the bootstrap.
