# NF1 legacy source disposition

GH-NF1-A03 review at 04c9d6af614ca941982fface62cdbf367ddba607.
This is a source review and ownership decision, not a successful legacy build.
No legacy files were deleted, reset, cleaned, or rewritten by this task.

| Material | Observed evidence | NF1 disposition |
|---|---|---|
| Root CMakeLists.txt | Two prototype executables select the same GlassEtch source; compiler is set after project(), OpenMP required, and entry source forced to CUDA even in the optional-CUDA configuration | Central GH M build owner replaces default wiring with linked native library/consumer and real registered tests. Retain old wiring only as explicit historical opt-in if it remains useful; do not count it as qualification. |
| src/models/GlassEtch | Long hypothetical autoencoder/GRN design claims masked reconstruction enforces causal inference; selected main.cc only includes GlassEtch.cuh | Preserve historical hypothesis and source for archaeology. Retire from default NF1 build. Never import reconstruction-as-causality assertions into scientific contracts. |
| src/Cell.hh | Fixed-width token pointer view, offset and dictionary translation; unmatched trailing endif | Historical token utility, not a domain/state identity contract. Do not repair merely to promote it to the new scientific Cell concept. Reuse requires separately scoped tests and ownership justification. |
| src/memory/SparseMatrix and src/include CUDA helpers | Generic CSR/coordinate storage and low-level numerical operations | Preserve for comparison; Cellerator owns any useful reusable algorithm. Review its current owner first, rehome only a demonstrated improvement under native scope, and validate real behavior before reuse. No parallel GH accelerator library. |
| src/memory/AsynchronousArray.hh and dataset utilities | Lazy-loading/chunked binary-file access | Historical storage overlap. Do not use this material to expand GH into CellShard storage/distribution work. |
| src/Sequence.hh and incomplete genome/tokenizer material | Sequence representations and incomplete experiments | Preserve; Baseplane remains sequence-grounding authority. No automatic public API promotion. |
| Classifier and data/preprocess | Hard-coded local datasets and output paths; classifier chooses CUDA or CPU opportunistically | Historical experiment tools outside default NF1; do not run dataset acquisition, select a fibroblast task, or copy silent CPU fallback into the real-CUDA default demo. |
| archive, .idea, incomplete fragments and empty GlassHelix model header | Historical, incomplete or empty material classified by repository-map.md | Retain; file presence does not settle model, binding, layout or runtime architecture. |
| planning/native-foundation-v1 and examples/native-foundation-v1 | Accepted scope, prospective contracts and demonstrations | Reuse as specifications and fixtures after checking each requirement. Reference-only output and declarations remain weaker than linked implementation evidence. |

The approved owner for new native scientific contracts is GlassHelix; Cellerator
owns operation semantics, prepared execution and derivative machinery. New CMake
registration, umbrella/package exports and integrated consumer wiring belong to
GH-NF1-L-M's central queue. Focused implementation lanes may own assigned build
fragments; request an authoritative narrow transfer when an early real test needs
central registration. An empty CTest run cannot discharge a prerequisite.

No source is declared tested-and-reused by this review. Numerical reuse must
carry input/effect and snapshot semantics, exact/tolerance qualification, tails,
nonfinite behavior and consumer-accessible source commits. This deliberate
absence of premature promotion preserves useful history without building on
unsupported legacy assumptions.
