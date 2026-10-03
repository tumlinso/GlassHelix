# Scientific specification and result seam

`GlassHelix/experiment/specification.hh` composes the existing immutable
`core::system_definition`, `models::candidate_set` and typed observation/provenance
contracts. State, parameters and measured quantities retain their declared roles.
It accepts externally propagated results, retains complete hypothesis/mechanism
and state tuples, and records the conditioning history from the existing
finite-candidate inference owner. A prediction is required for each alternative.

The Python `glasshelix.experiment` package adds typed declarations, owner-supplied
predictions, provenance and JSON result replay. `ExperimentSpecification.authorize`
uses the predecessor `EvidenceView`, `EvidencePermission` and `evidence_manifest`.
It therefore retains original masks, measured zero, permitted modalities, split
rules and prospective cutoff checks. Assay detection is a separate optional
boolean mask. The evidence checksum covers the original predecessor observation
manifest; detection is an additional declaration, not part of that checksum.

The supported fixed candidate set may be reweighted by its inference owner;
whole state coordinates and mechanism/parameter identities remain fixed through
this result seam. Changed state alternatives need a new explicit specification.
Parameter identities refer to external owners; no tensor, coefficient, optimizer
or tape storage is introduced. Replay reconstructs scientific declarations and
results, not a live numerical execution context.

The seam rejects dropped, duplicate or mixed alternatives, wrong measurement
roles/extents, missing provenance, unsupported global identification and claims
of individual histories from population snapshots. Biological status remains
`not_run`; an externally supplied audit does not qualify fitting. Numerical
precision and approximation decisions still belong to the referenced consumer
receipts. No new native CUDA, training, speed or biological claim is made.

Run from the assigned checkout:

```sh
/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -B \
  tests/substrate/science/check.py \
  --ce-include /home/tumlinson/Software/cellerator-ml2-cpp/include
```

The check compiles/runs the real C++ scientific contracts against installed CE
headers, exercises existing masked finite-candidate conditioning, then runs seven
Python tests through the actual predecessor evidence adapter. It prints exact
commands and source hashes. CPU checks use synthetic declarations and observations.

Shared integration requests: existing header installation already covers the new
C++ header. `GH-IS1-BUILD` should install the new Python package alongside
`glasshelix.learning`, expose the check through the supported test entry point,
and verify a clean installed import. No root CMake edit is required for this leaf.
