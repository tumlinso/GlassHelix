# Audited cohort adapter

The reusable `glasshelix.data` package stores normalized records and the exact
predecessor audit, source CSV hash, audit implementation hash and evidence IDs.
The thin `experiments.celltag_reprogramming.data_adapter` calls the original
`data.py` audit/read functions. Existing preprocessing and clone leakage rules
therefore remain authoritative. No second normalization or feature selector was
introduced.

RNA views use frozen training-only features and exclude other regimes/QC failures.
Validation/test evidence keeps its split, and ATAC needs an explicit independent
feature declaration. Measured zero, missing values and entirely unobserved cells
remain distinct. Views retain sample/cell/replicate, released clone membership and
original time labels. Donor identity is unavailable in the current standardized
format. Population snapshots gain neither individual histories nor presumed RNA
and ATAC pairing. No detection mask is synthesized from missingness.

`view.to_evidence(time_mapping, time_units, dtype=None)` uses the existing
`learning.EvidenceView`; numeric times require an explicit map. Float64 is the
default conversion, and callers can request float32. `view.evidence_manifest`
retains the units, original labels, map and dtype alongside the predecessor
observation manifest. All-missing views cannot be converted into usable evidence.

`AuditedCohort.dumps/loads` retains the full normalized cohort/audit and checks
canonical hashes. `data_adapter.replay_cohort(text, records_path)` additionally
re-runs the actual predecessor audit against the supplied source and rejects
changed source bytes or audit implementation. The embedded digest detects changed
contents; it is not an authenticated signature. A real cohort must carry explicit
independent input provenance references. An audit alone does not qualify a fit.

Actual acceptance input is the retained
`docs/learning/pilot-evidence/pilot.records.csv`. The current missing biological
inputs remain recorded in `experiments/celltag_reprogramming/missing-inputs.toml`.
Missing external files fail at their actual path; no fixture is substituted.
Biology remains `not_run`.

Gate from this checkout:

```sh
/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -B tests/substrate/data/check.py
```

Six CPU tests exercise the actual retained pilot audit/replay, mask handling,
units, independent validation, clone leakage, conflicting joins, input changes
and missing external files. The gate reports input/source hashes.

BUILD request: install `python/glasshelix/data` alongside `learning`; keep the
experiment adapter outside the core dependency graph and qualify the installed
library import. No root CMake or numerical provider change is needed.
