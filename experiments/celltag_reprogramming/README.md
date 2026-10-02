# CellTag experiment client

This independent package audits standardized long-form CSV records and supplies
a small Torch shared-support reference. Current qualification uses synthetic
fixtures. Actual CellTag inputs, corrected clone-call provenance and cohort
authorization are unavailable; biological fitting is **not_run**.

Install independently with `python -m pip install --no-deps ./experiments/celltag_reprogramming`.
The audit uses the Python standard library; reference tests require Torch 2.7.0.
Native GH/CelleraTorch integration belongs to later qualified public adapters.

Required CSV columns are `sample_id,cell_id,clone_id,time_label,modality,replicate,regime,split,feature_id,value`.
Empty `value` means missing; numeric zero is observed. `clone_id` may be empty.
Modalities are `RNA` or `ATAC`; splits are `train`, `validation` or `test`.
Use `regime=common` for the authorized common-forcing cohort. Other regimes,
including perturbations, are excluded from training. Optional `qc_pass` is
`true` or `false` (default true), with optional `qc_reason`. Biological QC
thresholds are upstream policy; the client applies declared exclusions.

The join key is sample, cell, modality, replicate and feature. Identical rows
collapse; conflicting duplicates or cell metadata fail. Clone IDs must be
globally unique in the standardized file and cannot span splits. Labels such
as `11` and `12` remain distinct. Coverage describes observed records, without
inventing paired modalities, trajectories or parent-child links.

```sh
python -m experiments.celltag_reprogramming.data --records /path/to/records.csv --output /path/to/audit.json --feature RNA_MODULE_1
python -m unittest experiments.celltag_reprogramming.test_data experiments.celltag_reprogramming.test_design -v
# After independent installation:
celltag-audit --records /path/to/records.csv --output /path/to/audit.json
```

Only observed, QC-passing training RNA in the common regime selects features.
An explicit whitelist must be supported by those records. Validation/test,
ATAC and perturbation values cannot enter training or feature selection.
The emitted feature manifest freezes that selection and hashes its training
records; reuse its feature list for later processing rather than refitting on
held-out data. No normalization or transformed features are fitted here.
Clone IDs are audit metadata, excluded from predictor inputs in the provisional
blind population-law arm. Clone-call provenance still needs independent review
for withheld-modality or expression-derived matching leakage.

The report includes raw source and canonical record SHA256 hashes, clone/time/
modality/replicate coverage, missingness, QC reasons and overlapping exclusion
counts. A successful audit is engineering evidence; it does not establish that
the cohort supports a biological fit. The supplied design reference is separate
from Cellerator/CelleraTorch native or GPU acceptance.

## Bounded population-response pilot

The pilot predicts two declared RNA module responses from three other RNA
modules and the released time label. It fits common-regime training observations
through an explicit cutoff and evaluates clone-isolated test observations.
This is a cross-sectional population response check. It does not infer cell
trajectories or forecast an individual's fate. Clone IDs serve only audit and
split checks; ATAC values stay withheld. An explicit frozen JSON feature manifest
has `inputs` (three distinct RNA features) and `targets` (two other RNA features).
Every selected feature must be observed in permitted training RNA through the
cutoff. Missing targets use observation masks; cells with missing inputs are
excluded and counted. Original time labels survive in the audit; the model's
declared context scale is the numeric label divided by 12.

```sh
python -m experiments.celltag_reprogramming.pilot --fixture --output /tmp/pilot-fixture.json
python -m unittest experiments.celltag_reprogramming.test_pilot -v
python -m experiments.celltag_reprogramming.pilot --records inputs/celltag-common-forcing.csv --feature-manifest inputs/training-only-features.json --cutoff 12 --output /tmp/pilot-real.json
```

The CPU fixture fits the structured Torch reference, a snapshot-only control
without time, and a flexible non-dynamical predictor with identical authorized
RNA/time inputs. Their scores are synthetic engineering diagnostics. Duplicate
edge coefficients identify only their sum; weight/activity scale can also remain
ambiguous. Sampled clone abundance includes growth, survival and sampling and
is not an individual transition probability. Missing real inputs fail explicitly
with biological status `not_run`.

Under the controller's GPU lease, add `--gpu` to compare installed CT
`SharedSupportRelation` against the exact same Torch graph, initial parameters,
masked objective and Adam updates. The fixture uses three sources, two targets,
five distinct edge identities with duplicate endpoints, batch eight and tail
three. The comparison includes source/target activity generators, full objective
gradients, genuinely fitted native edge values, setup and five timed complete
iterations including conversion, backward, guarded update and synchronization.
Peak memory includes both resident graphs and is reported with its baseline.
A slower result remains `evaluated_not_promoted`; even a faster tiny fixture
does not establish a general speedup. Public GH is imported from the installed
SDK via `--gh-prefix` (default `/tmp/gh-ml2-installed-sdk`); CT is installed
separately and requires `CELLERATORCH_NATIVE_LIBRARY` for GPU execution.
