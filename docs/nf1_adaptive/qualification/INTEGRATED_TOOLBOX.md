# NF1A integrated GlassHelix toolbox

GlassHelix consumes the installed Cellerator native foundation; scientific roles,
state/parameter identity and provenance do not prescribe a permanent IEEE storage
encoding. Numerical representation and promotion remain explicit Cellerator
capabilities. FP32 is the reliable baseline, with supported FP16 numerical routes;
unsupported response encodings refuse rather than silently substitute a format.
No quantization subsystem or new biological model is introduced here.

The integrated producers provide supplied-system assembly, direct-map and actual
four-stage RK4 evolution, candidate propagation through Cellerator, observations
and likelihoods with missingness/reporting contracts, supplied alternatives,
rank/nullspace diagnostics using LAPACK, and supplied reduction certificates.
Response composition differentiates the declared direct map, observation, RK4 step
or rollout. Retained state, parameter, forcing and activity owners are checked
against the actual primal operands before launch. Scientific errors, integration
errors, support decisions and numerical conversion errors remain distinct.

The native and installed consumer tests cover widths beyond legacy fixed limits,
finite-difference and adjoint counterexamples, live-owner invalidation, forcing at
RK4 stage times, nonlinear response composition, and inference collection after
successful native execution. The thin Python binding calls Cellerator and requires
contiguous float32 buffers, retaining their owners through native execution.
Replay records execute the supported native local-add JVP and bind the compiled
GlassHelix and installed Cellerator source revisions. Parsing rejects wrong
source/policy/schema/block, malformed numeric fields and trailing data; numerical
serialization round-trips all finite float values losslessly.

## Deliberate limits

Response checkpoints currently use saved primals; recomputation is explicitly
unsupported. The Python surface exposes host JVP, not host VJP or GPU interchange.
Replay supports its declared block, not arbitrary object serialization. Optional
CUDA replay optimization is not selected. Performance reports are bounded
measurements of supplied workloads on the recorded hardware, not claims of
speedup or evidence for selecting a biological system.

## Acceptance and publication

The authoritative GH-NF1A-ACCEPT receipt binds the exact integrated source pair,
configured build, executed CTest inventory, binary hashes and actual GPU lease.
The source-bound published evidence note carries that receipt, the Cellerator
qualification receipt, performance records and inherited-work dispositions.
Release gates additionally require both origin/main revisions to equal the
accepted source pair. The unrelated root preprint and preserved legacy work remain
outside this implementation delivery. No biological experiment is selected.
