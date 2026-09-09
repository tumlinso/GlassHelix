# T01 independent FP64 referees

`tests/native_foundation/fixtures/truth_known.hh` is test-only. It includes no
production owner, planning declaration header, optimized kernel, or production
index map. The named state fields identify driver, partner, hidden component and
reporter independently of packed storage. The executable constructs a separate
symbolic slot map and checks all 24 storage permutations at widths 0, 1, 16 and
33. These are oracle-adapter checks, not a claim to a production batch backend.

The equations match the prospective four-component nonlinear fixture: the
ordered triad is `tanh(driver) * partner/(1+partner^2) * hidden`, the reporter
field combines `(a+b)` times that triad with reporter decay, and upstream fields
are affine. The referee supplies FP64 fields and RK4 trajectories with the 0.75
forcing discontinuity aligned to a step boundary. Every stage reads a separate
state snapshot. A diagonal affine exact solution separately checks integration
accuracy and refinement. Tests also check the exact supplied `a+b` identity and
its analytic parameter derivative against a central finite difference.

The hidden-state pair has identical present reporter values and distinct future
reporters. A closed sum reduction is checked across several fibers using the
independent full equations; projection onto the reporter is explicitly nonclosed
because hidden coordinates change its derivative at fixed current observation.
This demonstrates a supplied reduction's closure domain, not mechanism recovery.

Eight deliberate defects must be rejected: Euler substituted for exact evolution,
reversed decay, swapped triad roles, a dropped tied contribution, in-place stage
contamination, merged hidden states, omitted reduced forcing, and a falsely
closed reporter projection. These controlled wrong variants establish sensitivity
of the referees. They are not represented as defects found in an existing
production implementation. Later consumer tests must feed real owner outputs
into the referees instead of treating this suite as production qualification.

The CTest `gh_nf1_t01` executes a real C++20 binary with FP64 arithmetic and
fast-math disabled. Comparisons require finite operands and use explicit absolute
plus relative tolerances: normally 2e-13, 2e-11 for the analytic integration
comparison, and 1e-11 for finite differences. Negative controls must differ by
more than 2e-9 absolute plus relative tolerance. Exact dimensional inventory and
invalid-step checks use integer/boolean assertions. No CUDA or performance
claim follows from these host-only tests.

For source-bound qualification, first commit the worktree, then populate an
external bindings JSON containing `build_dir` and an existing external
`evidence_dir`. Run `python3 -B tests/native_foundation/independent/prepare.py
--bindings <absolute file>`. The helper requires clean committed source, configures
with `NF1_HOST_VALIDATION=ON`, checks `CMAKE_HOME_DIRECTORY`, rebuilds the binary,
and records source/cache/bindings identity and actual build logs externally.
Then invoke the native acceptance gate using those same bindings. The native
runner separately verifies the exact test inventory, executable hashes and
non-skipped successful JUnit results. The helper's build receipt alone is not
passing test evidence.
