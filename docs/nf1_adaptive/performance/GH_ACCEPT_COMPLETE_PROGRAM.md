# GH ACCEPT complete-program measurement

`gh_nf1a_complete_program` measures the supplied GlassHelix direct-map plan and
its retained Cellerator response block. It selects five representative rows:
widths 16, 33, and 65; unary `tanh` and binary `multiply`; one and four supplied
candidates; and one and four steps. Unary rows execute one JVP direction and
binary rows execute two independent directions.

The delivered direct-map API has no compact activity execution route. The
`all`, `half`, and `churn` labels therefore describe an exact host activity map
applied to every state, forcing, and direction value before those values are
uploaded to the actual direct-map binding. `half` uploads its fixed alternating
map; `churn` uploads an alternating map whose phase changes at every step and
increments the retained GH activity generation. No memset, stamp-only refresh,
logical sparse label, or unimplemented route is presented as execution.

Every scenario emits five raw `cold` and five raw `resident` rows, followed by
per-phase medians. Cold creates fresh CUDA streams and resident CE vectors,
performs initial uploads, prepares the retained direct-map/response plan, runs
all requested steps, directions, and supplied candidates, then tears them down.
Resident creates that same setup once and reports its exact `setup / 5` share on
each reuse; every reuse still pays its own value refresh, activity retention,
forward launch and completion, saved-primal JVP launch and completion, full
output/response downloads, analytic checks, and GH observation/inference work.
Teardown is reported separately and split across resident rows.

Forward and JVP results are downloaded in full after each execution. `tanh`
uses the final differentiation fixture's prepared `tanh_forward` callback and
checks `(1 - tanh(state)^2) * direction`. Binary rows use a prepared multiply
callback and check `forcing * state_direction + state * forcing_direction`.
The actual downloaded forward output is passed through a GH `observation_map`
and `finite_candidates` propagation/assimilation with valid CE/GH provenance.
The benchmark rejects any disagreement; finite-only checks are insufficient.

The benchmark reports host allocation, initial upload, preparation, each-use
value refresh, actual-map retention, forward and response host-launch/device
completion, observation/download, teardown, and total costs. It makes no
speedup claim and labels compact support, FP16 response, capture, and device
interchange as unsupported delivered routes.

CMake snippet for the ACCEPT owner:

```cmake
add_executable(gh_nf1a_complete_program bench/native_foundation/nf1_complete_program.cu)
target_link_libraries(gh_nf1a_complete_program PRIVATE
  GlassHelix::foundation Cellerator::native_foundation CUDA::cudart)
set_target_properties(gh_nf1a_complete_program PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
add_test(NAME gh_nf1a_complete_program COMMAND gh_nf1a_complete_program)
```

## Leased V100 measurement

The accepted source candidate is `1c22d320fd65b21976f159792e2e93b2c4af7120`
(source fingerprint `7b38d584a5f65532fda46cb2a2578ec5c2d989ca50369eb7d8ec02a5e6b99f35`).
A CUDA-controller run completed successfully on one Tesla V100-SXM2-16GB
(compute capability 7.0, CUDA 12.9) after three clean quiescence samples. Its
controller evidence ID is `824295c5-ff13-4ff6-96fb-3822a02788dc`; the result
record is `/tmp/nf1a-gh-benchmark-result.json`. The raw five-sample CSV is
[foreground.stdout.txt](../../../.todo-orchestrator/runtime/background-artifacts/foreground/9077487f-0941-4bb5-8921-df0c93583590/foreground.stdout.txt)
(SHA-256 `0f35c9168a4f201ddfaae4578f302325c21810f3ca80c85af03eb19049748257`),
with the corresponding [lease receipt](../../../.todo-orchestrator/runtime/background-artifacts/foreground/9077487f-0941-4bb5-8921-df0c93583590/lease.json).
The binary SHA-256 was
`e2d36765f92e13188997192a85038cdb7f3bc2f0e0a38b0694bc55cc948ae6f5`.

All times below are microseconds. Ranges are the minimum--maximum of each
five-row raw distribution; the retained CSV includes every raw sample.

| Selected workload | Cold median (raw range) | Resident finite-lifecycle median (raw range) |
| --- | ---: | ---: |
| width 16, `multiply`, 1 candidate, all, 1 step, 2 directions | 391.864 (383.657--755.364) | 176.766 (175.478--193.553) |
| width 33, `tanh`, 1 candidate, half, 1 step, 1 direction | 319.610 (313.033--350.651) | 115.706 (113.436--126.477) |
| width 33, `multiply`, 4 candidates, half, 4 steps, 2 directions | 2487.870 (2457.764--2538.335) | 2121.425 (2116.426--2151.994) |
| width 65, `tanh`, 4 candidates, churn, 4 steps, 1 direction | 1487.143 (1463.542--1527.603) | 1136.661 (1110.585--1141.176) |
| width 65, `multiply`, 4 candidates, churn, 4 steps, 2 directions | 2526.272 (2518.457--2561.548) | 2153.613 (2139.353--2198.871) |

The resident value is finite lifecycle accounting: one fifth of measured
allocation, initial upload, preparation, and teardown plus every measured
per-use refresh, activity retention, forward, response, download, check, and
observation cost. It is neither a kernel-only timing nor a speedup comparison.
The width-16 cold distribution includes one 755.364 us first-use sample; its
other four raw cold samples span 383.657--401.397 us, and the table retains the
outlier rather than hiding it.

Every measured row completed the full forward-output and JVP analytic checks,
then completed GH observation and inference; no discrepancy was reported.
These five scenarios, one GPU, and five repetitions do not establish a
throughput crossover or a general speedup. The direct-map implementation still
has no compact active-only route, so `all`, `half`, and `churn` here measure
input-coefficient updates rather than compact support. FP16 response, graph
capture/replay, and device interchange remain unsupported and unmeasured.
