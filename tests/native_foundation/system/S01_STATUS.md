# S01 partial implementation evidence

The reusable assembled_mechanism owns signature/axis backing memory and immutable
scientific definitions, preserves explicit ordered n-ary roles and output effects,
and adapts actual compiled callbacks through Cellerator bind_compiled_stage and
execute_prepared_program_v2. Stage/candidate IDs do not change the supplied
scientific mechanism identity. No numerical evaluator, allocator or stream engine
is implemented in GlassHelix.

The real gh_nf1_s01 host test executes one joint nonlinear triad at width33 through
the existing CE program owner, validates its outputs, rejects arity mismatch and
unsupported derivative, and checks the mixed response cannot be a sum of separate
single-input contributions. The callback is the test-supplied scientific fixture;
it is not a general arithmetic provider or CUDA implementation.

An initial test failed because an explicitly zero-initialized output effect was
invalid; the API now requires caller-supplied effects and validates them with CE.
The corrected real test passes. Qualification compiles the actual canonical CE
schema/program sources into a test-only library; production linkage must use the
integrated CE owner target.

S01 remains incomplete pending the inspectable arithmetic assembly route. CE N04
owns the general batched arithmetic capability; CE N01 offers relation math only.
Do not replace that prerequisite with a GH scalar evaluator, fake arithmetic
provider or a descriptor-only completion. Complete the route and extend this test
once the CE arithmetic source/interface is available.
