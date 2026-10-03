# Existing model clients on the experiment surface

`glasshelix.models.ModelClient` delegates to actual supplied matrix/Sylvester,
actor-port and ML2 joint-product/population relation modules. Model choice,
whole-hypothesis/domain/parameter identities and state shapes are explicit.
Scientific loss and biological interpretation remain caller choices.

Existing arithmetic is reused in place. `load_predecessor` requires an explicit
source path and accepted SHA256; no hidden source fallback is used. Direct owners
such as `ProductHypothesis`, `PopulationRelation` or a CT module can be bound
with a declaration. Actor topology and repeated edges stay distinct; checkpoint
restore rejects changed topology or scientific identity before parameter updates.
Native restore remains its sanctioned owner route and is explicitly unavailable
through the generic framework checkpoint helper.

`to_hypothesis` and `to_prediction` join the SCIENCE package after its source merge.
Hypotheses preserve the entire supplied state and mechanism/parameter identities.
Batched ML2 clients require exactly one supplied row for a whole individual
snapshot. Predictions require an explicit caller-selected observable; a plain
matrix field is not automatically a measurement map. Actor clients retain their
actual supplied readout. No marginal coordinate recombination is offered.

Five tests execute the current matrix, Sylvester, actor-port, ProductHypothesis
and PopulationRelation owners on CPU float64, checking forward/VJP equivalence,
time derivatives, declared readout, repeated topology, checkpoint replay and
misuse rejection. Native CUDA/CT execution, capture, higher-order transforms,
mixed precision, throughput and biological interpretation remain unqualified.
`capability` reports framework ownership, not a blanket producer qualification.

Leaf gate:

```sh
/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -B tests/substrate/models/check.py
```

Required combined consumer after SCIENCE/MODELS source integration:

```sh
PYTHONPATH=python:. /home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python -B tests/substrate/models/combined_science.py -v
```

This leaf's base does not contain the independent SCIENCE implementation, so its
combined check is submitted for root integration rather than counted as passed.

Root acceptance request: `python/glasshelix/models/NUMERICAL_DEMANDS.md` gives exact
source/symbol/shape and derivative needs for CE numerical ownership. BUILD should
install `glasshelix.models`, qualify installed imports and explicitly package or
pin the existing optional predecessor model sources needed by the factories.
The core installed wrapper itself does not import repository experiments.
