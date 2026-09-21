# Observation and finite-candidate inference

`observation_map` reads a completed, provenance-bearing common-engine output.
It provides partial, linear and declared nonlinear readouts; a nonlinear map is
forward-only unless it explicitly advertises JVP/VJP capability.  It never
propagates state or implements a device backend.

`gaussian_noise` is an explicitly selected elementary observation model.  Its
seed controls only reproducible measurement sampling.  Missing channels make no
likelihood contribution, while a measured zero remains evidence.  An
`observation_record` retains its axes, time, units, modality, sampling ID and
individual/population interpretation from the scientific core contract.

`finite_candidates` accepts only supplied joint candidates.  Propagation is a
callback into the common Cellerator-backed engine, and stable log-sum-exp
reweighting preserves each candidate's joint state and mechanism identity.  A
zero total likelihood is an explicit error; inference is not a learner,
resampler, mechanism search, or population model.
