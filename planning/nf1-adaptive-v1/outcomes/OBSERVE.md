# GH-NF1A-OBSERVE: Deliver observation and finite-candidate evidence updates

Implement distinct observation records and simple declared likelihoods, then propagate and reweight supplied finite candidates without losing joint identity or reporter correlations.

**Why:** What determines the answer and what helps infer that determining state need not be the same information. This must remain executable, not merely documented.

**Completion prerequisites:** CE-NF1A-ADOPT, GH-NF1A-ADOPT, GH-NF1A-SYSTEM. These are enforced by the outcome gate; they are not all claim-time dependencies. Useful preliminary work is allowed before they are satisfied, but no capability may be declared delivered early.

**GH-OBSERVATION.** Partial, linear and nonlinear observations have explicit axes, time, units, missingness and individual/population record type. Simple declared noise/likelihood and reproducible observation sampling do not change deterministic process dynamics.

Evidence: Missing is not zero; population is not individual history; nonlinear/forward-only capability tests; likelihood/sampling comparisons and explicit invalid evidence.

**GH-INFERENCE.** Supplied finite candidates propagate through the common engine and use stable normalized likelihood updates; preserve joint state/mechanism identity, reporter correlations and evidence history.

Evidence: Indistinguishable observations retain alternatives; discriminating evidence changes weights; correlated/anticorrelated reporter test; zero-total likelihood and underflow handling.

## Execution latitude

Choose the local design, implementation/testing sequence and sensible delegation. Follow existing owners; use an authoritative narrow scope transfer when a better location lies outside the initial ownership. Do not duplicate an implementation to satisfy a directory proposal. Do not create separate records for ordinary inspect/code/test/retry steps. Publish a compact result: source and actual tests/evidence, material decisions, limitations and consumer impact.

This outcome subsumes 10 observed unfinished legacy records; `machine/legacy-disposition.json` gives the exact mapping. Completed legacy inputs remain successful historical records and are reused, not redone. `machine/requirements.json` is the complete acceptance inventory; old procedural sequencing is superseded.
