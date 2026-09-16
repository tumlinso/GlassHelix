# GH-NF1A-ACCESS: Deliver thin Python access, inspection and portable replay

Expose the same native implementation in Python, with correct array and asynchronous ownership, capability/failure inspection, logical replay artifacts and useful examples.

**Why:** Bind one library rather than duplicate its mathematics; replay logical inputs, not ephemeral pointers or opaque generations.

**Completion prerequisites:** CE-NF1A-ADOPT, GH-NF1A-ANALYZE, GH-NF1A-OBSERVE, GH-NF1A-SYSTEM. These are enforced by the outcome gate; they are not all claim-time dependencies. Useful preliminary work is allowed before they are satisfied, but no capability may be declared delivered early.

**GH-PYTHON.** A thin Python binding exposes the same native library with explicit host ingress, dtype/stride/copy policy, asynchronous owner/GIL/error lifetime and capability inspection.

Evidence: Native/Python parity in a fresh environment, bad dtype/stride and dropped-reference tests; no Torch-owned or Python-numerical substitute.

**GH-INTERCHANGE.** Host-first Python access is acceptable; GPU interchange is optional, explicitly unsupported unless stream/lifetime semantics are implemented and tested.

Evidence: No raw-pointer-only zero-copy claim. Enabled interchange requires actual ownership/readiness tests.

**GH-REPLAY.** Native/Python examples, capability/run/failure inspection and portable replay retain logical inputs, registered block definitions, identities, numerical policy and both source versions—not pointers.

Evidence: Fresh-process replay, altered-input variation, corrupt/missing/incompatible artifact tests; failed/pending execution distinguishable from valid empty output.

## Execution latitude

Choose the local design, implementation/testing sequence and sensible delegation. Follow existing owners; use an authoritative narrow scope transfer when a better location lies outside the initial ownership. Do not duplicate an implementation to satisfy a directory proposal. Do not create separate records for ordinary inspect/code/test/retry steps. Publish a compact result: source and actual tests/evidence, material decisions, limitations and consumer impact.

This outcome subsumes 13 observed unfinished legacy records; `machine/legacy-disposition.json` gives the exact mapping. Completed legacy inputs remain successful historical records and are reused, not redone. `machine/requirements.json` is the complete acceptance inventory; old procedural sequencing is superseded.
