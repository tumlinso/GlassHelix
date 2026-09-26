# GlassHelix after NF1A: review and proposed learning substrate

**Review date:** 26 September 2026  
**Status:** Read-only architectural/source review and design proposal. No repository mutation, new learner implementation, or biological fitting is claimed.

## 1. Recommendation

Preserve NF1A. Add an evidence-to-hypothesis learning layer above its execution and interrogation facilities, and develop the CellTag reprogramming study as an external consumer of that layer.

The missing transition is:

```
observations + scientific question + admissible hypothesis family
    -> fit/infer values, states, inputs, and optionally structure
    -> executable candidate explanations
    -> predictions, discrepancies, unresolved alternatives, and provenance
```

It is not another numerical engine, another automatic-differentiation system, or a single mandatory neural cell model. The first useful increment should fit a small, incompletely specified family through native evaluation, retain alternatives, and work in an installed external consumer.

Supplying a hypothesis class is a scientific assumption. Supplying every fitted system by hand should no longer be necessary.

## 2. Scope and evidence

Live project-control inspection observed GlassHelix at HEAD `da97825a18206f1ec727ea59439ca694995d7b94`, with a dirty worktree and Todo revision 237. The worktree fingerprint was `5f360b00686734110a869e2fce33da80fc5de5ed64ea6534b251df42d074e99d`. Cellerator was clean at `2c3b094aadcf024d04bb29213ee2dde6aa01fc00`, Todo revision 7417. Current source targets were read directly, not inferred from task titles.

The authoritative task read records GH-NF1A-ACCEPT as done/validated at the GlassHelix HEAD above. Qualification documents and test source were examined. A broad evidence query did not retrieve the complete underlying CUDA evidence inventory, so this report does not independently certify all binaries, receipts, or performance measurements. No upstream CTest or GPU campaign was rerun. The included numerical probes are standalone reproductions of two source algorithms, not upstream tests.

The source ledger at the end identifies inspected files and ranges. `G*` denotes GlassHelix, `C*` Cellerator, and `W*` public primary sources. Source-derived findings are distinguished below from proposed interfaces and proposed experimental choices.

## 3. What NF1A provides

### 3.1 Scientific descriptions and assembly

The core distinguishes dynamical state, observations, inferential information, parameters, forcing, and context. Quantities retain biological axis identity, structure epoch, units, scale, and optional biological labels. Scientific mechanisms support ordered n-ary arguments, repeated arguments, and explicit outputs. Immutable definitions and separately identified instances are already appropriate foundations for learned candidates. [G1–G3]

Assembly links scientific mechanisms to Cellerator compiled implementations or registered indexed evaluators. This is valuable, but does not itself generate a model from data or compile every possible trainable expression. Keep its separation of scientific occurrence and numerical realization. [G4]

### 3.2 Evolution

There are real resident direct-map and four-stage fixed-step RK4 execution paths. GlassHelix schedules the scientific calculation; Cellerator owns the arithmetic and resident vectors. Forcing stages, discontinuity scheduling, and primal retention have explicit contracts. [G5]

The delivered RK4 route is bounded: FP32 resident vectors, a field with one launch binding, and state-width forcing in that route. These are implementation capabilities, not requirements on all future biological models. Parameter dimensions, observation dimensions, and input dimensions must be allowed to differ as the fitting interface develops.

### 3.3 Differentiation and response

The response layer distinguishes the vector field, direct map, observation, implemented RK4 step, and rollout. It checks saved state/parameter/forcing/activity owners and generations, and composes JVP/VJP actions over supplied primitives. Recomputed primals are explicitly unsupported rather than silently reconstructed. [G6]

This is not yet a general training gradient. In the inspected RK4 buffers, directions and adjoints are exposed for state and stage forcing. Parameter provenance is checked, but a complete parameter-derivative path—with shared parameters accumulated over stages, observations, cells, and repeated arguments—is not supplied by those buffers.

The differentiation test uses real native actions and finite differences, but constructs the RK4 stage primals for its multiplication example explicitly. It does not demonstrate an automatic trace builder for an arbitrary newly fitted model. A learning session must obtain retained primals from the actual forward evaluation, not ask every experiment author to reconstruct them. [G6, G12]

### 3.4 Observation and candidate inference

The observation layer preserves missingness and measured-zero semantics, supports partial/linear/nonlinear readouts, and supplies an elementary Gaussian observation likelihood. However, `engine_output` and observation evaluation use host vectors. The linear map is evaluated by host loops. Nonlinear observation capability metadata is not, by itself, an executable derivative implementation. The native observation response plan is a separate route that needs coherent binding to the observation being fitted. [G7]

`finite_candidates` preserves supplied joint state/mechanism identities and scores their predicted observations. It neither learns candidate values nor proposes structures. Propagation caches outputs without replacing the candidate's joint state; it should not be mistaken for a general sequential state filter. The existing CE path collects candidate outputs to the host. [G8]

### 3.5 Interrogation, reduction, access, and qualification

There are small selected-sensitivity SVD diagnostics and atomic alternative comparison. Alternative comparison deliberately does not convert optimization-run counts into probability or local agreement into global identification. Its current purpose is bounded comparison of supplied alternatives, not mechanism search. [G9]

The reduction helper is a concrete two-component equal-rate example with error-provenance checks, not a general reduction-discovery system. Preserve the example as a regression/control; extract reusable certificate semantics only as another consumer needs them. [G10]

The Python surface is narrow: host JVP for add/multiply/tanh on contiguous one-dimensional FP32 buffers. Native VJP exists; Python VJP and CUDA interchange are not exposed. Replay concerns one registered local-add JVP, not a model checkpoint or training restart. [G11]

The integrated acceptance/benchmark documentation provides useful bounded evidence. Its small unary/binary workloads and full per-use downloads do not demonstrate scalable end-to-end learning. Activity labels in the reported benchmark are input-mask updates, not proof of compact active execution. [G13]

## 4. Correctness findings to address before richer inference

### R1 — Incorrect extraction of right-nullspace vectors

`interrogation/diagnostics.hh` calls Fortran DGESVD and then copies contiguous chunks of its `VT` output as if they were rows. In column-major storage, those chunks are columns. The right singular vectors are rows of `VT`. Correct extraction of vector `i` is `vt[i + j*ldvt]` across `j`. [G9, W4]

For the rank-one matrix

```
1 2 3
2 4 6
```

the included probe calls DGESVD with the reviewed `JOBU=N`, `JOBVT=A` settings. The reviewed indexing produces null-vector residual norms about 7.85 and 1.66 in this LAPACK build; correct indexing produces norms below 1e-14. Null bases can differ between LAPACK builds, but the storage error does not depend on a particular basis choice.

The inspected tests mostly exercise two-dimensional controls where this indexing can accidentally work. Add non-symmetric right-singular bases, rectangular matrices, multiple null directions, scaled-coordinate checks, and reconstruction residuals. Do not treat the existing local null reports as generally reliable until repaired. [G12]

### R2 — Finite candidate weights lose recoverable hypotheses numerically

`finite_candidates::assimilate` forms temporary log weights, normalizes them, then stores only ordinary floating-point weights. An extremely disfavored but finite candidate can underflow to zero. Later evidence then receives `log(0)`, and that candidate cannot recover. [G8]

The probe uses means 0 and 1, equal priors, and Gaussian sigma 0.02. Observing 0 then 1 leaves weights `[1,0]`; reversing the observation order leaves `[0,1]`. The total evidence is equal for both fixed candidates and persistent log weights recover approximately `[0.5,0.5]`.

Store log weights as the authoritative numerical state of this probabilistic strategy. Distinguish explicitly zero prior mass and deliberate pruning from finite evidence that merely makes a candidate small. Test evidence-order and batch-partition invariance for fixed candidates and the same conditionally independent evidence. These invariances do not imply arbitrary time reversal of a dynamical filtering problem.

### R3 — Learning requires stronger forward/response binding, not merely more derivative flags

The current response tests are valuable, but experiments should not author their own analytic stage primals or treat an advertised observation derivative as implementation. A prepared evaluation must return a trace/handle tied to the exact executed operator, values, forcing, observation mapping, support, and generations. Its pullback must reject a mismatched trace. General operator differentiation belongs in Cellerator, while GH assembles the scientific objective and selected unknowns. [G6–G7]

This is a missing consumer capability rather than a claim that the delivered primitive derivatives are generally incorrect.

## 5. The smallest coherent public design

I would initially expose five concepts, with a native evaluation session beneath them. The spellings are illustrative, not a proposed frozen ABI.

### 5.1 EvidenceView

An immutable, restricted view of observations and their relationships. It must retain biological feature identity, physical collection time, sampling identity, modality, missingness, replicate/context, and relationship semantics. Same cell measured repeatedly, different members of one clone, experimentally paired modalities, and inferred correspondences are different relationships.

Evidence must distinguish what exists in the dataset from what a particular fit or inference is allowed to see. A protocol should specify both collection times and permitted conditioning/target roles. A held-out future is not made available to local latent inference merely because it is present in the same file.

A partitioning layer may use barcode identifiers to keep related samples together while withholding those identifiers from the model. There is no contradiction: split construction and model inputs have different permissions.

The initial representation can be bounded tables plus explicit ragged groups, with a few needed relation kinds. It need not begin as a universal biological graph database. Numerical data can be views of standard scientific containers or Cellerator/CellShard-backed storage. The core should not become another AnnData replacement or single-cell preprocessing package.

Each derived feature representation needs its training partition, recipe, fitted values, and source provenance. A supplied normalized value, a raw count, an imputed feature, and a missing observation are not interchangeable evidence.

### 5.2 HypothesisFamily

A family specifies what may vary and how a candidate is evaluated. It can describe a direct finite-horizon predictor, a history-conditioned predictor, a latent dynamical model, or a structured mechanism family. It must not require every candidate to expose an ODE or assert that its latent coordinates are biological state.

Optional capabilities can include prediction, observation evaluation, state advance, history conditioning, initial-state inference, selected sensitivities, constraint evaluation, and intervention response. Algorithms ask for the capabilities they need. A missing capability returns an explicit unsupported result, not a fabricated zero gradient.

The family separates structural choices from fitted values. Adding a latent dimension, replacing an observation family, or inserting a mechanism is a structural proposal. Updating coefficients within the same structure is not. A family can initially have a fixed structure and still generate genuinely learned candidate systems by fitting its unknown values.

A fitted hypothesis should retain an interpretation profile: anonymous predictive coordinates, question-relative sufficient-state claims, effective response roles, molecular anchors, and tested intervention semantics. These are evidence-dependent claims, not an enum that automatically upgrades a model's scientific status.

### 5.3 LearningProblem

A learning problem combines the question, allowed evidence, hypothesis family, selected unknowns, fit criteria, and constraints.

Unknown metadata needs two separate axes:

- **Scientific role:** state, parameter, forcing, observation calibration, representation parameter, context, or inferential information.
- **Inferential role and scope:** fixed, fitted, bounded, marginalized, or conditioned; global, replicate-level, group-level, individual-level, or time-function-valued.

A global kinetic coefficient and a per-clone initial condition may both be optimized without being the same kind of biological quantity. An optimizer moment is neither one. Use Cellerator's existing parameter/storage descriptors and resident bindings for numerical values; GH adds scientific scope, ties, admissibility, and interpretation. [C4]

The objective should be a composition of explicit evidence terms. Some are likelihoods, some residuals, some population discrepancies, some hard feasibility constraints, and some scientific regularizers. A regularization preference is not new experimental evidence. Distinguish a good relative candidate score from absolute adequacy.

An evidence term declares which sampling relation it consumes. An individual transition loss must not silently accept two unrelated destructive snapshots. A clone-time population term may consume two bags of observations without pairing their rows. A likelihood over sampled cells does not itself assert intrinsic stochastic biological dynamics.

### 5.4 InferenceStrategy

A strategy chooses how candidate values and structures are proposed, fitted, conditioned, compared, and refined. It consumes the learning problem and evaluation capabilities rather than owning numerical buffers or its own tensor engine.

The first strategy can be very small: fixed-structure constrained fitting, optional alternation between global and local unknowns, a few starts or explicit alternative structures, and a candidate archive. No amortized encoder, Bayesian sampler, structural search engine, or universal optimizer framework is needed initially.

The distinction between global learning and local inference must be real. Fit the common law on training groups; freeze it for evaluation; infer a held-out group's unknown state or forcing from only its authorized measurements. An optimized latent fitted using that group's future target is retrospective reconstruction, not forecasting.

Later strategies can include derivative-free fitting, variational or particle methods, explicit compatible-set search, adversarial rival explanations, sparse structure proposals, and learned initialization/conditioning. These strategies do not require changing the scientific identity of evidence or duplicating the execution substrate.

### 5.5 FittedHypothesisSet

The output is one or more fitted candidates, their authorized evidence basis, diagnostics, and unresolved questions. It need not carry probabilities. When probabilities are provided, their likelihood/prior/approximation semantics must be explicit.

Distinguish at least: numerical failure, optimization not converged, no compatible candidate found, certified infeasibility within a declared class, candidate-supported ambiguity, and a supported prediction. Failing to find a fitting candidate does not by itself prove that the scientific hypothesis class is empty.

Each candidate preserves whole explanations. Do not independently mix favored edges from incompatible models. Compare different representations through common observables or declared alignments, not raw latent-coordinate distance.

A finite archive is a searched subset, not all compatible mechanisms. If two surviving candidates disagree, they witness ambiguity. If all found candidates agree, agreement is only as strong as search coverage. A min/max over found candidates is not automatically a conservative prediction interval over all possible candidates.

## 6. The internal evaluation session and ownership boundary

A native `EvaluationSession` should prepare one candidate structure against a compatible evidence layout, bind changing values, evaluate an objective, and expose selected derivative actions and diagnostics. It can be implemented over existing Cellerator prepared programs and GlassHelix evolution rather than a new general execution graph.

Conceptually:

```
problem = LearningProblem(question, evidence_view, family, unknowns, terms)
session = prepare(problem, numerical_policy)
result = strategy.fit(problem, session)
conditioned = strategy.condition(result, allowed_test_evidence)
answers = interrogate(conditioned, query)
```

This is design pseudocode. No such API is claimed to exist.

### A complete native fitting operation

The important new path is:

```
selected fitted values
 -> actual model/observation forward evaluation
 -> evidence discrepancy and constraints
 -> residual cotangents
 -> native pullbacks through the executed calculation
 -> gradients for the requested scientific unknowns
 -> proposed update and value publication
```

For a repeated discrete computation `x[k+1] = F(x[k], theta, u[k])`, the parameter gradient must include every use of the shared `theta`. Observation-map parameters require their own contributions, and losses attached at several times introduce adjoints at each time. State adjoints alone are not enough. The same applies to shared parameters across cloned instances and to repeated arguments within an n-ary mechanism.

A legitimate optimization step changes values and their generations while reusing structural preparation. A structural proposal changes the structure identity/epoch, causes appropriate preparation, and invalidates stale traces. Transferring values or optimizer state between representations needs an explicit mapping; a coincidentally equal shape is insufficient.

Forward evaluations used for gradients retain actual intermediate operands or employ a separately qualified recomputation scheme. Begin with saved primals for a short horizon. Do not start by implementing every adjoint method. Responses must differentiate the executed discrete RK4 calculation when that is what the objective evaluated, not an unrelated continuous approximation.

### Cellerator owns

Numeric operations and their derivatives; dense/sparse/relation contractions; gather/scatter and repeated-use accumulation; resident buffers, physical layouts, readiness, and value generations; numerical reductions; vector update arithmetic; generic executable preparation and scheduling; numerical policy and supported encodings.

Existing prepared relation/gradient/update facilities are relevant reuse candidates. Model-specific AdamW code also exists in old examples, but that does not make it a ready general optimizer API. Use a simple update first, and extract or extend Cellerator's reusable arithmetic only where the chosen fitting workload requires it. [C1–C5]

### GlassHelix owns

The scientific hypothesis and its unknowns; observation/evidence interpretation; how evidence constrains candidate explanations; scientific objective composition; fitting and candidate-selection policy; the relationship between inference and evolution; question-relative diagnostics; scientific provenance and claim scope.

Small host scalar decisions or existing LAPACK diagnostics do not amount to a competing GPU framework. But new repeated numerical kernels, a private AD tape, or a parallel matrix/storage abstraction in GH would be the wrong boundary.

### Experiment code owns

Dataset acquisition and reconstruction, cohort choices, normalization recipe, candidate scientific configurations, protocol arms, hyperparameter experiments, biological outcome definitions, grounding analyses, figures, and paper conclusions.

Registered scientific/custom model blocks remain possible. Their numerical implementations and derivative/effect capabilities must integrate through the common execution boundary, not bypass it with hidden untracked state or a second production solver.

### Runtime policy

The first useful GPU fitting session should amortize preparation and retain working values across iterations. Upload observations in bounded batches and download scalar metrics, selected diagnostics, or requested outputs, rather than every full state at every step. Do not promise entirely device-resident inference when small explicit host diagnostics remain.

Preflight metadata errors should not submit partial work. An asynchronous GPU failure cannot generally roll back already submitted arithmetic: mark the evaluation unusable and reconstruct from a valid accepted state as needed. Do not build an elaborate transaction layer merely to conceal this reality.

## 7. A scientifically neutral mathematical contract

Let `m` denote structure, `theta` global learned quantities, `eta` local initial/hidden quantities, `w` unresolved inputs, and `phi` observation/representation quantities. A problem can search within:

\[
\mathcal C_E = \{(m,\theta,\eta,w,\phi):
\text{declared constraints hold and evidence discrepancies meet declared tolerances}\}.
\]

A strategy may solve penalized objectives, constrained feasibility problems, or a probabilistic inference problem over these same scientific objects. These are distinct inferential choices.

For question `Q`, retain the answers produced by compatible candidates and the source of their disagreement. Explicit unresolved forcing can be represented by an admissible function family with bounds on magnitude, smoothness, shared structure, or innovations. An unconstrained forcing signal can explain arbitrary residuals, so the input class is a substantive scientific assumption, not an innocuous noise term.

Similarly, learning a flexible observation mapping together with dynamics introduces representation ambiguities. The first slice can hold an RNA feature transform fixed from the training data, then add learned observation maps when a specific experimental comparison needs them. It should not assume those features are the complete biological state.

A useful later method is constrained rival search: maximize disagreement on a query while requiring both candidates to retain an adequate fit. A rival is evidence of unresolved interpretation. Solver failure to find a rival is not a uniqueness theorem.

## 8. The smallest useful vertical slice

### 8.1 First engineering consumer: a learnable controlled family

Use a small truth-known, partially observed family to qualify the full fitting path. Unlike NF1A's supplied-candidate examples, its global coefficients and local initial quantities start unknown and must be inferred from observations. Include missing channels, two observation times, shared parameters across instances, one bounded unresolved input, and an explicit non-identifiable parameter combination.

The output should contain learned candidates, exact evidence identities, held-out predictions, and an ambiguity witness. Add a second observation that separates the deliberately indistinguishable alternatives. Test that the learner does not claim identification before that evidence is supplied.

The objective is not to publish a synthetic biological result. It is to prove that the library can perform an actual observation-to-fitted-hypothesis cycle, with verified gradients and persistence. The families supplied define the hypothesis class, not the fitted answer.

### 8.2 First biological consumer: bounded observable-horizon learning

After the data audit, start with train-only RNA features and clone-time observation bags. For a small initial implementation, use declared summaries or bounded representative sets. These describe the sampled clone-conditioned population; they are not measurements of one exact ancestral cell.

Fit a shared finite-horizon response model and a small competing augmented-state family. A horizon map need not implement a Markovian simulator. A latent dynamical family may choose a small controlled state-space realization, but that choice belongs to the candidate, not to the whole library.

Avoid unrestricted per-clone latent codes that memorize future outcomes. At evaluation, local unknowns must be conditioned on the authorized early evidence or retained as unresolved alternatives. A learned initializer can be added later; optimization over local unknowns is already a usable inference method.

Losses can initially compare observable features or population summaries. This is a deliberate resolution-limited experiment, not a full likelihood of the complete lineage process. Monitor how summaries obscure mixtures; preserve raw observations so the next representation can add moments, distributions, or representative states without changing the dataset identity.

Use actual clock times in the evidence, but do not demand identification of a fine continuous law from three sampling times. Different continuous laws can agree on those horizons. A direct horizon model is a valuable baseline and a legitimate first consumer.

### 8.3 Minimal deliverable

An independently built client imports evidence, defines an incompletely specified family, fits shared and local unknowns through the common numerical substrate, saves/reloads the result, conditions a held-out case without target leakage, and reports both predictions and remaining alternatives.

That is the next successful milestone. A full gene regulatory network, automatic molecular naming, general latent-state observability, or a large neural cell simulator is not required to reach it.

## 9. CellTag experiment as a separate package

A sibling package/repository such as `glasshelix-experiments` is a reasonable initial arrangement. It must link the installed GlassHelix/Cellerator interfaces; a developmental source override may point to a built package but should not compile private CE source into the experiment.

Illustrative organization:

```
glasshelix-experiments/
  celltag_reprogramming/
    data_manifest/       # accessions, hashes, joins, quality/coverage audit
    adapters/            # released data -> EvidenceView
    protocols/           # permitted evidence and held-out readouts
    model_recipes/       # provisional scientific model families
    analyses/            # fitting, conditioning, alternatives, comparisons
    reports/             # figures and scientific conclusions
    tests/               # joins, partitions, leakage and protocol tests
```

Nothing in the core library should mention a GEO accession, Hnf4a/Foxa1, day 3/12/21, iEP/dead-end labels, or a particular published clone table. Group ancestry, multimodal observation relations, and chronological restrictions can be generic concepts. Exact dataset definitions are experiment configuration.

Reusable methods should migrate from experiment code when another real consumer needs them. Generic capabilities can be designed now without prematurely extracting every biological analysis into a permanent framework.

### 9.1 Data readiness is real work

GEO superseries `GSE216521` links iEP RNA `GSE216518` and ATAC `GSE217119`; its H5 supplement is count data, with raw sequencing separately in SRA. The authors' repository provides analysis code and clone tables but says ready-made processed iEP objects are not on Figshare because of size. Reconstructing them from released inputs, or obtaining the authors' objects, is an explicit prerequisite, not a one-line guaranteed loader. [W1–W2]

Audit cell IDs against sample/replicate prefixes, gene/peak axes, clone calls, assay, time, missing channels, quality exclusions, and clone-by-time-by-modality coverage. Main-text/GEO time labels use day 12 while the Methods contains a day-11 reference; retain original labels and resolve the discrepancy from metadata rather than silently choosing a clock. [W1, W3]

The study's 1,422 state–fate clones include day 3 and at least one later time, not necessarily all three. RNA and ATAC siblings are usually different cells. There is also a day-21 paired Multiome comparison, not a paired longitudinal multiome series. [W3]

Do not use the released joint RNA–ATAC embedding as RNA-only input. Its integration and some biological annotation use accessibility information. Build fit features from the authorized training data and keep any grounding readout separate. [W3]

### 9.2 Protocols should be explicit alternatives

**Blind population-law protocol.** Fit from time-indexed RNA populations without lineage or ATAC features. Evaluate whether inferred restrictions agree with withheld relationships. This tests learned population dynamics, not individual hidden-state recovery from nonexistent histories.

**Retrospective reconstruction protocol.** Use early and later RNA siblings of a held-out clone to constrain an early hidden distinction, then compare with withheld early ATAC siblings. This is smoothing/reconstruction: the later evidence is intentionally allowed and the claim is not prospective prediction.

**Prospective family-history protocol.** Freeze the globally learned law; condition a held-out clone only on observations up to a declared cutoff; predict a later readout. Whether enough clones support a day-3/day-12-to-day-21 protocol is a data-audit question.

These protocols can share the same library interfaces but must not share ambiguous descriptions of what information the model received. They need not all be implemented first. Begin with the arm that has adequate actual coverage and the clearest claim.

### 9.3 Controls and meaningful outcomes

A static predictor with the same early features is required. A flexible non-dynamical predictor receiving the same complete authorized evidence is also required when the dynamical model sees multiple observations. Include same-time sibling pooling, random/unrelated sibling controls, and time/order ablations where their scientific meaning is clear. Equal input access matters more than a fashionable baseline name.

Keep clone-related cells out of both sides of an evaluation split. Include cross-replicate transfer where support permits; two biological replicates do not license a broad population-generalization claim.

Evaluate observable prediction, reconstruction of withheld RNA, and chromatin grounding separately. A calibrated auxiliary map from learned representation to ATAC must be fitted on separate calibration/training groups and tested on held-out groups. ATAC association grounds a molecular correlate; it is not direct evidence of a causal chromatin mechanism.

Do not replace mixed clones with modal-fate labels as the only target. Likewise, sampled descendant fractions combine state change, proliferation, survival, and sampling. They are useful observed population data, but not automatically probabilities of a single fate decision.

The principal outcome should be useful, validated narrowing of answer uncertainty under explicitly stated evidence and forcing assumptions. A model that predicts every possible outcome is not useful. A model that excludes the realized outcome without justified qualification is overclaiming. Apparent disagreement may still reflect search incompleteness or model inadequacy and must be labeled accordingly.

## 10. Necessary tests for the new slice

### Numeric and derivative tests

Check full-objective finite differences and adjoint identities with respect to initial state, global parameters, forcing coefficients, and observation parameters. Include parameters used at several times and in several instances, repeated n-ary arguments, zero-valued coefficients with nonzero derivatives, and missing-channel masks. A zero forward contribution must not erase derivative support.

Retained primal tests should run a true forward evaluation and then perturb a dependency/generation. Verify rejection or use of an explicitly retained historical snapshot. Include the actual composed model rather than only its primitive callbacks. Compare supported CPU/CUDA routes and distinguish tolerances/numerical policy from scientific uncertainty.

### Inference tests

Known identifiable and intentionally non-identifiable systems; all candidates poor despite a normalized score; bounded-forcing sensitivity; evidence partition/order consistency where mathematically required; inability to recover an omitted alternative falsely reported as uniqueness; solver failure distinct from no feasible explanation; candidate whole-branch preservation.

### Evidence and consumer tests

Reject a destructive population pair passed as a same-individual transition. Test clone/replicate split isolation, target access restrictions, preprocessing fit partitions, feature identity/order, measured zero versus missing, and out-of-protocol local latent fitting. The external consumer must build against the installed package and exercise a complete fit, not reimplement the runtime.

### Lifecycle and performance tests

Warm structural reuse across parameter updates, proper invalidation after structural change, multi-use parameter accumulation, bounded intermediate retention, explicit upload/download accounting, model save/reload/continuation, and shape-safe optimizer-state mapping. Measure complete fit iteration cost for the real consumer, not an unrelated scalar-kernel speedup.

## 11. Persistence and provenance

The narrow NF1 replay probe should remain a qualification artifact. Add a distinct fitted-hypothesis record: family and structure revisions, logical fitted values, observation/representation recipes, data and partition digests, permitted evidence, constraints/prior/forcing class, strategy settings and state, random seeds where used, CE/GH build identities, numerical policy, and diagnostic scope.

Do not serialize device addresses, streams, or borrowed primal pointers. Reconstruct preparation when loading. Resume requires compatible semantics; a newer source revision should require revalidation/migration rather than silently treating an old checkpoint as identical. Candidate provenance is separate from a mutable live execution handle.

No new file format is mandatory. Use existing serialization/storage facilities for values, with a small explicit metadata schema. The important feature is a reproducible scientific record, not a universal object serializer.

## 12. Extension paths without implementing them all now

**Learned representations:** introduce an encoder/observation family as a fitted operator using the same unknown/derivative contracts. Separate global representation learning from local hidden-state inference and record coordinate meaning.

**Memory models:** add a history-consuming capability and an execution provider when a real model requires it. A history-conditioned predictor must not be forced through a fictitious Markov state interface.

**Structural/mechanism learning:** propose changes to a declared library/grammar of candidate relations, latent dimensions, delays, modes, or observation maps. Each proposal obtains a fresh structural identity and explicit initialization/mapping of values. Compare through common questions, not raw graph shape alone.

**Uncertainty:** add set-valued, weighted, profile, or other search strategies. Preserve the guarantee type and distinguish computed candidate ranges from certified conservative sets.

**Multimodal and intervention evidence:** reuse typed evidence relationships; introduce the specific observation or intervention semantics needed by the dataset. A common treatment label does not assert identical realized intracellular forcing.

**Sequence grounding:** Baseplane-derived sequence constraints can inform candidate possibilities or test proposed mechanisms. A relation supplied as a prior must remain distinguished from a discovered relation. Nothing in the first learning interface should require genome input.

**Scale:** CellShard-backed evidence/value delivery and multi-device execution can be added through the existing storage and execution boundaries. No multi-GPU framework is a prerequisite for the first fitted model.

## 13. Suggested next development sequence

1. Repair the two reproduced correctness defects and qualify forward-to-response retention on one nontrivial composed family. Keep the accepted NF1A foundation intact.
2. Deliver a native evidence-to-objective fitting path with selected shared/local unknowns, genuine parameter accumulation, one small strategy, and logical save/reload. Use the truth-known partial-observation consumer to test it.
3. In parallel with that narrow library work, reconstruct/audit the external CellTag inputs and evidence partitions. This determines which scientific protocol has enough support.
4. Run the first bounded real-data consumer, compare it with equal-information baselines, and retain competing explanations. Promote only the reusable capabilities exposed by that work.

These are outcomes, not a proposed forest of coordination tasks. Do not require a universal latent-state framework, complete structure discovery, Bayesian inference, sequence model, generic JIT, new AD engine, multi-GPU runtime, or zero-copy framework interchange before this loop exists.

**The next deliverable is a small library that learns executable candidate explanations from evidence, demonstrated by an independent experiment client—not another supplied-system demonstration and not a one-off CellTag model embedded in the core.**

## Appendix A. Source ledger

Paths and ranges below were inspected through project-control on the stated review date. They identify review evidence, not a guarantee that every path in either repository was audited.

### GlassHelix

- **G1:** `README.md` 1–109; `include/GlassHelix/core/quantity.hh` 1–33.
- **G2:** `include/GlassHelix/core/system.hh` 1–22.
- **G3:** `include/GlassHelix/core/request.hh` 1–22; `core/observation.hh` 1–10; `core/provenance.hh` 1–3 (all under `include/GlassHelix/`).
- **G4:** `include/GlassHelix/system/assembly.hh` 1–38; `src/system/assembly.cc` 1–41.
- **G5:** `include/GlassHelix/dynamics/evolution.hh` 125–382 in the current read, plus earlier source context for scheduling/retention declarations.
- **G6:** `include/GlassHelix/dynamics/response_plan.hh` 145–320 in the current read, with preceding owner/retention declarations available in the supplied previous inspection.
- **G7:** `include/GlassHelix/observation/observation.hh` 1–161; `docs/nf1_adaptive/observation/README.md` 1–18.
- **G8:** `include/GlassHelix/inference/inference.hh` 1–134; `tests/native_foundation/inference/inference.cc` 1–101; `tests/native_foundation/inference/inference_cuda.cc` 1–20.
- **G9:** `include/GlassHelix/interrogation/diagnostics.hh` 1–15; `include/GlassHelix/interrogation/alternatives.hh` 1–54; `docs/nf1_adaptive/analysis/ANALYZE.md` 1–9.
- **G10:** `include/GlassHelix/reduction/contracts.hh` 1–27.
- **G11:** `bindings/python/nf1_module.cc` 1–18; `include/GlassHelix/artifacts/nf1_replay.hh` 1–36; `docs/nf1_adaptive/access/NF1_ACCESS.md` 1–26; `CMakeLists.txt` 1–158.
- **G12:** `tests/native_foundation/interrogation/diagnostics.cc` 1–41; `tests/native_foundation/interrogation/differentiate.cu` 1–521 read in overlapping bounded ranges; `tests/native_foundation/interrogation/CMakeLists.txt` 1–11.
- **G13:** `docs/nf1_adaptive/qualification/INTEGRATED_TOOLBOX.md` 1–46; `docs/nf1_adaptive/performance/GH_ACCEPT_COMPLETE_PROGRAM.md` 1–90; authoritative `GH-NF1A-ACCEPT` task inspection.
- **G14 (intent, not delivery evidence):** `planning/nf1-adaptive-v1/ARCHITECTURAL_INTENT.md` 1–44; `outcomes/ANALYZE.md` 1–33 and `outcomes/ACCEPT.md` 1–21 under that plan.

### Cellerator

- **C1:** `cmake/NativeFoundation.cmake` 1–169; `include/Cellerator/compute/operation/native_numeric/local_arithmetic.hh` 1–31; `native_numeric/device_linear.hh` 1–17.
- **C2:** `include/Cellerator/compute/operation/differential/local_arithmetic.hh` 1–83; `src/compute/operation/differential/local_arithmetic.cc` 1–127; its `CMakeLists.txt` 1–11.
- **C3:** `include/Cellerator/compute/operation/indexed_mechanism/evaluators.hh` 1–89; `indexed_mechanism/incidence.hh` 1–48.
- **C4:** `include/Cellerator/compute/operation/prepared_relation.hh` 1–113; `relation_update.hh` 1–92; `include/Cellerator/parameters.hh` 1–74.
- **C5:** `examples/models/developmental_time_model.cu` 480–540, illustrating model-specific update code rather than a general fitting API.

### Public primary sources

- **W1:** GEO GSE216521 and its iEP subseries, inspected 26 September 2026. `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE216521`
- **W2:** Authors' CellTag-multi analysis repository README, inspected 26 September 2026. `https://github.com/morris-lab/CellTag-multi-2023`
- **W3:** Jindal and colleagues, *Single-cell lineage capture across genomic modalities with CellTag-multi reveals fate-specific gene regulatory changes*, Nature Biotechnology, DOI `10.1038/s41587-023-01931-4`. Used for experimental sampling/assay semantics and leakage considerations, not as a preselected mechanistic answer.
- **W4:** LAPACK DGESVD documentation: right singular vectors are rows of `VT`. `https://netlib.org/lapack/explore-html/d1/d7f/group__gesvd_gac6bd5d4e645049e49bb70691180abf07.html`

## Appendix B. Included reproducibility files

`review_probes.py` calls system LAPACK and reproduces the two source-level numerical failure modes. `probe_results.json` records the actual standalone results in this environment. These files neither import GlassHelix nor modify its repositories. They are review evidence and seeds for upstream regression tests, not the proposed learning toolkit.
