# First consumer: CellTag-multi reprogramming

**Location:** `experiments/celltag_reprogramming`, independently built/installed; core does not depend on it. **Status:** proposed pilot; no dataset fitting has occurred in this package.

The supplied review identifies GEO superseries `GSE216521`, iEP RNA `GSE216518`, iEP ATAC `GSE217119`, and the authors' `morris-lab/CellTag-multi-2023` analysis repository. These are source references to audit, not assertions that ready-made processed iEP objects are already available locally. The review says some processed objects must be obtained or reconstructed. Avoid a large raw-sequencing pipeline unless required, tractable and explicitly selected.

Build a provenance-bearing coverage table before selecting the first protocol: clone × original time × modality × replicate, with counts, missing fields, QC exclusions and verified join keys. State–fate clone counts do not establish complete three-timepoint histories. Preserve original day labels; the review flags a day-11/day-12 discrepancy. Do not silently regularize times. Different siblings normally provide RNA and ATAC; a day-21 paired multiome comparison is not a longitudinal paired multiome dataset.

Choose one arm based on coverage:

**Blind population law:** time-indexed RNA only for fitting, lineage/ATAC withheld. Claims concern population constraints, not a private history for an individual RNA snapshot.

**Retrospective reconstruction:** permitted early and later RNA siblings constrain an early distinction; withheld early ATAC is grounding evidence. This is intentionally retrospective, not a forecast.

**Prospective family history:** train the common law on other clones; freeze it; condition a held-out clone only through its cutoff; predict a later readout. Sibling grouping is authorized input, not hidden leakage.

Start with modest training-only RNA features and bags, summaries or bounded representative sets. Preserve raw identity and mixtures. Clone summaries are observations of sampled descendants, not the exact state of an ancestor. Shared protocol is nominal common forcing, not proof of equal intracellular exposure. Unresolved state/forcing may remain explicit; unrestricted nuisance inputs cannot explain every failure.

Compare a small horizon predictor and a modest alternative/control. Include a flexible non-dynamical baseline with the same allowed inputs; examine whether gains reflect more RNA, sibling pooling or temporal information. Clone-related cells must not cross train/test. Use replicate transfer where coverage permits without overclaiming generalization from two replicates. ATAC probes need separate calibration and held-out testing; correlation is not causal chromatin identification.

If usable inputs are missing, ship the loader contract, explicitly synthetic format fixture, audit template, missing-input report and real-run configuration. Keep that engineering success separate from biological validation. Do not contact authors, spend money or initiate an unbounded data reconstruction merely to remove a status flag.
