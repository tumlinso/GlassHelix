# GlassHelix L1 — fast learning bootstrap

**Five GlassHelix outcomes, one optional Cellerator outcome. Native Todo plan schema v3.**

The goal is one real observation → fitted candidate → authorized conditioning → prediction → persistence loop, plus an independently buildable CellTag consumer. Preserve NF1A. Do not build a universal inference platform first.

## Use

Extract this `learning-v1` directory under **`<GlassHelix>/planning/`**. Nothing in the archive imports a plan, edits a database or starts an agent automatically. Keep the files intact until the initial integrity check and native preview are complete.

```sh
cd <GlassHelix>
python3 planning/learning-v1/scripts/check_package.py
project-control plan validate --project glasshelix \
  --file planning/learning-v1/machine/glasshelix.todo-plan.json
```

Use the existing verified Project Control launcher. Review the current target-specific plan preview/diff. When ready to adopt, apply through the same canonical front door:

```sh
project-control plan apply --project glasshelix \
  --file planning/learning-v1/machine/glasshelix.todo-plan.json
```

Then give Codex **`START_CONTROLLER.md`**. The run focus is `GH-L1-RUN-V1`; its first task is `GH-L1-REPAIR`. Importing is not a claim or evidence that anything has been implemented.

Only if the learner demonstrates a missing numerical capability, copy this package under `<Cellerator>/planning/learning-v1`, validate the optional CE plan against **cellerator**, review its current diff and apply it there. Do not import the CE plan into GH. Details are in `CROSS_AUTHORITY.md`.

## Outcomes

| Native task | Durable result |
|---|---|
| `GH-L1-REPAIR` | Correct native nullspace and log-weight regressions. |
| `GH-L1-DATA` | Isolated client, valid evidence adapter and actual coverage audit. |
| `GH-L1-LEARN` | A reusable, native end-to-end learning and conditioning slice. |
| `GH-L1-PILOT` | One runnable, scientifically explicit experimental arm. |
| `GH-L1-ACCEPT` | Installed-consumer qualification, measurements and clear remaining scope. |
| `CE-L1-ENABLE` — optional | Only the numerical capability the selected learner actually lacks. |

Each authority also has a closure-only epic. There are no inspect/build/test/retry microtasks, pre-created speculative branches, artificial foreign dependencies, or placeholder product-test commands.

## Read selectively

`START_CONTROLLER.md` is the launch instruction. `DESIGN.md` contains the decisions and rationale that should not be rediscovered. Native task briefs carry local acceptance. `ACCEPTANCE.md` explains how to turn those requirements into executable gates. `EXPERIMENT.md` scopes the independent client. `references/REVIEW_AND_DESIGN.md` is the full prior review, available on demand rather than mandatory reading on every task.

`validation/` records package/native preview checks, not a build or biological result. `scripts/check_package.py` is an integrity/lint check, **not** a substitute for the canonical validator or implementation tests.

**Current preview status:** GH passed on its actual authority. The optional CE payload passed the canonical schema validator, but CE target preview returned `bounded_read_failed`; validate/preview it successfully on CE before any CE import. See `validation/README.md`.
