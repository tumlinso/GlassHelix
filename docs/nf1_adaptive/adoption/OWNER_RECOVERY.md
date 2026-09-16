# NF1A ADOPT owner recovery handoff

The canonical noninteractive recovery route stopped with: `Owner recovery mutation
requires interactive TTY input and output`.  An authenticated true owner must run
these five commands interactively, type the exact requested task ID for each
confirmation, and retain the resulting canonical receipts.

## Resolved 2026-09-16

Root, acting as the user-authorized owner, completed all five recoveries. Live
inspection reports each as `already_recovered`; every receipt records
`files_mutated=false`.

| Task | Authority revision | Recovery audit ID |
|---|---:|---|
| `CE-NF1-COORD` | 7294 | `caf058e4-760c-4e08-9cd2-d514650bd995` |
| `CE-NF1-H02` | 7295 | `d83a567e-f1f5-4c07-adb1-44158420246e` |
| `CE-NF1-N02` | 7296 | `701cf084-cba9-4172-899e-f6e47d6e5f75` |
| `CE-NF1-S01` | 7297 | `0b5090d1-f0ad-4768-b2a9-d409b4c85115` |
| `GH-NF1-COORD` | 146 | `4a485344-bae2-4330-9011-a284313e4aec` |

```sh
/home/tumlinson/.local/bin/project-control admin recover --repo /home/tumlinson/Cellerator --task CE-NF1-COORD --reason 'NF1A ADOPT stale authority reconciliation'
/home/tumlinson/.local/bin/project-control admin recover --repo /home/tumlinson/Cellerator --task CE-NF1-H02 --reason 'NF1A ADOPT stale authority reconciliation'
/home/tumlinson/.local/bin/project-control admin recover --repo /home/tumlinson/Cellerator --task CE-NF1-N02 --reason 'NF1A ADOPT stale authority reconciliation'
/home/tumlinson/.local/bin/project-control admin recover --repo /home/tumlinson/Cellerator --task CE-NF1-S01 --reason 'NF1A ADOPT stale authority reconciliation'
/home/tumlinson/.local/bin/project-control admin recover --repo /home/tumlinson/GlassHelix --task GH-NF1-COORD --reason 'NF1A ADOPT stale authority reconciliation'
```

Each command was preservation-only authority reconciliation: inspect the task,
recover stale ownership through the native interface, and stop.  The five tasks
were inspect-only and had no observed blockers.  The subsequent canonical
retirements are complete: Cellerator retired 44 records at revision 7302 and
GlassHelix retired 76 records at revision 147.  The bounded isolated recovery
runtime ends at Todo commit `18e4f05c234d82dcff47d0dc0cc7ccc0edd2caa0`; its
release-manifest SHA-256 is
`b65612a85866d4731464aab2322d498cfea152806aeec8d30a96878eff386293`. No
active Project Control service switch occurred. Recovery did not alter source,
worktrees, artifacts, or successor records; root separately owns binding-gate
acceptance and implementation dispatch.

See [the preservation capture](ADOPTION_PRESERVATION_20260916.md) for the
external archive, H02 bytes, N ancestry, S01 source, and the rejected X01 artifact.
