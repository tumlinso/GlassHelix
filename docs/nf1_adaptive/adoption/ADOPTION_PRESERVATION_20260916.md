# NF1A adoption preservation evidence

Captured 2026-09-16 with the installed Project Control runtime
`/home/tumlinson/.local/state/project-control/venvs/investigator-f894427-688dcc2/bin/python`.
This is byte preservation and disposition evidence only.  It makes no integration,
claim-recovery, retirement, source, or test-acceptance assertion.

The owner-only recovery handoff is recorded in
[adoption/OWNER_RECOVERY.md](adoption/OWNER_RECOVERY.md).  Recovery is outside
this preservation report and was not attempted here.

## Import review recheck

The sealed reviews and applied receipts retain their recorded hashes:

| Project | Review SHA-256 | Applied-receipt SHA-256 | Current recheck |
|---|---|---|---|
| GlassHelix | `59bfc894c111116087e24cf9e4524976ea9cbf79f1ffa4f33a5acb84de48b04c` | `8b649ee8c871af5193c2b91331b25a51cbc81f8ca08ce018a7c9a6c047bbd386` | stopped fail-closed: `Existing task changed during import: GH-NF1-COORD` |
| Cellerator | `5ab40f674063112b8b3669302ff5de43258a3f1e0741940f096e768ac755a320` | `5d4d928fc09ab69dc23314c3c198134b9c5e1d882e5e6f2dba7fa243a234d834` | stopped fail-closed: `Existing task changed during import: CE-NF1-COORD` |

The recheck is not an import failure: it proves the reviewed preconditions are no
longer current after the coordinators changed.  Do not reuse either review for an
apply operation.  Current authority revisions captured below are Cellerator 7291
and GlassHelix 143.

## Current coordinator delta and successor state

The sealed reviews retain only canonical hashes for non-successful historical rows,
so they cannot reconstruct an exact field-by-field before image.  Their coordinator
hashes differ from the current rows: CE
`f3b1b8600959810aff3ae8a117be1286a6f7d980556df64aa4c6cbd20b585262` to
`8a61f32aba952987aea6bff7a5e13bbe5ccc77b897f8561f461886e70dc97b3e`; GH
`f038e70d435d723734bcb182ca2bb51cca90d7dc2d053f21556e0be682589995` to
`09f6bc4a94ca488ec9c939a13ca455319460d177af29cf1e384e1bd9b5f5d535`.
The available exact current fields are the same in both: `status=planned`,
`result=null`, `version=4`, `updated_at=2026-09-16T16:57:56Z`, with their original
NF1 parent and objective retained.  Field-level before values are not available in
the sealed review and are therefore not inferred here.

The current successor identities are `CE-NF1A-ADOPT` in active
`CE-NF1A-L-ROOT` and `GH-NF1A-ADOPT` in active `GH-NF1A-L-ROOT`, both position 0;
both are `in_progress`, version 2, with no native gate-binding rows yet.  All
historically successful rows remain byte-identical to their review baselines: 37 CE
and 15 GH mismatches out of 52 are zero.

Post-retirement recheck: GH revision 147 has old run `GH-NF1-RUN-V1=cancelled` and
all 15 preserved-successful rows unchanged. CE revision 7297 has all 37
preserved-successful rows unchanged but its old run remains `active`; no CE
retirement receipt is represented by this evidence.

Canonical retirement is now complete for both authorities: CE retired 44 records at
revision 7302, receipt `ce-retirement-receipt-reconciled.json` SHA-256
`c62d6ecbe8eef02347b26dc18e89db46046fefa606da649e236fa24ea71c942b`; GH
retired 76 at revision 147, receipt `gh-retirement-receipt-post-recovery.json`
SHA-256 `9aa996b6f9f39cb5816fb46785c46a8029764b9ebdddb64fb787256d9b3d7c12`.
Both external ADOPT bindings now reference these receipts. This records
administrative readiness only; root owns the gate invocation and acceptance.

## External immutable capture

- Receipt: `/home/tumlinson/.local/state/project-control/nf1a-bootstrap-20260916/preservation-adopt-20260916T1659Z/preservation.json`
- Receipt SHA-256: `8cb421345e1267e28b2a9992bf783d1ceb2ae90d645fd8200ed39087058e44ca`
- Archive: `/home/tumlinson/.local/state/project-control/nf1a-bootstrap-20260916/preservation-adopt-20260916T1659Z/dirty-source.tar`
- Archive SHA-256: `c26cccb514f426e8927f9508a93282bf73683b4fefac113bac502726156626ea`

The helper enumerated and captured every legacy NF1 workspace it found: eleven CE
and seven GH, eighteen total.  Their clean/dirty status hashes were rechecked by
the helper; no worktree moved or changed during capture.  No legacy NF1 worktree
is missing from the helper's `captured_nf1_worktrees` list.

## Source disposition

Every listed workspace remains in place.  “Successor” is the explicit owner from
the installed legacy disposition; it is a destination for reconciliation, not a
statement that the listed commit is integrated or qualified.

| Workspace | Source identity | Disposition and successor |
|---|---|---|
| `ce-nf1-a-v1` | `beef8a1bc0f5c1a0951a720feeeb2e16545efa4c` | Historical integrated artifact; `CE-NF1A-CORE`. |
| `ce-nf1-b-v1` | `8375dea1e72ebd77bda591fe497b1ee9fd01b9aa` | Queued commit artifact; `CE-NF1A-CORE`. |
| `ce-nf1-c-v1` | `7e586f653d1384aef0ab06d45f261579ff15570a` | Historical integrated artifact; `CE-NF1A-CORE`. |
| `ce-nf1-d-v1` | `5a6ff6e3cefdfd6f4949e3e76f49906f9aa40760` | Queued commit artifact; `CE-NF1A-RESPONSE`. |
| `ce-nf1-h-v1` | head `4cb93a590419c9857e85b769d86ae71d43ab4b36`; dirty bytes below | Preserved active H02 source; `CE-NF1A-MECHANISMS`. |
| `ce-nf1-m-v1` | `9280efd91f47851846ace3edda705a21bb4e4e43` | Active workspace without declared artifact; `CE-NF1A-CORE`. |
| `ce-nf1-n-v1` | `3a4813ef36eae359fccf501ddd7060afe7681b5c` | Active N source; `CE-NF1A-CORE`.  It is a descendant of V07 `d7b970d4d8898bcaf15a288b1249b78713a29cfc` (merge parent) and of its workspace base `9280efd91f47851846ace3edda705a21bb4e4e43`; prerequisite ancestry is present. |
| `ce-nf1-p-v1` | `18bff5f0b33e822b97435de8eab3082ecfd3f64b` | Active workspace without declared artifact; `CE-NF1A-CORE`. |
| `ce-nf1-s-v1` | `9280efd91f47851846ace3edda705a21bb4e4e43` | Active workspace without declared artifact; `CE-NF1A-SUPPORT`. |
| `ce-nf1-t-v1` | `cda15cc8f56f41cbbbea9568336aad5b62635a76` | Queued commit artifact; `CE-NF1A-QUALIFY`. |
| `ce-nf1-v-v1` | `d7b970d4d8898bcaf15a288b1249b78713a29cfc` | Queued V07 commit artifact; `CE-NF1A-CORE`. |
| `gh-nf1-a-v1` | `d7b85a797f42925906ae6b12bac8a756d885239a` | Historical integrated artifact; `GH-NF1A-SYSTEM`. |
| `gh-nf1-b-v1` | `b4bc9590b22601477e1c7105512f41e330504fec` | Active workspace without declared artifact; `GH-NF1A-SYSTEM`. |
| `gh-nf1-c-v1` | `37d06cfe56f980046560450fd5f3298fc2242fff` | Queued commit artifact; `GH-NF1A-SYSTEM`. |
| `gh-nf1-m-v1` | `3f05cf2f35ffeebbd903f2b53526c3a2c1028f88` | Active workspace without declared artifact; `GH-NF1A-SYSTEM`. |
| `gh-nf1-s-v1` | `9d6ac741a175bb284b8f0b28a9b4788f175ceb36` | Preserved partial S01 committed host assembly; `GH-NF1A-SYSTEM`.  Its parent is `3f05cf2f35ffeebbd903f2b53526c3a2c1028f88`. |
| `gh-nf1-t-v1` | `93988ed2fc7e179fa47fe71736935718e0d90538` | Queued commit artifact; `GH-NF1A-ACCEPT`. |
| `gh-nf1-x-v1` | `e84091bcbedbff04c6f7207d16c205d35729cfd4` | Preserve completed GH X01 evidence and reconcile its rejected M10 artifact under `GH-NF1A-SYSTEM`; do not repeat X01. |

### H02 byte inventory

All bytes are retained in the archive under `cellerator/14/files/{0,1,2}` with an
unstaged binary patch (`cellerator/14/unstaged.patch`, 700 bytes); the staged patch
is empty.

| Path | SHA-256 |
|---|---|
| `include/Cellerator/compute/operation/indexed_mechanism/evaluators.hh` | `17fc410434eeba723b152109ab898dbf872b97ebd95cea686d79de5cee34fd02` |
| `src/compute/operation/indexed_mechanism/CMakeLists.txt` | `0b87ffd1a0a23c7f89af94bcaa214dcdf38e886645c388ff09cc0a5ecf46c537` |
| `src/compute/operation/indexed_mechanism/evaluators.cc` | `91cce04f7cbefff260636e3dec87f99109d41c5e5269deeea9c25ef63f58b38b` |

### GH X01 rejected artifact

The captured authoritative patch record is `7d29515b-4c39-4cc5-842e-665897df345b`:
workspace `1bb3307b-aee1-43c1-93d6-1ddc452bc32c`, artifact
`e84091bcbedbff04c6f7207d16c205d35729cfd4`, base
`3f05cf2f35ffeebbd903f2b53526c3a2c1028f88`, state `rejected`, content hash
`c1455d64918b66b3d5b385a1c14706eed8c59e1009cbfbc7924480d7c651c43f`.
It requires reconciliation; this capture found no explanation for the rejection.
