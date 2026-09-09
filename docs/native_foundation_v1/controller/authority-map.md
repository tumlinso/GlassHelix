# Paired NF1 authority and execution map

Observed 2026-09-09. The accompanying authority-observation.json preserves
Project Control run, dispatch and component-authority fields at GlassHelix
revision 30 and Cellerator revision 7006. These are historical observations;
resume/sync and inspect live authority before every future assignment.

| Project | Canonical repository | Project UUID | Root / run |
|---|---|---|---|
| GlassHelix | /home/tumlinson/GlassHelix | b2c4b239-f18d-4252-b388-fb9889e75efd | GH-NF1-0000 / GH-NF1-RUN-V1 |
| Cellerator | /home/tumlinson/Cellerator | 0ccaac37-dbbf-448e-a5f8-def197a70aba | CE-NF1-0000 / CE-NF1-RUN-V1 |

The launched strategic controller occupies both project-qualified coordinator
seats, GH-NF1-L-COORD and CE-NF1-L-COORD. Each has its own native coordinator
task followed by the unfinished epic; planning/nf1-coordinator-correction-v1
records the reviewed correction. Dispatches are
31f2a61d-f1a5-41b7-9b88-56bfe8c761d0 and
825fe8a4-cbc4-4497-b072-b6132d39233f respectively. No delegate is needed and no
cross-project native parent exists. Their observation reported stale heartbeat,
which was reported to the controller for sync; it is not evidence of termination.

First-class A lanes are locally parented to their own coordinator, use native
claims/dispatches and managed isolated_merge worktrees. GH A workspace
87da49e8-1809-4941-afe2-692aee335a95 uses codex/gh-nf1-a-v1 and integration task
GH-NF1-M00. Project Control next_task acquires/resumes an exact task; inspect_task
reads its context and workspace. finish_task(action=complete) runs required gates.
An implementer cannot use coordinator-only run_gates as a workaround. Missing
workspace provisioning is handled by the controller through the installed native
WorkspaceService, with authoritative repository identity, exact shared base,
allowed new path, lane and explicit integration task. No database editing is
part of normal lane execution.

The installed service is systemd user project-control.service, whose observed
ExecStart is /home/tumlinson/.local/bin/project-control serve. That script invokes
/home/tumlinson/project-control/.venv/bin/project-control-release; its release
wrapper selects /home/tumlinson/project-control/.venv/runtime-skills and the
configured Python is /home/tumlinson/project-control/.venv/bin/python. These
paths were read from the running unit and installed wrappers, not guessed flags.
Use exposed Project Control tools for ordinary workflow mutation; collaboration
agents alone do not create native claims or workspaces. Never use the nested
extern/Cellerator gitlink as current producer authority.

Both repositories must serialize GPU evidence using an actual native lease and
one shared host lock: /tmp/nf1-20260908-v1-device-evidence.lock. This records the
controller-selected path, not a held lock or lease. CE A02 owns the installed
lease verification details; GPU qualification remains unavailable until its
live receipt and external execution bindings are integrated. Installed acquisition
is cuda_controller.py run --spec FILE --json; its child receives
TODO_GPU_LEASE_RECEIPT. CE A02 provides verify_gpu_lease.py with
--lease-receipt PATH --project-root ROOT --gpu-uuid UUID; the verifier checks
HostCoordinator(create=False).owner against active foreground state, project,
owner identity, live PID/start and reserved GPU membership. Its source was
inspected in the CE A workspace; pin its integrated commit before use. Seven
adversarial unit tests reported by that lane are synthetic verifier evidence,
not a held lease or device calculation. Governance review
needs no device lease. Select ready CPU work under the shared CPU/RAM budget;
30 declared lanes do not authorize simultaneous launch of all lanes.

Cross-project consumers verify integrated commits, dependency hashes, interface
versions, real test receipts and fresh producer completion/checkpoint evidence.
Developmental contracts release design only. The order remains CE M50, GH M60,
CE M90, GH M90. Unknown work and historical worktrees stay preserved; source
integration uses ordinary non-destructive Git plus native artifact/handoff state.
