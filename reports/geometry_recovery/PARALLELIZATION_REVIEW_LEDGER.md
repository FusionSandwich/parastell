# Prompt 7R parallelization review ledger

## 2026-08-26T23:09:23-04:00 - Phase 0 start

- Serial/root-owned: authoritative specification read, evidence hierarchy, principal worktree creation, candidate acceptance criteria, final geometry selection, integration, Git commits, pushes, and all scientific completion judgments.
- Safe read-only parallel work: public-reference CAD overlap review, WISTELL-D provenance/reconstruction review, Prompt-7A visual QA, and geometry-debug procedure review.
- Candidate jobs that may later run independently: clearance-constrained radial build, WISTELL-D modern reconstruction, conformal partition, and uniform-shield sensitivity, each in a separate worktree and create-only artifact root.
- Scheduler jobs: none launched. Local capability is sufficient for Phase 0 and source-CAD diagnosis. Remote execution is not currently needed.
- Delegated reviewers: R1 public-reference overlap, R2 WISTELL-D, R7 visual QA, followed by R4 DAGMC/OpenMC qualification. Reviewers have no write authority.
- Shared-state risks: Git index collision, artifact overwrite, DPA checkout contention, and false promotion of old negative geometry. Mitigation: only root writes the principal worktree; every candidate gets a unique output root; Prompt-7A and failed-feature artifacts are read-only; DPA writes remain locked until geometry acceptance.

## 2026-08-26T23:09:23-04:00 - Phase 0 evidence close

- Serial: freeze Phase-0 reports and choose which source-CAD candidates merit construction.
- Safe to parallelize: exact R1 Boolean intersection quantification, historical candidate re-audit, WISTELL-D metadata recovery, and geometry-debug command review.
- Independent scheduler work: none. Several unrelated WSL/DPA tasks are active and preserved.
- Risk update: project-control storage is under transient I/O contention. Exact-file reads and hash checks passed; broad enumeration was stopped. The tracker CSV and JSON agree for all 188 tasks.
- Visual QA update: all 70 Prompt-7A PNGs and two VTKs are hash-valid, but Prompt 7R visual evidence is not ready because WISTELL-D images are placeholders and the VTKs are point-only.

## 2026-08-27T01:43:00-04:00 - Phase 3 hourly review

- Serial/root-owned: the A1 physical interpretation, acceptance-criteria changes, source-CAD gate, candidate promotion, and final geometry selection.
- Safe independent reviews: adversarial A1 clearance review and Prompt 7R/project-control dependency review. Two projectless, read-only xhigh Codex tasks were created: `01a041be-7b66-76c1-b3cd-f02fcf674997` and `01a041bf-b263-7e63-9f74-b30255d898d9`. They received immutable paths and no edit/job authority.
- Safe compute lane: one exact local source-CAD audit. A second local CAD/DAGMC writer was initially withheld because the Docker audit used constrained RAM and a shared engine.
- Reviewer outcome: both reviews held the geometry gate closed. Their load-bearing findings were missing global continuous-clearance witnesses, unquantified breeder-volume/spatial changes, brittle raw STEP/source-mesh identity checks, unenforced acceptance criteria, and absent downstream DAGMC/OpenMC evidence.
- Root response: raw identity was replaced by timestamp-normalized STEP identity and semantic source-mesh identity; raw hashes remain provenance. Exact global vessel/magnet witnesses and checkpointed progress were added to the source-CAD audit. The full refined build remains locked pending the repaired source-CAD packet.
- Shared-state risks: reviewers inspected a live snapshot while root code and artifacts evolved. Their reports are therefore falsification inputs, not final acceptance evidence.

## 2026-08-27T01:55:00-04:00 - Authorized Bateman geometry lane

- The user separately authorized Polytechnique CPU-only work under 25% of physical cores and permitted multiple hosts. The user-wide routing rule selected `poly-bateman` for geometry-only work; no other Poly host was used.
- Live topology: 256 physical cores, one thread per core; strict cap 64. Initial ledger: 0 reserved, 0 observed-unreserved, 64 available. Fixed geometry environment imports passed without installing anything.
- Remote worktree and outputs are unique. A 4-core import gate and 4-core construct-only A1 build were attempted. The first import command failed safely because POSIX `sh` rejected `pipefail`; the corrected `bash -lc` import passed.
- The construct-only build completed with eight expected artifacts and `CONSTRUCTION_COMPLETE`. No full/refined DAGMC export was launched because independent-review gaps remain open.
- Broker/accounting risk: after guarded starts, the persistent broker channel closed and the wrapper lease disappeared while its child briefly continued. The exact child used only four cores and no other Codex work was active, so actual use stayed below the 64-core cap. A claim was attempted immediately; the child completed before it could be claimed. This operational defect is preserved in logs and prevents further remote CPU launch until the wrapper/lease behavior is made reliable or a command form is proven to retain the tracked PID.

## 2026-08-27T02:10:00-04:00 - Vanilla-reference priority review

- Serial/root-owned: reproduce the untouched public example from the new independent main repository, then bind its hashes and physical differences into the candidate comparison.
- Active compute: one local Docker build from `D:\parastell-reference-repos\parastell-vanilla-main-de7d297`, mounted read-only, writing only to `D:\parastell-artifacts\geometry-recovery-20260827\vanilla_main_reference\20260827T021126`.
- Stopped work: the original coarse source-CAD audit was terminated after nearly one hour because it computed 108 unnecessary exact distances and produced no checkpoint or terminal report. No material result was deleted. Its replacement is tested, checkpointed, and requests exact distance/witnesses only for the 18 protected vessel/magnet pairs.
- Safe read-only work while the vanilla build runs: review evidence, implement tests, and prepare hash-bound review prompts. No second local CAD/DAGMC job will contend with this build.
- Downstream lock: Prompt-1 rebinding, tallies, boundary export, transport, and bounded activation remain locked until the geometry gate passes.

## 2026-08-27T03:15:00-04:00 - Source-CAD hardening and shared-resource handoff

- Serial/root-owned: interpretation of the dense PCHIP result, acceptance-rule hardening, selection of the refined matrix, and all promotion decisions.
- Read-only reviewer finding: the first physical-change audit omitted `Surface.calculate_loci()` and could not produce valid affected-area evidence. Root stopped that container, preserved the incomplete directory with an explicit failure marker, fixed the call sequence, and added fail-closed tests.
- Gate hardening: manifests and every listed artifact are rehashed; source-CAD selection now requires the independent vanilla reference; 18/18 distance solutions and witnesses are mandatory; Boolean errors, invalid/too-small solids, component/magnet intersections, magnet/magnet intersections, and physical magnet mismatch all fail.
- Candidate result: the first coarse A1 matrix was rejected because dense cumulative PCHIP produced a 2.31 cm local breeder increase. A deterministic refinement removed all increases without crossing the 5 cm functional minimum. The fresh full coarse build is acceptance-criteria-bound and its physical-change gate passed.
- Safe parallel work now: light code review, report preparation, tests, and independent review only. Native source-CAD Boolean qualification remains the next heavy local geometry job.
- Shared-machine coordination: Lane-B requested a preregistered one-shot 2M-history OpenMC slab run using one logical CPU and a 4 GiB hard cap. Root opened an explicit launch window after the physical audit became terminal and committed not to start the next heavy geometry job until Lane-B reports terminal status.
- Downstream lock remains active: no candidate is accepted until exact source-CAD, native DAGMC/faceting, two OpenMC navigation seeds, and root freeze all pass.

## 2026-08-27T03:45:00-04:00 - Independent gate falsification and process isolation

- Independent xhigh review task `01a04212-a092-7ef1-a31c-39affc0d8bd0` rehashed all 14 vanilla and nine candidate artifacts and reproduced the dense-map metrics, but found decision-gate false-pass paths. Thresholds could be overridden separately from the hashed criteria, distance witnesses were not required, duplicate-shape distance was not enforced, and the source audit was not bound to the physical-change receipt.
- Root response: stopped the affected audit before promotion; thresholds are now read only from the preregistered criteria JSON; all 18 global distances require finite witness pairs; duplicate-shape checks, required manifest membership, physical-receipt binding, and implementation hashes are enforced. The hardened physical-change v1.1 receipt passed.
- Performance review: a four-thread OpenCascade pool remained at approximately one core because the binding serialized operations. The checkpointed run was stopped as nonselectable. Its replacement uses bounded worker processes; each process loads an independent read-only CAD copy and nested OpenCascade threading is disabled.
- Shared-machine coordination: Lane-A received the promised terminal handoff window for a one-worker, zero-history post-audit. Prompt 7R will not launch the process-isolated CAD successor until Lane-A reports a terminal process check. Fresh Lane-A transport remains locked.
- Serial/root-owned: acceptance interpretation, worker-count choice, scientific selection, and all later DAGMC/OpenMC gates. Safe parallel work remains limited to read-only reviews, tests, and report preparation in distinct paths.

## 2026-08-27T04:10:00-04:00 - Project-control ordering review and exact-solid audit

- Serial/root-owned: the hardened source-CAD decision, native DAGMC/faceting interpretation, OpenMC navigation interpretation, accepted-geometry freeze, and every downstream integration decision.
- Active compute: one local eight-CPU/16-GiB container runs the four-process exact-solid A1 audit in a unique create-only artifact root. Lane-A and Lane-B both reported their bounded jobs terminal and released the shared heavy-local window before this run began. No other Docker container or fresh transport job is active.
- Safe read-only parallel work: native DAGMC command review and OpenMC 0.16 geometry-debug preflight. Both reviewers have no write, launch, or acceptance authority.
- Project-control audit outcome: the required sequence is source CAD, native DAGMC plus two-resolution faceting, two bounded 4,000-history OpenMC seeds, and only then root-only selection and immutable freeze. Prompt-1 rebinding, tallies, boundary export, activation, and accepted-geometry visualization must wait for selection.
- Scheduler jobs: none. Local capacity is sufficient for the current audit; the Bateman lease/PID issue remains unresolved, so no new remote CPU job is authorized despite available core-policy headroom.
- Shared-state risks: concurrent OpenCascade writers, artifact-root collision, and premature downstream rebinding. Controls remain one heavy local owner, unique no-overwrite roots, immutable vanilla/candidate inputs, and a fail-closed downstream lock.

## 2026-08-27T04:22:00-04:00 - Independent native/OpenMC launch review

- Two read-only reviewers completed without creating artifacts or launching calculations. They independently confirmed the same ordering: immutable coarse/refined hashes, native reload/topology/watertightness and p1/p2/p4 overlaps, 18 signed and directed magnet envelopes, source-domain quadrature, faceting comparison, and only then two refined OpenMC replicas.
- Launch blockers discovered before execution: the existing wrapper enabled `auto_geom_ids`, the generator used one wrong hard-coded seed, the overlap parser covered only p2, envelope closure discarded triangle winding and signed-volume orientation, and no fail-closed two-seed terminal parser existed.
- Root response: native-ID-preserving wrapper IDs, winding-sensitive signatures, directed-edge/signed-volume checks, and a fail-closed OpenMC diagnostic parser are being added with adversarial tests. No DAGMC or OpenMC result will be promoted through the older helpers.
- Downstream read-only audit completed in parallel. Geometry-neutral producer utilities are reusable selectively, but every split casing/winding contract remains negative evidence. The accepted role will be one explicit `homogenized_magnet_outer` boundary per original solid if geometry selection passes.
- Active heavy work remains only the exact source-CAD audit. All reviewers are terminal; no scheduler or transport job is active.

## 2026-08-27T04:51:39-04:00 - A1 rejection and local retargeting review

- A1 completed 276 exact source-CAD pair checks with zero intersections, exact vanilla magnet/source identity, and complete distance witnesses, but its 4.493516669986348 cm global clearance is below the preregistered 5 cm minimum. Root rejected A1 before DAGMC; no native or transport job was launched on it.
- Serial/root-owned: interpret the exact clearance shortfall, choose the next locally constrained breeder-only target, inspect its exact global witnesses, and decide whether it may advance.
- Active compute: one single-process, one-core local measurement retargets the same frozen source inputs to a 5.75 cm directional constraint. It uses about 0.3-0.6 GiB and owns the heavy-local gate. Lane-A explicitly remains static-only; Lane-B has no successor running.
- Safe parallel work completed: an adversarial review of new qualification helpers identified launch false-pass paths. Root added or staged unique-seed aggregation, full source-reference binding, transport-time immutability, sparse/duplicate/positive native-ID controls, directed/signed/degenerate envelope checks, precision-header parsing, exact material counts, and create-only receipts. Pure regression status is 52 passed with one expected local skip for the qualified PyMOAB/OpenMC environment.
- Next independent work after this measurement: build one create-only source-CAD candidate, run only its 18 exact vessel/magnet distances, and avoid a full DAGMC export unless the global clearance passes. No remote scheduler job is needed.

## 2026-08-27T05:01:15-04:00 - Retarget exact-clearance gate and independent review

- Serial/root-owned: the exact global-clearance decision, the release of the shared heavy-local window, candidate promotion, and all later geometry selection and integration.
- Active compute: one create-only, process-isolated exact BRep clearance audit for the frozen 5.75 cm directional retarget. The container is bounded to eight CPUs and 12 GiB; it writes only the 18-magnet clearance receipt to its unique artifact root. No ParaStell successor or transport calculation is running.
- Safe parallel work: static regression review, report preparation, and an independent ChatGPT Pro falsification review. The review returned `PASS_AS_CANDIDATE_ONLY`, agreed that A1 was correctly rejected, and required the exact all-18 global witnesses plus the full hardened 276-pair audit before source-CAD acceptance.
- Shared-resource handoff: Lane A remains static-only and has requested one isolated Geant4 build/zero-history window. If the exact-clearance audit passes, Prompt 7R will explicitly release that window and wait for Lane A to report a terminal process check before starting the full coarse candidate build.
- Risks and controls: the retarget was chosen adaptively after A1 failed. The complete replacement matrix, construction manifest, criteria hash, attempt history, and audit procedure were frozen before measuring exact distance; the 5.75 cm construction target is not treated as evidence of the immutable 5.0 cm global clearance gate.

## 2026-08-27T05:17:00-04:00 - Full coarse build and second heavy-lane handoff

- Completed heavy owner: the acceptance-bound A1R coarse build (5/20 cm faceting) exited zero and produced a create-only H5M plus nine hash-declared artifacts. Manifest SHA-256 is `36fed787de68ba1ed963b8d0bf212e9f35615ad14abefad39b8ef3e16a755f91`; H5M SHA-256 is `8a2d1930cc03a82269feeedef60267197581559c99aa69a021235a7ac7fafa90`.
- Serial/root-owned: validate the build receipt, run the physical/source-CAD audits, choose whether the coarse geometry may advance, construct the refined candidate, and integrate every native/faceting/OpenMC gate.
- New independent reviews: three read-only reviewers falsified the candidate-only clearance receipt, native/faceting helpers, and OpenMC two-replica runner. They launched no jobs and made no acceptance decisions. Their findings tightened closest-point reconstruction/membership, native log grammar, deterministic OpenMC IDs, actual XML/statepoint/input binding, per-cell coverage, and aggregate re-derivation.
- Shared heavy-local window: after the coarse build became terminal, Prompt 7R explicitly released a separate one-thread planning-only CRN pilot window to Lane A. Prompt 7R remains static-only until Lane A reports terminal status and a clear Windows/WSL/Docker process audit.
- Safe parallel work during the Lane-A window: pure tests, parser/model-contract hardening, faceting-gate design, and PyMOAB-native topology design. No geometry, DAGMC, OpenMC, or activation calculation is running in this task.
- Remaining risks: the full 276-pair audit has not run on A1R; PyMOAB-native category/sense/group/triangle ownership and the two-resolution faceting comparator are still implementation blockers; OpenMC remains locked behind all earlier gates.

## 2026-08-27T05:35:00-04:00 - Static gate hardening during Lane-A pilot

- Heavy-local owner remains Lane A's explicitly granted one-thread Geant4 CRN pilot. It reported a live driver and native process, a consume-once claim, and no OpenMC or automatic successor. Prompt 7R has launched no overlapping geometry or transport process.
- Serial/root-owned: all physical geometry decisions, the A1R full physical/source-CAD audits, refined-build authorization, accepted-geometry selection, and downstream rebinding.
- Three read-only reviewers completed independent native-MOAB, OpenMC, and faceting falsification. They edited nothing and launched no jobs. The OpenMC review found two demonstrated false-failure paths against an actual 0.16 log; the faceting reviews prohibited cross-resolution canonical-fingerprint equality and coarse/refined deltas masquerading as certified error bounds; the native review required raw MOAB tag/sense/incidence/triangle/material evidence before PyDAGMC.
- Root corrected the OpenMC 0.16 banner and omitted-false-attribute parsers, changed stochastic per-cell zero coverage from a geometry failure to reported metadata, bound the expected source-domain receipt and four exact HDF5 libraries, expanded the model contract to wrapper/material content, added runtime version/commit and lost-restart gates, and replaced raw stochastic diagnostic comparison with a categorical signature. Focused tests pass, and the parser now accepts the preserved real 0.16 log while explicitly reporting its 13 zero-check cells.
- Safe continuing work: pure faceting aggregation, native audit integration, adversarial tests, and Prompt-7R/project-control gap review. Next heavy work remains the coarse physical-change audit followed by the hardened 276-pair source-CAD audit, only after Lane A reports terminal process clearance.

## 2026-08-27T05:56:00-04:00 - A1R physical gate and hash-bound source-CAD launch

- Lane A's one-thread CRN pilot finished with 85,000 histories and a terminal process-clear receipt. A Prompt 7R physical audit had started approximately two minutes before that terminal handoff; the small overlap is recorded as a coordination defect. Aggregate observed use remained bounded, but no overlapping-successor pattern is acceptable.
- The A1R coarse physical-change audit then completed. Receipt SHA-256 is `7411d2564b9210cf2d0c7d99e0691a346083fd5ca202c4529c2b609d12465141`; every frozen subgate passed. Only the breeder thickness matrix changed, the continuous minimum was 11.09681428133753 cm, the first wall/magnets/source remained canonically identical, all induced outer-layer CAD changes were quantified, and all input manifests/artifacts rehashed.
- An automatically started source-CAD successor used the older command line without caller-supplied expected hashes. Root stopped it after about one minute, preserved its partial progress directory as nonselectable evidence, and added mandatory expected hashes for the criteria and physical-change receipt.
- The first corrected launch failed before output creation because `PYTHONPATH=/work` was absent; the exited container is preserved. The second corrected launch is active under container `prompt7r-target575-source-cad-v12c-20260827T095600`, with network disabled, four isolated worker processes, an eight-CPU/16-GiB cap, read-only worktree, and a unique output directory. Its criteria and physical receipt are bound to `092315...34ca` and `7411d2...5141` respectively.
- Root preregistered the two-level faceting protocol before refined results; protocol SHA-256 is `d1f4216a279ffe6608e3b334d85828bb126036833e1d17aa241880c73ec8cacf`. The protocol explicitly rejects cross-resolution raw/canonical fingerprint equality as a gate and requires a certified refined facet-to-source-CAD boundary upper bound.
- Serial/root-owned: interpretation of the 276-pair report, refined-build authorization, all native/faceting/OpenMC decisions, geometry selection, and downstream rebinding. Safe parallel work remains pure tests, read-only review, and report preparation only.

## 2026-08-27T06:14:00-04:00 - Source-CAD I/O stop and receipt hardening

- Root stopped v13 after 12 minutes because it had not emitted a single pair record and the process was in uninterruptible `p9_client_rpc` sleep while reading a 57 MB STEP through the Windows bind mount. The container exited 137, was not OOM-killed, and its empty create-only directory and container remain preserved as nonselectable evidence.
- Independent review found that a successful v13 report still needed stronger promotion controls. The v1.3 successor now binds the exact vanilla reference path and manifest hash, rehashes physical-change support artifacts, records before/after/final input hashes, requires the exact unique 108 + 15 + 153 pair identities, rejects non-finite evidence, writes atomically with readback, and emits a hash sidecar.
- Serial/root-owned: seal and stage the v1.3 implementation, launch it against container-native read-only copies, interpret its physical evidence, and authorize the refined build only after a clean terminal report.
- Safe parallel review: one read-only third pass of the v1.3 contract. Native-DAGMC and OpenMC reviewers are terminal; their remaining hardening items are being addressed locally with focused tests.
- Scheduler jobs: none. The replacement remains local and bounded to eight CPUs/16 GiB. Lane A and Lane B are static-only, and no second heavy local calculation may overlap the replacement.
- Shared-state controls: source/reference inputs and the execution script/package will be copied once into named Docker volumes, hash-verified, then mounted read-only; only the small create-only output root remains a Windows write mount.

## 2026-08-27T06:52:00-04:00 - Reduced local audit and Bateman transfer review

- Serial/root-owned: interpret the source-CAD receipt, authorize any refined build, select the accepted geometry, and integrate all later work. The local v15 audit remains the only heavy local geometry process, bounded to six CPUs, 12 GiB, and three worker processes; it is progressing through the exact ordered 276-pair contract without malformed rows.
- Safe parallel work: two read-only agents are mapping the post-gate execution sequence and reassessing Prompt-7R/project-control parallel lanes. A third independent reviewer cleared the portable receipt-path adapter only after parent traversal and resolved symlink escape were rejected; the focused aggregate regression is 88 passed with two environment-dependent skips.
- Remote lane: Bateman was rechecked live at 256 scheduler-visible cores, a 64-core policy limit, zero reserved or observed unreserved Codex cores, and 64 available. No remote CPU job has launched. A raw 726,285,824-byte transfer was retired after only an incomplete `.part` file was created; the final remote path remained absent. A new 252,519,987-byte gzip bundle is transferring to a different create-only path and will claim no CPU unless it finishes while a remote audit would still shorten the gate.
- Remote provenance: the portable source-audit mode changes only absolute storage paths. It still requires the original physical-receipt logical paths plus exact criteria, physical receipt, reference manifest, candidate manifest, every declared artifact hash/size, support-artifact hashes, implementation hashes, and before/after/final immutability equality. Current portable script SHA-256 is `ea28aa0ac9c4529542d0e352b292f61e9d65b5b943979300c5e6e55d00d4484a`.
- Shared-resource risks: wasting time on a slower transfer, accidentally accepting a partial upload, path-remap provenance drift, and local memory regression. Controls are no remote CPU claim before a terminal hash-verified upload, create-only final paths, fail-closed remap containment, preservation of retired partial files, and continued host-memory monitoring. Lane A remains static-only during the local geometry audit.

## 2026-08-27T07:09:00-04:00 - Local OOM stop, remote lease proof, and faceting-plan correction

- The v15 local source-CAD attempt failed closed when one worker was OOM-killed during component-pair 7/15. Its 114 ordered progress rows are preserved, but it has no terminal report or seal and is nonselectable. The exact 18-magnet clearance evidence remains consistent at 5.23962648491591 cm; this is not a substitute for the incomplete 276-pair gate.
- The local heavy window was explicitly released to Lane A for one bounded one-worker build and zero-history smoke. Lane A reported terminal release at 07:08:09-04:00 with no Windows/WSL scientific process and no Docker container. Prompt 7R reclaimed the window but kept it idle because a one-worker local replay would approach the host memory floor and would not beat Bateman.
- Bateman's accounting defect was retested with a one-core, three-second foreground probe. Live status before launch was 256 total threads, 64 policy limit, one observed unreserved thread, zero reserved, and 63 available. Lease `36e76caef18841fc8feae2ad0a091bf7` tracked PID 753325; the log recorded start/end, the PID was terminal, and the lease returned to zero. No scientific remote CPU job has launched yet.
- A post-gate review found that the old coarse build used builder SHA `9d8f...`, while current code is `cedd9a...`; rebuilding only the refined level would therefore confound code and faceting. Root changed the plan to remesh the same qualified STEP solids byte-for-byte at both preregistered levels. A create-only refaceting path now inherits only a terminal hash-bound source-CAD packet, copies all physical STEP/source-mesh bytes exactly, and changes only Gmsh faceting settings.
- Remaining serial blockers: finish and hash-verify the Bateman upload; launch the full source-CAD audit under a fresh core ledger; root-review its report/seal; independently review the refaceting implementation; then generate matched coarse/refined H5Ms. Static work may continue on the missing faceting snapshot and certified deviation producers.

## 2026-08-27T07:55:29-04:00 - Source-CAD gate pass and post-gate ordering review

- The Bateman source-CAD audit is terminal. It ran under tracked lease `1ad9b210a9d303fbd15c0bba10f51c98`, reserved 64 of 256 scheduler-visible threads (the exact 25% policy ceiling), and used eight isolated Python workers. The lease has reaped: zero active leases, zero reserved/observed-unreserved threads, and 64 available.
- Root independently downloaded and rehashed the three small terminal artifacts. Audit SHA-256 is `e47f34884fc04b4a353b96a4e6a18928bdb4d501e26f3b6793b213c80ae2a9aa`, seal SHA-256 is `397968a853f7c5a5d75d95c3143cebaa73b2280e8b6be30475a3a9db3d5ddd3c`, and progress-ledger SHA-256 is `6ce7cfde1e2e982fbe15787c4c13d5e4d0e155e5c8f886d5fa53c86bf89785c8`.
- Exact source-CAD coverage is 108 magnet/component + 15 component/component + 153 magnet/magnet rows. All 276 identities are unique and complete; every intersection/duplicate/distance/witness gate passes. The exact minimum vessel-to-magnet clearance is 5.239626484915899 cm. The source-CAD physical gate is therefore accepted, but no H5M or OpenMC geometry is accepted yet.
- Serial/root-owned: formal A1R source-CAD promotion, integration of the faceting-evidence producer, matched coarse/refined H5M builds from immutable accepted STEP solids, native/faceting/OpenMC interpretation, and accepted-geometry selection.
- Safe parallel work: one isolated candidate-branch agent is implementing the static faceting snapshot/deviation producer and tests. Read-only reviews may inspect completed code or evidence, but only root may select geometry or integrate a commit. No agent writes the principal worktree or the external refacet artifact roots.
- Scheduler jobs: none active after the source audit. A refacet build may use Bateman later only through a fresh hash-bound `ssh-poly` lease and within the 64-thread cap. Local heavy work is idle after Lane A's explicit terminal release.
- Prompt-7R and project-control reassessment: B-05 through B-14 remain the controlling geometry sequence; E-05/E-06 require complete original homogenized-magnet outer envelopes after selection; C-01/C-03/C-04/C-06 and D-01 are rebinding work after geometry acceptance; I-14 remains blocked until an accepted bounded global field exists. The project plan's scalar-flux versus closed-boundary-bank firewall, global homogenized-magnet/local explicit-HTS split, no-weight-window rule, and DPA_workflow ownership of activation remain unchanged.
- Shared-state risks: a faceting producer could accidentally change solids, reorder material ownership, omit internal shared topology, or publish a seal before final immutability checks. Controls are exact source packet binding, source-solid-to-Gmsh bijection, source-derived 24-volume/142-surface/five-interface contract, bounded threads, explicit HDF5 output, final native readback, and terminal seal publication only after all checks pass.

## 2026-08-27T08:08:00-04:00 - Remote refacet separation correction

- The first `refacet_v1_remote_20260827T120500Z` setup failed before extraction because POSIX `sh` rejected `set -o pipefail`. A corrected extraction later reused that root. No scientific refacet or H5M was launched, but root reuse makes v1 permanently nonselectable; it is preserved as setup/failure evidence only.
- A fresh `refacet_v2_remote_20260827T122000Z` root was created with distinct code, control, log, and future output paths. The exact 337,775-byte code archive SHA-256 is `7bb63a84750b8a89fe14b1c4d57fe9e31c66625a6803bd99cf1d7d0c157dff8b`; its 21-line preflight binding receipt SHA-256 is `776cd6b636d83a655c18707b27f4f8d06ee7f9e02255b24202fd94aa05678511`.
- Before v2 setup, the `ssh-poly` ledger showed zero active leases, zero reserved threads, zero observed-unreserved threads, and 64 of 256 available under the 25% cap. A one-core import gate is the only current remote lease; scientific coarse/refined jobs remain unlaunched until it terminates successfully.
- Root remains the sole scientific selector. The independent faceting-evidence reviewer found reducer/schema, provenance, magnet-mapping, raw-ID, workload-bound, and final-immutability gaps; the isolated candidate agent is correcting them before root integration. This static work does not touch v2 artifacts.

## 2026-08-27T08:24:00-04:00 - Source-native prerequisite failure and minimal diagnosis

- v2 coarse failed closed before any Gmsh meshing or H5M creation. The refined start did not launch: after the broker channel closed, authoritative inspection found zero lease, PID, log content, temp path, or output. v2 is permanently nonselectable.
- A separate one-core minimal source-native reproducer loaded the exact source H5M (`8a2d1930...afa90`) without mutation and returned only `required_tag_error = TypeError: MOAB ErrorCode: MB_TYPE_OUT_OF_RANGE`; report SHA-256 is `5b01232fe6f036699c7fad72322d64e975725554b829ea948b5bdf0eadee2b85`.
- Binding introspection showed `tag_get_handle(name, size, tag_type, storage_type, create_if_missing=False, ...)` and constants `MB_TAG_SPARSE=1`, `MB_TAG_DENSE=2`, `MB_TAG_STORE=128`. The adapter had incorrectly ORed the non-storage query flag 128 into `storage_type`. This is local adapter logic, not an external PyMOAB/MOAB defect and not physical-geometry evidence.
- Root removed the invalid flag while retaining `create_if_missing=False`; 20 focused native/refacet tests pass with one expected environment skip. A fixed-code source-native diagnostic must pass completely before any fresh v3 scientific root is created.

## 2026-08-27T08:37:00-04:00 - Native prerequisite pass and v3 source/import mismatch

- The fresh `source_native_diagnostic_v3_20260827T124500Z` one-core audit terminally qualified the exact source H5M: 24 volumes, 142 surfaces, 147 incidences, five shared interfaces, exact material ownership, 187,576 uniquely owned triangles, valid senses, and zero closure failures. The source H5M remained byte-identical. This cleared the native topology/material-ordering prerequisite but made no refaceted-geometry decision.
- A fresh create-only `refacet_v3_remote_20260827T125000Z` root was bound to the corrected code and qualified diagnostic. Its coarse job ran under lease `78212c33cfc7707fd4579c5f3a95e7aa`, PID `775646`, and an eight-core request. It terminated before fragmentation, meshing, or H5M writing because source CadQuery solid 0 had zero matching Gmsh-import signatures.
- v3 is permanently nonselectable and is preserved as importer/signature failure evidence. Refined v3 was not launched. No geometry, native, faceting, or OpenMC result is promoted from this root.
- Serial/root-owned: diagnose whether the mismatch is a unit convention, Gmsh OCC bounding-box tolerance, center-of-mass convention, or import mutation; define a tested identity comparison that is strict enough to prove one-to-one source ownership; and launch any corrected scientific refacet only in a fresh v4-or-later root.
- Safe parallel work: the isolated faceting-evidence branch is correcting copied PyMOAB adapter defects and adding regression tests. Lane A owns one separately authorized one-worker local build/zero-history window; Prompt 7R remains remote-only until its explicit terminal release.
- Shared-state controls: no retry or tolerance relaxation in v3, no refined successor, no H5M creation, distinct future root/log/temp/output/PID/lease identities, and a fresh authoritative zero-ledger check before any remote CPU launch.

## 2026-08-27T08:52:00-04:00 - Source/import identity diagnosis and corrected proof

- The v3 mismatch was reproduced in two bounded, one-core, network-disabled local diagnostics using the existing qualified container and immutable named-volume source inputs. No meshing or H5M writing occurred. Diagnostic v5 SHA-256 is `594cc9808210117ef53932f1edcbc4f2af2ddc5c70047a54c5a170a349fafbdf`.
- All 24 source solids map bijectively to one Gmsh OCC volume when compared by OpenCascade volume, center of mass, and the full nine-component inertia tensor. Every selected diagonal pair is exact for all three invariants; every source has exactly one match; the source files remained byte-identical.
- CadQuery 2.7 `BoundingBox` values for nested, located STEP magnet solids are not location-normalized in this call path, while Gmsh OCC bounds are. The resulting diagnostic-only bound deltas reach 238.43942121244584 cm despite exact volume, center, and inertia. This is an adapter-invariant error, not evidence that Gmsh changed the solids and not an external-package patch requirement.
- The corrected contract uses volume, center, and full inertia for source-to-import ownership. It continues to require bounding boxes, volume, center, and inertia for Gmsh-import-to-fragment preservation, where both sides use the same location-normalized Gmsh OCC representation. Final faceting evidence must still project the written H5M boundary back to the immutable source BRep.
- Serial/root-owned: seal corrected code, create a fresh v4-or-later root, obtain an authoritative zero-ledger proof, and launch coarse only. Refined remains gated on terminal coarse H5M and native-contract validation.
- Safe parallel work: static integration/review of the isolated faceting-evidence collector. The local heavy window is reclaimed after Lane A's terminal v1.4 build receipt and final clear process audit.

## 2026-08-27T09:13:00-04:00 - v4 environment failure and runtime rebinding

- The fresh v4 coarse attempt used lease `4ecaf36ff1e356d3b9ebab3e1820522e`, PID `779518`, and eight requested cores after an authoritative ledger check. It failed before import matching, fragmentation, meshing, or H5M writing because the reopened shell's default `python3` lacked `pymoab`. Refined did not launch. v4 is permanently nonselectable and preserved as environment-selection failure evidence.
- The failure is not a geometry result and does not invalidate the exact source/import identity diagnostic. It demonstrates that shell-profile interpreter selection is not a stable runtime contract.
- The already-qualified PyMOAB installation resolves under `/home/apollon/josma/venvs/openmc-0.16.0-dagmc-aarch64/bin/python`, with its package at `/home/apollon/josma/opt/openmc-0.16.0-dagmc-aarch64/lib/python3.12/site-packages/pymoab`. That interpreter alone lacks the CadQuery/Gmsh stack, so the successor must use a single explicitly probed composite Python path rather than another default shell lookup.
- Serial/root-owned: generate a create-only runtime receipt that binds the exact interpreter path/hash plus CadQuery, cad_to_dagmc, Gmsh, PyMOAB, h5py, OCP, and NumPy imports/versions and an actual source-H5M reload; only then create a fresh v5-or-later scientific root and invoke that exact runtime.

## 2026-08-27 10:05 EDT — Runtime gate closed; refacet-v5 coarse launch review

- Serial/root-owned: accept the exact runtime receipt and independent same-runtime validation, freeze the refacet-v5 coarse control, perform the final live root/process/memory/disk/core-ledger check, and make the sole launch decision. Refined faceting remains dependent on coarse terminal scientific qualification.
- Safe to parallelize: read-only review of the frozen coarse control and launcher; static implementation of downstream native-qualification and faceting-evidence adapters in isolated branches; no other agent may select geometry or write the scientific attempt root.
- Scheduler/remote jobs: at most one 32-thread, three-hour-bounded coarse geometry-only refacet on Bateman. OpenMC transport, refined faceting, and automatic successors are forbidden at this stage.
- Shared-resource risks: a stale root, mismatched frozen control, another user's work reducing the 64-thread Codex allowance, or node memory falling below 128 GiB. Controls are an immediate authoritative core-budget query, exact root/process absence, live memory and disk thresholds inside the launcher, a unique PID/log/lease namespace, and no successor on completion.
- Independent launch review: `LAUNCH_REVIEW_PASS` after adding exact hashes for every extracted local Python source used by the refaceter, an atomic scientific-root claim, dangling-symlink rejection, and an exact launcher-hash check in the guarded invocation. Frozen control SHA-256 is `1d599ea9cb0357f5fb721d8bee5f805ea369cde7d7df6e82c26bde8c30d7184d`; launcher SHA-256 is `540387e76139820a250ffc4d8e478c654bc85a14b1d05b08c24384a38c01c848`.
- The sole coarse job started at 14:15:18Z under ssh-poly lease `764e2138e1930ffa2f3154712ccf780d`, session/timeout PID `803261`, and Python PID `803380`. It claimed 32 of 64 permitted threads after an immediate zero-ledger check; Bateman had 2,136,937,123,840 bytes available memory and 76,604,555,395,072 bytes free on the output filesystem. The scientific root was atomically created mode 0700. No refined or transport successor is authorized.

## 2026-08-27 14:32 EDT — Refacet-v5 terminal timeout separation

- Serial/root-owned: preserve v5 exactly, publish a separate terminal failure receipt, obtain independent failure qualification, and design a wholly fresh v6-or-later attempt with independently persisted exit/resource evidence. V5 can never be selected, reused, resumed, or relabeled.
- Safe to parallelize: read-only validation of the v5 terminal receipt and static adversarial review of a future v6 design. No reviewer may create a remote successor root or choose geometry.
- Terminal evidence: the 10,800-second bounded v5 process tree is gone and its lease is reaped. Exactly eight partial copied artifacts totaling 331,691,833 bytes remain. `dagmc.h5m`, candidate manifest, build seal, refined output, v6 output, and OpenMC successor are absent.
- Shared-resource state: authoritative ssh-poly status reports 256 total threads, 64-thread policy ceiling, zero active leases, zero reserved threads, one observed unreserved thread, and 63 available. No new remote launch is permitted during failure qualification/design review.
- V5 failure qualification is independently `PASS`; terminal receipt SHA-256 is `0fd3d7ca98440640312996fd48502d95d3a420ed9db72158f21596a93b9503ea` and independent qualification SHA-256 is `61d7e939535a4eda6c5e2fe158333b184c20676085b37e61636a8b1dfebaa7a2`.
- The corrected v6 recovery design is independently `DESIGN_REVIEW_PASS`: JSON SHA-256 `1484e3cc0997382b47e2084ec6c404d8550dc1b32bb43ec6982cd99f37bd0fea`, Markdown SHA-256 `b67fe0363a7520526535b9229727bf580bc092debd956bb78d9bd76a0c766167`. This permits local implementation/testing only; all remote gates remain closed.
- Safe parallel work completed: independent review of the faceting evidence producer/reducer found and root closed missing native hashes, magnet STEP reorder binding, physical/incidence count semantics, unbounded cost enumeration, level binding, and terminal-seal publication gaps. The focused producer/reducer suite is 47 passed; real H5M validation remains pending.
## 2026-08-27 09:31 EDT — Runtime-stack gate review

- Serial: the root agent alone owns accepted-runtime selection and any later refacet-v5 launch.
- Safe parallel work: read-only schema review and post-H5M G2–G6 execution mapping completed without shared writes.
- Independent jobs: only the bounded four-core Bateman runtime import/native-reload probe is active; no geometry meshing or transport is running.
- Delegated QA: one reviewer cleared the v1.1 receipt for the import-only probe; a second reviewer enumerated the exact six-control launch contract and downstream proof gaps.
- Shared-artifact risks: runtime-preflight-v1 and refacet-v1 through v4 are quarantined permanently; v2 has its own code, control, log, PID, lease, nonce, and receipt namespace. The future refacet-v5 root remains absent. Lane A has explicitly reconfirmed its read-only inventory window terminal and released.
- Gate: no successor may start until v2 is terminal, the receipt is collected and independently checked, the lease/PIDs are gone, and a fresh ssh-poly ledger shows zero active/reserved/used threads.
