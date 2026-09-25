# FES-II OpenMC compatibility and upstream-readiness audit

Date: 2026-09-25

## Decision

The ParaStell producer is source-compatible with the current OpenMC 0.16
release series after replacing literal `0.16.0` equality checks with a
fail-closed `>=0.16.0,<0.17.0` capability policy. The focused compatibility
suite passes 190 tests. Exact current OpenMC `develop` plus the two-file
candidate was built and passed the bounded DAGMC regression, physical-overlap
negative control, two-seed WISTELL-D geometry-debug check, and regular
surface-source smoke described below.

Final classification: `PR_READY_FOR_USER_REVIEW_AND_SUBMISSION`. This means
the code and evidence are ready for the user to review before submission; no
pull request or draft pull request has been opened.

The proposed OpenMC contribution is intentionally narrow: fix geometry-debug
false-positive overlap reports at a legitimate shared DAGMC facet. No
ParaStell-specific tally definitions, geometry, or research inputs belong in
the OpenMC pull request.

## Exact upstream identities

- Official repository: `openmc-dev/openmc`.
- Latest stable tag: v0.16.0,
  `617d35a5063c57796b43428bc401e627d2011046`.
- Audited live `develop`:
  `1d75981dbf3fd78962e516c12b550d034f2e7daa` (46 commits after v0.16.0).
- FES-II repository: `openmc-dev/fes-project-ii`, main
  `f1e3b53204f459870e96875af71c2f133e8cc1cb`.
- FusionSandwich OpenMC fork `develop`:
  `9a62e431d3101799e6179a6d0cf3b37440062e23`; it is 43 commits behind the
  audited official `develop` and has no unique commits in that comparison.

## Paul and Patrick fork disposition

The reviewed branches are useful historical evidence, not suitable PR bases:

- Paul Romano `surface-source-fix` is 841 official commits behind and has six
  unmerged patch commits. Its original all-active-batch banking change has
  since been superseded by the official per-batch/multi-file implementation
  (`9686851e7`, PR #3124) and the current unit/regression tests.
- Patrick Shriwise `dagmc-surf-source` is 701 official commits behind. Its key
  behavior was merged officially as `cb7ef009b` (PR #2857).
- Patrick's DAGMC-history reset work was merged officially as `8cd3911cb`
  (PR #3601).
- Surface-source half-space/direction handling is present officially as
  `94c0defae` (PR #4024).
- `MuSurfaceFilter`, surface-flux scoring, reaction filtering, particle
  production filtering, and PDG particle identities are all present on live
  `develop`; the old `dagmc_surf_norm` and `mu-surface` branches should not be
  transplanted wholesale.

## ParaStell compatibility changes

`parastell.openmc_compat` now provides one standard-library-only policy:

```text
minimum: 0.16.0
next unqualified series: 0.17.0
accepted examples: 0.16.0, 0.16.1.dev46+g1d75981db, 0.16.1-dev46
rejected examples: 0.15.3, 0.16.0rc1, 0.17.0, malformed/unversioned builds
```

Execution, the immutable one-period wrapper, surface-bank, statepoint,
checkpoint, activation, and downstream handoff gates use this policy and
continue to preserve the exact runtime version and artifact hashes. The
previous v0.16.0 geometry-debug acceptance
receipts remain pinned to their exact binary and commit; historical evidence
has not been silently relabeled as current-develop evidence.

Validation completed locally:

- 190 focused ParaStell compatibility and downstream tests pass, with one
  dependency-conditioned skip.
- The broad dependency-available suite passes 737 tests and skips four with
  the headless Matplotlib backend.
  Eight dependency-bound modules were excluded because this Windows runtime
  lacks PyMOAB/Gmsh/OpenMC or exposes incompatible legacy PyDAGMC and
  cad-to-DAGMC APIs; no dependency was installed or changed.
- Python compileall passes.
- `python -c "import parastell"` passes. The two CLI help probes remain
  dependency-bound on this Windows host because `cad_to_dagmc` is not
  installed; no dependency environment was modified for this audit.
- Black reports the changed compatibility files unchanged.
- `git diff --check` passes.
- A no-isolation source/wheel build passes from clean external staging; the
  resulting wheel has 241 entries and zero nested `build/` entries. The final
  wheel SHA-256 is
  `2a78385d6cff97ef86dffd33e31671c4747f3f7f18d6e0e451ba349677aa5125`;
  the source archive SHA-256 is
  `cd7bb23d32702a5e11cbf75d79aa68081cd8ce2067f7ac266ac923b8629ba5ae`.

## OpenMC candidate change

Local OpenMC branch:

```text
JS/fes-ii-dagmc-geometry-debug-neighbor-20260925
base: 1d75981dbf3fd78962e516c12b550d034f2e7daa
candidate commits:
  56eddcf6878d30055f265c756fcb55e3fc3e8cab
  92a3ccdaf4adbbf7ee644f3a99d65d3451149dc0
  095db1a9179071ba3b8e57e5181c07c9452bb69a
  43ce5bbe77a76646c2dfa324ffad57d2a61af3a5 (final)
```

The change makes `check_cell_overlap` exempt only the candidate DAGMC volume
that native `next_vol` proves is the current cell's topological neighbor across
the exact crossed DAGMC surface. It still checks every nonadjacent/third DAGMC
volume and every CSG cell. A regression test runs the official legacy DAGMC
model with `geometry_debug=True`; the unpatched v0.16.0 binary previously
failed this same model with `Overlapping cells detected: 1, 2 on universe 1`.

The candidate patch applies cleanly to live `develop`. A final fetch on
2026-09-25 reconfirmed that the official tip remains
`1d75981dbf3fd78962e516c12b550d034f2e7daa`. The staged patch SHA-256 is
`6f400835e678691bec154412b61b65210eae0614c92778f1cd96da400d201617`.
Python syntax and diff checks pass. The candidate branch is pushed to the
user-owned `FusionSandwich/openmc` fork. No pull request or draft pull request
was created, and the fork has no CI run for this branch.

An independent Sol review classified the original narrow implementation
`PASS` with no correctness blocker. The subsequent cleanup removed that new
helper entirely and reused OpenMC's existing `next_cell` DAGMC topology helper,
including the same one-based-to-zero-based conversion already used in
`particle.cpp`. This reduced the final production diff from 47 added C++ lines
to 13 without changing the accepted behavior.

## Current-develop runtime acceptance

The exact candidate was built natively on Bateman with GCC 13.3, DAGMC 3.2.4,
system HDF5, OpenMP, no MPI, and no UWUW. The executable reports OpenMC
`0.16.1-dev46` and exact commit `1d75981db`; its SHA-256 is
`be8a2477688265bcd04ccef248350708e90844d6ec786640cc276f4e093b248c`.
The final candidate `libopenmc.so` SHA-256 is
`08cab527e2ed9345336a17610ec93066da02948efc2fc50cc8c3fccf877d6c53`.
The bounded build used eight of 256 physical cores, a 16 GiB hard memory cap,
and disabled swap. Compilation completed in 2:19 with 358 MiB maximum RSS.
The review-polish incremental rebuild used four cores, an 8 GiB hard memory
cap, disabled swap, and completed in 50.02 seconds.

Runtime controls:

1. The unpatched exact current `develop` reproduces
   `Overlapping cells detected: 1, 2 on universe 1` on the official legacy
   DAGMC shared-face model. Restoring the candidate byte-for-byte makes the
   same 500-history geometry-debug model pass. The final candidate executable
   and both libraries were restored to their exact pre-comparison hashes.
2. The known nonadjacent physical-overlap H5M
   `8741dd48fded42e8411816e56e3e5e10a29db26ddb785b4a389f8a38b09707a0`
   remains rejected with
   `Overlapping cells detected: 26, 4 on universe 1`.
3. WISTELL-D geometry-debug seeds `8310101` and `8310102` each completed 400
   histories against accepted H5M
   `d31ea04e4d2fda78870db1688d6cc9215079e20408393c5f5bd374d58f43eaf3`.
   Neither log contains an OpenMC error, lost-particle message, or navigation
   error.
4. A regular (non-geometry-debug) DAGMC transport wrote a 100-record
   `surface_source.h5` bank. Every record was reconstructed on selected
   surface 1 at radius 7 cm, and the statepoint was written successfully.
5. Python compilation and `git diff --check` pass. The exact final commit also
   passes OpenMC's GitHub clang-format 18 workflow:
   `https://github.com/FusionSandwich/openmc/actions/runs/36157164939`.

The original full-build runtime receipt is
`/home/apollon/josma/data/codex-parastell/openmc-develop-fesii-compat-v8-20260925T145100Z/output/runtime-qualification-receipt.env`
(SHA-256
`7b2b3d691135b1c6ba0619551442c5db0d705ddf9bb2a5410f97ebf513215cfd`).
The unpatched/patched comparison receipt SHA-256 is
`418a325a55d851e32526e6f345863ec2ebeb02c3af6f3bf027d7948c9463523d`.
The patched regression log SHA-256 is
`b2d253ca6c0b4e00924d6f440c5c67f61a5c15339b18a61730fa08262e8a4c1d`;
the negative-control log SHA-256 is
`098eeff366c7c89a92f2592a09417f3343cb3b01a28e1b2ad93de2f82c1d2839`;
and the surface-source file SHA-256 is
`85aa4e9479f73270debbff336366f871a015c7e96679de0f137d79ee19d67fc6`.

The exact final minimized commit was rerun in a fresh combined control root:
`/home/apollon/josma/data/codex-parastell/openmc-develop-fesii-final-controls-v4-20260925T193000Z`.
Its receipt SHA-256 is
`0320b011a0bec1de671c0d43a10a7eca1736e08b390247d39525713de520d612`.
The shared-face regression, physical-overlap rejection, both WISTELL-D seeds,
and the 100-record surface-source reconstruction all passed. The immediately
preceding minimized `final-controls-v3` attempt was quarantined because it
compared OpenMC's one-based `next_cell` result directly to a zero-based model
cell index; the accepted change applies the same `- 1` conversion already used
by OpenMC particle transport.

The remaining pre-submission step is the user review requested here. The exact
final commit already passes the fork's OpenMC clang-format 18 workflow; the
full upstream CI matrix will run when the user authorizes PR submission. This
candidate is diagnostic infrastructure only; it is not authorization for
production transport.

## Proposed PR text

**Title:** Fix false geometry-debug overlaps at shared DAGMC boundaries

**Summary:** During a DAGMC surface crossing, the existing `next_cell` helper
identifies the topological destination volume while the particle remains
exactly on the shared facet. The generic overlap checker can therefore see
both legitimate neighbors as containing the point and report a false overlap.
This change skips only that topology-proven neighbor after applying OpenMC's
existing one-based-to-zero-based cell-index conversion; all other DAGMC and
CSG candidates remain checked.

**Regression:** Extend the existing legacy DAGMC regression model to run with
`geometry_debug=True`. The test fails on the unpatched behavior at the first
shared-facet crossing and passes with the fix.

**Scope:** Geometry-debug overlap diagnostics only; no transport scoring,
particle physics, surface-source serialization, or ParaStell geometry changes.
