# FES-II OpenMC compatibility and upstream-readiness audit

Date: 2026-09-25

## Decision

The ParaStell producer is source-compatible with the current OpenMC 0.16
release series after replacing literal `0.16.0` equality checks with a
fail-closed `>=0.16.0,<0.17.0` capability policy. The focused compatibility
suite passes 189 tests. OpenMC `develop` runtime qualification is still
required before claiming the current development binary is transport-qualified.

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

- 189 focused ParaStell compatibility and downstream tests pass, with one
  dependency-conditioned skip.
- The broad dependency-available suite passes 736 tests and skips four with
  the headless Matplotlib backend.
  Eight dependency-bound modules were excluded because this Windows runtime
  lacks PyMOAB/Gmsh/OpenMC or exposes incompatible legacy PyDAGMC and
  cad-to-DAGMC APIs; no dependency was installed or changed.
- Python compileall passes.
- Black reports the changed compatibility files unchanged.
- `git diff --check` passes.
- A no-isolation source/wheel build passes from clean external staging; the
  resulting wheel has 241 entries and zero nested `build/` entries.

## OpenMC candidate change

Local OpenMC branch:

```text
JS/fes-ii-dagmc-geometry-debug-neighbor-20260925
base: 1d75981dbf3fd78962e516c12b550d034f2e7daa
candidate commit: 56eddcf6878d30055f265c756fcb55e3fc3e8cab
```

The change makes `check_cell_overlap` exempt only the candidate DAGMC volume
that native `next_vol` proves is the current cell's topological neighbor across
the exact crossed DAGMC surface. It still checks every nonadjacent/third DAGMC
volume and every CSG cell. A regression test runs the official legacy DAGMC
model with `geometry_debug=True`; the unpatched v0.16.0 binary previously
failed this same model with `Overlapping cells detected: 1, 2 on universe 1`.

The candidate patch applies cleanly to live `develop`. Its staged SHA-256 is
`14663dff13781f054a259930da0d8ed83cfbeea0399092f49eaac24657a777ed`.
Python syntax and diff checks pass. A native current-develop build has not yet
completed because the shared Bateman persistent SSH channel closed after the
bounded inputs were staged; no build or test was falsely recorded as run.
The candidate branch is pushed to the user-owned `FusionSandwich/openmc` fork.
No pull request or draft pull request was created.

## Required runtime acceptance before PR submission

1. Build exact official `develop` plus the two-file candidate change using the
   qualified DAGMC 3.2.4/MOAB/HDF5 toolchain.
2. Prove the official legacy DAGMC geometry-debug reproducer fails unpatched
   and passes patched.
3. Re-run the known nonadjacent-overlap negative control and require it to
   remain rejected.
4. Run the added DAGMC regression test and relevant C++/Python format checks.
5. Run two bounded WISTELL-D geometry-debug seeds and a small unpatched regular
   transport/surface-bank smoke on the exact accepted H5M.
6. Obtain an independent Sol review of the final OpenMC diff and terminal
   receipts.

Until these gates pass, the branch is `PR_CANDIDATE_RUNTIME_VALIDATION_PENDING`,
not ready for submission and not a production runtime.

## Proposed PR text

**Title:** Fix false geometry-debug overlaps at shared DAGMC boundaries

**Summary:** During a DAGMC surface crossing, `next_vol` has already selected
the topological destination volume while the particle remains exactly on the
shared facet. The generic overlap checker can therefore see both legitimate
neighbors as containing the point and report a false overlap. This change
skips only the neighbor proven by DAGMC topology across the crossed surface;
all other DAGMC and CSG candidates remain checked.

**Regression:** Extend the existing legacy DAGMC regression model to run with
`geometry_debug=True`. The test fails on the unpatched behavior at the first
shared-facet crossing and passes with the fix.

**Scope:** Geometry-debug overlap diagnostics only; no transport scoring,
particle physics, surface-source serialization, or ParaStell geometry changes.
