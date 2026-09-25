# ParaStell port feature handoff

Date: 2026-09-25. Repository: `FusionSandwich/parastell`.
This document distinguishes implemented code, retained evidence, and unfinished
physical qualification. It is intended for a later Codex or ChatGPT Pro session
starting from the GitHub branches named below.

GitHub checkpoints: [integrated port branch](https://github.com/FusionSandwich/parastell/tree/JS/port-feature-integration-20260925)
and [PLC classifier hardening branch](https://github.com/FusionSandwich/parastell/tree/JS-plc-diagnostic-fail-closed).

## Branch and commit ledger

| Item | Identity | State |
| --- | --- | --- |
| Current target | `origin/main` at `de7d2978ff314b060ca2e6b10745a034e8b2a3c4` | No port modules at this commit |
| Integrated port stack | `JS/port-feature-integration-20260925`, first commit `ad09aa3c104c6280d1915ad3d26982916bb936e6` | Clean local integration of the 20 port commits from `3a0ce81` through `b9d6727` |
| PLC classifier hardening | `JS-plc-diagnostic-fail-closed` at `296b9f3` | Published separately; both fixes were transplanted onto integration as `4b4c49a` and `4ec392d` |
| In-situ aperture investigation | Source `ports/actual-plc-repair-20260824` at `b9d672764a804ea97fa311a45ce820a610dcb5e8` | Local work in progress; uncommitted geometry changes and tests must be reviewed before publication |

The local checkout is shallow at `main` and `e223061`; it cannot establish
the older common ancestor. The integrated branch was built from current
`main` using the port-specific tree changes. It excludes the unrelated COMSOL
and magnet handoff modules present in the older port branch. See
`docs/PORT_FEATURE_INTEGRATION_2026-09-25.md` for the exact transplant and
unit-test receipt. Do not merge the old `b9d6727` tree wholesale into main.

## What is implemented and checked

The integrated branch contains schema validation, layer-bounded CAD ducts,
separate void/liner/fill components, collision and clearance reporting,
surface-angle anchors, aperture loops, local views, native shared-facet
PyDAGMC H5M, discrete-PLC MOAB tetrahedralization, a global graveyard,
assembly/transport examples, ParaView export, and PLC diagnostics.
Native conformal export accepts one surface-anchored port.

The integration worktree reported 105 focused port/native/PLC tests passing,
13 visualization/shape tests passing, and nine selected magnet/in-vessel tests
passing from the required `tests/` working directory, using the already-local
`RadiantHTS-ParaStell-20260830` WSL runtime. No dependency was acquired.
An initial broad invocation failed collection because existing fixture paths
assume `tests/` as the working directory; the affected selectors passed when
rerun there.

The earlier actual radial PLC diagnostic failed at several resolutions:
13x49 had baseline and ported edge/facet crossings; ported 17x65 terminated
Gmsh with signal 11; 21x81 had boundary recovery failures. The subsequent
repair used localized radial diagonal policies and cyclic port-patch alignment.
Its six retained unported/ported cases at 13x49, 17x65, and 21x81 have zero
crossings and passing tetrahedron audits. An independent read-only audit
verified all 21 artifact hashes and byte counts in
`D:\openc-hts-dpa-data\geometry-diagnostics\actual-plc-repair-20260824-r2`
against master manifest SHA-256
`d1552af41800173495cabda768893fd99e7cebb91ab56d9a2cc94602759d8997`.
The run was at source commit `2fb5063`, not at the integrated branch tip.

The PLC classifier hardening closes a separate false-success path: six
malformed or duplicated zero-intersection payloads could previously produce
`PASS_NO_REPRODUCIBLE_PLC_FAILURE` without valid mesh results, and a child
could exit nonzero after writing a nominal payload. The dedicated branch
requires exact case/source identity, successful child exit, and a nonempty
clean mesh audit before a terminal PASS. An independent Sol review found two
further malformed-payload counterexamples: blank region tags and a boolean
false child exit code. Both now classify as blocked. The separate branch was
pushed to `origin`, and its two commits were cherry-picked onto integration.
The corrected focused suite passed 25 tests in the existing WSL runtime.
The same 25 tests passed after both commits were added to the integrated
branch. Applying that classifier to the six retained real case results and
their recorded child exit codes returned
`PASS_NO_REPRODUCIBLE_PLC_FAILURE`. This is a historical receipt check; the
actual matrix has not been rerun at the integrated branch tip.
These tests accept the diagnostic classifier, not a full geometry model.

## Attempts that did not establish physical validation

* The 26,000-history OpenMC result used a small magnet fixture translated
  1,000 cm from the in-vessel geometry. It supports navigation regression
  only; it does not measure physical port/magnet clearance or transport.
* The first actual-coordinate reactor placement did not resolve every native
  aperture loop, and the baseline radial PLC also contained crossings.
  The later PLC repair closes the six tested radial cases, but does not
  establish assembled magnet overlap or transport.
* The in-situ candidate search used fixed native magnet coordinates. Its
  50 cm reference shield intersects casing geometry; a 38.5 cm sensitivity
  passed an envelope check but prior real-VMEC aperture endpoint/loop
  construction failed. A local geometry-only patch and public-input test are
  in progress; no in-situ H5M or OpenMC result is accepted.
* A private in-situ YAML is ignored by Git. Do not copy it into the repository
  or infer physical acceptance from the public test geometry.

## Next actions and acceptance gates

1. Finish and review the in-situ aperture patch. Verify the real-VMEC endpoint,
   every inner and outer aperture ray, ordered non-self-intersecting loops,
   patch/liner topology, and counterexamples. Publish only reviewed code and
   public test inputs.
2. Rerun the six-case actual radial PLC matrix on the integrated branch with
   per-case diagonal policies and fresh output directories; verify process
   exits, source/input hashes, all mesh audit fields, and the terminal summary.
3. For the physical 40-coil model, qualify port/magnet clearance, actual
   overlaps, assembled DAGMC senses, interstitial/exterior behavior,
   `check_watertight`, and `overlap_check`.
4. Only then run OpenMC geometry-debug and bounded fixed-source transport with
   the real cross-section library. Require zero lost/navigation errors and a
   nonzero downstream tally before any full-reactor transport claim.

No full-reactor transport gate is complete. Large H5M, mesh, statepoint, and
diagnostic artifacts belong outside Git with hash-bound receipts. Do not touch
`svalinn/parastell`.

## Resource and test state at handoff

A broader repository test run was made from `tests/` on the integrated
branch with the existing WSL runtime, `PYTHONPATH=..`,
`PYTHONDONTWRITEBYTECODE=1`, `-q -x -p no:cacheprovider`, and a 20-minute
timeout. It reached **37 passed, one failed** in 217 seconds and stopped at
`test_nwl.py::test_nwl_io[ref_surf0]`: OpenMC's Python package was present
but its compiled `openmc` executable was absent from that WSL runtime's
`PATH`. This is an environment gate for the existing NWL test, not a port
geometry failure. The machine had 31.9 GB RAM with about 8.2 GB free while
another local Python geometry process was active. A fresh six-case Gmsh
matrix, which previously used a 6 GB/4 CPU limit, was not started
concurrently.

## Suggested continuation prompt

> Continue the ParaStell port feature from the
> `JS/port-feature-integration-20260925` branch of
> `FusionSandwich/parastell`. Read
> `docs/PORT_FEATURE_HANDOFF_2026-09-25.md` and
> `docs/PORT_FEATURE_INTEGRATION_2026-09-25.md` first. Verify remote branch
> SHAs and current main, then inspect the integrated PLC classifier hardening
> and its negative tests. Preserve
> uncommitted in-situ work only with its owner's review; never copy the
> ignored private YAML into Git. Re-run focused tests, then the actual
> six-case PLC matrix if local resources permit. Treat physical magnet
> overlap, exterior topology, and OpenMC transport as open gates. Record
> hashes and failed attempts, and do not claim full-reactor transport
> qualification from the translated fixture or from a PLC PASS.
