# ParaStell magnet test-spectrum, activation, and geometry foundation

Status: `PARTIAL_QUALIFICATION_NO_PRODUCTION_LAUNCH`.

This checkpoint completes the port-free producer/consumer interfaces and the
bounded analytic OpenMC software qualification. The historical real pilot is
retained as a hash-bound structural reader fixture; it is not activation,
SPECTRA-PKA, or production qualified.

## Repository checkpoints

| Repository | Feature branch | Implementation checkpoint before this evidence commit | Push status when recorded |
| --- | --- | --- | --- |
| ParaStell | `magnet-radiation-geometry-interface-20260826` | `d19c628` | pending final evidence commit and push |
| DPA_workflow | `codex/magnet-test-spectra-and-activation-20260826` | `d70ebda8` | pending final evidence commit and push |

ParaStell descends from the live-verified clean producer head
`744e1ab3cb7508aa30f11a3dcd9628cbf9e50430`; no known port-branch tip is an
ancestor. DPA_workflow starts from `cbb31c1e` with the three reviewed consumer
commits replayed at integration checkpoint `f4330525`. No PR, default-branch
merge, force push, or non-FusionSandwich remote is part of this work.

## Product identities

- Geometry interchange: `parastell.magnet_geometry_interchange/v1.0.0`.
- Activation-ready producer metadata:
  `parastell.activation_ready_metadata/v1.0.0`.
- Analytic fixture package:
  `parastell_damage.test_spectrum_fixture_package/v1.0.0`, nine fixtures,
  content SHA-256
  `68a97670ed23ae73f3fd48caa99e53c24ec797d38c53a855e498233122fbeb3b`.
- Corrected committed real-pilot receipt SHA-256:
  `5d5f6317c1dac0325fbaca36c856ab74b22f664224996092c4238a047bb86538`.
- External real-pilot bundle tree SHA-256:
  `ae4ac24edda9b7c710845d21c7fb222cf2b20233db3ba667756a9a8c3e47e5d3`;
  manifest SHA-256
  `7d9b1221f88092779703e2afc8a3a6a228046525649831d9066f6acbfa6f499a`.

Boundary phase space and volume scalar flux remain distinct product kinds.
The adapters reject surface current wherever scalar flux is required.

## OpenMC activation qualification

- API classification: `OPENMC_R2S_AVAILABLE`.
- Unchanged image default: `MISSING_CHAIN_OR_DATA` because no depletion chain
  is bound by default.
- Hash-bound thermal Cu-63 analytic run: `PASS` under OpenMC 0.16.0 commit
  `617d35a5063c57796b43428bc401e627d2011046`.
- Analytic report SHA-256:
  `8e8c95ce712733bc664ac57a75d05436bd0c83fc6b447b08173c7919e5f90c66`.
- R2S result SHA-256:
  `c44790ab4bed73fb1067de8163efd5f41b6c6ba379aea465ab2b4fb002cb852a`.
- IndependentOperator result SHA-256:
  `2af85e2734d3b7a8d314e02797eb85f1c61a48c27a257442dd89e4093334cdb6`.
- Protected total-atoms, activity, decay-heat, dominant-product, and
  decay-photon metrics agree exactly; declared relative tolerance is
  `1e-12`.
- Shutdown-photon statepoint SHA-256:
  `c87cae23c43fdad7a58208984de9c1c836a50017689564fa367dbc209ebede32`;
  tally `8.6220166020e17 +/- 3.2579602308e16` with 3.7787% relative error.
- The unpruned full-chain 14.1 MeV attempt is
  `DATA_RANGE_BLOCKED`; the passing source is deliberately 0.0253 eV. The
  thermal result does not qualify fusion-spectrum activation.

The six required material classes now have repository-grounded, unit-volume
test specs. The exact chain is complete for the declared representative
54-pair scope and the neutron index contains all required isotope identities.
The actual `get_microxs_and_flux` reaction axis remains
`not_evaluated_fail_closed`; explicit-layer cases remain virtual.

## Real-pilot and production status

The 5,000-history pilot contains 34 neutron and two photon boundary records,
but zero incoming photons. All nonzero whole-volume scalar-flux bins are
under-resolved; local meshes are spatially mixed and unfiltered; the bundle
predates activation-ready metadata and geometry interchange; whole-casing
scalar flux is absent. Real-pilot R2S, IndependentOperator, path-equivalence,
and shutdown-photon runs are therefore `NOT_RUN_FAIL_CLOSED`.

Evidence-based neutron planning estimates are:

- aggregate point estimate: 2,631,579 histories;
- aggregate conservative estimate: 4,370,920 histories;
- representative coil-0005 point/conservative estimates:
  7,142,858 / 17,766,010 histories.

No finite incoming-photon history estimate exists. S1 remains a planning
contract only: no hash-bound deck is prepared, no scheduler submission is
authorized, and weight windows remain disabled.

## Mandatory gates

| Gate | Status | Qualification boundary |
| --- | --- | --- |
| Clean ancestry and no port dependency | PASS | live remote head and ancestry verified; no-port test passes |
| Actual OpenMC activation capability | PASS_SCOPED | API and thermal bound path pass; default chain absent and 14.1 MeV full-chain path blocked |
| Activation-ready producer metadata | PASS_FOR_NEW_OUTPUTS | strict source-rate/material/volume/mesh/data manifests implemented; historical pilot predates it |
| Geometry interchange round trip | PASS | independent reader, metric arc/frame validation, boundary replay, and actual-H5M facet catalog pass |
| Analytic fixture package | PASS | all nine fixtures and eight reader acceptance checks pass |
| Real pilot fixture | STRUCTURAL_ONLY | reader/population plumbing passes; physics qualification blocked |
| Activation smoke with qualified data | PASS_ANALYTIC_BLOCKED_REAL | thermal analytic R2S and IndependentOperator pass; real pilot fails closed |
| Shutdown photons | PASS_ANALYTIC_BLOCKED_REAL | nonzero analytic tally; real pilot not run |
| Evidence-based production plan | PLAN_COMPLETE_LAUNCH_BLOCKED | neutron estimates exist; photon estimate and exact decks do not |
| Focused software gates | PASS | full repository baselines retain separately classified missing-data/dependency/hash-drift failures |

## Test and packaging evidence

- ParaStell required prompt tests: 34 passed.
- ParaStell combined relevant regression: 74 passed, one skipped.
- ParaStell dependency-complete collectable Docker subset: 194 passed, one
  skipped. Seven legacy modules requiring absent untracked `files_for_tests`
  data were excluded. Actual H5M facet smoke found 18 casing/winding pairs,
  36 components, and 403,636 unique canonical facet IDs.
- ParaStell wheel/sdist build: PASS; wheel SHA-256
  `ed66450c3b97e6995381c30a43173e8a9b5c47b718c3f549319a50c4f52f6cf8`,
  sdist SHA-256
  `2d7fb8363d62e02f2470375fb098af47cb1eae8155a3a57a0152bcd1b89bae9a`.
- DPA focused activation/fixture/geometry tests: 75 passed.
- DPA broad host suite after excluding 23 import-failing ML/atomistic modules:
  1,115 passed, 11 skipped, 46 failed. The failures are outside the new
  modules and separate into missing ASE/PyVista/Pandas, subprocess import-path
  assumptions, and pre-existing frozen-hash/plan drift.
- DPA wheel/sdist build: PASS; wheel SHA-256
  `476766c6cdeef057d662f220d5ff33d57967b5a0046184a08ab6fbe7f7df57e1`,
  sdist SHA-256
  `bf238a916539432513b4f3077da6c37c81de0dec0de069d721bec936bf9a9cb1`.
- Black, AST parsing, JSON parsing, JSON-Schema checks, and `git diff --check`
  pass for changed files.

No remote SSH or Slurm job was used: local existing runtimes were sufficient
for every authorized bounded task, while production launch gates remain open.

## Remaining blockers

1. Rerun a new bounded ParaStell pilot from the updated producer with whole
   casing flux, activation metadata, geometry interchange, component-filtered
   nonoverlapping local meshes, material-intersection volumes, and adequate
   neutron/photon statistics.
2. Audit the exact required-material MicroXS reaction axes and all referenced
   transport-table energy ranges for the fusion spectrum.
3. Select and hash-bind the actual production insulation material.
4. Obtain a nonzero incoming-photon calibration before resolving S1 histories
   or preparing exact unbiased decks.
5. Resolve unrelated repository dependency and frozen-manifest baseline
   failures separately; they are not hidden by this focused gate.
