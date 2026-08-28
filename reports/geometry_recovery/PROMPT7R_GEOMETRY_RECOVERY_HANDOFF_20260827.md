# Prompt 7R geometry-recovery and instrumentation handoff

Date: 2026-08-27  
Repository: `FusionSandwich/parastell`  
Current working branch: `magnet-radiation-valid-baseline-and-instrumentation-20260827`  
Required starting SHA, verified: `dd3c2c53474980b6527d6ad40e8e2a638337d107`  
Prompt specification: `C:\Users\joshu\Downloads\PROMPT7R_VALID_GEOMETRY_PROMPT1_PROMPT7_COMPLETION.md`  
Prompt SHA-256: `170d0073f00378e42b8ef0f6eebd41a933544fa520135d79b36b3882f2886acd`

## Executive summary

Prompt 7R is not complete. There is no accepted, magnet-inclusive, overlap-free H5M and therefore no geometry-qualified OpenMC producer yet. The correct current classification is:

```text
BLOCKED_GEOMETRY_PARITY
READY_FOR_SOURCE_MESH_AND_MEDIUM_STATISTICS = NO
```

The work exposed two separate problems that had been conflated:

1. **The current public ParaStell example is a software example, not a production-valid reactor reference.** Its own matched example VMEC and coil inputs produce four real vacuum-vessel/magnet intersections when the CAD-to-DAGMC example uses the shipped 50 cm shield and 40 x 50 cm magnets. This defect exists before DAGMC export. It is not caused by our tally, boundary-bank, casing, or H5M instrumentation code.
2. **The current public ParaStell example is not the WISTELL-D/StellD input pair.** The actual named WISTELL-D VMEC and coil inputs are present in the user's scratch/Blanket repositories and in `DPA_workflow`. They are physically and bytewise different from the public example inputs.

There was also a third, self-inflicted problem in the rejected feature line: a global casing/winding-pack subdivision changed the physical model, created/opened interfaces, and made the coupling-surface logic invalid. That feature must remain rejected and must not be used as the basis of the repair.

The most important next action is therefore **not another refacet of the clearance-repaired public example**. It is to freeze the intended WISTELL-D source identity and radial-build contract, reproduce that source-CAD model independently, and qualify its physical geometry before adding any metadata, tallies, or surface-bank instrumentation.

## Authoritative WISTELL-D update received after the initial handoff

The intended source contract is now identified. This section supersedes the earlier uncertainty about which WISTELL-D reconstruction to select.

Use this exact source set. Its existing validated STEP is a 45-degree source-CAD seed; the Prompt 7R transport target is now one complete 90-degree field period:

- Combined STEP: `D:\Scratch\wistell_d_parametric_parastell_20260827_01\nwl_baseline_bounded\wistell_d_model.step`
- VMEC: `D:\Scratch\stellarator_optimization\wistell-d_data\wout_wistell-d.nc`
- Coils: `D:\Scratch\stellarator_optimization\wistell-d_data\coils.wistell-d`
- Thickness arrays: `D:\Scratch\wistell_d_parametric_parastell_20260827_01\nwl_baseline_bounded\thickness_arrays_cm.npz`
- Source manifest: `D:\Scratch\wistell_d_parametric_parastell_20260827_01\nwl_baseline_bounded\manifest.json`
- Builder: `D:\Scratch\wistell_d_parametric_parastell_20260827_01\build_wistell_d.py`

The hash-bound contract is recorded in `reports/geometry_recovery/WISTELL_D_AUTHORITATIVE_SOURCE_CONTRACT.json`.

Independent local verification confirmed:

- all six files exist;
- VMEC SHA-256 `9231969001203a8133255ee0a275bf552b114cc12524dda0608ab2f12047f7ac`;
- coil SHA-256 `7748369407d28a70f35b5c4a7c0ab860495a08fd0030002112ea933fe570159b`;
- combined STEP SHA-256 `1e9d933e5fd4d393c82f47b26da82d9e58cfbb59e951e354a2f871ebd1920a62`;
- thickness NPZ SHA-256 `047585172bd682692b30787424a7e1977afdb60693b606b036dab3817b3bb501`;
- source manifest SHA-256 `f330bbd06a0c8234a3b52932ee48e8dcdec7e2842c3d12a5c75d3052028920b4`;
- builder SHA-256 `3702b76d3c5c0876119e99d16614069b6dc8181d9977c9b617dc013ac2eaf5c6`;
- every NPZ layer contains exactly 452 finite, strictly positive values;
- all nine source-CAD components are valid and positive-volume;
- the source manifest records zero Boolean failures and zero overlaps.

The verified radial stack, from plasma toward the magnet region, is:

| Layer | Thickness |
|---|---:|
| First wall | 4 cm |
| Breeder | 25–55 cm; mean 37.9758 cm |
| Back wall | 5 cm |
| High-temperature shield | 20 cm |
| Vacuum vessel | 10 cm |
| Low-temperature shield | 16.2009–116.7819 cm; mean 61.2592 cm |
| Vacuum gap | 6.2132–124.4057 cm; mean 28.2076 cm |
| Magnet layer | 30 cm |

The blanket through low-temperature shield reaches 97.5368–206.3693 cm from the plasma boundary. The magnet-boundary field reaches 103.7500–273.0804 cm. All radial-order residuals are at floating-point roundoff scale.

This evidence changes the geometry recovery decision:

```text
PUBLIC EXAMPLE: retain as a defective negative/software reference
PUBLIC 40 CM SHIELD FIX: retain as a public-example control only
A1R: retain as a clearance-method control only
WISTELL-D BOUNDED MODEL: selected source-CAD scientific baseline
```

Direct inspection resolves the period convention:

- the VMEC file has `nfp=4` and stellarator symmetry enabled (`lasym=false`);
- the coil file declares `periods 4`;
- ordinary rotational repetition is therefore every 90 degrees;
- 45 degrees is a stellarator half-period related to its neighbor by reflection, not by a simple rotation;
- the historical 45-degree transport code uses a custom `transformation` boundary that reflects particle position and direction;
- stock OpenMC 0.16.0 supports periodic/rotational planes but does not expose that custom transformation boundary type.

Therefore the general ParaStell + stock OpenMC 0.16.0 baseline must model **one full 90-degree period** and use ordinary rotational periodic planes at 0 and 90 degrees. The 452 independent half-period thickness values are expanded to a 31 x 31 full-period matrix using ParaStell helical symmetry. The retained `source_mesh.h5m` is now proven to cover only the 0–45° half-period. It is retained solely as the immutable parent input for the separately generated 0–90° source mesh; no second 45° transport or activation campaign is planned.

The selected WISTELL-D model represents the magnet region as **one continuous 30 cm winding-surface layer**. `coils.wistell-d` was used upstream to derive the blanket and magnet boundary arrays; it is not exported here as 18 explicit swept coil solids. Consequently:

- the baseline coupling interface is the complete inner boundary of the continuous magnet layer;
- the global producer must score and record crossings on that real interface;
- it must not fabricate 18 per-coil volumes or claim per-coil surface resolution that the geometry does not contain;
- an explicit per-coil extension, if later required, is a new geometry candidate that requires complete source-CAD, DAGMC, OpenMC, and visualization requalification.

The source-CAD physical gate is now a pass for this exact contract. The final H5M, native DAGMC, watertightness, overlap, interface-closure, OpenMC navigation, instrumentation, and bounded functional gates remain to be completed.

## The three geometry lines that must not be conflated

### Line R1: public ParaStell example — clean software reference, negative physical reference

Purpose: prove what untouched current ParaStell `main` actually builds.

- Clean, separate, push-disabled repository: `D:\parastell-reference-repos\parastell-vanilla-main-de7d297`
- Exact ParaStell commit: `de7d2978ff314b060ca2e6b10745a034e8b2a3c4`
- Rebuilt H5M: `D:\parastell-artifacts\geometry-recovery-20260827\vanilla_main_reference\20260827T021126\dagmc.h5m`
- H5M SHA-256: `17d0b406187e7fb49e5791b5973e2d057b8751d40365c71b9698337cc82e5303`
- Inputs: public `examples/wout_vmec.nc` plus `examples/coils.example`
- Public example magnet definition: one homogenized solid per magnet; 40 cm width, 50 cm thickness; no casing/winding split
- Public CAD-to-DAGMC radial build: 50 cm shield
- Result: 18 valid, mutually separated magnet solids, but four true magnet/vacuum-vessel intersections at magnets 5, 6, 11, and 12

This separate vanilla build satisfies the user's requirement to compare against unmodified ParaStell `main`. It remains the **software behavior authority**, but it is rejected as a physical transport baseline.

The exact Boolean intersection evidence is:

| Magnet | Intersection volume with vacuum vessel, shipped `sample_mod=6` | Approximate fraction of magnet volume |
|---:|---:|---:|
| 5 | 5,092.20 cm3 | 0.0899% |
| 6 | 21.5895 cm3 | 0.000376% |
| 11 | 21.5916 cm3 | 0.000376% |
| 12 | 5,113.58 cm3 | 0.0903% |

The intersections persist with all coil points (`sample_mod=1`), so coarse coil sampling is not their cause. The in-vessel solids are individually valid and adjacent layer pairs have zero common volume. The defect is the combination of the static radial build and finite magnet envelope, together with the example's lack of a clearance/overlap acceptance check.

### Line R1F: scratch repair of the public example — useful control, not automatically StellD

The user's scratch ParaStell repository contains a committed minimal correction:

- Repository: `D:\Scratch\parastell`
- Branch: `fix/example-shield-clearance-40cm`
- Base: `de7d2978ff314b060ca2e6b10745a034e8b2a3c4`
- Fix commit: `d480424e53747691075d6b0b074face03efd311f`
- Commit message: `Align example shield thickness with PyDAGMC`
- Working tree: clean except for untracked `examples/source_mesh.h5m`

The exact physical change is:

```text
shield thickness: 50 cm -> 40 cm
```

It is applied in:

- `examples/parastell_cad_to_dagmc_example.py`
- `examples/parastell_cubit_example.py`
- `examples/config.yaml`

This is an important naming correction. The saved change is to the `shield` layer in the in-vessel radial build. It is not a 50-to-40 cm change to the magnet thickness, and it is not a direct edit of the `breeder` entry. It is reasonable to describe it broadly as a blanket/radial-build correction, but the implementation must continue to name the exact component.

The scratch audit reconstructed this 40 cm envelope and found zero common volume for the four previously intersecting magnet/vessel pairs. By contrast, changing magnet thickness from 50 cm to 40 cm removes only the two tiny contacts and leaves approximately 1,269 and 1,261 cm3 at magnets 5 and 12. Therefore the committed 40 cm shield correction is the supported minimal public-example fix.

This lane should be retained as:

- a regression demonstrating the public-example fault and its minimal correction;
- an exporter/backend control;
- a possible generic ParaStell example repair.

It must not be called WISTELL-D merely because it is overlap-free. Its VMEC and coil inputs are still the public generic example pair.

### Line R2: named WISTELL-D/StellD assets — intended scientific source lane

The actual named WISTELL-D assets were found and are separate from R1.

Primary historical source checkout:

- Repository: `D:\Beyound DPA Repos\BlanketNeutronics`
- Branch: `t1e_main`
- SHA: `bc4ab3d0f27369d4eda908a3fc187a10b6c7fedb`
- Workflow root: `neutronics_wistelld_DCLL_1`
- VMEC: `plasma_configurations\plasma_wistelld.nc`
- Coils: `plasma_configurations\coils_wistelld.txt`
- Radial distances: `neutronics_wistelld_DCLL_1\radial_distances.csv`
- Original geometry definition: `neutronics_wistelld_DCLL_1\geometry.py`

Mirrored scratch checkout:

- Repository: `D:\Scratch\BlanketNeutronics`

Hash-identical staged copies already exist in:

- `D:\2026_DPA\DPA_workflow\examples\wistell_d\inputs`
- DPA_workflow SHA at inspection: `f9eef940ca6dffcd2cecdafa4fd286cb59c32d0b`
- DPA_workflow branch: `codex/master-consolidation-and-mlip-plan`
- That worktree has unrelated modified planning-inventory files and must be treated carefully.

Identity hashes:

| Input | SHA-256 |
|---|---|
| `plasma_wistelld.nc` | `9231969001203a8133255ee0a275bf552b114cc12524dda0608ab2f12047f7ac` |
| `coils_wistelld.txt` | `7748369407d28a70f35b5c4a7c0ab860495a08fd0030002112ea933fe570159b` |
| `radial_distances.csv` | `0e3c7d3fb6cd914abe04f1293ffb06e19db5a63289ca05c05d2aed70d9a9081c` |
| staged DPA ParaStell YAML | `18a15a00c45e5ccfecef3dbf99ab5ffdd5a30f1c1256e3eb4f36b224d8354277` |

The three historical source files are byte-identical to their DPA_workflow copies.

The WISTELL-D pair is materially different from the public ParaStell pair:

| Property | Named WISTELL-D | Public ParaStell example |
|---|---:|---:|
| VMEC SHA | `92319690...f7ac` | `1cebb8d4...4797` |
| Coil SHA | `77483694...159b` | `69f508b2...04ab` |
| Field periods | 4 | 4 |
| Major radius | about 10.0764 m | about 11.0845 m |
| Minor radius | about 1.48525 m | about 1.70452 m |
| Filaments | 48 | 40 |
| Points per filament | 384 | 128 |

Approximate centerline/plasma distances also prove that input pairing matters:

| VMEC | Coils | Approximate minimum distance |
|---|---|---:|
| WISTELL-D | WISTELL-D | 1.094 m |
| Public ParaStell | Public ParaStell | 1.272 m |
| WISTELL-D | public ParaStell | 0.0163 m |
| Public ParaStell | WISTELL-D | 0.00537 m |

Cross-pairing the files is physically invalid once finite magnets are added.

The historical WISTELL-D `geometry.py` uses `wall_s=1.2`, a 90-degree sector, source sampling `11 x 81 x 61`, a constant 50 cm `breeder` layer, several structural/gap/shield layers, and a radial `coils` layer of 50.5 cm. It references the named coil file, but the original Blanket case documentation also describes magnets as represented by radial-build layers rather than independently transported finite CAD magnets. The later DPA_workflow YAML is a DCLL-inspired adaptation with a variable breeder, a 51 cm TiH2 shield, explicit 40 x 50 cm coil solids, and a different source sampling definition. Those are not automatically identical physical models.

Before the later authoritative bounded-model handoff, the honest identity classification was:

```text
SEPARATE_WISTELL_D_ASSETS_FOUND
```

That uncertainty is now superseded for Prompt 7R by the hash-bound `nwl_baseline_bounded` source contract documented at the top of this handoff. The selected contract is confirmed as the intended WISTELL-D source-CAD baseline, while an exact published-paper identity claim still requires separate paper provenance.

## Controlling scientific goal

Build one no-port global ParaStell/OpenMC producer that:

1. uses one proven, overlap-free WISTELL-D source geometry;
2. preserves the selected source model's continuous 30 cm homogenized magnet layer;
3. adds stable identities, fingerprints, tallies, local meshes, and boundary recording without changing physical solids;
4. treats the complete inner boundary of the continuous magnet layer as the coupling interface;
5. produces both volume scalar-flux products and facet-complete boundary phase space, without substituting one for the other;
6. passes bounded OpenMC navigation and instrumentation tests;
7. stops at `READY_FOR_SOURCE_MESH_AND_MEDIUM_STATISTICS`, not production readiness.

The global model must not contain casing/winding-pack/tape subdivisions. Those belong in the downstream local-model repository.

## Work completed

### Evidence and repository control

- Prompt 7R was frozen by exact file hash.
- The requested starting SHA was verified.
- A clean standalone vanilla ParaStell repository was created from public `main`; it is push-disabled and separate from all feature worktrees.
- The vanilla public example was rebuilt without changing its physics inputs.
- Public-reference artifacts, candidate artifacts, attempted remote runs, runtime environments, leases, and artifact ownership were recorded under `reports/geometry_recovery` and `D:\parastell-artifacts\geometry-recovery-20260827`.
- No port branch, PR, default-branch merge, upstream modification, production transport, source-convergence campaign, or production-statistics campaign was performed.

### Public-example diagnosis

- The public example's four intersections were reproduced in smooth source CAD and bound to named magnet/vessel pairs.
- Magnet-to-magnet intersections were ruled out for all 153 pairs and periodic seam checks.
- Adjacent in-vessel layer intersections were ruled out.
- The same intersections remain at dense coil sampling, ruling out `sample_mod=6` as the cause.
- The scratch 40 cm shield correction was located as an actual commit and its semantics were distinguished from magnet-thickness and breeder-thickness changes.
- The absence of a full-assembly clearance/overlap test in the shipped example was identified as the workflow defect that allowed the invalid assembly to be presented as a working example.

### Rejected split-feature diagnosis

The failed feature branch `magnet-surface-field-visual-production-gate-20260826` at `df53ed28edc8d8123b6e42a9e4e5c4e970e69dc8` remains read-only rejection evidence.

Its physical geometry contained:

- 44 volumes and 269 surfaces;
- 18 casing volumes plus 18 winding-pack volumes;
- seven unintended nonadjacent-volume overlaps;
- 28 winding-pack faces directly adjacent to interstitial vacuum;
- only 2 of 18 strict external-only casing envelopes closed;
- an outer-interface declaration that included two winding-only surfaces.

The root error was not only an envelope selector bug. The global casing/winding split was an unnecessary physical subdivision for this producer, and it interacted with pre-existing public-example reactor/magnet intersections. Appending winding faces to force casing closure would have produced a mislabeled union, not a valid external casing manifold.

Permanent conclusion:

```text
Use the selected continuous homogenized magnet layer globally.
Use its complete inner boundary as the magnet-entry coupling interface.
Do not repair or reuse the split geometry.
```

### A1R clearance-constrained public-family candidate

A source-CAD candidate was constructed from the public example while preserving its magnets, source mesh, first wall, and other fixed identities. Only the breeder thickness field was reduced locally from measured coil clearance to satisfy a preregistered 5 cm clearance rule.

Key evidence:

- Candidate manifest SHA-256: `36fed787de68ba1ed963b8d0bf212e9f35615ad14abefad39b8ef3e16a755f91`
- Physical-change report SHA-256: `7411d2564b9210cf2d0c7d99e0691a346083fd5ca202c4529c2b609d12465141`
- Minimum breeder thickness: `11.09681428133753 cm`
- Minimum exact source-CAD vessel/magnet clearance: `5.239626484915899 cm`, at magnet 9
- All 18 clearances exceed 5 cm
- Full 276-pair source-CAD audit coverage: 108 in-vessel/magnet + 15 in-vessel/in-vessel + 153 magnet/magnet pairs
- Missing, duplicate, malformed, or failing pairs: zero
- Source-CAD decision: `SOURCE_CAD_PHYSICAL_GATE_PASS`

This candidate demonstrates a robust clearance-derived repair method. It does **not** yet have an accepted final H5M, and it remains based on the public example's non-WISTELL inputs. In light of the newly clarified device identity, preserve it as an engineering/control lane rather than promoting it as the final scientific baseline.

### Native source-H5M and mapping qualification

The A1R coarse source H5M was qualified as a source-CAD/native-topology basis only:

- SHA-256: `8a2d1930cc03a82269feeedef60267197581559c99aa69a021235a7ac7fafa90`
- 24 volumes
- 142 surfaces
- 147 incidences
- five material interfaces
- 187,576 triangles
- zero native topology/material-ordering failures
- Diagnostic report SHA-256: `6380ac6c8de8b53d8b35136077451484d85c767fa860795e7514a480ff2a924e`

The Gmsh source-solid import mapping was repaired to use volume, center of mass, and full inertia as the source-side invariant; imported bounding boxes are used only as part of imported-to-fragment matching. Local diagnostic v5 achieved a 24/24 bijection:

- `D:\parastell-artifacts\geometry-recovery-20260827\gmsh_import_diagnostic_v5_local_20260827T125100Z\import_mapping.json`
- SHA-256: `594cc9808210117ef53932f1edcbc4f2af2ddc5c70047a54c5a170a349fafbdf`

### Exact remote geometry runtime

After several fail-closed setup probes, runtime preflight v3 passed and was independently validated on Bateman:

- Runtime receipt SHA-256: `2f6260b1fd45b1b7e8a3d9d079a4faf97158e03106352c9699990d508f8c85c7`
- Independent validation SHA-256: `946ba2126615f13bbc14a8ed86cba461811d8d61bc9701ad5eb861ab40881da6`
- Canonical runtime SHA-256: `1a70f6fd02a7a8ddae152cee1b5be39c6c58cdeb806326f711f4f0c7728dcc2d`
- Python executable SHA-256: `63770468d7041b46aa7fc01ad9a17b4e616dbbb7d613f5470e1cdc5359c83a86`
- CadQuery 2.8
- cad_to_dagmc 0.11.5
- Gmsh 4.15.2
- PyMOAB 5.6.0
- h5py 3.16
- OCP 7.9.3.1
- NumPy 1.26.4

This closes the earlier ambiguity about the correct interpreter and native extension stack. It does not itself qualify a geometry.

### Hardened refacet-recovery design

Refacet v6 was designed but not launched. The design passed independent review:

- Design JSON SHA-256: `1484e3cc0997382b47e2084ec6c404d8550dc1b32bb43ec6982cd99f37bd0fea`
- Design Markdown SHA-256: `b67fe0363a7520526535b9229727bf580bc092debd956bb78d9bd76a0c766167`
- Review: `reports/geometry_recovery/REFACET_V6_DESIGN_REVIEW.json`
- `remote_launch_authorized=false`

The design adds create-only roots, exact runtime/input binding, host-local staging, pre/post-staging memory gates, separate 4-core import/fragment diagnostics, conditional later 32-core meshing, explicit Gmsh thread caps, shell-owned return-code and resource capture, hash-chained stage receipts, create-only publishing, manifest/seal-last publication, and no automatic successor.

This is reusable operational work, but a v6 public-family refacet should now remain paused until the WISTELL-D model-selection gate is resolved.

### Partial local code implementation

Uncommitted implementation exists in the principal worktree. Major additions include:

- clearance measurement and radial-build constraints;
- source-CAD candidate construction and full pair auditing;
- source identity and source-domain checks;
- native DAGMC topology/material ordering;
- DAGMC qualification and faceting evidence/convergence helpers;
- OpenMC geometry-debug model and replica helpers;
- Gmsh import diagnostics and source-CAD refaceting;
- create-only/refacet execution primitives and stage receipt chains.

The newest v6 implementation work added:

- `parastell/refacet_execution.py`
- shared import/fragment logic in `parastell/source_cad_refaceting.py`
- explicit Gmsh thread-option setting before import/fragment and again before meshing
- create-only JSON/byte publication helpers
- operational-root sealing/validation
- stage-receipt chaining
- post-exit evidence collection primitives
- focused tests in `tests/test_refacet_execution.py` and `tests/test_refacet_gmsh_pipeline.py`

The last fully observed focused result before a small compatibility patch was 20 passed and 1 skipped. One monkeypatched test then failed because its lambda did not accept the new `stage_callback`; that test was patched to accept `*_args, **_kwargs`, but the rerun was interrupted by this handoff request. The code is incomplete and must not be represented as a launch-ready v6 implementation.

## Attempts that failed and what each taught us

All roots listed below are permanently nonselectable. Never reuse or relabel them.

| Attempt | Result | Root cause / lesson |
|---|---|---|
| Public vanilla R1 | Physical rejection | Reproduced faithfully, but contains four real vessel/magnet intersections. Reproduction is not acceptance. |
| Split feature `df53ed28...` | Rejected | Changed the global model, retained/added overlaps, created invalid casing topology, and violated the homogenized-global/local-explicit firewall. |
| A1 candidate | Rejected before final | Clearance requirement was not met everywhere; led to stricter A1R construction. |
| Local source-CAD audit containers v13/v14 | Operationally retired | Windows bind-mount STEP reads caused severe I/O stalls; v14 also crossed the host-memory floor. Led to host-local staging and stricter memory policy. |
| Local v15 source-CAD audit | Source-CAD work only | Reduced resources and contributed evidence, but did not produce an accepted H5M. |
| Remote refacet v1 | Setup failure | `sh`/`pipefail` launch mismatch and then root reuse. Create-only attempt identity became mandatory. |
| Remote refacet v2 | False source-order failure | PyMOAB EntityHandle/tag adapter misuse. Native source topology needed an isolated diagnostic first. |
| Native diagnostic v2 | Failed | EntityHandle-array adapter error. Corrected only in a new v3 root. |
| Remote refacet v3 | Import-match failure | Nested CadQuery bounding boxes were not a stable source invariant; source solid 0 had no Gmsh match. |
| Remote refacet v4 | Runtime failure | Default `python3` lacked PyMOAB. Exact interpreter and full native stack must be bound before scientific work. |
| Runtime preflight v1 | Setup failure | SSH broker/shell launch ended before a selectable start receipt. |
| Runtime preflight v2 | Runtime-receipt failure | Treated `OCP` as a package and attempted `OCP.OCP`; it is a single extension module in this environment. |
| Runtime preflight v3 | Passed | Corrected extension discovery and exact seven-module inventory. |
| Refacet v5 coarse | Terminal incomplete failure | Copied eight inputs but produced no H5M, manifest, or seal before/around the bounded timeout. Python buffering and wrapper structure did not preserve exact terminal return code/resource use. |

Refacet v5 exact terminal classification:

```text
FAIL_CLOSED_TERMINAL_AFTER_TIMEOUT_BOUNDARY_INCOMPLETE
```

It left exactly eight partial input copies totaling 331,691,833 bytes and no `dagmc.h5m`, manifest, seal, refined result, or OpenMC successor. The preserved stale start receipt must remain unmodified. Honest terminal evidence is in:

- `reports/geometry_recovery/REFACET_V5_COARSE_TERMINAL_FAILURE.json`, SHA-256 `0fd3d7ca98440640312996fd48502d95d3a420ed9db72158f21596a93b9503ea`
- `reports/geometry_recovery/REFACET_V5_INDEPENDENT_FAILURE_QUALIFICATION.json`, SHA-256 `61d7e939535a4eda6c5e2fe158333b184c20676085b37e61636a8b1dfebaa7a2`
- stale start receipt `REFACET_V5_COARSE_ATTEMPT.json`, SHA-256 `f0cd6d30915245a21eda33c41147b8b716e285ea1528ff2df7af78797cf78739`

## What worked scientifically

- The untouched public model was independently reproduced, so its defect is not attributable to our feature code.
- The public model's exact intersections and their cause were localized.
- A minimal 50-to-40 cm public shield correction exists as a clean scratch commit and has supporting source-CAD evidence.
- Clearance-derived local breeder reduction produced a public-family source-CAD assembly with all 276 protected pair checks passing and at least 5 cm clearance.
- Original public magnets remained one homogenized solid each; no arbitrary transform was introduced.
- Native H5M material/volume ordering and source-solid mapping were independently repaired and qualified.
- The exact Bateman CAD/DAGMC Python stack was identified, hashed, and validated.
- Remote CPU leases respected the Polytechnique 25% aggregate limit; no automatic successor was launched.
- Failures were quarantined with distinct roots instead of being overwritten into apparent successes.
- The WISTELL-D input pair was found, hash-reconciled across repositories, and proven different from the public example pair.

## What has not been completed

### Geometry acceptance

- No WISTELL-D source-CAD candidate has yet been reconstructed and audited under the Prompt 7R acceptance criteria.
- No accepted final H5M exists.
- No accepted geometry fingerprint/manifest exists.
- No accepted H5M has passed matched coarse/refined faceting, native topology, watertightness, overlap, magnet-layer interface closure, source-domain, and OpenMC navigation gates.
- No accepted geometry has been compared against the separate vanilla public negative reference with matched cameras and semantic geometry tables.

### Prompt-1 and producer rebinding

- Geometry-neutral Prompt-1 utilities have not been selectively rebound to an accepted WISTELL-D H5M.
- Stable semantic IDs for all accepted WISTELL-D components and the continuous magnet layer have not been frozen in an accepted H5M.
- The accepted geometry interchange, activation-ready metadata, and test-fixture package do not exist.
- Old split-feature utilities have not been fully triaged into reusable geometry-neutral pieces versus rejected physical assumptions.

### Instrumentation

- Materials/nuclear data for the accepted geometry are not frozen.
- Blanket, shield, vessel, and continuous-magnet-layer neutron/photon tallies are not bound to accepted cells.
- Representative local meshes are not bound to accepted magnet frames.
- The boundary bank over the complete continuous magnet-entry interface is not implemented on accepted geometry.
- No accepted boundary bank has passed facet localization, signed-surface handling, raw-weight normalization, capacity/completeness classification, or same-run tally equality.

### Bounded OpenMC and activation

- Accepted-geometry S0 geometry debug has not run.
- Bounded S1 tally smoke, S2 boundary/coupled-photon smoke, and S3 independent replica have not run.
- Accepted-geometry MicroXS, cell-R2S, IndependentOperator, or shutdown-photon smoke has not run.
- Source-mesh convergence and medium statistics are intentionally not started.

### Visualization and software closure

- The Prompt 7R package of at least 100 high-resolution PNGs, inspectable VTK/XDMF products, and contact sheets is not complete.
- The final full ParaStell test, format, build, CLI/import, and clean-branch gates have not run.
- Nothing from this principal branch has been committed or pushed.
- The worktree is currently dirty with extensive modified and untracked source, tests, scripts, and reports.

## Current worktree state

Principal worktree:

```text
D:\parastell-worktrees\magnet-radiation-valid-baseline-and-instrumentation
```

The branch still points to `dd3c2c53474980b6527d6ad40e8e2a638337d107`. All Prompt 7R work is uncommitted. Preserve unrelated changes and do not run destructive Git cleanup.

Important report root:

```text
D:\parastell-worktrees\magnet-radiation-valid-baseline-and-instrumentation\reports\geometry_recovery
```

Important external artifact root:

```text
D:\parastell-artifacts\geometry-recovery-20260827
```

## Recommended restart decision

Pause the A1R/v6 **public-family** scientific lane. Do not discard its code or evidence. Retarget only its geometry-neutral execution, native-topology, faceting, OpenMC-debug, and evidence machinery to the selected WISTELL-D contract.

The selected source baseline is now:

```text
hash-bound WISTELL-D VMEC and coil source
+ 45-degree bounded NWL radial build
+ continuous 30 cm magnet winding-surface layer
+ 452-point positive thickness fields
+ source-CAD zero-overlap validation
+ immutable post-export H5M
+ instrumentation outside the H5M
```

Do not transplant the public example's 40 cm shield edit or A1R breeder field into this geometry. Do not replace its continuous magnet layer with 18 swept coils inside the baseline lane. The immediate task is to create and qualify a DAGMC H5M from the exact selected STEP/source contract.

## Ordered remaining work

### 1. Freeze the intended WISTELL-D source contract

- Treat `WISTELL_D_AUTHORITATIVE_SOURCE_CONTRACT.json` as the current identity receipt.
- Copy the six exact authoritative files and three boundary arrays into a create-only external reference root before H5M production.
- Rehash every copied input and require equality with the contract.
- Add the source-mesh definition and its hash when it is constructed from this same 45-degree VMEC contract.
- Record every deliberate difference from the historical Blanket workflow and the later DPA_workflow adaptation.
- Do not call the result exact paper WISTELL-D unless the paper/source metadata support that claim.

### 2. Reproduce vanilla WISTELL-D source CAD in a separate read-only lane

- Reproduce `nwl_baseline_bounded` from the frozen builder and inputs without instrumentation.
- Require the regenerated thickness arrays, component STEP files, combined STEP, and manifest to match the frozen contract or explain any deterministic serialization difference through semantic geometry checks.
- Repeat solid validity, positive volume, layer order, all protected source-CAD intersections, volumes, centroids, bounding boxes, and boundary residuals.
- Preserve the magnet representation as the continuous 30 cm winding-surface layer.
- Stop fail-closed if no source-CAD candidate passes.

### 3. Select one geometry candidate at source-CAD level

- The `nwl_baseline_bounded` source-CAD candidate is selected.
- Freeze its exact physical-change rationale, material/component order, 452-point fields, and source-CAD validation receipt.
- Retain the public vanilla, public 40 cm fix, and A1R only as negative/method controls.
- Do not reopen source-CAD model selection unless the selected candidate fails DAGMC or OpenMC gates for a proven physical reason rather than an exporter/runtime defect.

### 4. Produce and qualify the H5M

- Finish the v6 operational harness only after retargeting it to the selected WISTELL-D inputs.
- Rerun the interrupted focused test set first:

```text
python -m pytest -q tests/test_refacet_gmsh_pipeline.py tests/test_source_cad_refaceting.py tests/test_refacet_execution.py
```

- Complete runtime receipt v1.2, diagnostic runner/report, scratch staging, create-only publication, bounded shell launcher, exact return-code/resource capture, and post-exit collection.
- Run the import/fragment diagnostic first. Independently validate it.
- Run coarse refacet only after that diagnostic passes. Do not launch refined automatically.
- Run refined refacet only after coarse native qualification passes.
- Apply faceting convergence tolerances and select one H5M.

### 5. Complete geometry gates G2-G6

- PyMOAB and PyDAGMC reload.
- Native watertightness and topology.
- Native overlap check with every report spatially classified.
- Surface senses, material groups, volumes, areas, point-in-volume samples, and deterministic rays.
- The complete inner and outer boundaries of the continuous magnet layer close with correct senses and vector-area closure; the inner boundary is the producer coupling interface.
- Source mesh remains inside plasma/source domain with correct rate/sector semantics.
- Two independent bounded OpenMC geometry-debug seeds have zero lost particles and zero navigation errors.

### 6. Rebind Prompt-1 geometry-neutral capabilities

- Review individual utilities from prior branches; never cherry-pick the old feature history wholesale.
- Rebind geometry interchange, activation metadata, stable IDs, local frames, volume-field schema, boundary schema, and neutral radiation bundle to the accepted geometry fingerprint.
- Enforce `continuous_magnet_layer_inner_boundary` as the global coupling role.
- Reject any bundle that declares casing/winding interfaces for this global model.

### 7. Add accepted-geometry instrumentation

- Freeze global magnet material, insulation treatment, photon production, and exact nuclear-data hashes.
- Add all required volume tallies for blanket, shields, vessel, and the continuous magnet layer.
- Add surface-parametric/local mesh tallies on the magnet layer without claiming per-coil resolution.
- Add per-surface neutron/photon currents, spectra, `mu`, azimuth, heating, and uncertainty bookkeeping.
- Record one bank on the complete magnet-layer entry interface, preserving facet ID, barycentrics, global/local or surface-parametric coordinates, outward normal, particle, direction, energy, raw weight, history normalization, and signed crossing sense.
- Keep canonical bank weights raw OpenMC weight divided only by exact source histories.

### 8. Run only the bounded functional campaign

- S0 geometry debug.
- S1 small tally smoke.
- S2 boundary plus coupled-photon smoke.
- S3 independent-seed replica.
- Classify banks as complete, sampled, or truncated-invalid.
- Require same-run tally/bank agreement as an integrity check only.
- Stop before source-mesh convergence or medium statistics if any geometry or interface gate fails.

### 9. Run bounded activation rebinding

- Use volume scalar flux for activation; never substitute the boundary bank.
- Perform only the bounded MicroXS, cell-R2S, IndependentOperator, and shutdown-photon smokes required by Prompt 7R.
- Keep DPA_workflow edits in an isolated worktree and preserve its existing dirty planning files.

### 10. Build the visual evidence package

- Show the public 50 cm negative reference and its four intersections.
- Show the public 40 cm control correction.
- Show the actual WISTELL-D VMEC/coils and all candidate radial builds.
- Use matched cameras/colors for source, candidate, accepted, and instrumented comparisons.
- Show every accepted magnet envelope, surface IDs, normals, source mesh, overlap-clearance witnesses, tally meshes, and boundary-entry locations.
- Produce at least 100 PNGs, at least four VTK/XDMF products, and at least six contact sheets, each hash-bound to its input geometry.

### 11. Close tests, reports, and project control

- Run full supported tests, formatting, compile, package build, CLI, import, and diff checks.
- Commit only tested source, tests, compact reports, and selected figures.
- Push only the user-owned feature branch; no PR.
- Update `D:\2026_DPA\project_control` only after geometry and bounded functional gates are truly complete.
- Keep final classification at `READY_FOR_SOURCE_MESH_AND_MEDIUM_STATISTICS`; do not claim production readiness.

## Project-control implications

The project-control critical path is still correct but needs a sharper first step:

```text
select the correct WISTELL-D identity
-> accept its geometry
-> converge its source mesh
-> qualify global and boundary statistics
-> freeze volume field and external bank
-> unlock activation/PKA and local transport lanes
```

The currently completed cross-workflow capabilities are software/method assets, not end-to-end scientific completion:

- producer schemas for D-T sources, coupled neutron/photon tallies, volume fields, local fields, boundary phase space, and neutral bundles;
- bounded historical 5k, replica, photon-calibration, and 500k runs on rejected/unqualified geometry;
- bounded activation and ALARA data/software paths;
- planar local OpenMC/Geant4/MCNP verification assets;
- OpenSn software installation/tests and later normalization diagnosis;
- PKA/response-fold software and research outputs.

None removes the need for an accepted global WISTELL-D geometry. The accepted volume-field and boundary-bank hashes remain the synchronization point for the downstream lanes.

## Resource and coordination state

- No Prompt 7R local heavy job is currently represented as valid and running in this handoff.
- Refacet v5 is terminal; its PIDs are gone and its lease is reaped.
- Last authoritative Bateman accounting associated with this lane: 256 scheduler-visible threads, 64-thread Codex limit, zero active/reserved owner leases, one unrelated observed thread, 63 available. This is historical status, not authority for a new launch.
- Every new remote action requires a fresh live ledger, headroom, process, input-hash, and root-absence check.
- Use the installed `ssh-poly` and `parastell-bateman` procedures, never ad hoc repeated SSH.
- Do not exceed 25% of scheduler-visible cores on any Polytechnique host across all Codex tasks.
- No automatic successor is authorized.

## Evidence index

Primary Prompt 7R reports already present:

- `PREFLIGHT.md`
- `VANILLA_REFERENCE_REPOSITORY.json`
- `PUBLIC_REFERENCE_OVERLAP_QUANTIFICATION.md/.json`
- `HISTORICAL_ARTIFACT_REAUDIT.json`
- `CANDIDATE_ACCEPTANCE_CRITERIA.json`
- `CANDIDATE_A1R_SOURCE_CAD_ACCEPTANCE.md/.json`
- `FACETING_COMPARISON_PROTOCOL.json`
- `ABORTED_SOURCE_CAD_ATTEMPTS.json`
- `RUNTIME_STACK_PREFLIGHT_ATTEMPTS.json`
- `REMOTE_REFACETING_ATTEMPTS.json`
- `REFACET_V5_COARSE_TERMINAL_FAILURE.json`
- `REFACET_V5_INDEPENDENT_FAILURE_QUALIFICATION.json`
- `REFACET_V6_RECOVERY_DESIGN.md/.json`
- `REFACET_V6_DESIGN_REVIEW.json`
- `PARALLELIZATION_REVIEW_LEDGER.md`

Scratch identity and fault-analysis evidence:

- `D:\Scratch\wistell_d_blanket_parastell_comparison_20260827.md`
- `D:\Scratch\wistell_d_parastell_audit_20260827\reports\FINAL_WISTELLD_PARASTELL_AUDIT.md`
- `D:\Scratch\wistell_d_parastell_audit_20260827\reports\FINAL_WISTELLD_PARASTELL_AUDIT.json`
- `D:\Scratch\parastell`, branch `fix/example-shield-clearance-40cm`, commit `d480424e53747691075d6b0b074face03efd311f`
- `D:\Scratch\wistell_d_parametric_parastell_20260827_01`
- `D:\2026_DPA\DPA_workflow\examples\wistell_d`

## Non-negotiable warnings for the next agent

- Do not call the public ParaStell example WISTELL-D.
- Do not mix the public VMEC with the WISTELL-D coils, or vice versa.
- Do not treat byte-identical public reproduction as physical acceptance.
- Do not continue refaceting A1R merely because its source CAD passes; first confirm that it is the intended device.
- Do not change magnet thickness when applying the saved scratch fix; that commit changes the public `shield` from 50 to 40 cm.
- Do not assume the public 40 cm correction is automatically the right WISTELL-D repair.
- Do not restore the casing/winding split in the global model.
- Do not mutate an accepted H5M to add tally surfaces; use the complete native magnet boundaries and OpenMC configuration outside the H5M.
- Do not accept an H5M on watertightness or zero lost particles alone; overlap, topology, senses, envelope closure, source domain, and faceting convergence are all required.
- Do not launch refined meshing, OpenMC transport, source convergence, medium statistics, or production work automatically.
- Do not overwrite failed attempt roots or rewrite stale receipts into successes.
- Do not update project-control completion statuses from plans or code alone.

## Final handoff status

```text
PUBLIC_PARASTELL_VANILLA_REPRODUCED = YES
PUBLIC_PARASTELL_PHYSICALLY_VALID = NO
PUBLIC_50_TO_40_CM_SHIELD_FIX_LOCATED = YES
PUBLIC_40_CM_CONTROL_SOURCE_CAD_SUPPORTED = YES
WISTELL_D_INPUT_ASSETS_FOUND = YES
WISTELL_D_INPUTS_DIFFER_FROM_PUBLIC_EXAMPLE = YES
WISTELL_D_SOURCE_CONTRACT_FROZEN = YES
WISTELL_D_45_DEG_SOURCE_CAD_PHYSICAL_GATE = PASS
WISTELL_D_90_DEG_SOURCE_CAD_PHYSICAL_GATE = NOT_BUILT
WISTELL_D_TRANSPORT_EXTENT = ONE_FULL_90_DEG_PERIOD
WISTELL_D_MAGNET_REPRESENTATION = CONTINUOUS_30_CM_LAYER
SPLIT_CASING_WINDING_MODEL_REJECTED = YES
A1R_PUBLIC_FAMILY_SOURCE_CAD_GATE = PASS
A1R_ACCEPTED_H5M = NO
REFACET_V1_TO_V5_SELECTABLE = NO
REFACET_V6_DESIGN_REVIEW = PASS
REFACET_V6_IMPLEMENTATION_COMPLETE = NO
ACCEPTED_GLOBAL_H5M = NO
OPENMC_ACCEPTED_GEOMETRY_DEBUG = NOT_RUN
PROMPT1_REBIND_ON_ACCEPTED_GEOMETRY = NOT_RUN
ACCEPTED_GEOMETRY_INSTRUMENTATION = NOT_RUN
ACCEPTED_BOUNDARY_BANK = NOT_RUN
ACCEPTED_GEOMETRY_ACTIVATION_SMOKE = NOT_RUN
PROMPT7R_VISUAL_PACKAGE = INCOMPLETE
READY_FOR_SOURCE_MESH_AND_MEDIUM_STATISTICS = NO
FINAL_DECISION = BLOCKED_GEOMETRY_PARITY
```
