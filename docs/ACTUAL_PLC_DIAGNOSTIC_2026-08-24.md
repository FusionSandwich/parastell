# Actual ParaStell radial PLC diagnostic

Date: 2026-08-24

## Result

Primary classification: `BLOCKED_BASELINE_AND_PORTED_PLC_INTERSECTION`.

This result is a PLC/topology diagnostic. It does not qualify a transport
geometry. No magnets, full assembly, production volume mesh, OpenMC model,
particle histories, or reactor rendering were constructed.

The final diagnostic ran from branch
`ports/actual-plc-diagnostic-20260823` at commit
`b6568d0ddcbc1669ad2c80e1a972a24fda525003`. It consumed the repository VMEC,
radial-build YAML, canonical actual-port dictionary, and referenced coil file.
All inputs and relevant source modules are byte-hashed in the retained input
manifest. The already-local `parastell-damage-full:0.3.0` image was used with
network disabled and fixed 6 GB/4 CPU limits; no dependency was acquired.

## Six-case matrix

| Case | Port | Resolution | Retained crossings | Downstream result | Case classification |
|---|---:|---:|---:|---|---|
| A0 | no | 13x49 | 10 (evidence cap reached) | `PLC Error: A segment and a facet intersect at point` | `BLOCKED_BASELINE_RADIAL_PLC_INTERSECTION` |
| A1 | yes | 13x49 | 10 (evidence cap reached) | `PLC Error: A segment and a facet intersect at point` | `BLOCKED_PORT_STITCH_PLC_INTERSECTION` |
| B0 | no | 17x65 | 0 | Gmsh tetrahedralization and mesh validation passed | `PASS_NO_REPRODUCIBLE_PLC_FAILURE` |
| B1 | yes | 17x65 | 10 (evidence cap reached) | native Gmsh process terminated by signal 11 during boundary recovery | `BLOCKED_PORT_STITCH_PLC_INTERSECTION` |
| C0 | no | 21x81 | 0 | `Could not recover boundary mesh: error 2` | `BLOCKED_OTHER_REPRODUCIBLE_PLC_FAILURE` |
| C1 | yes | 21x81 | 10 (evidence cap reached) | `Could not recover boundary mesh: error 2` | `BLOCKED_PORT_STITCH_PLC_INTERSECTION` |

Success was determined from the returned mesh and validation result, not file
existence. The B1 signal termination is captured from a pre-Gmsh checkpoint,
process return code, stdout and stderr rather than misclassified as an input or
environment failure.

## Localization

The first A0 and A1 hit is identical. Edge 75 on `radial:plasma` crosses facet
2264 in A0 (2254 in A1) on `radial:sol` at
`[1265.3775675727838, 100.6184921509325, 149.22667648714304] cm`. The two
surface senses bind `plasma`/`sol` and `sol`/`first_wall`, respectively. It is
an adjacent-radial-surface crossing at toroidal cell 1 and poloidal cell 18,
with a 0.35295277340937475 cm local violation scale. It lies approximately
866.88 cm from the comparison patch boundary, proving that the 13x49 baseline
failure is outside the port patch.

The first B1 and C1 hit is also identical in physical coordinates. An edge and
facet on `radial:breeder` self-intersect at
`[443.0005469572796, 437.47029736404295, 0.5484655855004945] cm`. The facet
sense binds `breeder`/`back_wall`. The local violation scale is
0.2111081783976109 cm; the hit is 0.026133589524402995 cm from the liner
boundary and 2.0058905680969445 cm from the patch boundary. This retained
self-surface failure is present in the ported 17x65 and 21x81 cases but absent
from their unported intersection reports.

Every retained hit records vertex, edge and facet IDs; all edge-owning facets;
surface/component/layer and volume ownership; endpoints and triangle vertices;
normals and senses; Cartesian and local toroidal/poloidal/radial coordinates;
cell indices; port distances; violation scale; and the 1e-8 cm intersection
tolerance. Each crossing case exports only its implicated facets as JSON and
minimal legacy VTK.

## Verification and artifacts

The final immutable artifact directory is:

`D:\openc-hts-dpa-data\geometry-diagnostics\actual-plc-diagnostic-20260823-r3`

It contains 31 files totaling 486,650 bytes. The external master manifest is
12,842 bytes with SHA-256
`d36c9914d462584da80c445e5241946616ab5b5df5c94277b2cfa3de2a6cb497`.
An independent read-only audit recomputed and matched all 30 artifact records
in that manifest and every output record in all six receipts.

Focused diagnostic tests: 10 passed. Native-port and port regressions: 100
passed. Selected existing in-vessel/Gmsh regressions: 2 passed. The earlier
full scoped invocation also exposed 18 unrelated fixture path errors because
`test_parastell.py` expects the `tests` directory as its working directory;
rerunning its in-vessel selectors from that directory passed.

Two earlier immutable attempts are retained rather than overwritten. Their
manifests and supersession reasons are recorded in
`docs/geometry_diagnostic_manifest.json`.

## Next repair task

Repair the baseline radial surface complex first. Make the plasma/SOL boundary
use one shared vertex/edge/facet ledger with opposite volume senses, and verify
the A0 crossing disappears without changing tolerances. In the same repair
line, retain and diagnose the C0 21x81 boundary-recovery error because the
independent intersection test did not reduce it to a segment/facet crossing.
Only after the unported matrix passes should the port patch repair address the
`radial:breeder` self-intersection near the liner/patch boundary. Do not proceed
to magnets, external overlap qualification, OpenMC, or transport until those
PLC gates close.
