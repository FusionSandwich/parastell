# Actual ParaStell radial PLC repair

Date: 2026-08-24

## Result

The repaired six-case in-vessel radial PLC matrix passes. Every case has zero
retained edge/facet crossings, completes Gmsh discrete tetrahedralization, and
passes the independent volumetric mesh audit. The exact terminal classification
is `PASS_NO_REPRODUCIBLE_PLC_FAILURE`.

This is PLC/topology evidence only. It does not qualify a transport geometry.
No magnets, full assembly, OpenMC model, particle histories, or reactor
rendering were constructed.

The matrix ran from branch `ports/actual-plc-repair-20260824` at commit
`2fb50632ca47d2df7fa0e2afed4fd99bbe477ee9`. It used the same frozen local
`parastell-damage-full:0.3.0` image, network disabled, with 6 GB and 4 CPU
limits. Geometry and intersection tolerances were unchanged from the diagnostic.

## Repairs

Two independent defects were repaired in the authoritative native surface
complex builder.

First, radial quads now expose explicit shared-ledger diagonal policies. The
13x49 plasma/SOL crossing was caused by the primary a-c diagonal near the two
toroidal sector edges; the localized `sector_edge_alternate` policy replaces it
only in the implicated edge bands. The unported 21x81 boundary-recovery failure
closes under the alternate b-d split. Tests retain the primary policy as the
general default and verify the localized policy exactly.

Second, the port patch bridge no longer aligns its cyclic boundary from one
nearest three-dimensional point. It minimizes the complete cyclic mismatch in
the port's local transverse plane. The previous single-point anchor was three
vertices out of phase on the breeder loop and folded the bridge through itself.

Gmsh Algorithm 1 was retained. Probes showed that a single diagonal policy is
not robust for all six discrete PLCs: some otherwise clean layouts leave a
zero-tag cavity in Gmsh's returned tetra connectivity. The case matrix therefore
records its validated deterministic policy as an input: A0/A1 use
`sector_edge_alternate`, B0/C1 use `primary`, and B1/C0 use `alternate`. Invalid
zero-tag connectivity was not filtered or reconstructed; only layouts passing
the full existing mesh audit were accepted.

## Repaired matrix

| Case | Port | Resolution | Diagonal policy | Crossings | Tetrahedra | Mesh audit |
|---|---:|---:|---|---:|---:|---|
| A0 | no | 13x49 | `sector_edge_alternate` | 0 | 85,886 | pass |
| A1 | yes | 13x49 | `sector_edge_alternate` | 0 | 87,694 | pass |
| B0 | no | 17x65 | `primary` | 0 | 102,227 | pass |
| B1 | yes | 17x65 | `alternate` | 0 | 102,540 | pass |
| C0 | no | 21x81 | `alternate` | 0 | 118,118 | pass |
| C1 | yes | 21x81 | `primary` | 0 | 120,358 | pass |

Every mesh audit reports zero inverted or zero-volume tetrahedra, duplicate
tetrahedra, nonconformal interface faces, disconnected regions, and configured
quality-threshold failures.

## Verification and artifacts

The final immutable artifact directory is:

`D:\openc-hts-dpa-data\geometry-diagnostics\actual-plc-repair-20260824-r2`

It contains 22 files totaling 388,731 bytes. The external master manifest is
8,802 bytes with SHA-256
`d1552af41800173495cabda768893fd99e7cebb91ab56d9a2cc94602759d8997`.
An independent read-only audit recomputed and matched all 21 master-manifest
records and all 12 output records in the six case receipts.

Focused native-port, port, and diagnostic regressions: 105 passed. The first
immutable repair attempt is retained without overwrite; it proved the geometric
crossings were gone but exposed the zero-tag Gmsh connectivity failure under a
single global diagonal policy.

The next stage may use this result as the radial-PLC gate, but magnets, global
assembly overlap checks, graveyard/exterior qualification, OpenMC geometry
debugging, and transport validation remain separate required gates.
