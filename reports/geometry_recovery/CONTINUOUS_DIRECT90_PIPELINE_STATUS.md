# Authoritative continuous direct-90 pipeline

Status: `NATIVE_GEOMETRY_PASS_SOURCE_AND_OPENMC_GATES_IN_PROGRESS`

This lane binds the accepted WISTELL-D definition to one direct ParaStell
0--90 degree field period containing exactly nine nested radial volumes. The
last volume is the continuous 30 cm homogenized magnet layer. The coil file is
upstream provenance for the radial envelopes; it does not create 18 swept
global coil solids.

The source-CAD audit requires one valid positive-volume STEP solid for each
ordered component, exact zero-overlap common-volume checks for all eight
adjacent pairs, a SHA-256-chained cumulative-boundary construction proof, and
strictly positive intervening layers for all 28 nonadjacent pairs. Source
artifacts are hashed before and after the audit.

The DAGMC exporter ran create-only and imprinted exactly the nine radial
volumes. It requires only adjacent shared interfaces, verifies closed OCC
shells and both periodic ends, and writes no tally or coupling solid into the
H5M. The exact immutable H5M has SHA-256
`d31ea04e4d2fda78870db1688d6cc9215079e20408393c5f5bd374d58f43eaf3`.
It passed native reload, watertightness with zero unmatched edges and zero
unsealed surfaces/volumes, and overlap checks at precisions 1, 2, and 4 with
zero overlap locations. OpenMC navigation remains pending an accepted
contained source mesh. The existing
26-volume `export_parametric_direct90_dagmc.py` is an alternate swept-coil
candidate and is not accepted or relabelled by this lane. The existing
hash-bound, 32-core-lease, 21,600-second terminal wrapper now admits either
exporter by its exact repository filename while preserving distinct module
identities and the no-automatic-successor rule.

The surface manifest records every surface of the closed continuous magnet
volume for complete crossing capture. It separately selects the physical
vacuum-gap-to-magnet interface as
`continuous_magnet_layer_inner_boundary` for incoming replay. The inner-face
subset is explicitly not claimed to be a closed 3-D manifold; closure belongs
to the complete boundary containing inner, outer, and periodic/end surfaces.
Surface adjacency and outward signs come from native DAGMC forward/reverse
topology, and the H5M hash must remain unchanged.

The complete continuous-magnet capture boundary is surfaces 22, 25, 26, and
27; surface 22 is the separately declared incoming gap-to-magnet replay
interface. Its hash-bound surface manifest passed without changing the H5M.
No OpenMC transport or automatic production successor has been executed.
