# Authoritative continuous direct-90 pipeline

Status: `IMPLEMENTED_NOT_EXECUTED`

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

The DAGMC exporter is create-only, imprints exactly the nine radial volumes,
requires only adjacent shared interfaces, verifies closed OCC shells and both
periodic ends, and writes no tally or coupling solid into the H5M. It remains
blocked on native DAGMC and OpenMC qualification after export. The existing
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

No CAD build, DAGMC mesh, native overlap check, OpenMC run, or automatic
successor was executed in this checkpoint.
