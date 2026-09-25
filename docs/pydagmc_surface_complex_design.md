# PyDAGMC surface-complex API design note

This note records a possible reusable PyDAGMC API extracted from ParaStell's
native port assembly. It is a design proposal only; this branch does not
modify or submit changes to PyDAGMC.

## `Model.from_surface_complex(...)`

Accept an ordered vertex ledger, an ordered facet ledger, volume records, and
surface records. A surface record supplies one triangle-index array and up to
two volume senses. The constructor should create each physical surface exactly
once, reuse the caller's vertex identities, assign `GLOBAL_ID`, `CATEGORY`,
`NAME`, and `GEOM_DIMENSION`, and optionally attach material/component groups.
It should reject zero-area or duplicate facets before writing MOAB entities.

The generic portions of `NativePortSurfaceComplex._build_dagmc_model` can be
extracted: deterministic coordinate-to-vertex allocation, triangle creation,
surface meshset creation, volume parent/child links, sense assignment, and
group construction. Port loops, radial layers, and ParaStell component classes
must remain outside PyDAGMC.

## `Model.validate_geometry(...)`

Return a structured report and optionally raise on configured failures. The
generic checks are mandatory tag presence, unique IDs, parent/child
consistency, valid two-volume senses on internal surfaces, one ownership site
per triangle, positive triangle area, referenced vertices, duplicate
coordinate facets within a physical tolerance, and closed oriented volume
shells. External executables such as `check_watertight` and `overlap_check`
should be integrations that record command, version, return code, and output,
not hidden repair steps.

The implementation in `parastell.dagmc_assembly.audit_dagmc_model` is already
independent of ParaStell geometry classes and is the main extraction candidate.

## `Model.close_with_graveyard(...)`

Reject input models that already contain a conflicting `mat:Graveyard` group.
After all physical submodels are combined, find every physical exterior
surface with one missing sense, create one graveyard volume, assign it as the
missing sense, add a caller-selectable enclosing boundary, and leave only that
outer boundary one-sided. Create exactly one material group and return the new
volume/surface IDs and bounding box.

`parastell.dagmc_assembly.close_with_graveyard` contains no port-specific
types. Its axis-aligned box construction could move directly, while a reusable
API should also accept a callback or explicit outer surface complex for
non-box graveyards.

## Proposed data boundary

PyDAGMC should depend only on NumPy arrays, MOAB handles, stable string names,
integer geometry IDs, and material tags. It should not import `InVesselBuild`,
port specifications, CadQuery, Gmsh, or ParaStell's component ledger. ParaStell
would remain responsible for producing the conformal surface complex and for
mapping domain components to these generic records.
