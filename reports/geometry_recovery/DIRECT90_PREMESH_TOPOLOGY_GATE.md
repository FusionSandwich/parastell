# Direct-90 pre-mesh topology gate

Status: `IMPLEMENTED_LOCAL_TESTS_PASS_REMOTE_EXECUTION_PENDING`

This gate belongs to the direct ParaStell 0–90° WISTELL-D geometry lane. It
does not use, modify, or depend on `wistell-d-openmc`.

The accepted source-CAD manifest remains SHA-256
`b6e723cdb9ac95d789a838abbf44590d210c4fdbe718c3b459777d38768e0499`.
The no-imprint H5M SHA-256
`b8e4ca84d211db3e9efc7d69e37282f7befe91be4149d8cd22e4d1c5874c542f`
remains a quarantined negative control because its native gate found 30
unmatched edges, six unsealed surfaces, two unsealed volumes, and sampled
overlaps. It is not eligible for OpenMC.

Before another imprinted surface-mesh attempt, the exporter now writes a
create-only `PREMESH_TOPOLOGY.json` certificate. It requires:

- the nine source components and physical volumes to remain bijective;
- every OCC surface to have one or two known owning volumes;
- every shared surface to join only consecutive radial components;
- all eight intended radial interfaces to exist and be connected;
- all internal components to have no nonperiodic owner-one surface;
- every component to expose both 0° and 90° cut planes;
- matching 0°/90° cut-face areas within frozen tolerances;
- every volume boundary to agree between upward and downward incidence;
- every volume shell to be connected;
- every shell curve to occur exactly twice with cancelling orientation; and
- one connected nonperiodic outer magnet envelope.

The status is deliberately limited to
`PREMESH_OCC_INCIDENCE_AND_MANIFOLD_PASS`. It does not qualify Gmsh nodes or
triangles, DAGMC senses or watertightness, overlap, source containment, or
OpenMC navigation. Those remain mandatory serial successor gates.

Gmsh documents that all 2-D unstructured algorithms first create a Delaunay
mesh containing the 1-D points and recover missing edges. It also describes
MeshAdapt as the most robust option for very complex curved surfaces. The
previous edge-recovery failure therefore does not justify an unqualified
algorithm switch; the source OCC topology must be certified first.

Verification on the feature worktree:

```text
focused exporter/wrapper tests: 35 passed
related geometry/source/activation/handoff tests: 120 passed
Black: PASS
compileall: PASS
git diff --check: PASS
independent read-only review: PASS for limited pre-mesh OCC certificate
remote geometry job: NOT_RUN
OpenMC transport: NOT_RUN
```

Reference: <https://gmsh.info/doc/texinfo/gmsh.html#Choosing-the-right-unstructured-algorithm>
