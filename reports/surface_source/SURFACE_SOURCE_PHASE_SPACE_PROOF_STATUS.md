# Surface-source phase-space proof status

Current classification: `MECHANISM_PASS_GEOMETRY_PROOFS_PENDING`.

The exact OpenMC 0.16.0 runtime has passed two independent bounded checks. First,
OpenMC's native source serializer wrote neutron and photon records and the
geometry-neutral reader recovered position, direction, energy, flight time,
raw weight, delayed group, surface ID, and PDG particle identity exactly. This
used zero transport histories. Second, a 2,000-history vacuum-sphere transport
produced exactly 2,000 requested-surface crossings. The normalized bank current
was `1.0000000000000004` per source and the independently accumulated outgoing
surface-current tally was `1.0`; incoming current was zero. The configured bank
capacity was 4,000 records and was not reached.

The topology-localization unit contract also passes. It rejects global-coordinate
normal heuristics, requires normals derived from DAGMC forward/reverse topology,
maps each point to an existing facet, checks barycentric containment and
reconstruction residual, computes `mu = direction dot outward_normal`, classifies
incoming/outgoing/grazing crossings, and constructs a right-handed facet-local
frame.

These are mechanism proofs, not yet a claim about either requested reactor
geometry. The actual 90-degree WISTELL-D model remains gated on an accepted
watertight, overlap-free H5M and bounded OpenMC navigation. The private alternate
configuration is being tested only in external create-only artifacts; no private
input, result, or figure will enter Git. For each geometry, acceptance still
requires a closed envelope, exact requested surface IDs, a demonstrably complete
non-truncated bank, successful localization of every retained record, same-run
bank/tally integrity, and zero lost or navigation-error particles.

One native-format limitation is explicit: OpenMC 0.16 does not store a parent
history ID or polarization in `SourceParticle`. File/index record identity is
retained, but ancestry between multiple crossings from one transport history
cannot be reconstructed. This does not prevent replay of the requested scalar
neutron/photon phase-space measure; a future requirement for cross-record family
correlations would need separate history or collision instrumentation.

Machine-readable details and exact external receipt hashes are in
`SURFACE_SOURCE_PHASE_SPACE_PROOF_STATUS.json`.
