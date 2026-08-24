# Closed magnet boundary source (v2)

`parastell.magnet_boundary_source/v2.1.0` represents every retained crossing of a
closed magnet winding-pack envelope as one correlated record. Position,
direction, particle, energy, crossing sense, face role, and weight remain on
the same record. The multidimensional projection is derived from those records
without changing their weights. `v2.1.0` records whether optional fields such
as OpenMC history identity and per-record uncertainty are actually available;
unavailable fields are omitted instead of filled with fake values.

Canonical record weights are raw OpenMC source-bank transport weights divided
only by the exact source-history count. They are never rescaled to force tally
agreement. A tally-conditioned consumer distribution is allowed only as a
separate derived, noncanonical dataset with explicit provenance.

The tally and surface bank from one OpenMC run share histories. Their comparison
is therefore a **same-run integrity closure**, not an independent statistical
closure. Their uncertainties are reported separately because the covariance is
not available. Independent seeds are required for run-to-run reproducibility.

Every file reports one completeness status: `COMPLETE_CROSSING_BANK`,
`SAMPLED_CROSSING_BANK`, or `TRUNCATED_INVALID_BANK`. A cap-reached bank is
invalid for qualified replay. `time_s` means prompt particle flight time, not
irradiation, depletion, or cooling time.
and must conserve their integrated partial current.

The projection axes are fixed by the complete envelope plus the configured
spatial, particle-energy, and angular grids. Surfaces and bins with no sampled
crossings are retained explicitly with zero mean and uncertainty; projection
shape never depends on which bins happened to be populated in one Monte Carlo
run.

The production example retains a 1 cm radial clearance between the reactor
shield/vacuum-vessel build and the 5 cm magnet casing. Combined models default
to 5–20 cm CAD-to-DAGMC faceting; the looser 20–50 cm smoke tessellation can
hide near-tangent intersections and is not an overlap-acceptance mesh.

The normal is outward from the magnet DAGMC volume. Therefore
`mu = Omega dot n_outward` is negative for entry, positive for exit, and close
to zero for grazing crossings. No global coordinate sign is used to infer
sense. Every face stores its area, centroid, right-handed toroidal/poloidal/
normal frame, spatial patch edges, role, and OpenMC surface-sense sign.

The OpenMC-to-outward sign is derived from the target volume's DAGMC
forward/reverse sense. This is required for one-sided end faces, whose native
facet normal can point into the winding pack. Independent closure is read from
the named directional-current tally directly in the statepoint HDF5, so an
unrelated OpenMC 0.16 filter cannot prevent export. The manifest reports
entering, leaving, net, and total current with uncertainty and z-score for
every face and the whole envelope; the surface bank is never renormalized.

Large numeric record and projection arrays use lossless gzip compression.
Fixed zero bins remain present and deterministic solvers see the same dense
shape after HDF5 decompression.

Production neutron runs request OpenMC's exact `CCFE-709` edges. ParaStell
raises an error if that authoritative structure is absent. The default angular
grid has 26 mu bins, 16 azimuth bins, and six dedicated mu bins in
`abs(mu) <= 0.1`.

Absolute values are conditioned on an independent OpenMC current tally in
surface/particle/energy/sense strata. Records within each stratum retain their
joint distribution. A positive tally stratum with no records is a fatal error.

The source provenance contract requires a real `SourceMesh` generated from
the VMEC equilibrium with `default_plasma_conditions` and
`default_reaction_rate`; it records hashes and strength statistics. A fallback
point/ring source must be labelled software-only and cannot pass the production
source gate.

OpenMC transports neutrons and photons when photon data are configured. This
contract does not claim transport of charged secondaries. PKA, recoil,
charged-particle local deposition, and SRIM follow-on data are extension hooks,
not inferred DPA results.
