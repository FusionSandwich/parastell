# Closed magnet boundary source (v2)

`parastell.magnet_boundary_source/v2.0.0` represents every crossing of a
closed magnet winding-pack envelope as one correlated record. Position,
direction, particle, energy, crossing sense, face role, and weight remain on
the same record. The multidimensional projection is derived from those records
and must conserve their integrated partial current.

The normal is outward from the magnet DAGMC volume. Therefore
`mu = Omega dot n_outward` is negative for entry, positive for exit, and close
to zero for grazing crossings. No global coordinate sign is used to infer
sense. Every face stores its area, centroid, right-handed toroidal/poloidal/
normal frame, spatial patch edges, role, and OpenMC surface-sense sign.

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
