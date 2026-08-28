# WISTELL-D activation integration

The depletion data were found locally and their exact identities are now bound
to a new geometry-neutral ParaStell producer contract. No activation transport
or depletion calculation was launched.

The accepted producer observable is whole-volume neutron scalar flux from an
OpenMC track-length tally, normalized per source history and then multiplied by
the independently bound physical source rate. Magnet boundary-current banks are
reserved for downstream local tape transport and are rejected as activation
input.

The old ParaStell activation metadata cannot be reused unchanged because it
requires 18 casing/winding-pack pairs. The correct direct-90 global WISTELL-D
geometry has one continuous 30 cm `magnet_envelope`. The replacement
`parastell.activation_ready_metadata/v2.0.0` contract therefore uses generic
activation domains and separately records DAGMC volume IDs and actual OpenMC
cell IDs.

The local ENDF/B-VIII.0 fast depletion chain and all 728 payloads referenced by
the transport index are content-hashed. Existing old-geometry MicroXS/depletion artifacts may
exercise the software path only; they cannot supply final physical results.

The intended source-rate scope is the modeled 90° period. The candidate source
mesh and strengths hashes sum to `9.427053032700795e19 s^-1`, but remain
`PENDING_DOMAIN_AND_ELEMENT_ORDER_AUDIT` and cannot unlock activation until a
hash-bound audit receipt passes. The old
Prompt-1B value `2.693734274881251e20 s^-1` is rejected for this rebinding.

Bounded activation remains locked until the direct-90 geometry passes native
and two-seed OpenMC 0.16.0 navigation gates and a geometry-bound medium-history
scalar-flux field exists. Production activation remains separately blocked
because the mixed transport index lacks complete evaluation metadata for 315
audited rows.

After the geometry gate, the remaining sequence is:

1. Discover actual OpenMC cell/material IDs and freeze the nine-volume map.
2. Bind the source mesh and scalar-flux tally/statepoint hashes.
3. Prove scalar-flux normalization closure for the whole magnet envelope.
4. Audit requested nuclide/reaction MicroXS coverage; missing support is never
   interpreted as zero.
5. Pass the bounded R2S and `IndependentOperator` comparison in `DPA_workflow`.
6. Keep delayed-photon activation products distinct from prompt photon fields.
