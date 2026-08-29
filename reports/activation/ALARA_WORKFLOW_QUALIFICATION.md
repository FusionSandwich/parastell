# ALARA activation workflow qualification

The ParaStell-owned ALARA workflow now passes a bounded synthetic runtime
smoke. This proves the handoff, 175-group ordering, exact isotope naming,
activation schedule rendering, repaired ALARA executable, FENDL library
binding, activity/heat reporting, and delayed-photon export. It is not a
WISTELL-D activation result.

The accepted runtime is ALARA 2.9.2 from `FusionSandwich/ALARA` commit
`4d01679a9837d9e8a2882c7efa71bc0b5f9ade64`. The executable SHA-256 is
`a28b8413a829e2df22e2a6a26d67328275511bfd66b6f85271b108a1c831e2d0`.
The required isotope-name compiler repair is independently bound by patch
SHA-256
`cf93de3e57c52ed4a26a731daed94ceaf458ed79528b7dbcc78a173830d6fd0b`.
The four FENDL/A-2.0 plus FENDL/D-2.0 binary components and `nuclib.std` are
all hash-bound in the JSON qualification report.

Attempt 1 is preserved as a failed input-adapter reproducer. Rendering all
times as long raw-second strings overflowed ALARA's fixed-width report
formatter. The adapter now retains authoritative seconds in the JSON schedule
but renders exact compact ALARA tokens such as `1 d`, `1 w`, `1 y`, `5 y`, and
`10 y`. Attempt 2 used a fresh root and passed.

The passing one-zone Cu-63 smoke used synthetic VITAMIN-J-175 flux. It emitted
16 well-formed isotope labels, zero pre-irradiation activity and decay heat,
positive shutdown and one-day activity and heat, and a nonempty delayed-photon
source. The output SHA-256 is
`7b096ba18268e3888fcbf2e0a6ac8b538504a3119d1ddfad452a23fcf45394d5`;
the create-only result receipt SHA-256 is
`88ec3f7342183e1334a29a08f56d281bb7db0b6974abbd1e6578c5be6e4d5e38`.

One physical gate remains: the final swept-coil 90-degree DAGMC geometry must
pass native and OpenMC navigation qualification, then a bounded OpenMC 0.16.0
run must supply the exact VITAMIN-J-175 scalar spectra and material/volume
ledger. Until then the ALARA classification is
`ALARA_WORKFLOW_SMOKE_PASS_PHYSICAL_WISTELL_D_INPUT_PENDING`.

No result or source from `wistell-d-openmc` qualifies this lane. MCNP was not
run. No production activation was authorized.
