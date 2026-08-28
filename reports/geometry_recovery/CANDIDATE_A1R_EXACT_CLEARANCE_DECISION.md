# Candidate A1R exact-clearance decision

Decision: `PASS_EXACT_CLEARANCE_AS_CANDIDATE_ONLY`

The breeder-only 5.75 cm directional retarget is allowed to proceed to a full
source-CAD build and audit. It is not yet an accepted source-CAD model or an
accepted transport geometry.

The exact OpenCascade all-solid calculation measured 18 of 18 vessel-to-magnet
distances with no errors and valid closest-point witnesses. The global minimum
is 5.239626484915912 cm at the symmetry-related magnet-0008/magnet-0009 pair,
giving a 0.239626484915912 cm margin above the immutable 5.0 cm requirement.
Independent Euclidean reconstruction of every recorded witness pair agrees
with the reported distance; the maximum absolute residual is
3.552713678800501e-15 cm.

The receipt is
`D:/parastell-artifacts/geometry-recovery-20260827/candidate_A1_target575_exact_clearance_20260827T045721/exact_global_clearance.json`
(SHA-256
`0d263c410791ad6dafe823d17b1de96a825f6e19860b775cde1f1965f2a5573f`).
It is bound to construction manifest
`85b1c74fcd21122774649a60d993fa43562797d3ea468ca6a555a1f8d7227491`
and acceptance criteria
`092315725cfd06e64fa403cc8f484a8135f4920a5af93bcdccc797dfb41134ca`.

This decision does not substitute for the hardened 276-pair source-CAD audit,
the coarse/refined DAGMC gates, all-magnet envelope and ray checks, the source
domain audit, faceting parity, or the two bounded OpenMC navigation replicas.
