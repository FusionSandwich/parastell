# A1R source-CAD acceptance

Classification: `SOURCE_CAD_PHYSICAL_GATE_PASS`

The complete A1R source-CAD audit passed. It covered all 276 preregistered pairs: 108 magnet/component pairs, 15 component/component pairs, and 153 magnet/magnet pairs. There were no missing, unexpected, duplicate, malformed, or failing rows.

The smallest exact vacuum-vessel-to-magnet clearance is 5.239626484915899 cm, above the frozen 5.0 cm requirement. All 18 required closest-point witnesses are finite and valid. There are zero magnet/component, component/component, or magnet/magnet intersections, zero duplicate magnet shapes, zero distance errors, and zero witness failures.

The terminal audit SHA-256 is `e47f34884fc04b4a353b96a4e6a18928bdb4d501e26f3b6793b213c80ae2a9aa`; its seal SHA-256 is `397968a853f7c5a5d75d95c3143cebaa73b2280e8b6be30475a3a9db3d5ddd3c`. The seal binds the exact audit, implementation, frozen criteria, physical-change receipt, candidate/reference manifests, artifact hashes, and before/after/final immutability checks.

This accepts the source CAD, original homogenized magnet identities, and source-mesh identity only. It does not accept the existing candidate H5M or authorize transport. The next mandatory serial gate is to refacet the same immutable source CAD at both preregistered levels, then pass native topology/overlap, certified faceting, source-domain, and two-seed OpenMC navigation qualification.
