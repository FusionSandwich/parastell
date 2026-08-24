# Magnet Boundary Production Validation

This note records the 2026-08-24 production validation of the reactor-to-HTS
radiation-field workflow. Generated files live under the ignored directory
`validation_output/combined_production_multimagnet_20260824_v2` and are not
package data.

## Combined model and transport

- OpenMC 0.16.0, commit `617d35a5063c57796b43428bc401e627d2011046`.
- ParaStell VMEC D-T source mesh: 230,400 tetrahedra and
  `2.693734274881251e20` source particles/s.
- Combined sector: 42 physical volumes, one interstitial vacuum volume, and
  one post-assembly graveyard volume.
- Magnet inventory: 18 homogenized winding-pack volumes and 18 casings.
- DAGMC checks: watertight and native `overlap_check` reports no overlaps.
- OpenMC run: 250,000 histories, 10 batches, four threads, seed 24082026.
- Result: zero lost particles, 570 neutron and 34 photon surface records.

The independent neutron-incident `ParticleProductionFilter` tally scores
`6.52e-4 +/- 5.50e-5 photons/source` over all winding packs and
`1.12e-4 +/- 2.25e-5 photons/source` in magnet 20. The collision-track file
contains 5,335 records: 4,053 neutrons and 1,282 photons. Every photon record
has a parent ID and retains position, direction, energy, weight, material,
nuclide, and reaction MT. These collision records are diagnostic descendant
provenance; the production tally, not the collision count, is the production
estimator.

The explicit interstitial vacuum is required. Assigning every physical
one-sided surface directly to the graveyard kills valid source histories.
OpenMC's geometry-debug boundary-point check can report both adjacent DAGMC
cells at exact shared-surface crossings; the native DAGMC overlap check is the
authoritative overlap gate for this faceted model.

## Independent closure

An independent 250,000-history replicate used seed 8675311. The selected
magnet-20 envelope produced 176 comparisons. Four comparisons exceeded a raw
three-sigma criterion in the first run and two did so in the production
replicate. No comparison exceeded the two-sided Bonferroni familywise
threshold of 3.62939; the replicate maximum z-score was 3.14416. The bank was
not renormalized to the tally.

## Exported field

The bundle schema is `parastell.magnet_radiation_field_bundle/v1.0.0`; the
boundary schema is `parastell.magnet_boundary_source/v2.1.0`. The magnet-20
boundary preserves 87 continuous correlated records across ten surfaces and
16 local patches per surface. Each record retains particle PDG identity,
position, direction, energy, weight, surface, crossing sense, local frame,
and grazing metadata.

The volume field contains separate CCFE-709 and UKAEA-1102 neutron axes and a
42-group photon axis for all 18 magnets. Magnet 20 has an integrated neutron
scalar flux of `3.226459768861037e11 cm^-2 s^-1`. Independently scored heating
is 325.40034346032473 W from neutrons and 13313.31170364983 W from photons.

## Condensation and PKA interoperability

The measured-spectrum Pareto study selected 64 neutron groups at 0.141914%
maximum protected flux/heating-proxy error and 20 photon groups at 0.629040%
maximum protected error. Source-normalization errors are below `9e-16`.
These structures are qualified only for the measured scalar-flux,
energy-weighted-flux, and coarse heating-proxy responses used in the study.
Reaction-channel and PKA response preservation is not qualified.

SPECTRA-PKA commit `951d6fd82e29117cd97d72f9808c76f3de9d361c`
successfully folded the production CCFE-709 magnet-20 spectrum through its
bundled Zr recoil matrices. The reported total flux was `3.2265e11 cm^-2 s^-1`.
This is an integration proof, not an HTS PKA result: evaluated recoil matrices
for Cu, Ag, O, Y/RE, Ba, Ni, Cr, Fe, Mo, and the remaining configured tape
constituents were not available in the validation environment.

## Explicit multilayer replay

The continuous entering bank was conservatively projected to 16 ordinates and
replayed through Cu, Ag, REBCO, homogenized buffer, Hastelloy, rear Cu, solder,
and insulation. The reference solver supports unequal neutron/photon grids,
within-group and group-to-group scattering, neutron-to-photon production,
layer heating, and interface currents. The 64-neutron/20-photon production
replay converged in four source iterations with a particle-balance error of
`1.74987e-4`.

The replay uses bounded synthetic cross sections. It verifies coupling and
balance behavior but is not a design prediction. Evaluated multigroup
cross-section matrices and charged-particle transport remain future inputs.

## Software gates

- Final ParaStell suite: 178 passed, 8 skipped.
- Black: 92 files unchanged after formatting the new solver and ignored
  validation drivers.
- Source distribution and wheel builds pass; both CLI entry points and import
  pass.
- DPA transport/PKA/activation host subset: 199 passed, 3 skipped.
- DPA complete image suite: 968 passed, 20 skipped, 71 failed, with 23 modules
  additionally blocked at collection by absent `ase` and `torch`.
- The DPA failures are outside the radiation-bundle adapter and include
  atomistic/ML dependencies, frozen external artifact hashes, Docker-in-Docker
  workflows, and environment-specific paths.

## Scientific limitations

- Two winding packs had zero neutron score and four had zero photon score at
  250,000 histories; higher-statistics and variance-reduced runs are needed.
- The selected envelope has only 87 records, so fine spatial/angular bins are
  statistically sparse.
- Whole-winding-pack radial/toroidal/vertical frames do not resolve local tape
  twist continuously along a coil.
- HTS-constituent recoil matrices were unavailable, so no production PKA or
  DPA claim is made.
- The reference multilayer coefficients are synthetic and do not close a
  material-design heating or damage calculation.
- OpenMC transports neutrons and photons here, not all charged secondaries.
