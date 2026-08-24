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
The reaction/production product uses schema
`parastell.magnet_reaction_production/v1.0.0`. It records `0.015964`
reaction events/source and `6.52e-4` produced photons/source across all 18
magnets while distinguishing produced particles from transported particles.

## Condensation and PKA interoperability

The initial measured-spectrum Pareto study selected 64 neutron groups for
flux/heating responses, but a real SPECTRA-PKA fold rejected that grid: total
YBCO recoil-rate error was 18.24% and normalized recoil-distribution distance
was 23.24%. The production selector therefore promotes the smallest candidate
meeting 1% limits for all protected PKA observables. That grid has 256 neutron
groups; its recoil-rate error is 0.0134%, mean-recoil-energy error is 0.0133%,
distribution distance is 0.7614%, and maximum species-fraction error is
0.0091%. The selected photon grid remains 20 groups at 0.629040% maximum
protected error. Source-normalization errors are below `9e-16`.

SPECTRA-PKA commit `951d6fd82e29117cd97d72f9808c76f3de9d361c`
successfully folded the production magnet-20 spectrum through 13 natural
isotope recoil matrices for YBa2Cu3O7: O-16/17/18, Y-89, Ba-130/132/134-138,
and Cu-63/65. The retained TENDL/NJOY bundle contains 287 files totaling 9.37
GB. CCFE-709, independently tallied UKAEA-1102 rebinned to CCFE-709, and
64/128/192/256/384/512-group reconstructions were executed. UKAEA-1102 agrees
with the CCFE reference to `2.5e-7` in total recoil rate and `1.83e-4` in
distribution distance. SPECTRA-PKA is angularly marginalized and the 40 eV
displacement threshold used for diagnostics is not a validated
constituent-specific damage model; these results qualify transport-grid PKA
preservation, not a complete DPA or property-degradation prediction.

## Explicit multilayer replay

The continuous entering bank was conservatively projected to 16 ordinates and
replayed through Cu, Ag, REBCO, homogenized buffer, Hastelloy, rear Cu, solder,
and insulation. The reference solver supports unequal neutron/photon grids,
within-group and group-to-group scattering, neutron-to-photon production,
layer heating, interface currents, transmitted current, and reflected entrance
current. The 256-neutron/20-photon evaluated replay converged in ten source
iterations with a particle-balance error of `8.71e-11`.

The coefficients are collapsed from the exact NNDC OpenMC HDF5 library used by
transport for representative layer compositions. Non-absorption is currently
represented as within-group diagonal scattering; evaluated energy-angle
redistribution and neutron-to-photon production matrices are not yet supplied.
Heating is absorption-energy bookkeeping and excludes explicit electron
transport, so the replay remains a reference integration model rather than a
material-design prediction.

## Activation handoff

Magnet 20's CCFE-709 physical neutron spectrum was exported through the
`parastell.activation/v1.0.0` contract to versioned JSON, ALARA flux, and
FISPACT-II arbitrary-group flux files. All three preserve the
`3.226459768861037e11 cm^-2 s^-1` group-integrated normalization. The local
activation environment has a 16 GB TENDL-2017 FISPACT data package, but no
ALARA or FISPACT-II executable was found locally, in the full-run image, or in
the bounded Bateman audit. OpenMC 0.16.0 also lacks the optional newer R2S APIs
checked by the adapter.

OpenMC 0.16 independent depletion was executed separately using the exact
CCFE-709 magnet-20 spectrum, the NNDC transport library, and the local
ENDF/B-VIII.0 fast chain (SHA-256
`5eeb727498d824d7c951ad89864bbc1c2d76ec5e8c9097a820505213ba6a2bf3`).
The diagnostic schedule is one day irradiation followed by one- and seven-day
cooling intervals in a homogeneous one-cubic-centimeter YBCO reference. The
inventory contains 35 nonzero nuclides; leading end-of-irradiation products
include Cu-64, Y-90, Ba-131, Ba-133, metastable Ba-135, Ba-139, and Cu-66.
Twelve of 13 input isotopes have nonzero collapsed reactions. O-18 has zero
coverage because the selected NNDC transport library does not provide usable
O-18 neutron data. This is a real bounded OpenMC activation calculation, but
the schedule and homogeneous reference are diagnostic rather than a reactor
operating-history prediction. ALARA and FISPACT-II remain validated input-only
paths until their executables are supplied.

## Software gates

- Final ParaStell suite: 186 passed, 8 skipped.
- Black: 103 files unchanged after formatting the new solver and ignored
  validation drivers.
- Source distribution and wheel builds pass; both CLI entry points and import
  pass.
- Final DPA radiation-bundle/SPECTRA-PKA/activation subset: 156 passed, 1
  skipped.
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
- HTS YBCO recoil matrices are available and were folded, but reaction-channel
  fractions are not exposed by the current native-output parser and no
  production DPA or property-degradation claim is made.
- The evaluated reference solver omits energy-angle scattering transfer,
  evaluated neutron-to-photon matrices, and charged-particle transport.
- The OpenMC activation reference has no O-18 neutron reaction coverage and
  does not replace an operating-schedule, self-shielded activation analysis.
- OpenMC transports neutrons and photons here, not all charged secondaries.
