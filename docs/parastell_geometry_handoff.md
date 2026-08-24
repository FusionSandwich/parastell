# ParaStell reactor and magnet geometry handoff

## Purpose

This document hands off the current ParaStell reactor-to-magnet phase-space
work, with emphasis on the geometry problems encountered while constructing a
single reactor-sector DAGMC model containing the in-vessel build, magnet
casings, and homogenized winding packs.

The immediate geometry issue was apparent overlap between the reactor build
and magnets. That specific overlap has been resolved and independently checked.
Several broader production issues remain, particularly reproducible geometry
identity, representative material fidelity, response-preserving energy
condensation, and higher-fidelity deterministic transport through the explicit
HTS stack.

## Repository state

| Item | Value |
| --- | --- |
| Repository | `FusionSandwich/parastell` |
| Branch | `magnet-boundary-phase-space-production-20260822` |
| Starting SHA for this geometry investigation | `3b5ff2dbc81fddd4d7d372de003d4672181ac883` |
| Validated local checkpoint | `43aec9ba973e46e762c2ca0d3df8c74945643992` |
| Remote branch SHA | `3b5ff2dbc81fddd4d7d372de003d4672181ac883` |
| Push status | Not pushed because broader scientific gates remain incomplete |
| Upstream modifications | None |
| Pull request | None |

The validated checkpoint commit is:

```text
43aec9ba973e46e762c2ca0d3df8c74945643992
Validate overlap-free magnet phase-space workflow
```

## Scientific goal

The reactor-scale Monte Carlo model should represent the global fusion
environment without explicitly resolving every micrometre-scale HTS tape
layer. The intended workflow is:

```text
ParaStell VMEC equilibrium
-> spatially varying ParaStell D-T source mesh
-> reactor-sector DAGMC geometry
-> blanket, shield, vessel, structures, gaps, and homogenized magnets
-> OpenMC 0.16 coupled neutron-photon transport
-> closed boundary around a selected winding pack
-> correlated position, direction, energy, particle, surface, and weight records
-> deterministic explicit Cu/Ag/REBCO/buffer/substrate tape calculation
```

The canonical handoff must preserve continuous energy and direction. Grouped
energy, angular, and spatial fields are projections for deterministic solvers,
not replacements for the original surface-source records.

## Geometry construction path

The accepted combined model is generated through ParaStell's existing public
geometry paths:

```text
examples/wout_vmec.nc
-> Stellarator.construct_invessel_build(...)
-> Stellarator.construct_magnets_from_filaments(...)
-> CAD-to-DAGMC conversion
-> combined_reactor_magnet.h5m
```

The model contains:

| Component | Representation |
| --- | --- |
| Plasma chamber | DAGMC vacuum volume |
| First wall | In-vessel radial layer |
| Breeder/blanket | In-vessel radial layer |
| Back wall | In-vessel radial layer |
| Shield | In-vessel radial layer |
| Vacuum vessel | In-vessel radial layer |
| Magnet casing | Explicit global magnet solid |
| Winding pack | Explicit homogenized global magnet solid |
| HTS tape layers | Not present globally; resolved downstream |

The selected handoff volume is winding-pack volume `16`. Its closed envelope
contains surfaces `73, 74, 75, 76, 78, 79, 80, 81, 82, 83`.

## Original overlap problem

### Symptom

Earlier full 18-coil combined models reported overlaps between the outer
reactor structures and magnet casing volumes. Coarse models could appear to
pass even when a finer tessellation exposed a near-tangent intersection.

### Initial concern

It was reasonable to suspect that the spectral feature branch had broken the
basic ParaStell geometry kernel. A clean comparison was therefore made against
the untouched fork `main` geometry rather than continuing to tune the feature
model in isolation.

### Clean basic ParaStell reference

The untouched public basic example was generated from fork `main` at:

```text
de7d2978ff314b060ca2e6b10745a034e8b2a3c4
```

Reference result:

| Check | Result |
| --- | ---: |
| Combined volumes | 23 |
| Combined surfaces | 140 |
| Magnet volumes | 18 |
| Magnet surfaces | 124 |
| Checked edges | 11,772 |
| Unmatched edges | 0 |
| Watertight | Pass |
| Overlaps | 0 |

Reference artifact SHA-256:

```text
3634effa862b7092fb5a62c0ecbd33adc74f83bb0d37df4c6ea62bb119cc62d9
```

This established that the core filament magnet construction and the basic
in-vessel-plus-magnet assembly were sound.

## Root causes found

### Hidden casing default

The combined spectral builder silently defaulted to a `5 cm` magnet casing.
The basic public example did not apply that same hidden expansion. This made
the compared geometries physically different even when the visible coil width
and thickness looked identical.

The builder now defaults to zero casing unless casing is explicitly configured.
The production example explicitly requests:

```yaml
magnet_coils:
  width: 40.0
  thickness: 50.0
  case_thickness: 5.0
```

ParaStell's width and thickness parameters are the outer magnet dimensions.
The casing is subtracted internally to form the winding pack. They must not be
interpreted as winding-pack dimensions and expanded again.

### Insufficient physical clearance

The prior production configuration used a `50 cm` shield. With the explicit
`5 cm` casing and fine CAD-to-DAGMC faceting, this created an actual or
near-tangent intersection between the vacuum-vessel region and one magnet
casing.

Controlled reconstructions showed:

| Configuration | Faceting | Result |
| --- | --- | --- |
| Shield 50 cm, explicit casing | coarse | Overlap present |
| Shield 40 cm, explicit casing | 20-50 cm | No reported overlap, not trusted |
| Shield 40 cm, explicit casing | 5-20 cm | One near-tangent overlap |
| Shield 39 cm, explicit casing | 5-20 cm | Zero overlaps |

The accepted example uses a `39 cm` shield and retains approximately `1 cm`
clearance from the `5 cm` casing.

### Coarse faceting masked the problem

The former `20-50 cm` CAD-to-DAGMC mesh was adequate as a software smoke
setting but could hide a near-tangent intersection. Production defaults are now:

```text
minimum mesh size: 5 cm
maximum mesh size: 20 cm
```

Geometry acceptance must use the finer setting and the independent overlap
tool. Renderer appearance or a coarse mesh is not sufficient evidence.

## Attempts that did not solve the problem

### Shrinking the coil cross-section

A `30 x 40 cm` coil with a `5 cm` casing was tested against the `50 cm` shield.
This reduced the physical conductor dimensions and did not preserve the
intended ParaStell magnet semantics. It was rejected as the production fix.

### Trusting the coarse overlap result

The `40 x 50 cm` magnet, `5 cm` casing, and `40 cm` shield appeared clean under
coarse faceting. Fine `5-20 cm` faceting exposed a remaining casing-to-vessel
intersection. The coarse result was rejected.

### Separate reactor and magnet calculations

Previous work transported the reactor sector and a magnet-attached model in
separate OpenMC calculations. Those runs were useful prototypes, but they do
not prove navigation, shielding, secondary-photon transport, or current
closure in one combined physical model. They are not the production gate.

### Using the wrong Python bindings

The container initially used the official OpenMC `0.16.0` executable with
OpenMC `0.15.3` Python bindings. Transport completed, but reopening the
statepoint failed when the 0.16 `ReactionFilter` was encountered. This mixed
environment was rejected.

The accepted environment uses:

| Item | Accepted value |
| --- | --- |
| Python | 3.12.13 |
| OpenMC Python | 0.16.0 |
| OpenMC executable | 0.16.0 |
| OpenMC Git SHA | `617d35a5063c57796b43428bc401e627d2011046` |
| OpenMC Python environment | `/opt/openmc-v0.16.0-venv` |
| DAGMC | Enabled |
| MPI | Not enabled |
| OpenMP | 4 threads |

The exporter also now reads the named directional-current tally directly from
statepoint HDF5. An unrelated filter can no longer prevent boundary export.

## Accepted combined geometry result

Final OpenMC 0.16 production directory:

```text
C:\Users\joshu\OneDrive\Documents\ChatGPT\Parastell-validation-basic-20260823\production_transport_openmc016_final
```

Final geometry result:

| Check | Result |
| --- | ---: |
| Volumes | 42 |
| Surfaces | 267 |
| Checked edges | 48,132 |
| Unmatched edges | 0 |
| Unsealed surfaces | 0 |
| Unsealed volumes | 0 |
| Watertight | Pass |
| Overlaps | 0 |
| Lost particles | 0 |
| DAGMC navigation errors | 0 |

The independent tools reported:

```text
check_watertight:
0/48132 unmatched edges
0/267 unsealed surfaces
0/42 unsealed volumes

overlap_check -p 1:
No overlaps were found.
```

Final combined DAGMC SHA-256:

```text
22e30b10d5f57f8a12e8b6238287f40354b7e31d1513c6b552c96b81dabbeff3
```

## Surface-normal issue found during closure

One neutron crossing on surface `82` was initially assigned the opposite
entry/exit sense relative to OpenMC's `MuSurfaceFilter` result. This was not an
overlap and did not indicate a broken surface. Surface `82` is a valid one-sided
winding-pack end face.

The physical outward normal was correct, but `openmc_normal_sign` had been left
at `+1`. DAGMC volume `16` occupies the forward sense on surfaces `82` and
`83`, so OpenMC's native facet normal must be multiplied by `-1` to obtain the
physical outward normal.

The extractor now derives the conversion from the target volume's exact DAGMC
forward/reverse sense:

```text
target is reverse volume -> openmc_normal_sign = +1
target is forward volume -> openmc_normal_sign = -1
```

After correction, every populated surface passes entering, leaving, net, and
total-current closure without renormalization.

## Real OpenMC transport result

The accepted calculation used one combined reactor-plus-magnet model.

| Quantity | Result |
| --- | ---: |
| Histories | 500,000 |
| Batches | 10 |
| Particles per batch | 50,000 |
| OpenMP threads | 4 |
| Elapsed time | 248.39 s |
| Leakage | 0.20184 +/- 0.00056 |
| Surface-source records | 1,359 |
| Neutron records | 1,149 |
| Photon records | 210 |
| Lost particles | 0 |
| Navigation errors | 0 |

The source is ParaStell's VMEC-based spatial source, not a uniform or
monoenergetic substitute.

| Source property | Result |
| --- | ---: |
| Source mesh | 3 x 9 x 9 smoke mesh |
| Tetrahedra | 512 |
| Physical D-T rate | `2.414553994787296e20 /s` |
| Ion temperature range | 1,437.5 to 8,625 eV |
| D-T mean energy range | 14.05765 to 14.08059 MeV |
| D-T width range | 90.32 to 221.59 keV |

Coupled photon evidence:

| Evidence | Result |
| --- | ---: |
| Neutron-filtered photon production | `0.000914 +/- 0.0000509` per source |
| Transported photon crossings | 210 |
| Photon collision records with parent IDs | 3,169 |
| Electron production tally | 0 in this run |
| Positron production tally | 0 in this run |

## Same-run surface-current integrity closure

The surface bank is not scaled to force agreement. The bank and tally are
accumulated through separate OpenMC output mechanisms, but from the same
histories. This is a strong bookkeeping and software-integrity check, not an
independent statistical comparison. Their covariance is unavailable, so their
uncertainties must not be combined into an independent z-score.

| Quantity | OpenMC tally | Surface bank | Difference | Tally uncertainty |
| --- | ---: | ---: | ---: | ---: |
| Neutron entering | 0.001070 | 0.001070 | -2.17e-19 | 9.70e-05 |
| Neutron leaving | 0.001228 | 0.001228 | -2.17e-19 | 9.66e-05 |
| Neutron net outward | 0.000158 | 0.000158 | 0 | 1.37e-04 |
| Photon entering | 0.000210 | 0.000210 | -1.08e-19 | 2.92e-05 |
| Photon leaving | 0.000210 | 0.000210 | -1.08e-19 | 2.89e-05 |
| Photon net outward | approximately 0 | approximately 0 | 0 | 4.11e-05 |

All values are per OpenMC source history. Every face and whole-envelope
comparison passes the three-sigma criterion.

## Boundary handoff result

| Property | Result |
| --- | --- |
| Schema | `parastell.magnet_boundary_source/v2.0.0` |
| Handoff SHA-256 | `e299d77213908cff108bb1d3bcfb14e3e586307d84c9ca224873b7724e7be63d` |
| File size | 1,101,998 bytes |
| Projection shape | `10 x 9 x 42 x 416 x 2 x 3` |
| Mu bins | 26 |
| Azimuth bins | 16 |
| Particle axes | neutron and photon |
| Surface axes | all 10 selected envelope surfaces |
| Empty bins | Explicitly retained |
| Compression | Lossless HDF5 gzip |

The dense fixed-grid projection was initially 151.57 MB because 99.99% of its
bins were zero. Lossless compression reduced it to approximately 1.10 MB
without changing the schema, indexing, or values.

## Deterministic HTS replay result

The accepted handoff was consumed independently of the reactor geometry by an
explicit verification stack:

```text
Cu
Ag
REBCO
buffer stack
Hastelloy substrate
backside Cu
solder
insulation
```

| Replay quantity | Result |
| --- | ---: |
| Incoming current | 0.001280 per source |
| Transmitted current | 0.0012778878 per source |
| Removed current | 2.11218e-06 per source |
| Balance residual | 2.54e-19 per source |
| Relative balance error | 1.98e-16 |

This proves the software coupling and directional replay contract. It is not
yet a full radiation-damage or complete multigroup-scattering prediction.

## Software validation

The final suite was run with matching OpenMC 0.16 Python bindings:

```text
163 passed, 8 skipped
```

Additional gates:

| Gate | Result |
| --- | --- |
| Black | 75 files unchanged |
| `git diff --check` | Pass |
| Python compilation | Pass |
| sdist | Pass |
| wheel | Pass |
| `parastell -h` | Pass |
| `python -m parastell -h` | Pass |
| `import parastell` | Pass |

The recurring OpenMC duplicate-ID warnings are global Python object-registry
warnings in tests and model preparation. They did not produce duplicate DAGMC
entities, invalid tally IDs in the written model, overlaps, lost particles, or
navigation failures. They remain worth cleaning up to reduce diagnostic noise.

## Files changed in the validated checkpoint

```text
docs/magnet_boundary_envelope.md
examples/config.yaml
parastell/combined_openmc16_model.py
parastell/dagmc_envelope.py
parastell/magnet_boundary_envelope.py
parastell/magnet_handoff_cli.py
parastell/openmc16_export.py
tests/test_dagmc_envelope.py
tests/test_magnet_boundary_envelope.py
tests/test_magnet_phase_space_production.py
```

No core aperture, port-surgery, native PyDAGMC writer, or tetrahedralization
kernel was modified during this geometry correction.

## Important remaining geometry issues

### Bytewise DAGMC hashes are not reproducible

Equivalent regenerations produced the same topology counts, surface IDs,
volume IDs, closure results, and transport population but different raw H5M
SHA-256 values. MOAB entity ordering or file metadata is likely nondeterministic.

The current workflow records the exact file hash used by each transport run,
which is valid provenance. It does not yet provide a canonical geometry hash
that proves two independently generated files are topologically and
geometrically equivalent.

Recommended next implementation:

```text
canonical DAGMC fingerprint = hash(
  sorted volume IDs, material tags, volumes,
  sorted surface IDs, senses, areas, centroids,
  quantized sorted triangle coordinates/connectivity
)
```

The fingerprint tolerance must be explicit and smaller than the accepted
faceting tolerance.

### Clearance is configuration-specific

The `1 cm` clearance resolves this representative 90-degree sector. It should
not be treated as a universal stellarator design rule. Other VMEC equilibria,
coil sets, radial builds, ports, or full-device models need the same fine-mesh
overlap gate.

### Port-inclusive production geometry is not demonstrated here

This accepted workflow is deliberately port-free. Port geometry work exists on
separate branches and must not be merged into this spectral line without its
own full-assembly overlap, watertightness, and transport evidence.

### Material fidelity still needs review

The global winding pack is intentionally homogenized. Its composition must be
audited against the intended conductor, stabilizer, substrate, insulation,
coolant, and structural volume fractions before design conclusions are drawn.
The global model should remain homogenized; explicit tape layers belong in the
downstream deterministic model.

## Remaining scientific work

### Response-preserving energy condensation

The accepted transport uses a 7-group neutron smoke projection and a separate
42-group photon smoke projection while retaining continuous energies in the
canonical bank. Production work still needs:

```text
CCFE-709 neutron projection for SPECTRA-PKA interoperability
UKAEA-1102 neutron audit projection
response-adaptive neutron deterministic grid
response-adaptive photon deterministic grid
group-count versus response-error Pareto study
```

Protected responses should include current, surface flux, heating, capture,
elastic and inelastic scattering, gas production, damage energy, and
species-resolved PKA spectra.

### Higher-fidelity deterministic transport

The current internal solver exactly verifies uncollided/removal transport and
particle balance. It does not yet apply:

```text
within-group scattering redistribution
group-to-group downscatter
layer neutron-to-photon production
material-specific energy-dependent cross sections
charged-particle transport
temperature-dependent defect retention
```

These limitations must remain visible in output metadata.

### PKA and activation coupling

The next coupled analyses should consume the validated neutron field rather
than scalar DPA:

```text
CCFE-709 spectrum -> SPECTRA-PKA recoil matrices
OpenMC material spectra -> ALARA activation workflow
OpenMC material spectra -> FISPACT-II activation workflow
```

These tools require separate nuclear-data provenance, irradiation schedules,
cooling schedules, uncertainty treatment, and isotope inventories.

### Photon birth provenance

The neutron-filtered `ParticleProductionFilter` independently demonstrates
secondary-photon production, and collision tracks preserve photon parent IDs.
The available collision format does not directly provide a complete photon
birth record tied to a specific parent-neutron reaction for every photon.
Do not infer that diagnostic collision records are the primary production
estimator.

### Statistical covariance

Aggregate tally uncertainty is formed by quadrature over reported tally-bin
standard deviations because OpenMC statepoints do not provide the required
cross-bin covariance. This limitation should be stated whenever grouped or
whole-envelope uncertainty is reported.

## Recommended continuation order

1. Add a canonical DAGMC topology and geometry fingerprint and prove it is
   stable across two fresh equivalent builds.
2. Turn the clean fork-main versus feature-branch comparison into an automated
   regression summary containing volumes, surfaces, senses, materials, areas,
   and overlap results.
3. Remove or isolate OpenMC global-ID registry warnings while proving written
   tally/filter IDs remain unchanged.
4. Repeat the combined-model gate for at least one additional winding pack and
   one complete multi-magnet export.
5. Generate CCFE-709 and UKAEA-1102 neutron projections from the continuous
   bank without constructing impractically large uncompressed dense arrays.
6. Implement the response-preserving neutron and photon condensation study.
7. Upgrade deterministic replay with material-specific scattering and
   neutron-to-photon coupling data.
8. Run SPECTRA-PKA comparisons for the configured REBCO constituent set.
9. Connect the validated OpenMC spectra to the existing ALARA and FISPACT-II
   activation adapters.
10. Push the spectral branch only after the selected mandatory scientific gates
    are declared complete. Do not merge to `main` and do not open a PR as part
    of this handoff.

## Reproduction inputs

```text
Configuration:
C:\Users\joshu\OneDrive\Documents\ChatGPT\Parastell-magnet-boundary-phase-space-production\examples\config.yaml

VMEC equilibrium:
C:\Users\joshu\OneDrive\Documents\ChatGPT\Parastell-magnet-boundary-phase-space-production\examples\wout_vmec.nc

Coils:
C:\Users\joshu\OneDrive\Documents\ChatGPT\Parastell-magnet-boundary-phase-space-production\examples\coils.example

Cross sections:
C:\Users\joshu\Documents\2026_DPA\openc-hts-dpa\.data\openmc\cross_sections.xml

Container:
parastell-openmc:0.16.0

Python environment in container:
/opt/openmc-v0.16.0-venv/bin/python
```

## Validation artifacts

```text
Combined DAGMC:
C:\Users\joshu\OneDrive\Documents\ChatGPT\Parastell-validation-basic-20260823\production_transport_openmc016_final\combined_reactor_magnet.h5m

Statepoint:
C:\Users\joshu\OneDrive\Documents\ChatGPT\Parastell-validation-basic-20260823\production_transport_openmc016_final\statepoint.10.h5

Surface source:
C:\Users\joshu\OneDrive\Documents\ChatGPT\Parastell-validation-basic-20260823\production_transport_openmc016_final\surface_source.h5

Boundary handoff:
C:\Users\joshu\OneDrive\Documents\ChatGPT\Parastell-validation-basic-20260823\production_transport_openmc016_final\handoffs\magnet_boundary_winding-pack-16.h5

Deterministic replay:
C:\Users\joshu\OneDrive\Documents\ChatGPT\Parastell-validation-basic-20260823\production_transport_openmc016_final\hts_multilayer_replay.h5

Machine-readable validation summary:
C:\Users\joshu\OneDrive\Documents\ChatGPT\Parastell-validation-basic-20260823\production_transport_openmc016_final\scientific_validation_summary.json
```

Validation summary SHA-256:

```text
e788ea15eff9239c2bcbe34c43a34765131f9a75a3483dc44dce920c23d5e63a
```

## Acceptance status

| Gate | Status | Evidence or limitation |
| --- | --- | --- |
| Matching OpenMC 0.16 environment | Pass | Python and executable both 0.16.0 |
| ParaStell spatial D-T source | Pass | 512-tet spatial and temperature-dependent source |
| Combined reactor and magnets | Pass | 42-volume single DAGMC model |
| Watertight geometry | Pass | Zero unmatched edges and unsealed entities |
| No overlaps | Pass | Independent `overlap_check -p 1` |
| Closed winding-pack envelope | Pass | 10 surfaces with exact senses |
| Continuous neutron/photon phase space | Pass | 1,359 correlated records |
| Independent current closure | Pass | Every populated face and whole envelope |
| Secondary photons | Pass for production and transport | Production tally, crossings, and parent-bearing collision records |
| Explicit multilayer replay | Pass as verification model | Eight layers and machine-precision balance |
| Response-preserving energy architecture | Pending | Condensation and Pareto study not complete |
| Full scattering deterministic solver | Pending | Current model is removal/uncollided reference |
| SPECTRA-PKA production comparison | Pending | CCFE-709 comparison not complete |
| Canonical reproducible geometry hash | Pending | Raw H5M hashes differ between equivalent builds |
| Branch push | Withheld | Broader scientific gates remain incomplete |
