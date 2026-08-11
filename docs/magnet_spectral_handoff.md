# Reactor-to-magnet spectral handoff

## Purpose

The magnet spectral handoff connects a coarse, reactor-scale ParaStell/OpenMC
model to a separate deterministic model that resolves the individual layers of
an HTS winding pack. The reactor calculation remains responsible for the full
stellarator geometry, shielding, source distribution, and secondary-particle
production. The local deterministic calculation receives the radiation field
crossing a selected magnet interface and can then transport that field through
explicit copper, silver, REBCO, buffer, substrate, solder, insulation, and
structural layers.

This implementation adds two complementary products:

1. Energy-resolved OpenMC tallies for reproducible, low-storage spectral
   summaries and absolute normalization.
2. An optional native OpenMC surface-source bank, converted to a documented
   ParaStell HDF5 phase-space contract for high-fidelity downstream boundary
   conditions.

The normal tally path is the default scientific product. The phase-space bank
is opt-in because its size can be large and OpenMC may limit the number of
records retained.

## Required reactor-scale model

The handoff operates on an existing OpenMC model. That model is expected to
contain:

- a ParaStell-generated DAGMC reactor or sector geometry;
- materials and a fixed neutron source, normally the ParaStell tetrahedral
  source mesh;
- OpenMC settings suitable for the reactor-scale calculation;
- OpenMC cell and surface IDs identifying each selected magnet interface.

The first implementation deliberately keeps interface metadata external to the
DAGMC file. Cell IDs, surface IDs, volumes, areas, normal orientations, and
local coordinate frames are supplied in the handoff YAML. This avoids assuming
one metadata-tagging convention. Automatic discovery from DAGMC/UW² metadata is
a separate integration step.

## Generated OpenMC tallies

Each configured magnet region can receive the following deterministic-ID
Tallies:

| Role | Filters | Score | Primary use |
|---|---|---|---|
| `cell_flux` | cell, particle, energy, optional time | `flux` | Spectrum within the coarse magnet region |
| `mesh_flux` | regular/unstructured mesh, particle, energy, optional time | `flux` | Spatially resolved spectrum around or within the magnet |
| `boundary_current` | surface, particle, energy, surface-relative mu, optional time/global angles | `current` | Incoming/outgoing interface spectrum and phase-space normalization |
| `heating` | cell, particle, energy, optional time | `heating` | Nuclear heating by incident energy and particle |
| `damage_energy` | cell, neutron, energy, optional time | `damage-energy` | Damage-energy source for downstream DPA/PKA work |
| `gas_production` | cell, neutron, energy, optional time | selected production scores | H/He production source terms |

OpenMC 0.15.1 or later is required because `MuSurfaceFilter` is used to split
surface crossings by the cosine relative to the OpenMC surface normal. The
statepoint exporter also includes a compatibility path for the incomplete
unstructured-mesh DataFrame support in OpenMC 0.15.x.

### Incoming and outgoing convention

`MuSurfaceFilter` uses the OpenMC surface normal. The YAML field
`surface_normal_signs` maps that normal to the magnet-outward normal:

- `+1`: the OpenMC normal points out of the selected magnet region;
- `-1`: the OpenMC normal points into the selected magnet region.

A negative cosine relative to the magnet-outward normal is labelled
`incoming`; a positive cosine is labelled `outgoing`. If no orientation sign is
supplied, the tally export retains the raw positive/negative-mu label without
claiming a magnet-relative direction.

Optional `polar_bounds_rad` and `azimuthal_bounds_rad` add global-direction
filters. These can create very large tallies. The phase-space bank is generally
preferable when the local deterministic solver needs the full direction vector.

## Surface-source phase space

`configure_surface_source` uses the native OpenMC surface-source writer:

- `incoming` selects `cellto`;
- `outgoing` selects `cellfrom`;
- `both` selects `cell`;
- `all` selects all crossings of the configured surfaces without a cell-side
  selector.

For a single run that must capture both directions, use `both`. Separate
incoming and outgoing runs are also possible. Only one OpenMC
`surf_source_write` definition can be active in a model, so multiple independent
magnet interfaces require separate runs or a combined surface selection that is
subsequently separated by surface ID.

The converted file defaults to `hts_phase_space_source.h5` and contains:

- global and magnet-local position in centimetres;
- global and magnet-local direction unit vectors;
- energy, time, statistical weight, delayed group, and surface ID;
- raw OpenMC particle code plus normalized particle name and PDG code;
- source-file and source-record indices plus a unique output `record_id`;
- source-region, magnet, coil, and winding-pack identifiers;
- optional global/local outward normals and `mu_outward`;
- an `incoming`, `outgoing`, `grazing`, or `unknown` direction label;
- a `direction_label_basis` stating whether that label came from a configured
  geometric normal, the OpenMC `cellto`/`cellfrom` selector, or neither;
- complete configuration and source-file metadata.

OpenMC 0.15 source files encode particle type with a small enum, whereas newer
OpenMC source files use PDG identifiers. ParaStell preserves the raw value and
normalizes both representations.

The native OpenMC source-bank format does **not** provide true history ID,
parent ID, cell ID, or material ID. The exporter lists these unavailable fields
rather than fabricating them. A future OpenMC/core extension would be required
for exact event genealogy.

`surface_outward_normals_global` is optional. A constant normal is valid for a
planar interface or a deliberately local planar proxy. It is not an exact
normal field for a curved coil surface. For an `incoming` or `outgoing` bank,
records without a configured normal still inherit the direction guaranteed by
OpenMC `cellto` or `cellfrom`; `mu_outward` remains unavailable. For `both` or
`all`, the label remains unknown without a geometric normal. The full direction
vector and the surface-relative-mu tally remain available in every case.

## Normalization

The number of records in a surface-source file is not an absolute crossing
rate. OpenMC can cap and sample the bank, and `max_particles` applies per MPI
process. The companion `boundary_current` tally is therefore authoritative.

For a source rate \(S\) in particles per second:

- cell or mesh flux is the raw track-length score divided by volume and
  multiplied by \(S\);
- boundary current density is the raw crossing score divided by surface area
  and multiplied by \(S\);
- heating power is the raw heating score multiplied by the electronvolt-to-
  joule conversion and by \(S\).

When `normalization.source_rate_per_s` is omitted, all exported results remain
per source particle. This is appropriate when full-device/sector source
normalization has not yet been fixed.

For a deterministic boundary condition, the surface-current tally supplies the
absolute normalization while the phase-space records supply the sampled joint
distribution in position, direction, particle, energy, and time. A downstream
sampler should preserve that separation rather than treating each stored record
as one physical particle per second.

## Configuration

A complete example is provided at
`examples/magnet_spectral_handoff.yaml`. The machine-readable schema is at
`schemas/magnet_spectral_handoff.schema.json`.

The most important region fields are:

- `cell_ids`: OpenMC cells used for volume tallies;
- `surface_ids`: the interface surfaces used for current tallies and banking;
- `phase_space_cell_id`: required when the directional cell selector is not
  unambiguous from a single `cell_ids` entry;
- `cell_volumes_cm3` or `volume_cm3`: required for normalized cell flux;
- `surface_areas_cm2`: required for current density;
- `surface_normal_signs`: required for magnet-relative incoming/outgoing labels;
- `coordinate_frame`: origin and right-handed local axes;
- `mesh`: optional regular or unstructured spatial tally mesh. Relative
  unstructured-mesh filenames are resolved from the YAML file directory.

Tally IDs are generated deterministically from `tally_id_base`. The default
range begins at 9,000,000. A model that already uses this range must supply a
different base.

## Command-line workflow

### Prepare an OpenMC model

```bash
parastell-magnet-handoff prepare \
  --config examples/magnet_spectral_handoff.yaml \
  --model reactor_openmc_model \
  --output-dir reactor_magnet_run \
  --region winding_pack_01 \
  --direction both
```

`--model` may be a combined `model.xml` or a directory containing the usual
OpenMC XML files. ParaStell loads it from its own directory so relative DAGMC,
source, and other external paths are resolved before the augmented model is
exported elsewhere. The command appends the handoff tallies, enables photon
transport when photons are requested, configures HDF5 surface-source writing,
exports the augmented model, and writes a manifest.

### Prepare, run, and export in one command

```bash
parastell-magnet-handoff prepare \
  --config examples/magnet_spectral_handoff.yaml \
  --model reactor_openmc_model \
  --output-dir reactor_magnet_run \
  --region winding_pack_01 \
  --direction both \
  --run \
  --threads 16
```

For a simple MPI launch, add `--mpi-processes 64` and optionally
`--mpi-launcher srun`. Large reactor calculations should normally be submitted
through the site's scheduler using the prepared XML files, then post-processed
separately.

A fresh run directory remains preferable. For a local `--run`, the command
snapshots existing `surface_source*.h5` files and accepts only newly created or
modified banks, preventing unchanged files from being treated as current
output.

### Post-process an HPC run

```bash
parastell-magnet-handoff postprocess \
  --config examples/magnet_spectral_handoff.yaml \
  --statepoint reactor_magnet_run/statepoint.100.h5 \
  --surface-source reactor_magnet_run/surface_source.h5 \
  --output-dir reactor_magnet_handoff \
  --region winding_pack_01 \
  --selection both
```

Multiple MPI surface-source files can be passed after `--surface-source`.

The main outputs are:

- `magnet_spectra.h5`: tidy tally rows, derived units, uncertainty, and
  normalization columns;
- `hts_phase_space_source.h5`: phase-space records and local-coordinate
  metadata;
- `magnet_spectral_handoff_manifest.json`: configuration, IDs, units, and data
  contract.

The two HDF5 products are written through temporary files and atomically
renamed after successful completion, so a failed conversion does not leave a
partially written final product.

## Programmatic use

```python
import openmc

from parastell import MagnetSpectralHandoff

handoff = MagnetSpectralHandoff.from_yaml("handoff.yaml")
model = openmc.Model.from_model_xml("model.xml")

handoff.attach_to_model(model)
handoff.configure_surface_source(
    model.settings,
    "winding_pack_01",
    direction="both",
    max_particles=2_000_000,
    max_source_files=8,
)
model.export_to_xml(directory="reactor_magnet_run")
```

After OpenMC finishes:

```python
handoff.export_statepoint(
    "reactor_magnet_run/statepoint.100.h5",
    "reactor_magnet_handoff/magnet_spectra.h5",
)
handoff.export_surface_source(
    [
        "reactor_magnet_run/surface_source.1.h5",
        "reactor_magnet_run/surface_source.2.h5",
    ],
    "reactor_magnet_handoff/hts_phase_space_source.h5",
    region_name="winding_pack_01",
    selection="both",
)
```

## Executable scientific validation gates

The branch includes an executable, fail-closed validation sequence rather than
only unit-level arithmetic checks:

```bash
parastell-magnet-handoff-validate \
  --output-dir magnet_handoff_validation \
  --assets-dir tests/files_for_tests \
  --planar-particles 5000 \
  --sector-particles 2000 \
  --threads 2
```

The sequence performs three coupled checks:

1. **Planar interface closure.** A monodirectional 14.1 MeV neutron source
   crosses an exactly planar void interface. OpenMC's direction-resolved
   current tally, native surface-source bank, and ParaStell HDF5 export must
   close to one crossing per source particle.
2. **Real ParaStell sector.** The repository's DAGMC sector asset,
   tetrahedral source mesh, source strengths, VMEC equilibrium, and test
   cross-section library are used in an actual OpenMC run. The first-wall
   surface is explicitly classified as a magnet-interface software proxy; it
   is not represented as a resolved magnet.
3. **Explicit multilayer replay.** The weighted boundary measure is replayed
   through separately represented copper, silver, REBCO, buffer, Hastelloy,
   copper, solder, and insulation layers. The output axes are
   `x_bin,y_bin,layer,particle,energy_group`. A transparent response library
   verifies current closure, and a separate constant-coefficient problem is
   checked against the exact exponential characteristic solution.

The surface-source record count is never interpreted as an absolute source
rate. Records are reweighted within particle/energy/surface bins to the
companion OpenMC current tally. For the curved sector gate, a per-record VMEC
reference-surface normal is reconstructed and stored with the exported phase
space before replay.

Each run writes `validation_attestation.json`, per-gate JSON reports, OpenMC
statepoints and source banks, normalized handoff files, and multilayer HDF5
response files. The GitHub workflow uploads these products as validation
evidence and exits non-zero on any failed tolerance.

The multilayer operator is deliberately scoped to exact planar
uncollided/removal characteristics. Non-zero scattering redistribution,
secondary-particle production, physical material-response coefficients, PKA
recoil matrices, and defect-retention physics must be supplied by later
response modules; the validation code does not fabricate them.

## Deliberate first-version limits

This branch does not yet:

- discover magnet cells/surfaces automatically from DAGMC metadata;
- build the heterogeneous HTS tape geometry itself;
- convert damage energy or incident spectra directly into PKA recoil matrices;
- calculate NRT-DPA, arc-DPA, defect survival, or superconducting performance;
- preserve unavailable OpenMC event genealogy;
- provide an exact curved-surface normal at every banked crossing.

Those are downstream or follow-on capabilities. The implemented boundary is
intended to make the reactor calculation reproducible and to provide a stable,
versioned input contract for those later stages.
