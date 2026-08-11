# Open-source COMSOL-ready fusion magnet models

ParaStell does not require COMSOL for these examples. The implementation
creates neutral CAD files, named solid parts, machine-readable manifests, and
small COMSOL model files for Java that import the geometry and save an `.mph`
model. COMSOL is required only for the final import, physics setup, meshing,
and solution.

The default dimensions are generic workflow-test values. They do not reproduce
proprietary or reactor-authoritative magnet geometry.

## Public model and method survey

The public COMSOL material separates into one downloadable tutorial model and
several fusion-specific papers or presentations that document methods but do
not expose redistributable `.mph` files.

| Resource | Public artifact | Reusable content |
| --- | --- | --- |
| COMSOL **Superconducting Wire**, Application ID 689 | COMSOL tutorial download, subject to COMSOL Access and licensing | Baseline Magnetic Field Formulation model for a nonlinear superconducting E-J law |
| Strasser and Friedel, **Electromagnetic Modeling of HTS Superconductors — A Benchmark Collection** (2025) | Poster | Twelve-model formulation map, including H, H-phi, T-A, homogenized T-A, CORC cable, and nonplanar stellarator examples |
| Amardas and Dwivedi, **Magneto-structural Analysis of Fusion grade Superconducting Toroidal Field Coils** (2009) | Paper | Magnetostatic-to-Lorentz-force-to-structural coupling for tokamak TF coils |
| Cocilovo, **Analysis of D-Shaped Toroidal Superconductive Coils for Medium Size Fusion Experiment Facility** (2018) | Paper and poster | D-shaped TF geometry and coupled magnetic/mechanical design workflow |
| De Marzi et al., **Electromagnetic Analysis of the Superconducting Magnet System of the Divertor Tokamak Test Facility** (2020) | Paper and poster | 3D TF plus 2D-axisymmetric PF/CS decomposition and field mapping to feeders |
| Spruijtenburg et al., **Modelling superconductor AC losses in the STEP TF magnet during plasma initiation** (2025) | Paper | Quasi-2.5D center-column slices and coupled H-H0-phi formulation |
| CFS, **Designing HTS-Based Magnets for Fusion Devices with COMSOL** (2024) | Keynote video | SPARC TF model-coil quench, AC-loss, Lorentz-load, and diagnostic workflow |
| Chan et al., **3D Mixed-Dimensional Quench Model of a High Aspect Ratio HTS Coated Conductor Tape** (2010) | Paper | Explicit thick layers, 2D superconducting film, and thin-interface resistance treatment |

The source pages are listed in each generated model manifest. The geometry in
this repository is newly generated and uses the sources only as method and
model-family precedents.

## Generated model families

### `tokamak_tf_d_shape`

A generic D-shaped toroidal-field winding pack. The recommended first model is
stationary Magnetic Fields coupled to Solid Mechanics through Lorentz loads.
A Heat Transfer interface can consume ParaStell nuclear-heating fields.

### `spherical_tokamak_tf`

A low-aspect-ratio TF surrogate partitioned into a named center column and an
outer return limb. The partition supports the quasi-2.5D approach used for STEP:
solve representative center-column slices with Magnetic Field Formulation and
supply the CS/PF background field separately.

### `central_solenoid_pf`

Six modular central-solenoid winding packs plus an upper and lower PF coil. The
geometry is directly suited to a 2D-axisymmetric electromagnetic model or to a
3D imported-geometry check. Current waveforms can be coupled through the
Electrical Circuit interface.

### `demountable_tf_joint`

A D-shaped TF surrogate with a separated coil body, joint blocks, and an
explicit thin contact layer. The contact layer can be assigned a measured or
parametric electrical contact resistivity and coupled to heat transfer and
contact-pressure mechanics.

## Generate the files

After installing ParaStell, run:

```bash
parastell-comsol-fusion-magnets \
  --design all \
  --output-dir comsol_fusion_magnet_models \
  --format step
```

Generate both STEP and STL outputs by repeating `--format`:

```bash
parastell-comsol-fusion-magnets \
  --design spherical_tokamak_tf \
  --output-dir comsol_fusion_magnet_models \
  --format step \
  --format stl
```

The Python module can also be invoked directly:

```bash
python -m parastell.comsol_fusion_magnets --design all
```

Each design directory contains:

- one named multi-part STEP assembly when STEP output is requested;
- one neutral CAD file per solid part;
- `model_manifest.json`, with dimensions, part volumes, bounding boxes,
  intended physics, references, and limitations;
- a COMSOL model file for Java that imports the STEP assembly and saves an
  `.mph` file.

STEP files declare metres as their geometry unit. STL files are unitless; treat
their coordinates as metres, as recorded in the manifest.

## Use inside COMSOL

The generated Java source follows the COMSOL model-file-for-Java pattern. It
accepts the STEP path as its first argument and the output `.mph` path as its
second argument. Compile or run it using the COMSOL API tools supplied with the
installed COMSOL version, or copy the import sequence into the Application
Builder. The importer deliberately stops after geometry creation. This avoids
silently assigning unvalidated materials, currents, mechanical constraints, or
superconducting constitutive laws.

For the electromagnetic model, add the relevant interface from the manifest.
For HTS transient loss calculations, use a measured or cited E-J law and a
field-, angle-, and temperature-dependent critical-current model appropriate to
the selected conductor. Do not use the generic geometry defaults as design
allowables.

## Connect the ParaStell radiation handoff

The generated manifests list four supported field classes:

1. energy-group neutron or photon flux;
2. signed boundary current by energy group;
3. nuclear heating;
4. species-resolved PKA source terms.

Import spatial fields as COMSOL interpolation functions, tables, or mapped
fields. Use General Extrusion operators when a lower-dimensional HTS slice is
fed by a 3D ParaStell field. Preserve the handoff coordinate system, energy-group
ordering, units, and normalization. For a quasi-2.5D model, retain the spatial
weight assigned to each center-column slice rather than averaging all slices
before the electromagnetic or thermal solve.

## Validation boundary

The open-source validation covers:

- parameter validation;
- positive-volume solid generation for every named part;
- deterministic STEP and STL export;
- JSON manifest generation;
- COMSOL Java importer generation;
- ordinary ParaStell packaging and regression tests.

The Java files have not been compiled or solved in COMSOL in open-source CI.
The next proprietary-software gate is a geometry-import smoke test followed by
a mesh-only check. Electromagnetic, thermal, structural, and quench results must
remain explicitly unvalidated until those studies are independently configured
and benchmarked.
