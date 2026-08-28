# Parametric Geometry Infrastructure

## Scope

This checkpoint provides a device-neutral, create-only ParaStell source-CAD
lane. It does not select or qualify an H5M and it does not run OpenMC. The
existing WISTELL-D recovery/acceptance lane remains fail-closed and separate.

The public entry points are:

- `parastell.parametric_geometry.load_plan`
- `parastell.parametric_geometry.resolve_geometry`
- `parastell.parametric_geometry.build_source_cad`
- `scripts/build_parametric_geometry.py`

The schema is `parastell.parametric_geometry_plan/v1.0.0`. Every input is
role-named, relative to an explicit input root, and SHA-256 bound. A plan may
use arbitrary device filenames and one of these sector definitions:

- one stellarator half-period, when live VMEC metadata confirms stellarator
  symmetry;
- one or more complete field periods, calculated from live `nfp`;
- an explicit 0-degree-anchored direct sector.

Nonzero phase origins remain blocked until ParaStell has a qualified
phase-origin CAD implementation.

## Geometry controls

The ordered radial build accepts constant thicknesses or named NPY/NPZ arrays.
Resolved arrays must exactly match the requested CAD grid, contain only finite
positive values, and close at the poloidal seam. The chamber material and every
layer material are explicit.

Exactly one magnet representation is selected:

- `radial_envelope`: the final radial-build layer is the homogenized global
  magnet region;
- `swept_filaments`: a hash-bound coil file plus explicit width, thickness,
  material, sampling, header-line, and scale conventions drives ParaStell's
  filament sweeps.

The modes cannot share their magnet-specific fields. The core filament-sector
predicate was corrected so ordinary angular intervals use intersection logic
and intervals crossing 0/360 degrees use union logic. This is necessary for
non-WISTELL-D coil files; the old predicate accepted almost every coil.

Ports, imported physical solids, post-build transforms, and overwrite outputs
are schema-invalid. The Docker command is an argument vector rather than a
shell command, binds source and inputs read-only, disables networking, uses an
immutable image ID, and applies equal hard memory and memory-plus-swap caps.
The source-CAD subcommand rejects host execution and requires the exact
container attestation emitted by `docker-command`. The command is printed by
default; execution additionally requires the CLI's explicit `--execute` flag.
The host must be a clean checkout at the declared revision, and the container
recomputes a canonical digest over every Python source file under `parastell/`
plus the build CLI. A source change after command generation therefore fails
before the create-only output root is created.

## Concrete WISTELL-D full-period profile

`configs/wistell_d_parametric_full_period.json` is the concrete 90-degree
profile. A live read-only validation against
`D:\2026_DPA\wistell-d-parastell-90deg` passed:

- VMEC `nfp = 4`, so one full field period is exactly 90 degrees;
- VMEC `lasym = false`, explicitly interpreted as stellarator symmetry;
- direct CAD grid: 80 by 90;
- all three input SHA-256 values match;
- all eight thickness matrices are 80 by 90, finite, strictly positive, and
  poloidally closed;
- continuous homogenized magnet envelope thickness: 30 cm;
- source mesh definition: 11 by 81 by 61.

The resolved plan SHA-256 is
`22c8feb53e3a29cfff0fc27aaec96cdb28a4369b486503dbc21aac95fdd3ae5f`.
This validation proves the input/configuration contract only. It does not
supersede pairwise CAD, imprint, native DAGMC, source-containment, or OpenMC
navigation gates.

## Use

Validate any profile without building geometry:

```text
python scripts/build_parametric_geometry.py validate \
  --config <plan.json> \
  --input-root <hash-bound-input-root>
```

Print the bounded Docker invocation without executing it:

```text
python scripts/build_parametric_geometry.py docker-command \
  --config <plan.json> \
  --input-root <input-root> \
  --output-root <new-create-only-root> \
  --repository-root <this-repository> \
  --source-revision <full-40-character-git-sha>
```

Adding `--execute` is a distinct authorization boundary. A successful source
CAD build is still labeled `SOURCE_CAD_BUILT_GEOMETRY_GATES_PENDING` and
`transport_eligible: false`.

## Remaining infrastructure gates

The reusable source-CAD specification/build layer is present. Before a model
may feed OpenMC, the workflow still needs to bind this generic lane to the
existing complete source-CAD pair audit, imprinted DAGMC export, native
watertight/overlap/sense audit, source-domain containment, and two-seed OpenMC
0.16 navigation smoke. Surface-current and downstream activation/replay
contracts remain consumers of an accepted geometry; they do not relax these
geometry gates.
